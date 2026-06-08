from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

from mjlab.entity import Entity
from mjlab.envs.mdp.actuator_torque import (
  read_torque_and_limits,
  resolve_torque_channels,
)
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _posture_speed_masks(
  command: torch.Tensor,
  *,
  walking_threshold: float,
  running_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Compute turning-first posture regime masks from the current command."""
  linear_speed = torch.norm(command[:, :2], dim=1)
  angular_speed = torch.abs(command[:, 2])
  total_speed = linear_speed + angular_speed

  turning_mask = (linear_speed < walking_threshold) & (
    angular_speed > walking_threshold
  )
  standing_mask = (~turning_mask) & (total_speed < walking_threshold)
  walking_mask = (
    (~turning_mask)
    & (total_speed >= walking_threshold)
    & (total_speed < running_threshold)
  )
  running_mask = (~turning_mask) & (total_speed >= running_threshold)

  return (
    standing_mask.float(),
    turning_mask.float(),
    walking_mask.float(),
    running_mask.float(),
  )


def foot_height(
  env: ManagerBasedRlEnv,
  sensor_name: str | SceneEntityCfg | None = None,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Return per-foot height using either a terrain sensor or asset site positions.

  ``sensor_name`` preserves the historical observation-manager contract used by
  velocity configs. Passing a ``SceneEntityCfg`` positionally still selects the
  asset-based site-height path.
  """
  if isinstance(sensor_name, SceneEntityCfg):
    asset_cfg = sensor_name
    sensor_name = None

  if sensor_name is not None:
    sensor = env.scene[sensor_name]
    assert isinstance(sensor, TerrainHeightSensor), (
      f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
    )
    return sensor.data.heights

  resolved_asset_cfg = _DEFAULT_ASSET_CFG if asset_cfg is None else asset_cfg
  asset: Entity = env.scene[resolved_asset_cfg.name]
  return asset.data.site_pos_w[
    :, resolved_asset_cfg.site_ids, 2
  ]  # (num_envs, num_sites)


def foot_height_terrain(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height_terrain requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  return sensor.data.heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def normalized_actuator_torque(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  eps: float = 1.0e-6,
) -> torch.Tensor:
  """Return torque normalized by actuator force limits.

  Regular actuators use ``asset.data.actuator_force``. Custom actuators
  that expose ``last_motor_torque`` are normalized in motor space by their
  ``force_limit`` so the critic observes motor effort rather than projected
  ankle/joint torque. The result is intended as a critic-only privileged
  observation.
  """
  asset: Entity = env.scene[asset_cfg.name]
  selection = resolve_torque_channels(
    asset, asset_cfg, asset.data.actuator_force.device
  )
  actuator_force, effort_limit = read_torque_and_limits(env, asset, selection)
  selected_id_to_col = selection.selected_id_to_col
  actuators = getattr(asset, "actuators", ())
  if not isinstance(actuators, (list, tuple)):
    actuators = ()
  for actuator in actuators:  # Custom motor-space override.
    motor_torque = getattr(actuator, "last_motor_torque", None)
    force_limit = getattr(actuator, "force_limit", None)
    if not isinstance(motor_torque, torch.Tensor) or force_limit is None:
      continue

    src_cols: list[int] = []
    dst_cols: list[int] = []
    for src_col, local_id in enumerate(actuator.ctrl_ids.tolist()):
      dst_col = selected_id_to_col.get(local_id)
      if dst_col is not None:
        src_cols.append(src_col)
        dst_cols.append(dst_col)
    if not src_cols:
      continue

    src_idx = torch.tensor(src_cols, device=actuator_force.device, dtype=torch.long)
    dst_idx = torch.tensor(dst_cols, device=actuator_force.device, dtype=torch.long)
    actuator_force[:, dst_idx] = motor_torque.to(
      device=actuator_force.device, dtype=actuator_force.dtype
    )[:, src_idx]

    if isinstance(force_limit, torch.Tensor):
      limit = force_limit.to(device=actuator_force.device, dtype=actuator_force.dtype)
    else:
      limit = torch.as_tensor(
        force_limit, device=actuator_force.device, dtype=actuator_force.dtype
      )
    if limit.ndim == 1:
      limit = limit.unsqueeze(-1)
    effort_limit[:, dst_idx] = torch.abs(limit[:, src_idx])

  return actuator_force / torch.clamp(effort_limit, min=eps)


# --------- ADD OBSERVATIONS ---------


def lin_vel_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Linear velocity tracking error (B, 3): [cmd_x - actual_x, cmd_y - actual_y, -actual_z].

  Target is 0 for all components (xy tracks command, z should be zero).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = asset.data.root_link_lin_vel_b
  # Error: always negative if actual != command
  error = torch.zeros_like(actual)
  error[:, :2] = -torch.square(command[:, :2] - actual[:, :2])
  error[:, 2] = -torch.square(actual[:, 2])  # z velocity should be 0
  return error


def ang_vel_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Angular velocity tracking error (B, 3): [actual_x, actual_y, cmd_z - actual_z].

  Target is 0 for all components (xy should be zero, z tracks command).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = asset.data.root_link_ang_vel_b
  # Error: always negative if actual != command
  error = torch.zeros_like(actual)
  error[:, :2] = -torch.square(actual[:, :2])  # xy angular velocity should be 0
  error[:, 2] = -torch.square(command[:, 2] - actual[:, 2])
  return error


def orientation_error(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Orientation error from upright (B, num_bodies * 2): projected gravity xy per body.

  If asset_cfg.body_ids is specified, computes for those bodies.
  Otherwise uses root link.

  Target is 0 (gravity aligned with -z in body frame).
  """
  from mjlab.utils.lab_api.math import quat_apply_inverse

  asset: Entity = env.scene[asset_cfg.name]

  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    num_bodies = len(asset_cfg.body_ids)
    gravity_w = asset.data.gravity_vec_w.unsqueeze(1).expand(
      -1, num_bodies, -1
    )  # [B, N, 3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, N, 3]
    # Reduce dimension: return negative squared norm of xy components per body
    return -torch.sum(torch.square(projected_gravity_b[..., :2]), dim=-1)  # [B, N]
  else:
    return -torch.sum(
      torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1, keepdim=True
    )


def base_height_error(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Base height error (B, 1): actual - target.

  Target is 0.
  """
  asset: Entity = env.scene[asset_cfg.name]
  # Error: always negative squared error if deviating from target height
  height_error = -torch.square(asset.data.root_link_pos_w[:, 2] - target_height)
  return height_error.unsqueeze(-1)


def action_rate_slack(
  env: ManagerBasedRlEnv,
  threshold: float = 1.0,
) -> torch.Tensor:
  """Slack: Per-joint action rate violation (B, num_actions).

  0 if |action_rate| < threshold, negative (threshold - |rate|) otherwise.
  """
  diff = env.action_manager.action - env.action_manager.prev_action
  return (threshold - torch.abs(diff)).clamp(max=0.0)


def joint_limit_slack(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Slack: Per-joint limit violations (B, 2*num_joints).

  0 if within limits, negative if violating [lower_violations, upper_violations].
  """
  asset: Entity = env.scene[asset_cfg.name]
  soft_limits = asset.data.soft_joint_pos_limits

  if soft_limits is None:
    return torch.zeros(env.num_envs, len(asset_cfg.joint_ids) * 2, device=env.device)

  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  lower_limits = soft_limits[:, asset_cfg.joint_ids, 0]
  upper_limits = soft_limits[:, asset_cfg.joint_ids, 1]

  # Lower violations (where pos < lower): negative value
  violation_lower = (joint_pos - lower_limits).clamp(max=0.0)
  # Upper violations (where pos > upper): negative value
  violation_upper = (upper_limits - joint_pos).clamp(max=0.0)

  return torch.cat([violation_lower, violation_upper], dim=-1)


def self_collision_slack(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Slack: Per-body contact status (0 if no collision, -1 if collision).

  Returns [B, N] where N is the number of bodies tracked by the sensor.
  """
  sensor: ContactSensor = env.scene[sensor_name]

  if sensor.data.found is None:
    return torch.zeros(env.num_envs, 1, device=env.device)

  # Return 0 if found == 0, -1 if found > 0
  found = sensor.data.found
  if found.ndim == 1:
    found = found.unsqueeze(-1)

  return -(found > 0).float()


def foot_clearance_error(
  env: ManagerBasedRlEnv,
  target_height: float,
  vel_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Foot clearance error during swing phase (B, num_feet).

  Returns absolute deviation from target height if in swing phase, 0 otherwise.
  Swing phase determined by foot xy velocity > vel_threshold.

  Args:
    target_height: Target foot height during swing phase.
    vel_threshold: Velocity threshold to determine swing phase.
    asset_cfg: Asset config with site_names for foot sites.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]

  # Swing phase mask
  is_swing = (vel_norm > vel_threshold).float()  # [B, N]

  # Error: negative absolute difference from target
  error = -torch.abs(foot_z - target_height) * is_swing  # [B, N]

  return error


def foot_slip_slack(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Slack: Foot slip violation per axis during contact (B, num_feet * 2).

  0 if |vel_xy| < threshold (or not in contact), negative if slipping.
  Output: [left_x, left_y, right_x, right_y, ...]
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]

  if contact_sensor.data.found is None:
    num_feet = len(asset_cfg.site_ids) if asset_cfg.site_ids else 2
    return torch.zeros(env.num_envs, num_feet * 2, device=env.device)

  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]

  # Slack: threshold - |vel| per axis, clamped to max 0
  violation = (threshold - torch.abs(foot_vel_xy)).clamp(max=0.0)  # [B, N, 2]

  # Apply contact mask: only penalize during contact
  violation = violation * in_contact.unsqueeze(-1)  # [B, N, 2]

  # Flatten to [B, N*2]
  return violation.view(env.num_envs, -1)


def foot_flat_error(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_ratio: float = 0.4,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Error: Feet orientation deviation from flat during contact (B, num_feet * 2).

  Returns projected gravity xy components per foot.
  Output: [left_grav_x, left_grav_y, right_grav_x, right_grav_y, ...]
  Target is 0 (feet flat when in contact).

  Args:
    threshold_ratio: Force threshold as a ratio of total robot weight. Default 0.4.
  """
  from mjlab.utils.lab_api.math import quat_apply_inverse

  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]

  num_feet = len(asset_cfg.body_ids) if asset_cfg.body_ids else 2

  if contact_sensor.data.found is None:
    return torch.zeros(env.num_envs, num_feet * 2, device=env.device)

  # Calculate force threshold dynamically to match reward: total_mass * gravity * ratio
  all_body_ids = asset.indexing.body_ids
  body_masses = env.sim.model.body_mass
  if not isinstance(body_masses, torch.Tensor):
    body_masses = torch.tensor(body_masses, device=env.device, dtype=torch.float32)

  if body_masses.ndim == 2:
    total_mass = torch.sum(body_masses[:, all_body_ids], dim=1)
    if total_mass.ndim == 1:
      total_mass = total_mass.unsqueeze(-1)
  else:
    total_mass = torch.sum(body_masses[all_body_ids])

  gravity = 9.81
  force_threshold = total_mass * gravity * threshold_ratio

  # Check if bearing weight
  forces = contact_sensor.data.force
  if forces is None:
    return torch.zeros(env.num_envs, num_feet * 2, device=env.device)

  force_magnitude = torch.norm(forces, dim=-1)  # [B, N_feet]
  in_contact = (force_magnitude > force_threshold).float()

  # Get foot quaternions
  body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]

  # Gravity in world frame
  gravity_w = asset.data.gravity_vec_w.unsqueeze(1).expand(
    -1, num_feet, -1
  )  # [B, N, 3]

  # Project gravity to body frame
  projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, N, 3]

  # Reduce dimension: return negative squared norm of xy components per foot
  xy_squared = -torch.sum(torch.square(projected_gravity_b[..., :2]), dim=-1)  # [B, N]

  # Apply contact mask
  xy_squared = xy_squared * in_contact

  return xy_squared


def body_ang_vel_error(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Error: Body angular velocity xy components (B, 2).

  Target is 0 (no roll/pitch rotation). Z (yaw) is excluded.
  """
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]  # [B, N, 3]
  ang_vel = ang_vel.squeeze(1)  # [B, 3] assuming single body
  # Error: always negative squared error if xy angular velocity is non-zero
  return -torch.square(ang_vel[:, :2])  # [B, 2] - xy only, z excluded


def angular_momentum_error(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Error: Angular momentum xyz components (B, 3).

  Target is 0 (zero angular momentum for natural walking).
  """
  from mjlab.sensor import BuiltinSensor

  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data  # [B, 3]
  # Error: always negative squared error if angular momentum is non-zero
  return -torch.square(angmom)


class variable_posture_error:
  """Normalized deviation from default pose with speed-dependent tolerance.

  Follows the same logic as mdp.variable_posture reward but returns the
  weighted squared error per joint.
  """

  def __init__(self, cfg: ObservationTermCfg | RewardTermCfg, env: ManagerBasedRlEnv):
    asset_name = cfg.params["asset_cfg"].name
    asset: Entity = env.scene[asset_name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    turning_cfg = cfg.params.get("std_turning", cfg.params["std_walking"])
    _, _, std_turning = resolve_matching_names_values(
      data=turning_cfg,
      list_of_strings=joint_names,
    )
    self.std_turning = torch.tensor(std_turning, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing: dict[str, float],
    std_turning: dict[str, float] | None,
    std_walking: dict[str, float],
    std_running: dict[str, float],
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_turning, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    standing_mask, turning_mask, walking_mask, running_mask = _posture_speed_masks(
      command,
      walking_threshold=walking_threshold,
      running_threshold=running_threshold,
    )

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_turning * turning_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]

    # Return negative normalized squared error per joint: -(joint_error^2 / std^2)
    # This gives [B, num_joints] dimension.
    return -torch.square(current_joint_pos - desired_joint_pos) / (std**2)
