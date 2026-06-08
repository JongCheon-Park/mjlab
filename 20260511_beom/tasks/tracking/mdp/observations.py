from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.envs.mdp.observations import (
  generated_commands as _generated_commands_default,
)
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_inv,
  quat_mul,
  subtract_frame_transforms,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.observation_manager import ObservationTermCfg

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _validate_future_len(future_len: int) -> None:
  if future_len < 0:
    raise ValueError(f"future_len must be >= 0, got {future_len}")


def _future_offsets(future_len: int) -> range:
  # Convention:
  #   - future_len == 0 -> one frame at base index (backward-compatible)
  #   - future_len == N (>0) -> exactly N frames: [base + 0, ..., base + (N-1)]
  return range(1) if future_len == 0 else range(future_len)


def _require_motion_command(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
  command_term = env.command_manager.get_term(command_name)
  if not isinstance(command_term, MotionCommand):
    raise TypeError(
      f"Observation term requires MotionCommand, got {type(command_term).__name__} "
      f"for command '{command_name}'."
    )
  return command_term


def generated_commands(
  env: ManagerBasedRlEnv,
  command_name: str,
  future_len: int = 0,
  use_pre_shift: bool = False,
) -> torch.Tensor:
  _validate_future_len(future_len)
  command_term = env.command_manager.get_term(command_name)
  if not isinstance(command_term, MotionCommand):
    if future_len != 0 or use_pre_shift:
      raise TypeError(
        "future_len/use_pre_shift options are only supported for MotionCommand."
      )
    return _generated_commands_default(env, command_name)
  return command_term.get_future_command(
    future_len=future_len,
    use_pre_shift=use_pre_shift,
  )


def motion_anchor_pos_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  future_len: int = 0,
  use_pre_shift: bool = False,
) -> torch.Tensor:
  command = _require_motion_command(env, command_name)
  _validate_future_len(future_len)

  robot_anchor_pos_w = command.robot_anchor_pos_w
  robot_anchor_quat_w = command.robot_anchor_quat_w
  obs_chunks: list[torch.Tensor] = []
  for step_offset in _future_offsets(future_len):
    pos, _ = subtract_frame_transforms(
      robot_anchor_pos_w,
      robot_anchor_quat_w,
      command.get_anchor_pos_w(step_offset=step_offset, use_pre_shift=use_pre_shift),
      command.get_anchor_quat_w(step_offset=step_offset, use_pre_shift=use_pre_shift),
    )
    obs_chunks.append(pos.view(env.num_envs, -1))

  return torch.cat(obs_chunks, dim=1)


def motion_anchor_ori_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  future_len: int = 0,
  use_pre_shift: bool = False,
) -> torch.Tensor:
  command = _require_motion_command(env, command_name)
  _validate_future_len(future_len)

  robot_anchor_pos_w = command.robot_anchor_pos_w
  robot_anchor_quat_w = command.robot_anchor_quat_w
  obs_chunks: list[torch.Tensor] = []
  for step_offset in _future_offsets(future_len):
    _, ori = subtract_frame_transforms(
      robot_anchor_pos_w,
      robot_anchor_quat_w,
      command.get_anchor_pos_w(step_offset=step_offset, use_pre_shift=use_pre_shift),
      command.get_anchor_quat_w(step_offset=step_offset, use_pre_shift=use_pre_shift),
    )
    mat = matrix_from_quat(ori)
    obs_chunks.append(mat[..., :2].reshape(mat.shape[0], -1))
  return torch.cat(obs_chunks, dim=1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


# --------- ADD OBSERVATIONS ---------

# --- Slack Variables ---


def action_rate_slack(env: ManagerBasedRlEnv, threshold: float = 1.0) -> torch.Tensor:
  """Slack: Per-joint action rate violation (0 if below threshold, negative if above)."""
  diff = env.action_manager.action - env.action_manager.prev_action
  # Per-joint violation: 0 if |diff| < threshold, negative (threshold - |diff|) otherwise
  return (threshold - torch.abs(diff)).clamp(max=0.0)


def joint_limit_slack(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Slack: Per-joint lower and upper limit violations (0 if good, negative if violating)."""
  asset: Entity = env.scene[asset_cfg.name]
  soft_limits = asset.data.soft_joint_pos_limits

  if soft_limits is None:
    # Return correctly shaped zeros: (N, 2 * num_joints)
    return torch.zeros(env.num_envs, len(asset_cfg.joint_ids) * 2, device=env.device)

  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  lower_limits = soft_limits[:, asset_cfg.joint_ids, 0]
  upper_limits = soft_limits[:, asset_cfg.joint_ids, 1]

  # Lower violations (where pos < lower): negative value
  violation_lower = (joint_pos - lower_limits).clamp(max=0.0)

  # Upper violations (where pos > upper): negative value
  violation_upper = (upper_limits - joint_pos).clamp(max=0.0)

  # Concatenate (N, 2 * num_joints)
  return torch.cat([violation_lower, violation_upper], dim=-1)


class torque_limit_slack:
  """Slack: Per-actuator lower/upper torque limit violations.

  Returns concatenated violations with shape ``(N, 2 * num_actuators)``:
  ``[lower_violation, upper_violation]`` where each entry is 0 if within limit
  and negative when violating that side.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv) -> None:
    asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    self._asset_name = asset_cfg.name
    self._default_soft_ratio = cfg.params.get("soft_ratio", 0.9)
    self._validate_soft_ratio(self._default_soft_ratio)

    asset: Entity = env.scene[self._asset_name]
    if isinstance(asset_cfg.actuator_ids, slice):
      num_actuators = asset.data.actuator_force.shape[1]
      local_ctrl_ids = torch.arange(num_actuators, device=env.device, dtype=torch.long)[
        asset_cfg.actuator_ids
      ]
    else:
      local_ctrl_ids = torch.as_tensor(
        asset_cfg.actuator_ids, device=env.device, dtype=torch.long
      )
    self._local_ctrl_ids = local_ctrl_ids
    self._global_ctrl_ids = asset.data.indexing.ctrl_ids[local_ctrl_ids]

    selected_id_to_col = {
      ctrl_id: col for col, ctrl_id in enumerate(self._local_ctrl_ids.tolist())
    }
    self._explicit_overrides: list[tuple[object, torch.Tensor, torch.Tensor]] = []
    has_motor_space = False
    for actuator in asset.actuators:
      force_limit = getattr(actuator, "force_limit", None)
      if force_limit is None:
        continue
      if hasattr(actuator, "last_motor_torque"):
        has_motor_space = True

      src_cols: list[int] = []
      dst_cols: list[int] = []
      for j, local_id in enumerate(actuator.ctrl_ids.tolist()):
        col = selected_id_to_col.get(local_id)
        if col is not None:
          src_cols.append(j)
          dst_cols.append(col)
      if src_cols:
        self._explicit_overrides.append(
          (
            actuator,
            torch.tensor(src_cols, device=env.device, dtype=torch.long),
            torch.tensor(dst_cols, device=env.device, dtype=torch.long),
          )
        )
    self._has_motor_space = has_motor_space

  @staticmethod
  def _validate_soft_ratio(soft_ratio: float) -> None:
    if not (0.0 < soft_ratio <= 1.0):
      raise ValueError(f"'soft_ratio' must be in (0, 1], got {soft_ratio}.")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    soft_ratio: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del asset_cfg  # Mapping is cached at initialization.
    ratio = self._default_soft_ratio if soft_ratio is None else soft_ratio
    self._validate_soft_ratio(ratio)

    asset: Entity = env.scene[self._asset_name]
    if self._has_motor_space:
      torque = asset.data.actuator_force[:, self._local_ctrl_ids].clone()
    else:
      torque = asset.data.actuator_force[:, self._local_ctrl_ids]

    forcerange = env.sim.model.actuator_forcerange
    if forcerange.ndim == 3:
      effort_limit = torch.abs(forcerange[:, self._global_ctrl_ids, 1])
    elif forcerange.ndim == 2:
      effort_limit = torch.abs(forcerange[self._global_ctrl_ids, 1]).unsqueeze(0)
      effort_limit = effort_limit.expand(env.num_envs, -1)
    else:
      raise ValueError(
        "Expected 'actuator_forcerange' to have 2 or 3 dims, "
        f"got shape {tuple(forcerange.shape)}."
      )

    # For fourbar actuators, replace ankle-space torque with motor-space
    # torque so that force and limit are in the same (motor) space.
    for actuator, src_cols, dst_cols in self._explicit_overrides:
      if self._has_motor_space:
        motor_torque = actuator.last_motor_torque
        if motor_torque is not None:
          torque[:, dst_cols] = motor_torque[:, src_cols]
      force_limit = actuator.force_limit
      if force_limit is None:
        continue
      if force_limit.ndim == 1:
        force_limit = force_limit.unsqueeze(-1)
      effort_limit[:, dst_cols] = torch.abs(force_limit[:, src_cols])

    soft_limit = effort_limit * ratio
    lower_limits = -soft_limit
    upper_limits = soft_limit

    # Match joint_limit_slack pattern:
    # - lower_violation < 0 when tau < lower_limit
    # - upper_violation < 0 when tau > upper_limit
    violation_lower = (torque - lower_limits).clamp(max=0.0)
    violation_upper = (upper_limits - torque).clamp(max=0.0)
    return torch.cat([violation_lower, violation_upper], dim=-1)


class torque_rate_slack:
  """Slack: Per-actuator normalized torque rate violation.

  Returns ``(N, num_actuators)`` where each entry is 0 when the normalized
  rate ``|Δτ| / (limit * soft_ratio)`` is below *threshold*, and negative
  (``threshold - normalized_rate``) when it exceeds the threshold.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv) -> None:
    asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    self._asset_name = asset_cfg.name
    self._default_soft_ratio = cfg.params.get("soft_ratio", 1.0)
    self._default_threshold = cfg.params.get("threshold", 0.5)
    self._default_eps = cfg.params.get("eps", 1.0e-6)

    asset: Entity = env.scene[self._asset_name]
    if isinstance(asset_cfg.actuator_ids, slice):
      num_actuators = asset.data.actuator_force.shape[1]
      local_ctrl_ids = torch.arange(num_actuators, device=env.device, dtype=torch.long)[
        asset_cfg.actuator_ids
      ]
    else:
      local_ctrl_ids = torch.as_tensor(
        asset_cfg.actuator_ids, device=env.device, dtype=torch.long
      )
    self._local_ctrl_ids = local_ctrl_ids
    self._global_ctrl_ids = asset.data.indexing.ctrl_ids[local_ctrl_ids]

    selected_id_to_col = {
      ctrl_id: col for col, ctrl_id in enumerate(self._local_ctrl_ids.tolist())
    }
    self._explicit_overrides: list[tuple[object, torch.Tensor, torch.Tensor]] = []
    has_motor_space = False
    for actuator in asset.actuators:
      force_limit = getattr(actuator, "force_limit", None)
      if force_limit is None:
        continue
      if hasattr(actuator, "last_motor_torque"):
        has_motor_space = True

      src_cols: list[int] = []
      dst_cols: list[int] = []
      for j, local_id in enumerate(actuator.ctrl_ids.tolist()):
        col = selected_id_to_col.get(local_id)
        if col is not None:
          src_cols.append(j)
          dst_cols.append(col)
      if src_cols:
        self._explicit_overrides.append(
          (
            actuator,
            torch.tensor(src_cols, device=env.device, dtype=torch.long),
            torch.tensor(dst_cols, device=env.device, dtype=torch.long),
          )
        )
    self._has_motor_space = has_motor_space

    nu = len(self._local_ctrl_ids)
    self._prev_torque = torch.zeros(
      (env.num_envs, nu), device=env.device, dtype=torch.float
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    threshold: float | None = None,
    soft_ratio: float | None = None,
    eps: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del asset_cfg  # Mapping is cached at initialization.
    thresh = self._default_threshold if threshold is None else threshold
    ratio = self._default_soft_ratio if soft_ratio is None else soft_ratio
    denom_eps = self._default_eps if eps is None else eps

    asset: Entity = env.scene[self._asset_name]
    if self._has_motor_space:
      torque = asset.data.actuator_force[:, self._local_ctrl_ids].clone()
    else:
      torque = asset.data.actuator_force[:, self._local_ctrl_ids]

    forcerange = env.sim.model.actuator_forcerange
    if forcerange.ndim == 3:
      effort_limit = torch.abs(forcerange[:, self._global_ctrl_ids, 1])
    elif forcerange.ndim == 2:
      effort_limit = torch.abs(forcerange[self._global_ctrl_ids, 1]).unsqueeze(0)
      effort_limit = effort_limit.expand(env.num_envs, -1)
    else:
      raise ValueError(
        "Expected 'actuator_forcerange' to have 2 or 3 dims, "
        f"got shape {tuple(forcerange.shape)}."
      )

    # For fourbar actuators, replace ankle-space torque with motor-space
    # torque so that force and limit are in the same (motor) space.
    for actuator, src_cols, dst_cols in self._explicit_overrides:
      if self._has_motor_space:
        motor_torque = actuator.last_motor_torque
        if motor_torque is not None:
          torque[:, dst_cols] = motor_torque[:, src_cols]
      force_limit = actuator.force_limit
      if force_limit is None:
        continue
      if force_limit.ndim == 1:
        force_limit = force_limit.unsqueeze(-1)
      effort_limit[:, dst_cols] = torch.abs(force_limit[:, src_cols])

    denom = effort_limit * ratio + denom_eps
    delta = torque - self._prev_torque
    normalized_rate = torch.abs(delta) / denom
    self._prev_torque.copy_(torque)

    # 0 when within threshold, negative when exceeding
    return (thresh - normalized_rate).clamp(max=0.0)

  def reset(self, env_ids: torch.Tensor | slice) -> None:
    self._prev_torque[env_ids] = 0.0


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


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


@torch.jit.script
def quat_to_tan_norm(q: torch.Tensor) -> torch.Tensor:
  ref_tan = torch.zeros_like(q[..., 0:3])
  ref_tan[..., 0] = 1
  tan = quat_apply(q, ref_tan)

  ref_norm = torch.zeros_like(q[..., 0:3])
  ref_norm[..., -1] = 1
  norm = quat_apply(q, ref_norm)

  norm_tan = torch.cat([tan, norm], dim=-1)
  return norm_tan


# --- Anchor Errors ---


def motion_global_anchor_position_error(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Error vector: Motion Anchor Pos - Robot Anchor Pos (Global)"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  # We want (Reference - Agent)
  # The rewards used (command.anchor_pos_w - command.robot_anchor_pos_w)
  return command.anchor_pos_w - command.robot_anchor_pos_w


def motion_global_anchor_orientation_error(
  env: ManagerBasedRlEnv, command_name: str, representation: str = "6d"
) -> torch.Tensor:
  """Error vector: Orientation difference between Motion Anchor and Robot Anchor.

  MimicKit style: Compute tan_norm for each quaternion, then subtract.
  This ensures the diff is 0 when orientations match.

  Args:
      env: The environment object.
      command_name: The name of the command.
      representation: The representation of the orientation.
          - "6d": Tangent and Normal vectors (default, MimicKit style).
          - "3d": Imaginary part of quaternion difference.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  if representation == "6d":
    # MimicKit style: tan_norm(demo) - tan_norm(robot)
    motion_tan_norm = quat_to_tan_norm(command.anchor_quat_w)
    robot_tan_norm = quat_to_tan_norm(command.robot_anchor_quat_w)
    return motion_tan_norm - robot_tan_norm

  # Fallback: quaternion diff imaginary part
  q_diff = quat_mul(command.anchor_quat_w, quat_inv(command.robot_anchor_quat_w))
  return q_diff[:, 1:]


# --- Body Errors (Relative to Anchor) ---


def motion_relative_body_position_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Error vector: Motion Body Pos (Rel) - Robot Body Pos (Rel)"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)

  # Shape: (NumEnvs, NumBodies, 3)
  diff = (
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes]
  )
  # Flatten to (NumEnvs, NumBodies * 3)
  return diff.view(env.num_envs, -1)


def motion_relative_body_orientation_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...] | None = None,
  representation: str = "6d",
) -> torch.Tensor:
  """Error vector: Motion Body Ori (Rel) - Robot Body Ori (Rel)

  MimicKit style: Compute tan_norm for each quaternion, then subtract.
  This ensures the diff is 0 when orientations match.

  Args:
      env: The environment object.
      command_name: The name of the command.
      body_names: Names of the bodies to track.
      representation: The representation of the orientation.
          - "6d": Tangent and Normal vectors (default, MimicKit style).
          - "3d": Imaginary part of quaternion difference.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)

  q_motion_rel = command.body_quat_relative_w[:, body_indexes]
  q_robot_rel = command.robot_body_quat_w[:, body_indexes]

  if representation == "6d":
    # MimicKit style: tan_norm(demo) - tan_norm(robot)
    motion_tan_norm = quat_to_tan_norm(q_motion_rel)
    robot_tan_norm = quat_to_tan_norm(q_robot_rel)
    diff = motion_tan_norm - robot_tan_norm
    return diff.reshape(env.num_envs, -1)

  # Fallback: quaternion diff imaginary part
  q_diff = quat_mul(q_motion_rel, quat_inv(q_robot_rel))
  return q_diff[..., 1:].reshape(env.num_envs, -1)


# --- Joint Errors ---


def joint_pos_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  joint_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Error vector: Motion Joint Pos - Robot Joint Pos"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  if joint_names is None:
    return command.joint_pos - command.robot_joint_pos

  # Find joint indices using the scene entity
  robot = env.scene[command.cfg.entity_name]
  joint_indices, _ = robot.find_joints(joint_names)
  return command.joint_pos[:, joint_indices] - command.robot_joint_pos[:, joint_indices]


def joint_vel_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  joint_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Error vector: Motion Joint Vel - Robot Joint Vel"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  if joint_names is None:
    return command.joint_vel - command.robot_joint_vel

  # Find joint indices using the scene entity
  robot = env.scene[command.cfg.entity_name]
  joint_indices, _ = robot.find_joints(joint_names)
  return command.joint_vel[:, joint_indices] - command.robot_joint_vel[:, joint_indices]


# --- Global Velocity Errors ---
# Typically ADD also includes global velocity errors


def motion_global_body_lin_vel_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Error vector: Motion Body Lin Vel - Robot Body Lin Vel"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)

  diff = (
    command.body_lin_vel_w[:, body_indexes]
    - command.robot_body_lin_vel_w[:, body_indexes]
  )
  return diff.view(env.num_envs, -1)


def motion_global_body_ang_vel_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Error vector: Motion Body Ang Vel - Robot Body Ang Vel"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)

  diff = (
    command.body_ang_vel_w[:, body_indexes]
    - command.robot_body_ang_vel_w[:, body_indexes]
  )
  return diff.view(env.num_envs, -1)
