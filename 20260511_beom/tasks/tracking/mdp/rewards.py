from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


class joint_effort_limits:
  """Penalize effort overflow above a soft fraction of actuator limits."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    self._asset_name = asset_cfg.name
    self._default_soft_ratio = cfg.params.get("soft_ratio", 0.9)
    self._validate_soft_ratio(self._default_soft_ratio)
    self._default_power = cfg.params.get("power", 1.0)
    self._validate_power(self._default_power)

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

    # Cache mapping from selected local ctrl ids to columns for explicit overrides.
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

  @staticmethod
  def _validate_power(power: float) -> None:
    if power < 1.0:
      raise ValueError(f"'power' must be >= 1.0, got {power}.")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    soft_ratio: float | None = None,
    power: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del asset_cfg  # Mapping is cached at initialization.
    ratio = self._default_soft_ratio if soft_ratio is None else soft_ratio
    self._validate_soft_ratio(ratio)
    effort_power = self._default_power if power is None else power
    self._validate_power(effort_power)

    asset: Entity = env.scene[self._asset_name]
    actuator_force = torch.abs(asset.data.actuator_force[:, self._local_ctrl_ids])

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

    # Explicit actuators may use runtime limits (force_limit) that diverge from
    # model.actuator_forcerange after DR; prefer those runtime values.
    # For fourbar actuators, also replace ankle-space torque with motor-space
    # torque so that force and limit are in the same (motor) space.
    for actuator, src_cols, dst_cols in self._explicit_overrides:
      if self._has_motor_space:
        motor_torque = actuator.last_motor_torque
        if motor_torque is not None:
          actuator_force[:, dst_cols] = torch.abs(motor_torque[:, src_cols])
      force_limit = actuator.force_limit
      if force_limit is None:
        continue
      if force_limit.ndim == 1:
        force_limit = force_limit.unsqueeze(-1)
      effort_limit[:, dst_cols] = torch.abs(force_limit[:, src_cols])

    soft_limit = effort_limit * ratio
    overflow = (actuator_force - soft_limit).clamp(min=0.0)
    if effort_power == 1.0:
      return torch.sum(overflow, dim=1)
    return torch.sum(torch.pow(overflow, effort_power), dim=1)


class joint_torque_rate_l2:
  """Penalize normalized torque slew rate.

  Computes ``sum(((tau_t - tau_{t-1}) / (limit * soft_ratio + eps))^2)``.
  This normalizes each actuator by its effort limit so heterogeneous actuators
  contribute on a comparable scale.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    self._asset_name = asset_cfg.name
    self._default_soft_ratio = cfg.params.get("soft_ratio", 1.0)
    self._validate_soft_ratio(self._default_soft_ratio)
    self._default_eps = cfg.params.get("eps", 1.0e-6)
    self._validate_eps(self._default_eps)

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

  @staticmethod
  def _validate_soft_ratio(soft_ratio: float) -> None:
    if not (0.0 < soft_ratio <= 1.0):
      raise ValueError(f"'soft_ratio' must be in (0, 1], got {soft_ratio}.")

  @staticmethod
  def _validate_eps(eps: float) -> None:
    if eps <= 0.0:
      raise ValueError(f"'eps' must be > 0, got {eps}.")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    soft_ratio: float | None = None,
    eps: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del asset_cfg  # Mapping is cached at initialization.
    ratio = self._default_soft_ratio if soft_ratio is None else soft_ratio
    self._validate_soft_ratio(ratio)
    denom_eps = self._default_eps if eps is None else eps
    self._validate_eps(denom_eps)

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
    rate = torch.sum(torch.square(delta / denom), dim=1)
    self._prev_torque.copy_(torque)
    return rate

  def reset(self, env_ids: torch.Tensor | slice) -> None:
    self._prev_torque[env_ids] = 0.0


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_joint_pos_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  joint_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  if joint_names is None:
    error = torch.sum(torch.square(command.joint_pos - command.robot_joint_pos), dim=-1)
  else:
    # Find joint indices using the scene entity
    robot = env.scene[command.cfg.entity_name]
    joint_indices, _ = robot.find_joints(joint_names)
    error = torch.sum(
      torch.square(
        command.joint_pos[:, joint_indices] - command.robot_joint_pos[:, joint_indices]
      ),
      dim=-1,
    )

  return torch.exp(-error / std**2)


def motion_joint_vel_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  joint_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  if joint_names is None:
    error = torch.sum(torch.square(command.joint_vel - command.robot_joint_vel), dim=-1)
  else:
    # Find joint indices using the scene entity
    robot = env.scene[command.cfg.entity_name]
    joint_indices, _ = robot.find_joints(joint_names)
    error = torch.sum(
      torch.square(
        command.joint_vel[:, joint_indices] - command.robot_joint_vel[:, joint_indices]
      ),
      dim=-1,
    )

  return torch.exp(-error / std**2)


def motion_global_anchor_position_xy_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square((command.anchor_pos_w - command.robot_anchor_pos_w)[..., :2]), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)
