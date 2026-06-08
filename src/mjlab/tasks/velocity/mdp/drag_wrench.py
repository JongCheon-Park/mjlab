from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

REGIME_STANDING = 0
REGIME_TURNING = 1
REGIME_MOVING = 2
_REGIME_NAME_TO_CODE = {
  "standing": REGIME_STANDING,
  "turning": REGIME_TURNING,
  "moving": REGIME_MOVING,
}


def _regime_codes(
  regime: str | list[str] | torch.Tensor,
  num_envs: int,
  device: torch.device | str,
) -> torch.Tensor:
  if isinstance(regime, str):
    return torch.full(
      (num_envs,),
      _REGIME_NAME_TO_CODE[regime],
      device=device,
      dtype=torch.int8,
    )
  if isinstance(regime, list):
    return torch.tensor(
      [_REGIME_NAME_TO_CODE[name] for name in regime],
      device=device,
      dtype=torch.int8,
    )
  return regime.to(device=device, dtype=torch.int8)


def _xy_slice(actual_lin_vel: torch.Tensor) -> torch.Tensor:
  return actual_lin_vel[:, :2] if actual_lin_vel.shape[-1] > 2 else actual_lin_vel


def compute_velocity_drag_wrench(
  regime: str | list[str] | torch.Tensor,
  command: torch.Tensor,
  actual_lin_vel: torch.Tensor,
  is_active: torch.Tensor,
  drag_force: torch.Tensor,
  drag_damping: torch.Tensor,
  force_clip_xy: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute command-opposed horizontal drag for moving environments only."""
  num_envs = command.shape[0]
  device = command.device
  dtype = command.dtype
  regime_codes = _regime_codes(regime, num_envs, device)
  moving_mask = regime_codes == REGIME_MOVING

  force = torch.zeros((num_envs, 3), device=device, dtype=dtype)
  torque = torch.zeros((num_envs, 3), device=device, dtype=dtype)

  active_mask = moving_mask & is_active.to(device=device)
  if not active_mask.any():
    return force, torque

  command_xy = command[:, :2]
  direction = command_xy / torch.linalg.vector_norm(
    command_xy, dim=-1, keepdim=True
  ).clamp_min(1e-6)
  actual_lin_xy = _xy_slice(actual_lin_vel)[:, :2]
  forward_speed = torch.clamp(
    torch.sum(actual_lin_xy * direction, dim=-1),
    min=0.0,
  )
  magnitude = (
    drag_force.to(device=device, dtype=dtype)
    + drag_damping.to(device=device, dtype=dtype) * forward_speed
  )
  magnitude = torch.clamp(magnitude, min=0.0, max=force_clip_xy)

  force_xy = -magnitude.unsqueeze(-1) * direction
  force[active_mask, :2] = force_xy[active_mask]
  return force, torque


@dataclass(kw_only=True)
class VelocityDragWrenchCfg:
  asset_cfg: SceneEntityCfg = field(
    default_factory=lambda: SceneEntityCfg("robot", body_names=("base_link",))
  )
  enable_prob: float = 0.5
  drag_force_range: tuple[float, float] = (10.0, 20.0)
  drag_damping_range: tuple[float, float] = (1.0, 3.0)
  force_clip_xy: float = 100.0
  start_steps: int = 0

  def __post_init__(self) -> None:
    if not 0.0 <= self.enable_prob <= 1.0:
      raise ValueError("enable_prob must be in [0, 1].")
    if self.drag_force_range[1] < self.drag_force_range[0]:
      raise ValueError("drag_force_range must be ordered as (min, max).")
    if self.drag_damping_range[1] < self.drag_damping_range[0]:
      raise ValueError("drag_damping_range must be ordered as (min, max).")
    if self.force_clip_xy <= 0.0:
      raise ValueError("force_clip_xy must be > 0.")
    if self.start_steps < 0:
      raise ValueError("start_steps must be >= 0.")


class VelocityDragWrench:
  def __init__(self, cfg: VelocityDragWrenchCfg, env) -> None:
    self.cfg = cfg
    self._env = env
    self.device = env.device
    self._asset_cfg = deepcopy(cfg.asset_cfg)
    self._asset_cfg.resolve(env.scene)
    self.robot = env.scene[self._asset_cfg.name]

    if isinstance(self._asset_cfg.body_ids, slice):
      body_ids = list(range(self.robot.num_bodies))[self._asset_cfg.body_ids]
    else:
      body_ids = list(self._asset_cfg.body_ids)
    if len(body_ids) != 1:
      raise ValueError(
        "VelocityDragWrench requires exactly one target body, got "
        f"{len(body_ids)} from asset_cfg={self._asset_cfg}."
      )
    self._body_ids = body_ids

    num_envs = env.num_envs
    self.is_active = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
    self.drag_force = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
    self.drag_damping = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
    self.regime_codes = torch.full(
      (num_envs,), REGIME_STANDING, device=self.device, dtype=torch.int8
    )

    self._zero_force = torch.zeros((num_envs, 1, 3), device=self.device)
    self._zero_torque = torch.zeros((num_envs, 1, 3), device=self.device)
    self._wrench_is_zeroed = True

  def _enabled(self) -> bool:
    common_step_counter = getattr(self._env, "common_step_counter", 0)
    if not isinstance(common_step_counter, Integral):
      common_step_counter = 0
    return common_step_counter >= self.cfg.start_steps

  def _write_zero_wrench_if_needed(self) -> None:
    if self._wrench_is_zeroed:
      return
    self.robot.write_external_wrench_to_sim(
      self._zero_force,
      self._zero_torque,
      body_ids=self._body_ids,
    )
    self._wrench_is_zeroed = True

  def set_regime_from_command(
    self,
    command: torch.Tensor,
    is_standing_env: torch.Tensor,
  ) -> None:
    lin_speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
    self.regime_codes.fill_(REGIME_TURNING)
    self.regime_codes[is_standing_env] = REGIME_STANDING
    self.regime_codes[(~is_standing_env) & (lin_speed > 1e-6)] = REGIME_MOVING

  def set_regime(self, regime: str | list[str] | torch.Tensor) -> None:
    self.regime_codes = _regime_codes(regime, self._env.num_envs, self.device)

  def on_command_resample(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return

    self.is_active[env_ids] = (
      torch.rand(len(env_ids), device=self.device) < self.cfg.enable_prob
    )

    force_min, force_max = self.cfg.drag_force_range
    self.drag_force[env_ids] = force_min + (force_max - force_min) * torch.rand(
      len(env_ids), device=self.device
    )

    damping_min, damping_max = self.cfg.drag_damping_range
    self.drag_damping[env_ids] = damping_min + (damping_max - damping_min) * torch.rand(
      len(env_ids), device=self.device
    )

  def apply(self, command: torch.Tensor) -> None:
    if not self._enabled():
      self._write_zero_wrench_if_needed()
      return

    force_b, torque_b = compute_velocity_drag_wrench(
      regime=self.regime_codes,
      command=command,
      actual_lin_vel=self.robot.data.root_link_lin_vel_b,
      is_active=self.is_active,
      drag_force=self.drag_force,
      drag_damping=self.drag_damping,
      force_clip_xy=self.cfg.force_clip_xy,
    )

    if not torch.any(force_b):
      self._write_zero_wrench_if_needed()
      return

    heading = self.robot.data.heading_w
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    force = torch.zeros_like(force_b)
    force[:, 0] = cos_h * force_b[:, 0] - sin_h * force_b[:, 1]
    force[:, 1] = sin_h * force_b[:, 0] + cos_h * force_b[:, 1]
    force[:, 2] = 0.0

    self.robot.write_external_wrench_to_sim(
      force.unsqueeze(1),
      torque_b.unsqueeze(1),
      body_ids=self._body_ids,
    )
    self._wrench_is_zeroed = False
