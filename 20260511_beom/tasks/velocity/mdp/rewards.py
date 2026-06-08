from __future__ import annotations

import math
import numbers
from dataclasses import replace
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _safe_common_step_counter(env: ManagerBasedRlEnv) -> int:
  """Return integer step counter when available; otherwise disable caching."""
  step = getattr(env, "common_step_counter", -1)
  return int(step) if isinstance(step, numbers.Real) else -1


def _get_or_init_cache_dict(
  env: ManagerBasedRlEnv, attr_name: str
) -> dict[tuple[object, ...], object]:
  """Get a dict cache stored on env, creating one when absent/invalid."""
  cache = getattr(env, attr_name, None)
  if not isinstance(cache, dict):
    cache = {}
    setattr(env, attr_name, cache)
  return cache


class _ContactEventSnapshot(NamedTuple):
  just_landed: torch.Tensor
  just_lifted: torch.Tensor
  last_air_time: torch.Tensor
  last_contact_time: torch.Tensor
  current_air_time: torch.Tensor
  current_contact_time: torch.Tensor


class _CommandSnapshot(NamedTuple):
  command: torch.Tensor | None
  cmd_x: torch.Tensor | None
  cmd_y: torch.Tensor | None
  cmd_speed: torch.Tensor | None


class _BipedCommandBaseSnapshot(NamedTuple):
  total_speed: torch.Tensor
  standing_mask: torch.Tensor
  linear_speed: torch.Tensor
  angular_speed: torch.Tensor
  vx: torch.Tensor
  vy: torch.Tensor
  yaw: torch.Tensor


class _BipedCommandRegimeSnapshot(NamedTuple):
  total_speed: torch.Tensor
  standing_mask: torch.Tensor
  turning_mask: torch.Tensor
  walking_mask: torch.Tensor
  running_mask: torch.Tensor
  lateral_only_mask: torch.Tensor
  vy: torch.Tensor
  yaw: torch.Tensor


def _get_contact_event_snapshot(
  env: ManagerBasedRlEnv,
  *,
  sensor_name: str,
) -> _ContactEventSnapshot:
  """Shared per-step contact-event snapshot for gait reward terms."""
  step = _safe_common_step_counter(env)
  cache = _get_or_init_cache_dict(env, "_gait_contact_event_cache")
  cache_key = (sensor_name,)
  if step >= 0:
    cached = cache.get(cache_key)
    if (
      isinstance(cached, tuple)
      and len(cached) == 2
      and cached[0] == step
      and isinstance(cached[1], _ContactEventSnapshot)
    ):
      return cached[1]

  contact_sensor: ContactSensor = env.scene[sensor_name]
  just_landed = contact_sensor.compute_first_contact(dt=env.step_dt).float()
  just_lifted = contact_sensor.compute_first_air(dt=env.step_dt).float()
  sensor_data = getattr(contact_sensor, "data", None)
  last_air_candidate = (
    getattr(sensor_data, "last_air_time", None) if sensor_data is not None else None
  )
  last_contact_candidate = (
    getattr(sensor_data, "last_contact_time", None) if sensor_data is not None else None
  )
  current_air_candidate = (
    getattr(sensor_data, "current_air_time", None) if sensor_data is not None else None
  )
  current_contact_candidate = (
    getattr(sensor_data, "current_contact_time", None)
    if sensor_data is not None
    else None
  )
  last_air = (
    last_air_candidate
    if isinstance(last_air_candidate, torch.Tensor)
    else torch.zeros_like(just_landed)
  )
  last_contact = (
    last_contact_candidate
    if isinstance(last_contact_candidate, torch.Tensor)
    else torch.zeros_like(just_lifted)
  )
  current_air = (
    current_air_candidate
    if isinstance(current_air_candidate, torch.Tensor)
    else torch.zeros_like(just_lifted)
  )
  current_contact = (
    current_contact_candidate
    if isinstance(current_contact_candidate, torch.Tensor)
    else torch.zeros_like(just_landed)
  )

  snapshot = _ContactEventSnapshot(
    just_landed=just_landed,
    just_lifted=just_lifted,
    last_air_time=last_air,
    last_contact_time=last_contact,
    current_air_time=current_air,
    current_contact_time=current_contact,
  )
  if step >= 0:
    cache[cache_key] = (step, snapshot)
  return snapshot


def _post_reward_event_mask(
  *,
  event_mask: torch.Tensor,
  phase_time: torch.Tensor,
  step_dt: float,
  post_reward_steps: int,
  post_reward_inv_std_sq: float,
) -> torch.Tensor:
  """Return a decayed post-event mask, preserving legacy event-only behavior.

  ``post_reward_steps`` counts the event step itself. With
  ``post_reward_steps=5``, phase ages 0, 1, 2, 3, 4 are active and age 5 is
  inactive. A phase time of exactly one env step is treated as event age 0
  because contact sensors usually report phase duration after the sim step.
  """
  if post_reward_steps <= 1:
    return event_mask

  eps = max(float(step_dt) * 1.0e-6, 1.0e-9)
  phase_age_steps = torch.floor(
    torch.clamp((phase_time - eps) / max(float(step_dt), 1.0e-12), min=0.0)
  )
  phase_active = (phase_time > 0.0) & (phase_age_steps < float(post_reward_steps))
  phase_decay = torch.exp(-phase_age_steps.square() * post_reward_inv_std_sq)
  phase_mask = phase_decay * phase_active.float()
  # Keep event-step rewards even if a mock or older sensor backend lacks
  # current_*_time. Event masks may be fractional after min-phase filtering.
  return torch.maximum(event_mask, phase_mask)


def _get_biped_command_base_snapshot(
  env: ManagerBasedRlEnv,
  *,
  command_name: str | None,
  command_threshold: float,
) -> _BipedCommandBaseSnapshot:
  """Shared per-step base command snapshot for biped rewards."""
  step = _safe_common_step_counter(env)
  cache = _get_or_init_cache_dict(env, "_biped_command_base_cache")
  cache_key = (command_name, float(command_threshold))
  if step >= 0:
    cached = cache.get(cache_key)
    if (
      isinstance(cached, tuple)
      and len(cached) == 2
      and cached[0] == step
      and isinstance(cached[1], _BipedCommandBaseSnapshot)
    ):
      return cached[1]

  command_snapshot = _get_command_snapshot(env, command_name=command_name)
  zero_speed = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  if command_snapshot.command is None:
    snapshot = _BipedCommandBaseSnapshot(
      total_speed=zero_speed,
      standing_mask=torch.ones(env.num_envs, device=env.device, dtype=torch.bool),
      linear_speed=zero_speed,
      angular_speed=zero_speed,
      vx=zero_speed,
      vy=zero_speed,
      yaw=zero_speed,
    )
  else:
    linear_speed = torch.norm(command_snapshot.command[:, :2], dim=1)
    angular_speed = torch.abs(command_snapshot.command[:, 2])
    total_speed = linear_speed + angular_speed
    snapshot = _BipedCommandBaseSnapshot(
      total_speed=total_speed,
      standing_mask=total_speed <= command_threshold,
      linear_speed=linear_speed,
      angular_speed=angular_speed,
      vx=command_snapshot.cmd_x if command_snapshot.cmd_x is not None else zero_speed,
      vy=command_snapshot.cmd_y if command_snapshot.cmd_y is not None else zero_speed,
      yaw=command_snapshot.command[:, 2],
    )

  if step >= 0:
    cache[cache_key] = (step, snapshot)
  return snapshot


def _get_command_snapshot(
  env: ManagerBasedRlEnv,
  *,
  command_name: str | None,
) -> _CommandSnapshot:
  """Shared per-step command snapshot for gait reward terms."""
  if command_name is None:
    return _CommandSnapshot(command=None, cmd_x=None, cmd_y=None, cmd_speed=None)

  step = _safe_common_step_counter(env)
  cache = _get_or_init_cache_dict(env, "_gait_command_snapshot_cache")
  cache_key = (command_name,)
  if step >= 0:
    cached = cache.get(cache_key)
    if (
      isinstance(cached, tuple)
      and len(cached) == 2
      and cached[0] == step
      and isinstance(cached[1], _CommandSnapshot)
    ):
      return cached[1]

  command = env.command_manager.get_command(command_name)
  if command is not None:
    cmd_x = command[:, 0]
    cmd_y = command[:, 1]
    cmd_speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
  else:
    cmd_x = None
    cmd_y = None
    cmd_speed = None
  snapshot = _CommandSnapshot(
    command=command,
    cmd_x=cmd_x,
    cmd_y=cmd_y,
    cmd_speed=cmd_speed,
  )
  if step >= 0:
    cache[cache_key] = (step, snapshot)
  return snapshot


def _get_biped_command_regime_snapshot(
  env: ManagerBasedRlEnv,
  *,
  command_name: str | None,
  command_threshold: float,
  running_threshold: float,
) -> _BipedCommandRegimeSnapshot:
  """Shared per-step command/regime snapshot for biped reward terms."""
  step = _safe_common_step_counter(env)
  cache = _get_or_init_cache_dict(env, "_biped_command_regime_cache")
  cache_key = (command_name, float(command_threshold), float(running_threshold))
  if step >= 0:
    cached = cache.get(cache_key)
    if (
      isinstance(cached, tuple)
      and len(cached) == 2
      and cached[0] == step
      and isinstance(cached[1], _BipedCommandRegimeSnapshot)
    ):
      return cached[1]

  base_snapshot = _get_biped_command_base_snapshot(
    env,
    command_name=command_name,
    command_threshold=command_threshold,
  )
  zero_bool = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  if command_name is None:
    snapshot = _BipedCommandRegimeSnapshot(
      total_speed=base_snapshot.total_speed,
      standing_mask=base_snapshot.standing_mask,
      turning_mask=zero_bool,
      walking_mask=zero_bool,
      running_mask=zero_bool,
      lateral_only_mask=zero_bool,
      vy=base_snapshot.vy,
      yaw=base_snapshot.yaw,
    )
  else:
    turning_mask = (base_snapshot.linear_speed < command_threshold) & (
      base_snapshot.angular_speed > command_threshold
    )
    walking_mask = (
      (~base_snapshot.standing_mask)
      & (~turning_mask)
      & (base_snapshot.total_speed < running_threshold)
    )
    running_mask = (
      (~base_snapshot.standing_mask)
      & (~turning_mask)
      & (base_snapshot.total_speed >= running_threshold)
    )
    lateral_only_mask = (
      (torch.abs(base_snapshot.vx) < command_threshold)
      & (torch.abs(base_snapshot.vy) >= command_threshold)
      & (torch.abs(base_snapshot.yaw) < command_threshold)
    )
    snapshot = _BipedCommandRegimeSnapshot(
      total_speed=base_snapshot.total_speed,
      standing_mask=base_snapshot.standing_mask,
      turning_mask=turning_mask,
      walking_mask=walking_mask,
      running_mask=running_mask,
      lateral_only_mask=lateral_only_mask,
      vy=base_snapshot.vy,
      yaw=base_snapshot.yaw,
    )

  if step >= 0:
    cache[cache_key] = (step, snapshot)
  return snapshot


def _select_speed_dependent_std(
  command: torch.Tensor,
  *,
  std: float,
  std_turning: float | None = None,
  std_walking: float,
  std_running: float,
  walking_threshold: float,
  running_threshold: float,
) -> torch.Tensor:
  """Return a per-env std tensor using turning-aware speed classification.

  Buckets: std (below walking_threshold) → std_turning (turn-in-place) →
  std_walking → std_running (above running_threshold).
  """
  if std_turning is None:
    std_turning = std_walking

  linear_speed = torch.norm(command[:, :2], dim=1)
  angular_speed = torch.abs(command[:, 2])
  total_speed = linear_speed + angular_speed
  turning_mask = (linear_speed < walking_threshold) & (
    angular_speed > walking_threshold
  )
  return torch.where(
    turning_mask,
    torch.full_like(total_speed, std_turning),
    torch.where(
      total_speed < walking_threshold,
      torch.full_like(total_speed, std),
      torch.where(
        total_speed < running_threshold,
        torch.full_like(total_speed, std_walking),
        torch.full_like(total_speed, std_running),
      ),
    ),
  )


def _select_biped_air_time_std(
  total_speed: torch.Tensor,
  turning_mask: torch.Tensor,
  *,
  std: float,
  std_turning: float | None,
  std_walking: float,
  std_running: float,
  walking_threshold: float,
  running_threshold: float,
) -> torch.Tensor:
  """Return per-env std for biped air time using command-regime masks."""
  if std_turning is None:
    std_turning = std_walking

  return torch.where(
    turning_mask,
    std_turning,
    torch.where(
      total_speed < walking_threshold,
      std,
      torch.where(
        total_speed < running_threshold,
        std_walking,
        std_running,
      ),
    ),
  )


def _biped_air_time_command_regimes(
  command: torch.Tensor,
  *,
  command_threshold: float,
  running_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Classify commands for biped air-time as standing/turning/walking/running.

  Turning follows the user's intended in-place criterion:
  low linear speed and any non-trivial yaw above the command threshold.
  """
  linear_speed = torch.norm(command[:, :2], dim=1)
  angular_speed = torch.abs(command[:, 2])
  total_speed = linear_speed + angular_speed

  standing_mask = total_speed <= command_threshold
  turning_mask = (linear_speed < command_threshold) & (
    angular_speed > command_threshold
  )
  walking_mask = (~standing_mask) & (~turning_mask) & (total_speed < running_threshold)
  running_mask = (~standing_mask) & (~turning_mask) & (total_speed >= running_threshold)

  return total_speed, standing_mask, turning_mask, walking_mask, running_mask


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


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  std_walking: float,
  std_running: float,
  walking_threshold: float = 0.5,
  running_threshold: float = 1.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  Uses 3-bucket speed classification: std / std_walking / std_running.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  selected_std = _select_speed_dependent_std(
    command,
    std=std,
    std_walking=std_walking,
    std_running=std_running,
    walking_threshold=walking_threshold,
    running_threshold=running_threshold,
  )
  return torch.exp(-lin_vel_error / (selected_std**2))


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float | None = None,
  low_std: float | None = None,
  high_std: float | None = None,
  speed_threshold: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  if low_std is not None or high_std is not None:
    if low_std is None or high_std is None:
      raise ValueError("Both 'low_std' and 'high_std' must be provided together.")
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    selected_std = torch.where(
      total_speed < speed_threshold,
      low_std,
      high_std,
    )
  elif std is not None:
    selected_std = std
  else:
    raise ValueError("Either 'std' or both 'low_std'/'high_std' must be provided.")
  return torch.exp(-ang_vel_error / (selected_std**2))


def _projected_gravity_xy(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return projected gravity xy in the requested body/root frame."""
  asset: Entity = env.scene[asset_cfg.name]

  if isinstance(asset_cfg.body_ids, list) and asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    gravity_w = asset.data.gravity_vec_w.unsqueeze(1).expand(
      -1, len(asset_cfg.body_ids), -1
    )
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
    return projected_gravity_b[..., :2].mean(dim=1)
  return asset.data.projected_gravity_b[:, :2]


def flat_orientation(
  env: ManagerBasedRlEnv,
  std: float | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  std_xy: tuple[float, float] | None = None,
  weight_xy: tuple[float, float] = (1.0, 1.0),
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.

  Two modes (mutually exclusive):

  - **Combined** (provide ``std``): ``exp(-(gx²+gy²) / std²)``, output ∈ [0, 1].
  - **Per-component** (provide ``std_xy=(std_x, std_y)``):
    ``weight_xy[0] * exp(-gx²/std_x²) + weight_xy[1] * exp(-gy²/std_y²)``,
    output ∈ [0, weight_xy[0] + weight_xy[1]].
  """
  gxy = _projected_gravity_xy(env, asset_cfg)

  if std_xy is not None:
    return weight_xy[0] * torch.exp(
      -torch.square(gxy[:, 0]) / std_xy[0] ** 2
    ) + weight_xy[1] * torch.exp(-torch.square(gxy[:, 1]) / std_xy[1] ** 2)
  if std is None:
    raise ValueError(
      "flat_orientation: provide either 'std' (combined) or 'std_xy' (per-component)."
    )
  return torch.exp(-torch.sum(torch.square(gxy), dim=1) / std**2)


class startup_pitch_upright:
  """Penalize pitch deviation around standing-to-moving startup transitions.

  On the first moving step after standing, applies a one-shot penalty equal to
  the mean pitch cost over the last ``pre_window_steps`` standing steps. Then
  applies the current pitch cost for ``post_window_steps`` moving steps,
  including the transition step. Use with a negative weight.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._was_standing = torch.ones(
      env.num_envs,
      device=env.device,
      dtype=torch.bool,
    )
    self._post_steps_left = torch.zeros(
      env.num_envs,
      device=env.device,
      dtype=torch.int32,
    )
    self._pre_window_steps = max(int(cfg.params.get("pre_window_steps", 0)), 0)
    self._pre_cost_sum = torch.zeros(
      env.num_envs,
      device=env.device,
      dtype=torch.float32,
    )
    self._pre_counts = torch.zeros(
      env.num_envs,
      device=env.device,
      dtype=torch.int32,
    )
    self._buf_idx = 0
    if self._pre_window_steps > 0:
      self._pre_cost_buffer = torch.zeros(
        (env.num_envs, self._pre_window_steps),
        device=env.device,
        dtype=torch.float32,
      )
    else:
      self._pre_cost_buffer = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float,
    pre_window_steps: int,
    post_window_steps: int,
    std: float,
    axis: int = 0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    if axis not in (0, 1):
      raise ValueError(f"startup_pitch_upright: axis must be 0 or 1, got {axis}.")
    if int(pre_window_steps) != self._pre_window_steps:
      raise ValueError(
        "startup_pitch_upright: pre_window_steps cannot change after initialization."
      )

    command_snapshot = _get_biped_command_base_snapshot(
      env,
      command_name=command_name,
      command_threshold=command_threshold,
    )
    standing_mask = command_snapshot.standing_mask
    entering_moving = self._was_standing & (~standing_mask)

    gxy = _projected_gravity_xy(env, asset_cfg)
    current_cost = torch.square(gxy[:, axis]) / std**2

    pre_reward = torch.zeros(env.num_envs, device=env.device)
    if self._pre_window_steps > 0:
      pre_count_f = self._pre_counts.to(dtype=current_cost.dtype)
      valid_pre = self._pre_counts > 0
      pre_mean_cost = torch.where(
        valid_pre,
        self._pre_cost_sum / torch.clamp(pre_count_f, min=1.0),
        torch.zeros_like(current_cost),
      )
      pre_reward = torch.where(
        entering_moving & valid_pre,
        pre_mean_cost,
        torch.zeros_like(current_cost),
      )

    post_steps = max(int(post_window_steps), 0)
    decayed_post = torch.where(
      standing_mask,
      torch.zeros_like(self._post_steps_left),
      torch.clamp(self._post_steps_left - 1, min=0),
    )
    reset_post = torch.full_like(decayed_post, post_steps, dtype=decayed_post.dtype)
    self._post_steps_left = torch.where(
      entering_moving & (reset_post > 0),
      reset_post,
      decayed_post,
    )
    post_reward = torch.where(
      (self._post_steps_left > 0) & (~standing_mask),
      current_cost,
      torch.zeros_like(current_cost),
    )

    if self._pre_window_steps > 0:
      assert self._pre_cost_buffer is not None
      moving_mask = ~standing_mask
      if torch.any(moving_mask):
        self._pre_cost_buffer[moving_mask] = 0.0
        self._pre_cost_sum[moving_mask] = 0.0
        self._pre_counts[moving_mask] = 0

      if torch.any(standing_mask):
        old_cost = self._pre_cost_buffer[standing_mask, self._buf_idx]
        new_cost = current_cost[standing_mask]
        self._pre_cost_buffer[standing_mask, self._buf_idx] = new_cost
        self._pre_cost_sum[standing_mask] += new_cost - old_cost
        self._pre_counts[standing_mask] = torch.clamp(
          self._pre_counts[standing_mask] + 1,
          max=self._pre_window_steps,
        )
        self._buf_idx = (self._buf_idx + 1) % self._pre_window_steps

    self._was_standing = standing_mask
    return pre_reward + post_reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._was_standing[env_ids] = True
    self._post_steps_left[env_ids] = 0
    self._pre_cost_sum[env_ids] = 0.0
    self._pre_counts[env_ids] = 0
    if self._pre_cost_buffer is not None:
      self._pre_cost_buffer[env_ids] = 0.0


class upright:
  """Reward for keeping the base upright.

  Without ``terrain_sensor_names``, penalizes tilt relative to world up (correct for
  flat ground).

  With ``terrain_sensor_names``, penalizes tilt relative to the terrain surface normal.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
      "terrain_sensor_names"
    )
    self._debug_vis_enabled = True
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
      body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    else:
      body_quat_w = asset.data.root_link_quat_w  # [B, 4]

    if terrain_sensor_names is not None:
      terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)  # [B, 3]
      # Project terrain normal into body frame. When aligned with the terrain surface
      # this should be (0, 0, 1); XY measures tilt.
      target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
      xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
    else:
      gravity_w = asset.data.gravity_vec_w  # [3]
      projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Unused.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled or self._terrain_sensor_names is None:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]

    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
    if self._asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
    else:
      body_quat_w = asset.data.root_link_quat_w
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
      body_quat_w[:, :3]
    )
    body_up_w = quat_apply(body_quat_w, up_local)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    offset = np.array([0.0, 0.3, 0.0])
    terrain_normal_np = terrain_normal.cpu().numpy()
    body_up_np = body_up_w.cpu().numpy()
    scale = 0.25

    for i in env_indices:
      origin = positions[i] + offset
      # Terrain normal (magenta).
      visualizer.add_arrow(
        start=origin,
        end=origin + terrain_normal_np[i] * scale,
        color=(0.8, 0.2, 0.8, 0.8),
        width=0.01,
      )
      # Body up (orange).
      visualizer.add_arrow(
        start=origin,
        end=origin + body_up_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.8),
        width=0.01,
      )


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
  return data.found.sum(dim=-1).float()


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def _get_total_mass(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  step = _safe_common_step_counter(env)
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = tuple(int(i) for i in asset.indexing.body_ids)
  cache = _get_or_init_cache_dict(env, "_total_mass_cache")
  cache_key = (asset_cfg.name, body_ids)
  if step >= 0:
    cached = cache.get(cache_key)
    if (
      isinstance(cached, tuple)
      and len(cached) == 2
      and cached[0] == step
      and isinstance(cached[1], torch.Tensor)
    ):
      return cached[1]

  body_masses = env.sim.model.body_mass
  if not isinstance(body_masses, torch.Tensor):
    body_masses = torch.tensor(body_masses, device=env.device, dtype=torch.float32)

  if body_masses.ndim == 2:
    total_mass = torch.sum(body_masses[:, body_ids], dim=1)
  else:
    total_mass = torch.sum(body_masses[list(body_ids)])
    total_mass = torch.full(
      (env.num_envs,), total_mass, device=env.device, dtype=torch.float32
    )

  if step >= 0:
    cache[cache_key] = (step, total_mass)
  return total_mass


class biped_air_time:
  """Dense lift reward with speed-aware aggregation for biped gait phases."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg  # Unused.
    self._ema_left = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    self._ema_right = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    # Landing cooldown mask for biped_air_time (env-level).
    self._landing_mask_steps_left = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.int32
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_air_time: float = 0.5,
    std: float = 0.31622776601683794,
    std_turning: float | None = None,
    std_walking: float = 0.3872983346207417,
    std_running: float = 0.5,
    command_name: str | None = None,
    command_threshold: float = 0.05,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.0,
    post_landing_mask_steps_low_speed: int = 0,
    post_landing_mask_steps_high_speed: int = 0,
    symmetry_penalty_weight: float = 0.0,
    symmetry_ema_alpha: float = 0.05,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None  # [B, 2], >= 0 always

    if command_name is None:
      return torch.zeros(env.num_envs, device=env.device)
    command_snapshot = _get_biped_command_regime_snapshot(
      env,
      command_name=command_name,
      command_threshold=command_threshold,
      running_threshold=running_threshold,
    )
    total_command = command_snapshot.total_speed
    standing_mask = command_snapshot.standing_mask
    turning_mask = command_snapshot.turning_mask
    walking_mask = command_snapshot.walking_mask
    running_mask = command_snapshot.running_mask
    lateral_only_mask = command_snapshot.lateral_only_mask

    selected_std = _select_biped_air_time_std(
      total_command,
      turning_mask,
      std=std,
      std_turning=std_turning,
      std_walking=std_walking,
      std_running=std_running,
      walking_threshold=walking_threshold,
      running_threshold=running_threshold,
    )

    zero_reward = torch.zeros(env.num_envs, device=env.device)
    landing_mask_active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    # Optional landing cooldown for non-running regimes only.
    # This creates a short dead-zone after touchdown so the policy does not
    # instantly lift the opposite foot, giving double-support time to form.
    low_speed_steps = int(post_landing_mask_steps_low_speed)
    high_speed_steps = int(post_landing_mask_steps_high_speed)
    low_speed_steps = max(low_speed_steps, 0)
    high_speed_steps = max(high_speed_steps, 0)

    # Speed-branch landing cooldown:
    # - total_command < walking_threshold: low-speed cooldown
    # - walking_threshold <= total_command < running_threshold: high-speed cooldown
    # - total_command >= running_threshold: disabled
    if max(low_speed_steps, high_speed_steps) > 0:
      first_contact = sensor.compute_first_contact(dt=env.step_dt)
      just_landed = torch.any(first_contact, dim=1)
      non_running = ~running_mask
      low_speed = total_command < walking_threshold

      decayed = torch.clamp(self._landing_mask_steps_left - 1, min=0)
      low_steps_t = torch.full_like(decayed, low_speed_steps, dtype=decayed.dtype)
      high_steps_t = torch.full_like(decayed, high_speed_steps, dtype=decayed.dtype)
      reset_steps = torch.where(low_speed, low_steps_t, high_steps_t)
      apply_reset = just_landed & non_running & (reset_steps > 0)
      self._landing_mask_steps_left = torch.where(apply_reset, reset_steps, decayed)

      landing_mask_active = (self._landing_mask_steps_left > 0) & non_running

    reward = zero_reward.clone()
    active_reward_mask = (~standing_mask) & (~landing_mask_active)
    active_foot_reward_full = None
    if torch.any(active_reward_mask):
      current_air_time_active = current_air_time[active_reward_mask]
      selected_std_active = selected_std[active_reward_mask]
      in_air_active = current_air_time_active > 0.0
      air_time_error_sq_active = torch.square(current_air_time_active - target_air_time)
      foot_reward_active = torch.exp(
        -air_time_error_sq_active / torch.square(selected_std_active).unsqueeze(1)
      )
      active_foot_reward = foot_reward_active * in_air_active.float()
      num_in_air_active = torch.sum(in_air_active, dim=1)
      single_support_reward = torch.where(
        num_in_air_active == 1,
        torch.sum(active_foot_reward, dim=1),
        torch.zeros_like(selected_std_active),
      )
      running_reward = torch.max(active_foot_reward, dim=1).values
      reward_active = torch.where(
        running_mask[active_reward_mask],
        running_reward,
        torch.where(
          (turning_mask | walking_mask)[active_reward_mask],
          single_support_reward,
          torch.zeros_like(single_support_reward),
        ),
      )
      reward[active_reward_mask] = reward_active
      if symmetry_penalty_weight != 0.0:
        active_foot_reward_full = torch.zeros_like(current_air_time)
        active_foot_reward_full[active_reward_mask] = active_foot_reward

    # Keep symmetry branch consistent with dead-zone semantics:
    # when landing mask is active, freeze EMA state and skip penalty.
    symmetry_active = (
      (walking_mask | running_mask) & (~landing_mask_active) & (~lateral_only_mask)
    )
    if symmetry_penalty_weight != 0.0 and torch.any(symmetry_active):
      alpha = float(symmetry_ema_alpha)
      alpha = max(0.0, min(1.0, alpha))

      if active_foot_reward_full is None:
        active_foot_reward_full = torch.zeros_like(current_air_time)
      left_reward = active_foot_reward_full[:, 0]
      right_reward = active_foot_reward_full[:, 1]
      symmetry_active_f = symmetry_active.float()

      self._ema_left = torch.where(
        symmetry_active,
        (1.0 - alpha) * self._ema_left + alpha * left_reward,
        self._ema_left,
      )
      self._ema_right = torch.where(
        symmetry_active,
        (1.0 - alpha) * self._ema_right + alpha * right_reward,
        self._ema_right,
      )

      symmetry_penalty = torch.abs(self._ema_left - self._ema_right) * symmetry_active_f
      reward = reward - symmetry_penalty_weight * symmetry_penalty

    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._ema_left[env_ids] = 0.0
    self._ema_right[env_ids] = 0.0
    self._landing_mask_steps_left[env_ids] = 0


class biped_first_swing_foot:
  """One-shot reward for matching the commanded first swing foot after standing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg  # Unused.
    self._was_standing = torch.ones(
      env.num_envs,
      device=env.device,
      dtype=torch.bool,
    )
    self._armed = torch.zeros(
      env.num_envs,
      device=env.device,
      dtype=torch.bool,
    )
    self._desired_foot = torch.full(
      (env.num_envs,),
      -1,
      device=env.device,
      dtype=torch.int64,
    )
    self._armed_steps = torch.zeros(
      env.num_envs,
      device=env.device,
      dtype=torch.int32,
    )

  def _load_transfer_force_diff_norm(
    self,
    env: ManagerBasedRlEnv,
    sensor: ContactSensor,
  ) -> torch.Tensor | None:
    forces = sensor.data.force
    if forces is None:
      return None
    if forces.ndim != 3 or forces.shape[1] < 2 or forces.shape[-1] != 3:
      return None

    desired_foot = self._desired_foot
    valid = desired_foot >= 0
    swing_foot = torch.clamp(desired_foot, min=0, max=1)
    support_foot = 1 - swing_foot
    batch_ids = torch.arange(env.num_envs, device=env.device)

    force_mag = torch.norm(forces[:, :2, :], dim=-1)
    swing_force = force_mag[batch_ids, swing_foot]
    support_force = force_mag[batch_ids, support_foot]
    body_weight = _get_total_mass(env) * 9.81
    force_diff_norm = (support_force - swing_force) / torch.clamp(
      body_weight,
      min=1.0e-6,
    )
    return torch.where(valid, force_diff_norm, torch.zeros_like(force_diff_norm))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str | None = None,
    command_threshold: float = 0.05,
    wrong_swing_penalty: float = 1.0,
    timeout_steps: int = 0,
    no_swing_penalty: float = 0.0,
    min_first_swing_air_time: float = 0.0,
    load_transfer_weight: float = 0.0,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None

    zero_reward = torch.zeros(env.num_envs, device=env.device)
    if command_name is None:
      return zero_reward
    command_snapshot = _get_biped_command_base_snapshot(
      env,
      command_name=command_name,
      command_threshold=command_threshold,
    )
    vy = command_snapshot.vy
    yaw = command_snapshot.yaw
    standing_mask = command_snapshot.standing_mask
    moving_mask = ~standing_mask

    entering_moving = self._was_standing & moving_mask
    arm_now = entering_moving
    if torch.any(arm_now):
      abs_vy = torch.abs(vy)
      abs_yaw = torch.abs(yaw)
      use_vy = (abs_vy >= command_threshold) & (abs_vy >= abs_yaw)
      use_yaw = (~use_vy) & (abs_yaw >= command_threshold)

      desired_foot = torch.full_like(self._desired_foot, -1)
      desired_foot = torch.where(
        (use_vy & (vy > 0.0)) | (use_yaw & (yaw > 0.0)),
        torch.zeros_like(desired_foot),
        desired_foot,
      )
      desired_foot = torch.where(
        (use_vy & (vy < 0.0)) | (use_yaw & (yaw < 0.0)),
        torch.ones_like(desired_foot),
        desired_foot,
      )
      self._desired_foot = torch.where(arm_now, desired_foot, self._desired_foot)
      self._armed = self._armed | arm_now
      self._armed_steps = torch.where(
        arm_now,
        torch.zeros_like(self._armed_steps),
        self._armed_steps,
      )

    qualified_air = (current_air_time > 0.0) & (
      current_air_time >= float(min_first_swing_air_time)
    )
    num_qualified = torch.sum(qualified_air, dim=1)
    any_qualified = num_qualified > 0
    first_swing_now = self._armed & (num_qualified == 1)
    lifted_foot = torch.argmax(qualified_air.to(torch.int64), dim=1)
    directional_start = self._desired_foot >= 0
    correct_first_swing = (
      first_swing_now & directional_start & (lifted_foot == self._desired_foot)
    )
    wrong_first_swing = (
      first_swing_now & directional_start & (lifted_foot != self._desired_foot)
    )
    timed_out = self._armed_steps >= int(timeout_steps)
    timeout_penalty_active = self._armed & moving_mask & (~any_qualified) & timed_out
    reward = (
      correct_first_swing.float()
      - wrong_first_swing.float() * float(wrong_swing_penalty)
      - timeout_penalty_active.float() * float(no_swing_penalty)
    )

    load_transfer_weight_f = float(load_transfer_weight)
    if load_transfer_weight_f != 0.0:
      force_diff_norm = self._load_transfer_force_diff_norm(env, sensor)
      if force_diff_norm is not None:
        dense_active = self._armed & moving_mask & directional_start & (~any_qualified)
        load_transfer_score = torch.clamp(force_diff_norm, min=-1.0, max=1.0)
        load_transfer_reward = (
          dense_active.float() * load_transfer_weight_f * load_transfer_score
        )
        reward = reward + load_transfer_reward

    waiting_for_swing = self._armed & moving_mask & (~any_qualified)
    self._armed_steps = torch.where(
      waiting_for_swing,
      self._armed_steps + 1,
      self._armed_steps,
    )

    clear_state = standing_mask | any_qualified
    self._armed = torch.where(clear_state, torch.zeros_like(self._armed), self._armed)
    self._desired_foot = torch.where(
      clear_state,
      torch.full_like(self._desired_foot, -1),
      self._desired_foot,
    )
    self._armed_steps = torch.where(
      clear_state,
      torch.zeros_like(self._armed_steps),
      self._armed_steps,
    )
    self._was_standing = standing_mask

    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._was_standing[env_ids] = True
    self._armed[env_ids] = False
    self._desired_foot[env_ids] = -1
    self._armed_steps[env_ids] = 0


class biped_double_support_time:
  """Dense reward for current double-support duration during walking motion."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg  # Unused.
    self._current_double_support_time = torch.zeros(
      env.num_envs,
      device=env.device,
      dtype=torch.float32,
    )
    self._has_seen_first_walking_swing = torch.zeros(
      env.num_envs,
      device=env.device,
      dtype=torch.bool,
    )
    self._was_walking_motion = torch.zeros(
      env.num_envs,
      device=env.device,
      dtype=torch.bool,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_double_support_time: float = 0.15,
    std: float = 0.2,
    std_walking: float = 0.35,
    min_first_swing_air_time: float = 0.02,
    command_name: str | None = None,
    command_threshold: float = 0.05,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.0,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None

    if command_name is None:
      return torch.zeros(env.num_envs, device=env.device)
    command_snapshot = _get_biped_command_regime_snapshot(
      env,
      command_name=command_name,
      command_threshold=command_threshold,
      running_threshold=running_threshold,
    )
    total_command = command_snapshot.total_speed
    standing_mask = command_snapshot.standing_mask
    turning_mask = command_snapshot.turning_mask
    walking_mask = command_snapshot.walking_mask
    running_mask = command_snapshot.running_mask
    del standing_mask, turning_mask, running_mask  # Encoded into walking_mask.

    walking_motion_mask = walking_mask
    entering_walking_motion = walking_motion_mask & (~self._was_walking_motion)

    self._current_double_support_time = torch.where(
      entering_walking_motion,
      torch.zeros_like(self._current_double_support_time),
      self._current_double_support_time,
    )
    self._has_seen_first_walking_swing = torch.where(
      entering_walking_motion,
      torch.zeros_like(self._has_seen_first_walking_swing),
      self._has_seen_first_walking_swing,
    )

    in_air = current_air_time > 0.0
    single_support_now = walking_motion_mask & (torch.sum(in_air, dim=1) == 1)
    single_support_air_time = torch.max(current_air_time, dim=1).values
    single_foot_swing = single_support_now & (
      single_support_air_time >= min_first_swing_air_time
    )
    self._has_seen_first_walking_swing = (
      self._has_seen_first_walking_swing | single_foot_swing
    )

    selected_std = torch.where(
      total_command < walking_threshold,
      std,
      std_walking,
    )

    both_contact = (current_air_time <= 0.0).all(dim=1)
    active_double_support = (
      walking_motion_mask & self._has_seen_first_walking_swing & both_contact
    )
    current_double_support_time = torch.where(
      active_double_support,
      self._current_double_support_time + env.step_dt,
      torch.zeros_like(self._current_double_support_time),
    )
    self._current_double_support_time = current_double_support_time
    self._has_seen_first_walking_swing = torch.where(
      walking_motion_mask,
      self._has_seen_first_walking_swing,
      torch.zeros_like(self._has_seen_first_walking_swing),
    )
    self._was_walking_motion = walking_motion_mask

    error_sq = torch.square(current_double_support_time - target_double_support_time)
    reward = torch.exp(-error_sq / torch.square(selected_std))
    reward = torch.where(active_double_support, reward, torch.zeros_like(reward))

    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._current_double_support_time[env_ids] = 0.0
    self._has_seen_first_walking_swing[env_ids] = False
    self._was_walking_motion[env_ids] = False


def biped_standing_stability(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
  standing_weight: float = 1.5,
  standing_air_penalty_weight: float = 0.1,
  standing_action_rate_weight: float = 0.01,
) -> torch.Tensor:
  """Reward quiet double support while standing and suppress idle gaiting."""
  sensor: ContactSensor = env.scene[sensor_name]
  current_air_time = sensor.data.current_air_time
  assert current_air_time is not None

  command_snapshot = _get_biped_command_base_snapshot(
    env,
    command_name=command_name,
    command_threshold=command_threshold,
  )
  standing_mask = command_snapshot.standing_mask
  both_contact = (current_air_time <= 0.0).all(dim=1)
  any_in_air = (current_air_time > 0.0).any(dim=1)
  action_delta = env.action_manager.action - env.action_manager.prev_action
  action_rate_cost = torch.sum(torch.square(action_delta), dim=1)

  reward = (
    standing_weight * both_contact.float()
    - standing_air_penalty_weight * any_in_air.float()
    - standing_action_rate_weight * action_rate_cost
  )
  return torch.where(standing_mask, reward, torch.zeros_like(reward))


def toe_off_push(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.05,
  push_window_steps: int = 2,
  eps: float = 1.0e-6,
  max_reward: float = 1.0,
) -> torch.Tensor:
  """Reward forward/lateral toe-off push from pre-detachment contact force.

  The reward activates only when planar command magnitude exceeds
  ``command_threshold`` and exactly one foot detached in the current step.
  It averages the pre-detachment planar force history excluding the current
  detached sample, projects the mean force onto the unit command direction,
  and normalizes the positive projection by body weight.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  current_air_time = sensor.data.current_air_time
  current_contact_time = sensor.data.current_contact_time
  force_history = sensor.data.force_history
  assert current_air_time is not None
  assert current_contact_time is not None
  if force_history is None:
    return torch.zeros(env.num_envs, device=env.device)

  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  planar_command = command[:, :2]
  planar_norm = torch.norm(planar_command, dim=1)
  active = planar_norm > command_threshold
  if not torch.any(active):
    return torch.zeros(env.num_envs, device=env.device)

  first_detach = sensor.compute_first_air(dt=env.step_dt)
  assert first_detach is not None
  detached_count = torch.sum(first_detach, dim=1)
  single_detach = active & (detached_count == 1)
  if not torch.any(single_detach):
    return torch.zeros(env.num_envs, device=env.device)

  if force_history.ndim != 4 or force_history.shape[-1] != 3:
    raise ValueError("force_history must have shape [B, N, H, 3].")
  if force_history.shape[2] <= 1:
    return torch.zeros(env.num_envs, device=env.device)

  body_weight = _get_total_mass(env) * 9.81
  cmd_dir = planar_command / torch.clamp(planar_norm.unsqueeze(-1), min=eps)

  batch_ids = torch.arange(env.num_envs, device=env.device)
  detached_foot = torch.argmax(first_detach.float(), dim=1)
  window_steps = min(push_window_steps, force_history.shape[2] - 1)
  if window_steps <= 0:
    return torch.zeros(env.num_envs, device=env.device)

  pre_detach_force = force_history[
    batch_ids,
    detached_foot,
    1 : window_steps + 1,
    :2,
  ]
  raw_planar_force = torch.mean(pre_detach_force, dim=1)
  # With primary=foot and secondary=terrain, MuJoCo contact force follows the
  # reference-side convention, so negate it to score the foot reaction force.
  foot_planar_force = -raw_planar_force
  projected_force = torch.sum(foot_planar_force * cmd_dir, dim=1)
  reward = torch.relu(projected_force / torch.clamp(body_weight, min=eps))
  reward = torch.clamp(reward, max=max_reward)
  return reward * single_detach.float()


def standing_air_time_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.3,
) -> torch.Tensor:
  """Return a unit positive cost when any foot leaves the ground while standing."""
  sensor: ContactSensor = env.scene[sensor_name]
  current_air_time = sensor.data.current_air_time
  assert current_air_time is not None

  total_command = torch.zeros(env.num_envs, device=env.device)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])

  is_standing = total_command <= command_threshold
  any_in_air = (current_air_time > 0.0).any(dim=1)
  return is_standing.float() * any_in_air.float()


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


# class gait_cycle:
#   """Compact biped gait timing reward: event cycle + no-air penalty."""

#   # Command gating.
#   _CMD_THRESHOLD = 0.05
#   _YAW_WEIGHT = 1.0

#   # Valid same-foot stride (HS->HS) range.
#   _STRIDE_RANGE_S = (0.3, 2.5)

#   # Fixed reference period mapping (fast, slow) and normalization speed range.
#   _PERIOD_REF_RANGE_S = (0.6, 2.0)
#   _PERIOD_SPEED_RANGE = (0.20, 2.50)
#   _PERIOD_BAND_RATIO = 0.2
#   _PERIOD_BAND_MIN_S = 0.05

#   # Phase shaping.
#   _PHASE_SIGMA_RANGE = (0.10, 0.50)  # straight -> turn-dominant

#   # Turn dominance detection.
#   _TURN_YAW_RANGE = (0.10, 0.50)
#   _TURN_LIN_SPEED_RANGE = (0.00, 0.20)

#   # No-air penalty.
#   _NO_AIR_HORIZON_S = 0.50

#   # Reward weights.
#   _W_PERIOD_EVENT = 0.25
#   _W_PHASE_EVENT = 0.50
#   _W_NO_AIR = 1.00

#   _EPS = 1.0e-6

#   def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
#     self._num_envs = env.num_envs
#     self._device = env.device
#     self._step_dt = env.step_dt

#     sensor_name = cfg.params.get("sensor_name")
#     if sensor_name is None:
#       raise ValueError("gait_cycle requires 'sensor_name' in params.")

#     contact_sensor: ContactSensor = env.scene[sensor_name]
#     current_air_time = contact_sensor.data.current_air_time
#     if current_air_time is None:
#       raise RuntimeError(
#         f"Contact sensor '{sensor_name}' must have track_air_time=True for gait_cycle."
#       )

#     self._num_feet = current_air_time.shape[1]
#     self._is_biped = self._num_feet == 2

#     # Unpack tuple ranges for readability and lower indexing overhead.
#     self._stride_min_s, self._stride_max_s = self._STRIDE_RANGE_S
#     self._period_fast_s, self._period_slow_s = self._PERIOD_REF_RANGE_S
#     self._period_speed_min, self._period_speed_max = self._PERIOD_SPEED_RANGE
#     self._phase_sigma_straight, self._phase_sigma_turn = self._PHASE_SIGMA_RANGE
#     self._turn_yaw_min, self._turn_yaw_max = self._TURN_YAW_RANGE
#     self._turn_lin_min, self._turn_lin_max = self._TURN_LIN_SPEED_RANGE
#     self._clock = torch.zeros(self._num_envs, device=self._device)
#     self._last_strike_time = torch.full(
#       (self._num_envs, 2), float("nan"), device=self._device
#     )
#     self._stride_time = torch.full(
#       (self._num_envs, 2),
#       0.5 * (self._stride_min_s + self._stride_max_s),
#       device=self._device,
#     )
#     self._no_air_time = torch.zeros(self._num_envs, device=self._device)

#   @staticmethod
#   def _linear_ramp(
#     x: torch.Tensor, lo: float, hi: float, eps: float = 1.0e-6
#   ) -> torch.Tensor:
#     denom = max(hi - lo, eps)
#     return torch.clamp((x - lo) / denom, 0.0, 1.0)

#   def _compute_period_ref(self, linear_speed: torch.Tensor) -> torch.Tensor:
#     """Map translational speed to a fixed target period (slow->fast)."""
#     speed_gate = self._linear_ramp(
#       linear_speed,
#       self._period_speed_min,
#       self._period_speed_max,
#       self._EPS,
#     )
#     period_ref = self._period_slow_s + (
#       self._period_fast_s - self._period_slow_s
#     ) * speed_gate
#     return torch.clamp(period_ref, min=self._stride_min_s, max=self._stride_max_s)

#   def _compute_turn_gate(
#     self,
#     linear_speed: torch.Tensor,
#     yaw_speed: torch.Tensor,
#   ) -> torch.Tensor:
#     """High for high yaw OR low-linear+moderate-yaw turn-dominant cases."""
#     yaw_gate_strong = self._linear_ramp(
#       yaw_speed, self._turn_yaw_min, self._turn_yaw_max, self._EPS
#     )
#     low_lin_gate = 1.0 - self._linear_ramp(
#       linear_speed, self._turn_lin_min, self._turn_lin_max, self._EPS
#     )
#     # Lower yaw thresholds for low-linear turning regime.
#     yaw_gate_combo = self._linear_ramp(
#       yaw_speed, 0.5 * self._turn_yaw_min, 0.5 * self._turn_yaw_max, self._EPS
#     )
#     turn_combo = yaw_gate_combo * low_lin_gate
#     return torch.clamp(torch.maximum(yaw_gate_strong, turn_combo), 0.0, 1.0)

#   def __call__(
#     self,
#     env: ManagerBasedRlEnv,
#     sensor_name: str,
#     command_name: str,
#     phase_target: float,
#   ) -> torch.Tensor:
#     self._clock += self._step_dt
#     reward = torch.zeros(self._num_envs, device=self._device)

#     # Biped-only term.
#     if not self._is_biped:
#       return reward

#     command = env.command_manager.get_command(command_name)
#     if command is None:
#       self._no_air_time.zero_()
#       return reward

#     contact_sensor: ContactSensor = env.scene[sensor_name]
#     current_air_time = contact_sensor.data.current_air_time
#     assert current_air_time is not None

#     phase_target = float(phase_target)
#     phase_target = phase_target % 1.0

#     linear_speed = torch.norm(command[:, :2], dim=1)
#     yaw = command[:, 2]
#     yaw_speed = torch.abs(yaw)
#     cmd_speed = linear_speed + self._YAW_WEIGHT * yaw_speed
#     active = cmd_speed > self._CMD_THRESHOLD

#     # No command -> fully disabled.
#     if not active.any():
#       self._no_air_time.zero_()
#       return reward

#     # Turn-aware modulation.
#     turn_gate = self._compute_turn_gate(linear_speed, yaw_speed)
#     phase_sigma_eff = self._phase_sigma_straight + (
#       self._phase_sigma_turn - self._phase_sigma_straight
#     ) * turn_gate
#     phase_target_eff = torch.full_like(linear_speed, phase_target)

#     # No-air penalty branch.
#     has_air_time = (current_air_time > 0.0).any(dim=1)

#     self._no_air_time = torch.where(
#       active & (~has_air_time),
#       self._no_air_time + self._step_dt,
#       torch.zeros_like(self._no_air_time),
#     )
#     no_air_penalty = torch.clamp(
#       self._no_air_time / self._NO_AIR_HORIZON_S,
#       min=0.0,
#       max=1.0,
#     )
#     reward -= self._W_NO_AIR * no_air_penalty * active.float()

#     # Event reward on first contact.
#     # Use command activity only; first-contact already defines the landing event.
#     first_contact = contact_sensor.compute_first_contact(dt=self._step_dt)
#     strike_now = first_contact & active.unsqueeze(-1)
#     prev_last_strike = self._last_strike_time.clone()

#     for foot_idx in range(2):
#       strike_env_ids = strike_now[:, foot_idx].nonzero(as_tuple=False).squeeze(-1)
#       if strike_env_ids.numel() == 0:
#         continue

#       now = self._clock[strike_env_ids]
#       prev_same = prev_last_strike[strike_env_ids, foot_idx]
#       observed_stride = now - prev_same
#       valid_stride = (
#         torch.isfinite(prev_same)
#         & (observed_stride >= self._stride_min_s)
#         & (observed_stride <= self._stride_max_s)
#       )

#       # Update strike timestamp for all strike envs after reading prev snapshot.
#       self._last_strike_time[strike_env_ids, foot_idx] = now
#       if valid_stride.any():
#         valid_env_ids = strike_env_ids[valid_stride]
#         stride = observed_stride[valid_stride]

#         period_ref = self._compute_period_ref(linear_speed[valid_env_ids])
#         period_band = torch.clamp(
#           period_ref * self._PERIOD_BAND_RATIO,
#           min=self._PERIOD_BAND_MIN_S,
#         )
#         period_dev = torch.abs(stride - period_ref) - period_band
#         period_dev = torch.clamp(period_dev, min=0.0)
#         period_score = torch.exp(
#           -torch.square(period_dev / (period_band + self._EPS))
#         )

#         # Phase score for strike against opposite-foot cycle.
#         other_idx = 1 - foot_idx
#         other_last = prev_last_strike[valid_env_ids, other_idx]
#         other_stride = self._stride_time[valid_env_ids, other_idx]
#         phase_score = torch.zeros_like(period_score)
#         valid_phase = (
#           torch.isfinite(other_last)
#           & (other_stride >= self._stride_min_s)
#           & (other_stride <= self._stride_max_s)
#         )
#         if valid_phase.any():
#           phase_env_ids = valid_env_ids[valid_phase]
#           phase_dt = now[valid_stride][valid_phase] - other_last[valid_phase]
#           phase = torch.remainder(
#             phase_dt / (other_stride[valid_phase] + self._EPS), 1.0
#           )
#           phase_err = torch.abs(phase - phase_target_eff[phase_env_ids])
#           phase_err = torch.minimum(phase_err, 1.0 - phase_err)
#           phase_score[valid_phase] = torch.exp(
#             -torch.square(phase_err / (phase_sigma_eff[phase_env_ids] + self._EPS))
#           )

#         cycle_score = (
#           self._W_PERIOD_EVENT * period_score
#           + self._W_PHASE_EVENT * phase_score
#         )
#         reward[valid_env_ids] += cycle_score
#         self._stride_time[valid_env_ids, foot_idx] = stride

#     return reward

#   def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
#     if env_ids is None:
#       env_ids = slice(None)
#     self._clock[env_ids] = 0.0
#     self._last_strike_time[env_ids] = float("nan")
#     self._stride_time[env_ids] = 0.5 * (self._stride_min_s + self._stride_max_s)
#     self._no_air_time[env_ids] = 0.0


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"feet_clearance requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
  )
  foot_height = height_sensor.data.heights  # [B, F]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
  delta = torch.abs(foot_height - target_height)  # [B, F]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      f"feet_swing_height requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    foot_heights = height_sensor.data.heights
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.peak_heights[env_ids] = 0.0


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class _FootPitchPatternCore:
  """Shared core for foot-pitch pattern terms to avoid duplicated compute per step."""

  def __init__(
    self,
    *,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    sensor_name: str,
    command_name: str | None,
    window_steps: int,
    temporal_std_steps: float,
  ):
    self._asset_cfg = asset_cfg
    self._sensor_name = sensor_name
    self._command_name = command_name
    self._window_steps = window_steps

    self._num_feet = len(self._asset_cfg.body_ids)
    self._ones_env = torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
    self._cached_step: int = -1
    self._cached_state: _FootPitchPatternState | None = None
    self._compute_calls: int = 0
    self._buf_idx = 0

    if window_steps > 0:
      # Ring buffer: [B, N_feet, window_steps]
      self._buffer = torch.zeros(
        (env.num_envs, self._num_feet, window_steps),
        device=env.device,
        dtype=torch.float32,
      )
      self._buf_idx = 0
      # Pre-compute temporal weights: index 0 = oldest, -1 = most recent.
      offsets = torch.arange(window_steps, 0, -1, dtype=torch.float32)
      self._temporal_w = torch.exp(-offsets.square() / (temporal_std_steps**2)).to(
        env.device
      )  # [W]
      temporal_w_by_buf_idx = []
      for buf_idx in range(window_steps):
        ordered_idx = (
          torch.arange(window_steps, device=env.device) + buf_idx
        ) % window_steps
        weights_by_slot = torch.empty_like(self._temporal_w)
        weights_by_slot[ordered_idx] = self._temporal_w
        temporal_w_by_buf_idx.append(weights_by_slot)
      self._temporal_w_by_buf_idx = torch.stack(temporal_w_by_buf_idx, dim=0)
    else:
      self._temporal_w = None
      self._temporal_w_by_buf_idx = None

  def _compute(self, env: ManagerBasedRlEnv) -> _FootPitchPatternState:
    self._compute_calls += 1
    asset: Entity = env.scene[self._asset_cfg.name]
    contact_event = _get_contact_event_snapshot(env, sensor_name=self._sensor_name)

    # Current gx
    body_quat_w = asset.data.body_link_quat_w[
      :, self._asset_cfg.body_ids, :
    ]  # [B, N, 4]
    gravity_w = asset.data.gravity_vec_w.unsqueeze(1).expand(-1, self._num_feet, -1)
    gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
    gx = gravity_b[..., 0]  # [B, N]

    # Update ring buffer before transition check.
    if self._window_steps > 0:
      self._buffer[:, :, self._buf_idx] = gx
      self._buf_idx = (self._buf_idx + 1) % self._window_steps

    # Transition masks and phase durations are shared per-step by sensor name.
    just_landed = contact_event.just_landed
    just_lifted = contact_event.just_lifted
    last_air = contact_event.last_air_time
    last_contact = contact_event.last_contact_time
    current_air = contact_event.current_air_time
    current_contact = contact_event.current_contact_time

    # Command snapshot (term-specific gating is applied outside the core).
    command_snapshot = _get_command_snapshot(env, command_name=self._command_name)
    cmd_x = command_snapshot.cmd_x
    cmd_speed = command_snapshot.cmd_speed

    use_window = self._window_steps > 0
    window_buf: torch.Tensor | None = self._buffer if use_window else None

    return _FootPitchPatternState(
      gx=gx,
      window_buf=window_buf,
      buf_idx=self._buf_idx,
      just_landed=just_landed,
      just_lifted=just_lifted,
      last_air_time=last_air,
      last_contact_time=last_contact,
      current_air_time=current_air,
      current_contact_time=current_contact,
      cmd_x=cmd_x,
      cmd_speed=cmd_speed,
      use_window=bool(use_window),
    )

  def compute(self, env: ManagerBasedRlEnv) -> _FootPitchPatternState:
    step = _safe_common_step_counter(env)
    if step >= 0 and self._cached_step == step and self._cached_state is not None:
      return self._cached_state
    state = self._compute(env)
    if step >= 0:
      self._cached_step = step
      self._cached_state = state
    return state

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Clear cached state and ring-buffer rows for reset environments."""
    if env_ids is None:
      env_ids = slice(None)
    self._cached_step = -1
    self._cached_state = None
    if self._window_steps > 0:
      self._buffer[env_ids] = 0.0
      # Full reset: restart ring-buffer write pointer.
      if isinstance(env_ids, slice) and env_ids == slice(None):
        self._buf_idx = 0


class foot_pitch_pattern:
  """Reward heel-strike at landing and toe-off at liftoff using foot pitch proxy.

  Same core state/compute can be shared across multiple reward terms that use
  identical foot-pitch state parameters. This enables clean split terms
  (heel-only / toe-only) without duplicate expensive state extraction.

  Sign convention used by this term:
  - ``gx`` is the projected gravity x-component in the foot body frame.
  - ``direction`` is +1 for forward commands (cmd_x >= 0), -1 for backward.
  - ``direction * gx`` normalizes motion direction so one formula works for
    both forward and backward walking.
  - ``heel`` mode rewards negative signed gx near landing (heel-down trend).
  - ``toe`` mode rewards positive signed gx near liftoff (toe-down trend).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self._asset_cfg: SceneEntityCfg = p.get("asset_cfg", _DEFAULT_ASSET_CFG)
    self._sensor_name: str = p["sensor_name"]
    self._command_name: str | None = p.get("command_name")
    self._command_threshold: float = p.get("command_threshold", 0.05)
    self._running_threshold: float | None = p.get("running_threshold")
    self._mode: str = str(p.get("mode", "heel")).lower()
    if self._mode not in {"heel", "toe"}:
      raise ValueError(
        f"foot_pitch_pattern mode must be one of ('heel', 'toe'), got: {self._mode}"
      )
    self._target_gx: float = math.sin(math.radians(p.get("target_angle_deg", 10.0)))
    self._std: float = p.get("std", math.sin(math.radians(5)))
    self._inv_std_sq: float = 1.0 / max(self._std**2, 1.0e-12)
    # Backward compatibility:
    # - legacy: min_phase_steps applied to both landing and liftoff.
    # - new:    min_air_steps (landing), min_contact_steps (liftoff).
    legacy_min_phase_steps: int = int(p.get("min_phase_steps", 0))
    self._min_air_steps: int = int(p.get("min_air_steps", legacy_min_phase_steps))
    self._min_contact_steps: int = int(
      p.get("min_contact_steps", legacy_min_phase_steps)
    )
    self._min_air_time: float = (
      self._min_air_steps * env.step_dt if self._min_air_steps > 0 else 0.0
    )
    self._min_contact_time: float = (
      self._min_contact_steps * env.step_dt if self._min_contact_steps > 0 else 0.0
    )
    self._window_steps: int = p.get("window_steps", 0)
    self._temporal_std_steps: float = p.get("temporal_std_steps", 3.0)
    self._post_reward_steps: int = max(int(p.get("post_reward_steps", 1)), 1)
    self._post_reward_std_steps: float = max(
      float(p.get("post_reward_std_steps", 1.0)), 1.0e-6
    )
    self._post_reward_inv_std_sq: float = 1.0 / (self._post_reward_std_steps**2)

    cache_key = (
      self._asset_cfg.name,
      tuple(int(b) for b in self._asset_cfg.body_ids),
      self._sensor_name,
      self._command_name,
      int(self._window_steps),
      float(self._temporal_std_steps),
    )
    cache_attr = "_foot_pitch_pattern_core_cache"
    core_cache = _get_or_init_cache_dict(env, cache_attr)
    if cache_key not in core_cache:
      core_cache[cache_key] = _FootPitchPatternCore(
        env=env,
        asset_cfg=self._asset_cfg,
        sensor_name=self._sensor_name,
        command_name=self._command_name,
        window_steps=self._window_steps,
        temporal_std_steps=self._temporal_std_steps,
      )
    self._core = core_cache[cache_key]

    self._symmetry_penalty_weight: float = p.get("symmetry_penalty_weight", 0.0)
    self._symmetry_ema_alpha: float = p.get("symmetry_ema_alpha", 0.2)
    self._ema_0 = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    self._ema_1 = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    mode: str = "heel",
    heel_strike_weight: float = 1.0,
    toe_off_weight: float = 1.0,
    command_name: str | None = None,
    command_threshold: float = 0.05,
    running_threshold: float | None = None,
    target_angle_deg: float = 10.0,
    std: float = math.sin(math.radians(5)),
    window_steps: int = 0,
    temporal_std_steps: float = 3.0,
    post_reward_steps: int = 1,
    post_reward_std_steps: float = 1.0,
    min_phase_steps: int = 0,
    min_air_steps: int = 0,
    min_contact_steps: int = 0,
    symmetry_penalty_weight: float = 0.0,
    symmetry_ema_alpha: float = 0.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del (
      sensor_name,
      mode,
      heel_strike_weight,
      toe_off_weight,
      command_name,
      command_threshold,
      running_threshold,
      target_angle_deg,
      std,
      window_steps,
      temporal_std_steps,
      post_reward_steps,
      post_reward_std_steps,
      min_phase_steps,
      min_air_steps,
      min_contact_steps,
      symmetry_penalty_weight,
      symmetry_ema_alpha,
      asset_cfg,
    )  # Params are resolved once at init via cfg.
    state = self._core.compute(env)
    gx = state.gx
    window_buf = state.window_buf
    buf_idx = state.buf_idx
    just_landed = state.just_landed
    just_lifted = state.just_lifted
    last_air_time = state.last_air_time
    last_contact_time = state.last_contact_time
    current_air_time = state.current_air_time
    current_contact_time = state.current_contact_time
    cmd_x = state.cmd_x
    cmd_speed = state.cmd_speed
    use_window = state.use_window

    ones_env = self._core._ones_env
    direction = ones_env
    active = ones_env
    if cmd_x is not None:
      active = (torch.abs(cmd_x) > self._command_threshold).float()
      # direction: +1 forward, -1 backward.
      direction = torch.where(cmd_x < 0.0, -ones_env, ones_env)
    hs_scale: torch.Tensor | None = None
    if self._mode == "heel":
      hs_scale = ones_env
      if self._running_threshold is not None and cmd_speed is not None:
        hs_scale = (cmd_speed <= self._running_threshold).float()
      # Landing event filter: ignore too-short preceding air phases.
      landing_valid = torch.ones_like(just_landed)
      if self._min_air_time > 0.0:
        landing_valid = (last_air_time >= self._min_air_time).float()
      just_landed = just_landed * landing_valid
      just_landed = (
        _post_reward_event_mask(
          event_mask=just_landed,
          phase_time=current_contact_time,
          step_dt=env.step_dt,
          post_reward_steps=self._post_reward_steps,
          post_reward_inv_std_sq=self._post_reward_inv_std_sq,
        )
        * landing_valid
      )
    else:
      # Liftoff event filter: ignore too-short preceding contact phases.
      liftoff_valid = torch.ones_like(just_lifted)
      if self._min_contact_time > 0.0:
        liftoff_valid = (
          liftoff_valid * (last_contact_time >= self._min_contact_time).float()
        )
      # Also filter liftoffs with no prior air phase (standing → first step).
      if self._min_air_time > 0.0:
        liftoff_valid = liftoff_valid * (last_air_time >= self._min_air_time).float()
      just_lifted = just_lifted * liftoff_valid
      just_lifted = (
        _post_reward_event_mask(
          event_mask=just_lifted,
          phase_time=current_air_time,
          step_dt=env.step_dt,
          post_reward_steps=self._post_reward_steps,
          post_reward_inv_std_sq=self._post_reward_inv_std_sq,
        )
        * liftoff_valid
      )

    # Multi-step post rewards rescore the current state each step. The legacy
    # temporal window remains available for event-only terms.
    use_temporal_window = use_window and self._post_reward_steps <= 1
    if use_temporal_window:
      assert window_buf is not None
      dir_exp = direction.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
      assert self._core._temporal_w_by_buf_idx is not None
      temporal_w = self._core._temporal_w_by_buf_idx[buf_idx].view(1, 1, -1)
      if self._mode == "heel":
        # Heel-strike should favor heel-down (forward cmd -> gx < 0).
        err = -dir_exp * window_buf - self._target_gx
        score_buf = torch.exp(-err.square() * self._inv_std_sq)
        assert hs_scale is not None
        raw_score = (score_buf * temporal_w).sum(dim=-1)
        score = raw_score * just_landed
        reward = (hs_scale.unsqueeze(-1) * score).sum(dim=1)
      else:
        # Toe-off should favor toe-down (forward cmd -> gx > 0).
        err = dir_exp * window_buf - self._target_gx
        score_buf = torch.exp(-err.square() * self._inv_std_sq)
        raw_score = (score_buf * temporal_w).sum(dim=-1)
        score = raw_score * just_lifted
        reward = score.sum(dim=1)
    else:
      if self._mode == "heel":
        # Heel-strike should favor heel-down (forward cmd -> gx < 0).
        err = -direction.unsqueeze(-1) * gx - self._target_gx
        raw_score = torch.exp(-err.square() * self._inv_std_sq)
        score = raw_score * just_landed
        assert hs_scale is not None
        reward = (hs_scale.unsqueeze(-1) * score).sum(dim=1)
      else:
        # Toe-off should favor toe-down (forward cmd -> gx > 0).
        err = direction.unsqueeze(-1) * gx - self._target_gx
        raw_score = torch.exp(-err.square() * self._inv_std_sq)
        score = raw_score * just_lifted
        reward = score.sum(dim=1)

    # Event-based symmetry penalty for biped (N_feet == 2).
    # EMA updates only when an event fires AND the reward term is active
    # (standing/running-gated envs are excluded to avoid EMA contamination).
    if self._symmetry_penalty_weight != 0.0 and raw_score.shape[1] == 2:
      event = just_landed if self._mode == "heel" else just_lifted
      symmetry_active = active
      if self._mode == "heel" and hs_scale is not None:
        symmetry_active = symmetry_active * hs_scale
      symmetry_active_bool = symmetry_active > 0.5
      update_0 = (event[:, 0] > 0) & symmetry_active_bool
      update_1 = (event[:, 1] > 0) & symmetry_active_bool
      self._ema_0 = torch.where(
        update_0,
        (1.0 - self._symmetry_ema_alpha) * self._ema_0
        + self._symmetry_ema_alpha * raw_score[:, 0],
        self._ema_0,
      )
      self._ema_1 = torch.where(
        update_1,
        (1.0 - self._symmetry_ema_alpha) * self._ema_1
        + self._symmetry_ema_alpha * raw_score[:, 1],
        self._ema_1,
      )
      symmetry_penalty = torch.abs(self._ema_0 - self._ema_1) * symmetry_active
      reward = reward - self._symmetry_penalty_weight * symmetry_penalty

    return reward * active

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Delegate reset to shared core (buffer/cache cleanup)."""
    if env_ids is None:
      env_ids = slice(None)
    self._core.reset(env_ids=env_ids)
    self._ema_0[env_ids] = 0.0
    self._ema_1[env_ids] = 0.0


class _FootPitchPatternState(NamedTuple):
  gx: torch.Tensor
  window_buf: torch.Tensor | None
  buf_idx: int
  just_landed: torch.Tensor
  just_lifted: torch.Tensor
  last_air_time: torch.Tensor
  last_contact_time: torch.Tensor
  current_air_time: torch.Tensor
  current_contact_time: torch.Tensor
  cmd_x: torch.Tensor | None
  cmd_speed: torch.Tensor | None
  use_window: bool


class _JointGaitPatternState(NamedTuple):
  joint_pos: torch.Tensor
  window_buf: torch.Tensor | None
  buf_idx: int
  just_landed: torch.Tensor
  just_lifted: torch.Tensor
  last_air_time: torch.Tensor
  last_contact_time: torch.Tensor
  current_air_time: torch.Tensor
  current_contact_time: torch.Tensor
  cmd_x: torch.Tensor | None
  cmd_y: torch.Tensor | None
  cmd_speed: torch.Tensor | None
  use_window: bool


class _JointGaitPatternCore:
  """Shared core for joint gait-pattern terms to avoid duplicated event extraction."""

  def __init__(
    self,
    *,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    sensor_name: str,
    command_name: str | None,
    window_steps: int,
    temporal_std_steps: float,
  ):
    self._asset_cfg = asset_cfg
    self._sensor_name = sensor_name
    self._command_name = command_name
    self._window_steps = window_steps
    self._ones_env = torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
    self._cached_step: int = -1
    self._cached_state: _JointGaitPatternState | None = None
    self._compute_calls: int = 0
    self._buf_idx = 0

    asset: Entity = env.scene[self._asset_cfg.name]
    self._num_joints = int(asset.data.joint_pos[:, self._asset_cfg.joint_ids].shape[1])

    if window_steps > 0:
      # Ring buffer: [B, N_joint, window_steps]
      self._buffer = torch.zeros(
        (env.num_envs, self._num_joints, window_steps),
        device=env.device,
        dtype=torch.float32,
      )
      self._buf_idx = 0
      offsets = torch.arange(window_steps, 0, -1, dtype=torch.float32)
      self._temporal_w = torch.exp(-offsets.square() / (temporal_std_steps**2)).to(
        env.device
      )
      temporal_w_by_buf_idx = []
      for buf_idx in range(window_steps):
        ordered_idx = (
          torch.arange(window_steps, device=env.device) + buf_idx
        ) % window_steps
        weights_by_slot = torch.empty_like(self._temporal_w)
        weights_by_slot[ordered_idx] = self._temporal_w
        temporal_w_by_buf_idx.append(weights_by_slot)
      self._temporal_w_by_buf_idx = torch.stack(temporal_w_by_buf_idx, dim=0)
    else:
      self._temporal_w = None
      self._temporal_w_by_buf_idx = None

  def _compute(self, env: ManagerBasedRlEnv) -> _JointGaitPatternState:
    self._compute_calls += 1
    asset: Entity = env.scene[self._asset_cfg.name]
    contact_event = _get_contact_event_snapshot(env, sensor_name=self._sensor_name)

    joint_pos = asset.data.joint_pos[:, self._asset_cfg.joint_ids]  # [B, N_joint]

    if self._window_steps > 0:
      self._buffer[:, :, self._buf_idx] = joint_pos
      self._buf_idx = (self._buf_idx + 1) % self._window_steps

    just_landed = contact_event.just_landed
    just_lifted = contact_event.just_lifted
    last_air = contact_event.last_air_time
    last_contact = contact_event.last_contact_time
    current_air = contact_event.current_air_time
    current_contact = contact_event.current_contact_time

    command_snapshot = _get_command_snapshot(env, command_name=self._command_name)
    cmd_x = command_snapshot.cmd_x
    cmd_y = command_snapshot.cmd_y
    cmd_speed = command_snapshot.cmd_speed

    use_window = self._window_steps > 0
    window_buf: torch.Tensor | None = self._buffer if use_window else None

    return _JointGaitPatternState(
      joint_pos=joint_pos,
      window_buf=window_buf,
      buf_idx=self._buf_idx,
      just_landed=just_landed,
      just_lifted=just_lifted,
      last_air_time=last_air,
      last_contact_time=last_contact,
      current_air_time=current_air,
      current_contact_time=current_contact,
      cmd_x=cmd_x,
      cmd_y=cmd_y,
      cmd_speed=cmd_speed,
      use_window=bool(use_window),
    )

  def compute(self, env: ManagerBasedRlEnv) -> _JointGaitPatternState:
    step = _safe_common_step_counter(env)
    if step >= 0 and self._cached_step == step and self._cached_state is not None:
      return self._cached_state
    state = self._compute(env)
    if step >= 0:
      self._cached_step = step
      self._cached_state = state
    return state

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._cached_step = -1
    self._cached_state = None
    if self._window_steps > 0:
      self._buffer[env_ids] = 0.0
      if isinstance(env_ids, slice) and env_ids == slice(None):
        self._buf_idx = 0


class joint_gait_pattern:
  """Reward target joint angle at toe-off or heel-strike events.

  This term mirrors the event-driven structure used by ``foot_pitch_pattern``:
  - ``toe`` mode: evaluate at first-air (toe-off) events.
  - ``heel`` mode: evaluate at first-contact (heel-strike) events.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self._asset_cfg: SceneEntityCfg = p.get("asset_cfg", _DEFAULT_ASSET_CFG)
    self._sensor_name: str = p["sensor_name"]
    self._command_name: str | None = p.get("command_name")
    self._command_threshold: float = p.get("command_threshold", 0.05)
    self._forward_only: bool = p.get("forward_only", False)
    self._mode: str = str(p.get("mode", "toe")).lower()
    if self._mode not in {"heel", "toe"}:
      raise ValueError(
        f"joint_gait_pattern mode must be one of ('heel', 'toe'), got: {self._mode}"
      )
    self._target_angle_rad: float = math.radians(p.get("target_angle_deg", 30.0))
    self._std: float = float(p.get("std", math.radians(10.0)))
    self._std_running: float = float(p.get("std_running", self._std))
    self._running_threshold: float | None = p.get("running_threshold")
    self._window_steps: int = p.get("window_steps", 0)
    self._temporal_std_steps: float = p.get("temporal_std_steps", 3.0)
    self._post_reward_steps: int = max(int(p.get("post_reward_steps", 1)), 1)
    self._post_reward_std_steps: float = max(
      float(p.get("post_reward_std_steps", 1.0)), 1.0e-6
    )
    self._post_reward_inv_std_sq: float = 1.0 / (self._post_reward_std_steps**2)
    self._min_air_steps: int = int(p.get("min_air_steps", 0))
    self._min_contact_steps: int = int(p.get("min_contact_steps", 0))
    self._min_air_time: float = (
      self._min_air_steps * env.step_dt if self._min_air_steps > 0 else 0.0
    )
    self._min_contact_time: float = (
      self._min_contact_steps * env.step_dt if self._min_contact_steps > 0 else 0.0
    )
    self._inv_std_sq: float = 1.0 / max(self._std**2, 1.0e-12)
    self._inv_std_running_sq: float = 1.0 / max(self._std_running**2, 1.0e-12)

    joint_ids = self._asset_cfg.joint_ids
    joint_ids_key: tuple[object, ...]
    if isinstance(joint_ids, slice):
      joint_ids_key = ("slice", joint_ids.start, joint_ids.stop, joint_ids.step)
    else:
      joint_ids_key = tuple(int(j) for j in joint_ids)

    cache_key = (
      self._asset_cfg.name,
      joint_ids_key,
      self._sensor_name,
      self._command_name,
      int(self._window_steps),
      float(self._temporal_std_steps),
    )
    cache_attr = "_joint_gait_pattern_core_cache"
    core_cache = _get_or_init_cache_dict(env, cache_attr)
    if cache_key not in core_cache:
      core_cache[cache_key] = _JointGaitPatternCore(
        env=env,
        asset_cfg=self._asset_cfg,
        sensor_name=self._sensor_name,
        command_name=self._command_name,
        window_steps=self._window_steps,
        temporal_std_steps=self._temporal_std_steps,
      )
    self._core = core_cache[cache_key]

    selected_joint_names = self._selected_joint_names(env)
    num_joints = len(selected_joint_names)
    self._axis_signs = self._resolve_axis_signs(
      p.get("axis_signs", 1.0),
      joint_names=selected_joint_names,
      num_joints=num_joints,
      device=env.device,
    )

    self._symmetry_penalty_weight: float = p.get("symmetry_penalty_weight", 0.0)
    self._symmetry_ema_alpha: float = p.get("symmetry_ema_alpha", 0.2)
    self._ema_0 = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    self._ema_1 = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  def _selected_joint_names(self, env: ManagerBasedRlEnv) -> list[str]:
    """Resolve selected joint names in resolved order; fallback for mock envs."""
    asset = env.scene[self._asset_cfg.name]
    all_joint_names = getattr(asset, "joint_names", None)
    joint_ids = self._asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
      if isinstance(all_joint_names, (list, tuple)):
        return list(all_joint_names)
      joint_dim = int(asset.data.joint_pos.shape[1])
      return [f"joint_{i}" for i in range(joint_dim)]
    if isinstance(all_joint_names, (list, tuple)):
      return [all_joint_names[int(i)] for i in joint_ids]
    return [f"joint_{int(i)}" for i in joint_ids]

  @staticmethod
  def _resolve_axis_signs(
    raw_axis_signs: float | int | tuple[float, ...] | list[float] | dict[str, float],
    *,
    joint_names: list[str],
    num_joints: int,
    device: str | torch.device,
  ) -> torch.Tensor:
    if num_joints == 0:
      return torch.zeros(0, device=device, dtype=torch.float32)

    if isinstance(raw_axis_signs, (int, float)):
      return torch.full(
        (num_joints,), float(raw_axis_signs), device=device, dtype=torch.float32
      )

    if isinstance(raw_axis_signs, (list, tuple)):
      if len(raw_axis_signs) != num_joints:
        raise ValueError(
          "joint_gait_pattern axis_signs list length must match selected joints: "
          f"len(axis_signs)={len(raw_axis_signs)}, num_joints={num_joints}."
        )
      return torch.tensor(raw_axis_signs, device=device, dtype=torch.float32)

    if isinstance(raw_axis_signs, dict):
      _, _, signs = resolve_matching_names_values(
        data=raw_axis_signs,
        list_of_strings=joint_names,
      )
      return torch.tensor(signs, device=device, dtype=torch.float32)

    raise TypeError(
      "joint_gait_pattern axis_signs must be float/int, list/tuple, or dict."
    )

  @staticmethod
  def _align_event_mask(event_mask: torch.Tensor, num_joints: int) -> torch.Tensor:
    """Align contact-event mask shape [B, N_event] to joint shape [B, N_joint]."""
    if event_mask.shape[1] == num_joints:
      return event_mask
    if num_joints == 1:
      return torch.any(event_mask > 0.0, dim=1, keepdim=True).float()
    if event_mask.shape[1] == 1:
      return event_mask.expand(-1, num_joints)
    raise ValueError(
      "joint_gait_pattern requires matching counts between selected joints and "
      f"contact sensor elements, got joints={num_joints}, events={event_mask.shape[1]}."
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    mode: str = "toe",
    command_name: str | None = None,
    command_threshold: float = 0.05,
    forward_only: bool = False,
    running_threshold: float | None = None,
    target_angle_deg: float = 30.0,
    std: float = math.radians(10.0),
    std_running: float | None = None,
    window_steps: int = 0,
    temporal_std_steps: float = 3.0,
    post_reward_steps: int = 1,
    post_reward_std_steps: float = 1.0,
    min_air_steps: int = 0,
    min_contact_steps: int = 0,
    axis_signs: float | int | tuple[float, ...] | list[float] | dict[str, float] = 1.0,
    symmetry_penalty_weight: float = 0.0,
    symmetry_ema_alpha: float = 0.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del (
      sensor_name,
      mode,
      command_name,
      command_threshold,
      forward_only,
      running_threshold,
      target_angle_deg,
      std,
      std_running,
      window_steps,
      temporal_std_steps,
      post_reward_steps,
      post_reward_std_steps,
      min_air_steps,
      min_contact_steps,
      axis_signs,
      symmetry_penalty_weight,
      symmetry_ema_alpha,
      asset_cfg,
    )  # Params are resolved once at init via cfg.
    state = self._core.compute(env)
    joint_pos = state.joint_pos
    window_buf = state.window_buf
    buf_idx = state.buf_idx
    just_landed = state.just_landed
    just_lifted = state.just_lifted
    last_air_time = state.last_air_time
    last_contact_time = state.last_contact_time
    current_air_time = state.current_air_time
    current_contact_time = state.current_contact_time
    cmd_x = state.cmd_x
    cmd_y = state.cmd_y
    cmd_speed = state.cmd_speed
    use_window = state.use_window

    if joint_pos.shape[1] == 0:
      return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    if self._mode == "heel":
      landing_valid = torch.ones_like(just_landed)
      if self._min_air_time > 0.0:
        landing_valid = (last_air_time >= self._min_air_time).float()
      just_landed = just_landed * landing_valid
      event_mask = (
        _post_reward_event_mask(
          event_mask=just_landed,
          phase_time=current_contact_time,
          step_dt=env.step_dt,
          post_reward_steps=self._post_reward_steps,
          post_reward_inv_std_sq=self._post_reward_inv_std_sq,
        )
        * landing_valid
      )
    else:
      liftoff_valid = torch.ones_like(just_lifted)
      if self._min_contact_time > 0.0:
        liftoff_valid = (
          liftoff_valid * (last_contact_time >= self._min_contact_time).float()
        )
      # Also filter liftoffs with no prior air phase (standing → first step).
      if self._min_air_time > 0.0:
        liftoff_valid = liftoff_valid * (last_air_time >= self._min_air_time).float()
      just_lifted = just_lifted * liftoff_valid
      event_mask = (
        _post_reward_event_mask(
          event_mask=just_lifted,
          phase_time=current_air_time,
          step_dt=env.step_dt,
          post_reward_steps=self._post_reward_steps,
          post_reward_inv_std_sq=self._post_reward_inv_std_sq,
        )
        * liftoff_valid
      )
    event_mask = self._align_event_mask(event_mask, joint_pos.shape[1])

    active = self._core._ones_env
    if cmd_x is not None:
      # Only active for forward commands; backward gait patterns differ
      # biomechanically and are left to free learning for now.
      # TODO: add backward-specific target angles for hip (direction-sensitive)
      # and knee (direction-insensitive).
      active = (cmd_x > self._command_threshold).float()
    if self._forward_only and cmd_y is not None:
      active = active * (torch.abs(cmd_y) < self._command_threshold).float()

    # Multi-step post rewards rescore the current state each step. The legacy
    # temporal window remains available for event-only terms.
    use_temporal_window = use_window and self._post_reward_steps <= 1
    if use_temporal_window:
      assert window_buf is not None
      signed_joint_buf = window_buf * self._axis_signs.view(1, -1, 1)
      assert self._core._temporal_w_by_buf_idx is not None
      temporal_w = self._core._temporal_w_by_buf_idx[buf_idx].view(1, 1, -1)
      if self._running_threshold is not None and cmd_speed is not None:
        inv_std_sq = torch.where(
          cmd_speed >= self._running_threshold,
          self._inv_std_running_sq,
          self._inv_std_sq,
        ).view(-1, 1, 1)
        score_buf = torch.exp(
          -torch.square(signed_joint_buf - self._target_angle_rad) * inv_std_sq
        )
      else:
        score_buf = torch.exp(
          -torch.square(signed_joint_buf - self._target_angle_rad) * self._inv_std_sq
        )
      score = (score_buf * temporal_w).sum(dim=-1)
    else:
      signed_joint_pos = joint_pos * self._axis_signs.unsqueeze(0)
      if self._running_threshold is not None and cmd_speed is not None:
        inv_std_sq = torch.where(
          cmd_speed >= self._running_threshold,
          self._inv_std_running_sq,
          self._inv_std_sq,
        ).unsqueeze(1)
        score = torch.exp(
          -torch.square(signed_joint_pos - self._target_angle_rad) * inv_std_sq
        )
      else:
        score = torch.exp(
          -torch.square(signed_joint_pos - self._target_angle_rad) * self._inv_std_sq
        )

    reward = torch.sum(score * event_mask, dim=1)

    # Event-based symmetry penalty for biped (N_joints == 2).
    # EMA updates only when an event fires AND the reward is active
    # (standing envs are excluded to avoid EMA contamination).
    if self._symmetry_penalty_weight != 0.0 and score.shape[1] == 2:
      active_bool = active > 0.5
      update_0 = (event_mask[:, 0] > 0) & active_bool
      update_1 = (event_mask[:, 1] > 0) & active_bool
      self._ema_0 = torch.where(
        update_0,
        (1.0 - self._symmetry_ema_alpha) * self._ema_0
        + self._symmetry_ema_alpha * score[:, 0],
        self._ema_0,
      )
      self._ema_1 = torch.where(
        update_1,
        (1.0 - self._symmetry_ema_alpha) * self._ema_1
        + self._symmetry_ema_alpha * score[:, 1],
        self._ema_1,
      )
      symmetry_penalty = torch.abs(self._ema_0 - self._ema_1)
      reward = reward - self._symmetry_penalty_weight * symmetry_penalty

    return reward * active

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Delegate reset to shared core (cache cleanup)."""
    if env_ids is None:
      env_ids = slice(None)
    self._core.reset(env_ids=env_ids)
    self._ema_0[env_ids] = 0.0
    self._ema_1[env_ids] = 0.0


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Four speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_turning (low linear speed, high angular speed): Turning-specific tolerance.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  Regime-specific scalar weights modulate the final reward importance.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
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

    _, _, std_turning = resolve_matching_names_values(
      data=cfg.params["std_turning"],
      list_of_strings=joint_names,
    )
    self.std_turning = torch.tensor(std_turning, device=env.device, dtype=torch.float32)

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

    self.weight_standing = float(cfg.params.get("weight_standing", 1.0))
    self.weight_turning = float(cfg.params.get("weight_turning", 1.0))
    self.weight_walking = float(cfg.params.get("weight_walking", 1.0))
    self.weight_running = float(cfg.params.get("weight_running", 1.0))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std_standing=None,
    std_turning=None,
    std_walking=None,
    std_running=None,
    weight_standing: float | None = None,
    weight_turning: float | None = None,
    weight_walking: float | None = None,
    weight_running: float | None = None,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del (
      std_standing,
      std_turning,
      std_walking,
      std_running,
      weight_standing,
      weight_turning,
      weight_walking,
      weight_running,
    )  # Unused.

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
    weight = (
      self.weight_standing * standing_mask
      + self.weight_turning * turning_mask
      + self.weight_walking * walking_mask
      + self.weight_running * running_mask
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1)) * weight


def flat_orientation_multi(
  env: ManagerBasedRlEnv,
  std: float | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  weights: list[float] | None = None,
  std_xy: tuple[float, float] | None = None,
  weight_xy: tuple[float, float] = (1.0, 1.0),
) -> torch.Tensor:
  """Reward flat base orientation for multiple bodies with per-body weights.

  Iterates over each body in ``asset_cfg.body_ids``, calls ``flat_orientation``
  for each, and returns the weighted sum.

  Args:
    std: Standard deviation for combined mode ``exp(-(gx²+gy²)/std²)``.
    weights: Per-body scalar weights. Defaults to 1.0 for each body.
    std_xy: ``(std_x, std_y)`` for per-component mode (roll, pitch).
    weight_xy: ``(weight_x, weight_y)`` per-component term weights.
  """
  if weights is None:
    weights = [1.0] * len(asset_cfg.body_ids)

  return sum(
    flat_orientation(
      env, std, replace(asset_cfg, body_ids=[body_id]), std_xy, weight_xy
    )
    * weight
    for body_id, weight in zip(asset_cfg.body_ids, weights, strict=False)
  )


def base_height_l2(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  std: float | None = None,
  std_walking: float | None = None,
  std_running: float | None = None,
  walking_threshold: float = 0.5,
  running_threshold: float = 1.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target base height.

  When speed-dependent stds are provided, the squared height error is scaled by
  the commanded locomotion regime, mirroring ``track_linear_velocity``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  height_error = torch.square(asset.data.root_link_pos_w[:, 2] - target_height)

  if std is None and std_walking is None and std_running is None:
    return height_error

  if command_name is None:
    raise ValueError("'command_name' must be provided when using speed-dependent stds.")
  if std is None or std_walking is None or std_running is None:
    raise ValueError(
      "'std', 'std_walking', and 'std_running' must all be provided together."
    )

  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  selected_std = _select_speed_dependent_std(
    command,
    std=std,
    std_walking=std_walking,
    std_running=std_running,
    walking_threshold=walking_threshold,
    running_threshold=running_threshold,
  )
  return torch.exp(-height_error / (selected_std**2))


def feet_orientation_flat(
  env: ManagerBasedRlEnv,
  std: float,
  sensor_name: str,
  stance_force_ratio: float = 0.4,
  target_height: float | None = None,
  settle_steps: int = 0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat foot orientation during stance and/or swing.

  Two activation conditions (OR logic):
  - **Stance**: contact force > ``stance_force_ratio`` × robot weight.
  - **Swing** (optional): foot body height >= ``target_height``.

  Args:
    std: Gaussian standard deviation for orientation error.
    sensor_name: Name of the contact sensor.
    stance_force_ratio: Force threshold as a ratio of total robot weight.
    target_height: If given, also activate when foot height >= this value (m).
    settle_steps: Number of steps after landing to suppress the stance
      reward. This prevents penalising the tilted foot pose during
      heel-strike. The grace period is ``settle_steps * step_dt``.
    asset_cfg: Asset configuration. Must provide ``body_ids`` for the feet.
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]

  num_feet = len(asset_cfg.body_ids)

  # --- Stance mask (force-based) ---
  all_body_ids = asset.indexing.body_ids
  body_masses = env.sim.model.body_mass
  if not isinstance(body_masses, torch.Tensor):
    body_masses = torch.tensor(body_masses, device=env.device, dtype=torch.float32)
  if body_masses.ndim == 2:
    total_mass = torch.sum(body_masses[:, all_body_ids], dim=1)
  else:
    total_mass = torch.sum(body_masses[all_body_ids])
  force_threshold = total_mass * 9.81 * stance_force_ratio
  if force_threshold.ndim == 1:
    force_threshold = force_threshold.unsqueeze(-1)
  forces = contact_sensor.data.force
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N_feet]
  stance_mask = force_magnitude > force_threshold  # [B, N_feet]

  # --- Landing grace period ---
  if settle_steps > 0:
    grace_time = settle_steps * env.step_dt
    contact_time = contact_sensor.data.current_contact_time  # [B, N_feet]
    last_air = contact_sensor.data.last_air_time  # [B, N_feet]
    # Chattering (short prior air phase): suppress foot_flat entirely.
    is_chattering = (last_air > 0) & (last_air < grace_time)
    # Real landing: grace period for heel-strike transition.
    recently_landed = (contact_time > 0) & (contact_time < grace_time)
    stance_mask = stance_mask & ~recently_landed & ~is_chattering

  # --- Swing mask (height-based, optional) ---
  if target_height is not None:
    foot_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]  # [B, N_feet]
    swing_mask = foot_z >= target_height
  else:
    swing_mask = torch.zeros_like(stance_mask)

  mask = (stance_mask | swing_mask).float()

  # --- Projected gravity in foot frame ---
  body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N_feet, 4]
  gravity_w = asset.data.gravity_vec_w.unsqueeze(1).expand(-1, num_feet, -1)
  projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)

  xy_squared = torch.sum(
    torch.square(projected_gravity_b[..., :2]), dim=-1
  )  # [B, N_feet]
  reward_per_foot = torch.exp(-xy_squared / std**2)

  return torch.sum(reward_per_foot * mask, dim=1)


def _compute_thigh_swing_target_scores(
  env: ManagerBasedRlEnv,
  *,
  std: float,
  sensor_name: str,
  command_name: str,
  lin_vel_y_threshold: float,
  yaw_threshold: float | None,
  target_angle_deg: float,
  axis_signs: float | int | tuple[float, ...] | list[float] | dict[str, float],
  asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]

  current_air_time = contact_sensor.data.current_air_time
  assert current_air_time is not None, (
    f"Contact sensor '{sensor_name}' must have track_air_time=True."
  )
  in_air = (current_air_time > 0.0).float()

  command = env.command_manager.get_command(command_name)
  assert command is not None
  small_vy = (torch.abs(command[:, 1]) < lin_vel_y_threshold).float().unsqueeze(-1)

  cmd_mask = small_vy
  if yaw_threshold is not None:
    small_yaw = (torch.abs(command[:, 2]) < yaw_threshold).float().unsqueeze(-1)
    cmd_mask = cmd_mask * small_yaw

  mask = in_air * cmd_mask

  body_names = asset.body_names
  resolved_body_ids = asset_cfg.body_ids
  if isinstance(resolved_body_ids, slice):
    resolved_body_ids = list(range(len(body_names)))[resolved_body_ids]

  num_legs = len(resolved_body_ids)
  body_quat_w = asset.data.body_link_quat_w[:, resolved_body_ids, :]
  gravity_w = asset.data.gravity_vec_w.unsqueeze(1).expand(-1, num_legs, -1)
  projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)

  selected_body_names = [body_names[int(body_id)] for body_id in resolved_body_ids]
  if num_legs == 0:
    target_signs = torch.zeros(0, device=env.device, dtype=torch.float32)
  elif isinstance(axis_signs, (int, float)):
    target_signs = torch.full(
      (num_legs,), float(axis_signs), device=env.device, dtype=torch.float32
    )
  elif isinstance(axis_signs, (list, tuple)):
    if len(axis_signs) != num_legs:
      raise ValueError(
        "thigh_swing_target axis_signs list length must match selected bodies: "
        f"len(axis_signs)={len(axis_signs)}, num_bodies={num_legs}."
      )
    target_signs = torch.tensor(axis_signs, device=env.device, dtype=torch.float32)
  elif isinstance(axis_signs, dict):
    _, _, signs = resolve_matching_names_values(
      data=axis_signs,
      list_of_strings=selected_body_names,
    )
    target_signs = torch.tensor(signs, device=env.device, dtype=torch.float32)
  else:
    raise TypeError(
      "thigh_swing_target axis_signs must be float/int, list/tuple, or dict."
    )

  target_deg = target_signs * float(target_angle_deg)
  target_gy = torch.sin(torch.deg2rad(target_deg)).unsqueeze(0)
  gy_err_sq = torch.square(projected_gravity_b[..., 1] - target_gy)
  reward_per_leg = torch.exp(-gy_err_sq / std**2)
  return reward_per_leg, mask, cmd_mask.squeeze(-1)


class thigh_swing_target:
  """Reward lateral thigh target posture during swing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    p = cfg.params or {}
    self._std: float = p.get("std", math.sin(math.radians(10)))
    self._sensor_name: str = p.get("sensor_name", "feet_ground_contact")
    self._command_name: str = p.get("command_name", "twist")
    self._lin_vel_y_threshold: float = p.get("lin_vel_y_threshold", 0.05)
    self._yaw_threshold: float | None = p.get("yaw_threshold")
    self._target_angle_deg: float = p.get("target_angle_deg", 0.0)
    self._axis_signs = p.get("axis_signs", 1.0)
    self._asset_cfg: SceneEntityCfg = p.get("asset_cfg", _DEFAULT_ASSET_CFG)
    self._symmetry_penalty_weight: float = p.get("symmetry_penalty_weight", 0.0)
    self._symmetry_ema_alpha: float = p.get("symmetry_ema_alpha", 0.1)
    self._ema_0 = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    self._ema_1 = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float = math.sin(math.radians(10)),
    sensor_name: str = "feet_ground_contact",
    command_name: str = "twist",
    lin_vel_y_threshold: float = 0.05,
    yaw_threshold: float | None = None,
    target_angle_deg: float = 0.0,
    axis_signs: float | int | tuple[float, ...] | list[float] | dict[str, float] = 1.0,
    symmetry_penalty_weight: float = 0.0,
    symmetry_ema_alpha: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del (
      std,
      sensor_name,
      command_name,
      lin_vel_y_threshold,
      yaw_threshold,
      target_angle_deg,
      axis_signs,
      symmetry_penalty_weight,
      symmetry_ema_alpha,
      asset_cfg,
    )  # Params are resolved once at init via cfg.

    reward_per_leg, mask, active = _compute_thigh_swing_target_scores(
      env,
      std=self._std,
      sensor_name=self._sensor_name,
      command_name=self._command_name,
      lin_vel_y_threshold=self._lin_vel_y_threshold,
      yaw_threshold=self._yaw_threshold,
      target_angle_deg=self._target_angle_deg,
      axis_signs=self._axis_signs,
      asset_cfg=self._asset_cfg,
    )
    raw_score = reward_per_leg * mask
    reward = torch.sum(raw_score, dim=1)

    if self._symmetry_penalty_weight != 0.0 and raw_score.shape[1] == 2:
      active_bool = active > 0.5
      update_0 = (mask[:, 0] > 0) & active_bool
      update_1 = (mask[:, 1] > 0) & active_bool
      alpha = max(0.0, min(1.0, float(self._symmetry_ema_alpha)))
      self._ema_0 = torch.where(
        update_0,
        (1.0 - alpha) * self._ema_0 + alpha * raw_score[:, 0],
        self._ema_0,
      )
      self._ema_1 = torch.where(
        update_1,
        (1.0 - alpha) * self._ema_1 + alpha * raw_score[:, 1],
        self._ema_1,
      )
      symmetry_penalty = torch.abs(self._ema_0 - self._ema_1) * active.float()
      reward = reward - self._symmetry_penalty_weight * symmetry_penalty

    return reward * active

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._ema_0[env_ids] = 0.0
    self._ema_1[env_ids] = 0.0


def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  diff_angle = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  reward = torch.sum(torch.square(diff_angle), dim=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command <= command_threshold).float()
      reward *= scale
  return reward


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
