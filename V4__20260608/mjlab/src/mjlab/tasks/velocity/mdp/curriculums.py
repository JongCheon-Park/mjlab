from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class _VelocityStageOptional(TypedDict, total=False):
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class VelocityStage(_VelocityStageOptional):
  step: int


class ResamplingStage(TypedDict):
  step: int
  resampling_time_range: tuple[float, float] | None


class MotionMixStage(TypedDict):
  step: int
  rel_standing_envs: float | None
  rel_turn_in_place_envs: float | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


class RewardParamStage(TypedDict):
  step: int
  param: str
  value: float



def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage] | None = None,
  resampling_stages: list[ResamplingStage] | None = None,
  motion_mix_stages: list[MotionMixStage] | None = None,
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages or ():
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  for stage in resampling_stages or ():
    if env.common_step_counter >= stage["step"]:
      if stage["resampling_time_range"] is not None:
        cfg.resampling_time_range = stage["resampling_time_range"]
        command_term.time_left.clamp_(max=cfg.resampling_time_range[1])
  for stage in motion_mix_stages or ():
    if env.common_step_counter >= stage["step"]:
      if stage["rel_standing_envs"] is not None:
        cfg.rel_standing_envs = stage["rel_standing_envs"]
      if stage["rel_turn_in_place_envs"] is not None:
        cfg.rel_turn_in_place_envs = stage["rel_turn_in_place_envs"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
    "resampling_time_min": torch.tensor(cfg.resampling_time_range[0]),
    "resampling_time_max": torch.tensor(cfg.resampling_time_range[1]),
    "rel_standing_envs": torch.tensor(cfg.rel_standing_envs),
    "rel_turn_in_place_envs": torch.tensor(cfg.rel_turn_in_place_envs),
  }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])


def reward_param(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  param_stages: list[RewardParamStage],
) -> torch.Tensor:
  """Update a reward term's params dict based on training step stages.

  Use ``reward_weight`` to change the outer term weight instead.

  Example cfg::

    CurriculumTermCfg(
      func=mdp.reward_param,
      params={
        "reward_name": "standing_air_time_penalty",
        "param_stages": [
          {"step": 0,        "param": "target_air_time", "value": 0.3},
          {"step": 72_000,   "param": "target_air_time", "value": 0.4},
          {"step": 192_000,  "param": "target_air_time", "value": 0.5},
        ],
      },
    )
  """
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in param_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.params[stage["param"]] = stage["value"]
  first_param = param_stages[0]["param"] if param_stages else None
  current = reward_term_cfg.params.get(first_param, 0.0) if first_param else 0.0
  return torch.tensor([current])
