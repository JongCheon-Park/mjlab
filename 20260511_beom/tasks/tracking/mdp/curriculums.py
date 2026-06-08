from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict

import torch

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class PreShiftStage(TypedDict):
  step: int
  range_s: tuple[float, float]
  apply_prob: NotRequired[float]


def pre_shift_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: list[PreShiftStage],
) -> dict[str, torch.Tensor]:
  """Stage-based curriculum for MotionCommand pre-shift range/probability."""
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  if not isinstance(command_term, MotionCommand):
    raise TypeError(
      f"Expected MotionCommand for '{command_name}', got {type(command_term).__name__}."
    )

  target_range_s = command_term.cfg.pre_shift_range_s
  target_apply_prob = command_term.cfg.pre_shift_apply_prob
  for stage in stages:
    if env.common_step_counter >= stage["step"]:
      target_range_s = stage["range_s"]
      if "apply_prob" in stage:
        target_apply_prob = stage["apply_prob"]

  low_s, high_s = target_range_s
  if low_s < 0.0 or high_s < low_s:
    raise ValueError(
      f"Invalid pre-shift curriculum range {target_range_s}. Expected 0 <= min <= max."
    )
  if target_apply_prob < 0.0 or target_apply_prob > 1.0:
    raise ValueError(
      f"Invalid pre-shift curriculum apply_prob {target_apply_prob}. "
      "Expected value in [0, 1]."
    )

  command_term.cfg.pre_shift_range_s = target_range_s
  command_term.cfg.pre_shift_apply_prob = target_apply_prob
  return {
    "pre_shift_min_s": torch.tensor(low_s, device=env.device),
    "pre_shift_max_s": torch.tensor(high_s, device=env.device),
    "pre_shift_apply_prob": torch.tensor(target_apply_prob, device=env.device),
  }
