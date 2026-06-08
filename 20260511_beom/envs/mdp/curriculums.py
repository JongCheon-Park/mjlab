from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


# Stage schemas.


class _RewardCurriculumStageOptional(TypedDict, total=False):
  weight: float
  params: dict[str, Any]


class RewardCurriculumStage(_RewardCurriculumStageOptional):
  step: int


class _TerminationCurriculumStageOptional(TypedDict, total=False):
  params: dict[str, Any]
  time_out: bool


class TerminationCurriculumStage(_TerminationCurriculumStageOptional):
  step: int


class _ActuatorMotorBandwidthStageOptional(TypedDict, total=False):
  enabled: bool
  cutoff_hz_range: tuple[float, float]


class ActuatorMotorBandwidthStage(_ActuatorMotorBandwidthStageOptional):
  step: int


# Shared engine.  Stage dicts are passed directly from the public TypedDict
# schemas.  Any key that isn't "step" or "params" is treated as a top-level
# field on the target term config (e.g. "weight" on RewardTermCfg).

_RESERVED_KEYS = {"step", "params"}


def _validate_stages(
  term_cfg: Any,
  term_name: str,
  stages: Sequence[Any],
) -> None:
  """Validate stage ordering, field existence, and param keys."""
  for i in range(1, len(stages)):
    if stages[i]["step"] < stages[i - 1]["step"]:
      raise ValueError(
        f"Curriculum stages must be in nondecreasing step order,"
        f" but stage {i} has step"
        f" {stages[i]['step']} < {stages[i - 1]['step']}."
      )
  for stage in stages:
    for key in stage:
      if key not in _RESERVED_KEYS and not hasattr(term_cfg, key):
        raise AttributeError(
          f"Field '{key}' does not exist on the resolved term config for '{term_name}'."
        )
  for stage in stages:
    unknown = stage.get("params", {}).keys() - term_cfg.params.keys()
    if unknown:
      raise KeyError(
        f"Stage at step {stage['step']} sets unknown param(s)"
        f" {unknown} on term '{term_name}'. Check for typos."
      )


def _apply_stages(
  term_cfg: Any,
  step_counter: int,
  stages: Sequence[Any],
) -> dict[str, torch.Tensor]:
  """Apply staged updates and return a logging snapshot."""
  for stage in stages:
    if step_counter >= stage["step"]:
      for key, value in stage.items():
        if key not in _RESERVED_KEYS:
          setattr(term_cfg, key, value)
      if "params" in stage:
        term_cfg.params.update(stage["params"])
  # Only log values that stages actually reference.
  logged_fields: set[str] = set()
  logged_params: set[str] = set()
  for stage in stages:
    for key in stage:
      if key not in _RESERVED_KEYS:
        logged_fields.add(key)
    for key in stage.get("params", {}):
      logged_params.add(key)
  result: dict[str, torch.Tensor] = {}
  for key in logged_fields:
    value = getattr(term_cfg, key)
    if isinstance(value, (int, float, bool)):
      result[key] = torch.tensor(value)
    elif isinstance(value, torch.Tensor):
      result[key] = value
  for key in logged_params:
    v = term_cfg.params[key]
    if isinstance(v, (int, float, bool)):
      result[key] = torch.tensor(v)
    elif isinstance(v, torch.Tensor):
      result[key] = v
  return result


# Public wrappers.


class reward_curriculum:
  """Update a reward term's weight and/or params based on training steps.

  Each stage specifies a ``step`` threshold and optionally a ``weight``
  and/or ``params`` dict.  When ``env.common_step_counter`` reaches a
  stage's ``step``, the corresponding values are applied.  Later stages
  take precedence when multiple thresholds are reached.

  Example::

    CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "joint_vel_hinge",
        "stages": [
          {"step": 0, "weight": -0.01},
          {"step": 12000, "weight": -0.1},
          {"step": 24000, "weight": -1.0, "params": {"max_vel": 1.0}},
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    reward_name: str = cfg.params["reward_name"]
    stages: list[RewardCurriculumStage] = cfg.params["stages"]
    self._term_cfg = env.reward_manager.get_term_cfg(reward_name)
    self._stages = stages
    _validate_stages(self._term_cfg, reward_name, self._stages)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    stages: list[RewardCurriculumStage],
  ) -> dict[str, torch.Tensor]:
    del env_ids, reward_name, stages
    return _apply_stages(self._term_cfg, env.common_step_counter, self._stages)


class termination_curriculum:
  """Update a termination term's params based on training steps.

  Each stage specifies a ``step`` threshold and a ``params`` dict.  When
  ``env.common_step_counter`` reaches a stage's ``step``, the params are
  applied.  Later stages take precedence.

  Example::

    CurriculumTermCfg(
      func=mdp.termination_curriculum,
      params={
        "termination_name": "energy",
        "stages": [
          {"step": 12000, "params": {"threshold": 1000.0}},
          {"step": 24000, "params": {"threshold": 700.0}},
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    termination_name: str = cfg.params["termination_name"]
    stages: list[TerminationCurriculumStage] = cfg.params["stages"]
    self._term_cfg = env.termination_manager.get_term_cfg(termination_name)
    self._stages = stages
    _validate_stages(self._term_cfg, termination_name, self._stages)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    termination_name: str,
    stages: list[TerminationCurriculumStage],
  ) -> dict[str, torch.Tensor]:
    del env_ids, termination_name, stages
    return _apply_stages(self._term_cfg, env.common_step_counter, self._stages)


class actuator_motor_bandwidth_curriculum:
  """Stage finite-motor-bandwidth DR from disabled/weak to slower cutoff ranges.

  This curriculum targets the already-instantiated
  ``dr.actuator_motor_bandwidth`` event term. It is designed to run before
  ``EventManager.reset()``, so range changes are used when the event term
  resamples per-environment bandwidth on reset.

  Example::

    CurriculumTermCfg(
      func=mdp.actuator_motor_bandwidth_curriculum,
      params={
        "event_name": "actuator_motor_bandwidth",
        "stages": [
          {"step": 0, "enabled": False},
          {"step": 120_000, "enabled": True, "cutoff_hz_range": (15.0, 30.0)},
          {"step": 480_000, "cutoff_hz_range": (3.0, 15.0)},
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._event_name: str = cfg.params.get("event_name", "actuator_motor_bandwidth")
    self._stages: list[ActuatorMotorBandwidthStage] = cfg.params["stages"]
    self._term_cfg = env.event_manager.get_term_cfg(self._event_name)
    self._term = self._term_cfg.func
    if not hasattr(self._term, "set_enabled") or not hasattr(
      self._term,
      "set_cutoff_hz_range",
    ):
      raise TypeError(
        f"Event term '{self._event_name}' does not support actuator bandwidth "
        "curriculum updates."
      )
    self._validate_stages()

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    stages: list[ActuatorMotorBandwidthStage],
  ) -> dict[str, torch.Tensor]:
    del env_ids, event_name, stages
    active_stage = None
    for stage in self._stages:
      if env.common_step_counter >= stage["step"]:
        active_stage = stage
    if active_stage is not None:
      if "cutoff_hz_range" in active_stage:
        self._term.set_cutoff_hz_range(
          active_stage["cutoff_hz_range"],
          invalidate_alpha=False,
        )
      if "enabled" in active_stage:
        self._term.set_enabled(active_stage["enabled"])
    return self._log_state(env)

  def _validate_stages(self) -> None:
    for i in range(1, len(self._stages)):
      if self._stages[i]["step"] < self._stages[i - 1]["step"]:
        raise ValueError(
          f"Curriculum stages must be in nondecreasing step order,"
          f" but stage {i} has step"
          f" {self._stages[i]['step']} < {self._stages[i - 1]['step']}."
        )
    for stage in self._stages:
      if "enabled" not in stage and "cutoff_hz_range" not in stage:
        raise ValueError(
          "actuator_motor_bandwidth_curriculum stages must set at least one of "
          "'enabled' or 'cutoff_hz_range'."
        )
      if "cutoff_hz_range" in stage:
        self._validate_cutoff_hz_range(stage["cutoff_hz_range"], stage["step"])

  @staticmethod
  def _validate_cutoff_hz_range(
    cutoff_hz_range: tuple[float, float],
    step: int,
  ) -> None:
    if not isinstance(cutoff_hz_range, (tuple, list)) or len(cutoff_hz_range) != 2:
      raise ValueError(
        "cutoff_hz_range curriculum stages must provide a tuple/list of length "
        f"2 at step {step}, got {cutoff_hz_range!r}."
      )
    low = float(cutoff_hz_range[0])
    high = float(cutoff_hz_range[1])
    if low <= 0.0:
      raise ValueError(f"cutoff_hz_range[0] must be > 0.0 at step {step}, got {low}.")
    if low > high:
      raise ValueError(
        f"Invalid cutoff_hz_range at step {step}: low ({low}) > high ({high})."
      )

  def _log_state(self, env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    result = {
      "enabled": torch.tensor(float(self._term.enabled), device=env.device),
    }
    cutoff_hz_range = self._term.cutoff_hz_range
    if cutoff_hz_range is not None:
      result["cutoff_hz_min"] = torch.tensor(cutoff_hz_range[0], device=env.device)
      result["cutoff_hz_max"] = torch.tensor(cutoff_hz_range[1], device=env.device)
    return result
