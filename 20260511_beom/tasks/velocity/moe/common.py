"""Common utilities for velocity-task MoE data collection and evaluation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rsl_rl.models import MoEMLPModel

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

REGIME_NAMES = ("stand", "turn", "vx", "vy")
CMD_REGIME_TURN_RATIO_BASE = 2.0


def _apply_overrides_recursive(
  target: Any,
  overrides: dict[str, Any],
  *,
  path: str,
) -> None:
  """Apply nested dict overrides onto an object or dict in-place."""
  for key, value in overrides.items():
    if isinstance(target, dict):
      if key not in target:
        raise KeyError(f"Unknown override key '{path}.{key}' for dict target.")
      child = target[key]
    else:
      if not hasattr(target, key):
        raise KeyError(f"Unknown override key '{path}.{key}' for object target.")
      child = getattr(target, key)

    if isinstance(value, dict):
      if child is None:
        raise ValueError(f"Cannot apply nested override to None at '{path}.{key}'.")
      _apply_overrides_recursive(child, value, path=f"{path}.{key}")
      continue

    if isinstance(target, dict):
      target[key] = value
    else:
      setattr(target, key, value)


def apply_twist_command_overrides(
  env_cfg: Any,
  command_overrides: dict[str, Any] | Any | None,
) -> None:
  """Apply user-provided overrides to `env_cfg.commands['twist']`.

  Supported inputs:
  - `None`: no-op
  - `dict`: recursive attribute/key overrides on the existing twist config
  - non-dict object: replaces the twist cfg entirely
  """
  if command_overrides is None:
    return
  if "twist" not in env_cfg.commands:
    raise KeyError("Environment config has no 'twist' command to override.")

  if isinstance(command_overrides, dict):
    twist_cfg = env_cfg.commands["twist"]
    _apply_overrides_recursive(twist_cfg, command_overrides, path="twist")
  else:
    env_cfg.commands["twist"] = command_overrides


def build_velocity_moe_policy(
  task_id: str,
  checkpoint_file: str | None,
  device: str,
  num_envs: int,
  *,
  play: bool,
  command_overrides: dict[str, Any] | Any | None = None,
) -> tuple[RslRlVecEnvWrapper, MjlabOnPolicyRunner, MoEMLPModel]:
  """Construct env + runner and load a cmd_regime MoE policy checkpoint."""
  ckpt_path: Path | None = None
  if checkpoint_file is not None:
    ckpt_path = Path(checkpoint_file)
    if not ckpt_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

  env_cfg = load_env_cfg(task_id, play=play)
  env_cfg.scene.num_envs = int(num_envs)
  apply_twist_command_overrides(env_cfg, command_overrides)
  rl_cfg = load_rl_cfg(task_id)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(vec_env, asdict(rl_cfg), log_dir=None, device=device)
  if ckpt_path is not None:
    # For evaluation/collection we only need actor weights.
    # This keeps scripts backward-compatible with checkpoints across model/optimizer changes.
    runner.load(
      str(ckpt_path),
      load_cfg={
        "actor": True,
        "critic": False,
        "optimizer": False,
        "iteration": False,
        "rnd": False,
      },
      map_location=device,
      strict=False,
    )
  policy = runner.get_inference_policy(device=device)

  if not isinstance(policy, MoEMLPModel):
    raise TypeError(
      f"Loaded actor is {type(policy).__name__}, expected MoEMLPModel for this workflow."
    )
  if policy.router_mode != "cmd_regime":
    raise ValueError(f"Expected cmd_regime router mode, got '{policy.router_mode}'.")

  return vec_env, runner, policy


def flatten_actor_obs(obs: torch.Tensor | dict, obs_groups: list[str]) -> torch.Tensor:
  """Flatten actor observations in model group order."""
  obs_list = [obs[group] for group in obs_groups]
  return torch.cat(obs_list, dim=-1)


def require_router_slice(model: MoEMLPModel) -> tuple[int, int]:
  """Return validated (start, end) router slice for cmd_regime command input."""
  router_slice = getattr(model, "_router_input_slice", None)
  if router_slice is None:
    raise ValueError(
      "MoEMLPModel has no router input slice; expected cmd_regime setup."
    )
  start, end = router_slice
  if end - start != 3:
    raise ValueError(
      f"Router input slice width must be 3 for (vx, vy, yaw), got {end - start}."
    )
  return int(start), int(end)


def classify_cmd_regime(
  router_input: torch.Tensor,
  *,
  cmd_threshold: float,
  turn_ratio: float = CMD_REGIME_TURN_RATIO_BASE,
) -> torch.Tensor:
  """Classify command regimes to match cmd_regime routing logic.

  Returns integer regime IDs:
  - 0: stand
  - 1: turn
  - 2: vx-dominant
  - 3: vy-dominant
  """
  if router_input.shape[-1] != 3:
    raise ValueError(f"router_input last dim must be 3, got {router_input.shape[-1]}.")

  command = router_input[..., :3]
  vx_abs = torch.abs(command[..., 0])
  vy_abs = torch.abs(command[..., 1])
  yaw_abs = torch.abs(command[..., 2])
  lin_mag = torch.sqrt(vx_abs * vx_abs + vy_abs * vy_abs)
  total_mag = vx_abs + vy_abs + yaw_abs

  standing_mask = total_mag < cmd_threshold
  turning_mask = (
    (~standing_mask)
    & (yaw_abs > cmd_threshold)
    & ((lin_mag < cmd_threshold) | (yaw_abs > turn_ratio * lin_mag))
  )
  vx_mask = (~standing_mask) & (~turning_mask) & (vx_abs >= vy_abs)

  regime = torch.full(
    (command.shape[0],),
    3,
    dtype=torch.long,
    device=command.device,
  )
  regime[vx_mask] = 2
  regime[turning_mask] = 1
  regime[standing_mask] = 0
  return regime


def classify_cmd_regime_from_policy(
  policy: MoEMLPModel,
  router_input: torch.Tensor,
) -> torch.Tensor:
  """Classify regimes using the loaded policy's own routing boundary logic.

  This keeps offline data collection aligned with checkpoint behavior,
  including learnable regime boundaries when enabled.
  """
  if policy.router_mode != "cmd_regime":
    raise ValueError(f"Expected cmd_regime router mode, got '{policy.router_mode}'.")
  if router_input.shape[-1] != 3:
    raise ValueError(f"router_input last dim must be 3, got {router_input.shape[-1]}.")

  classifier = getattr(policy, "_classify_cmd_regime", None)
  if classifier is not None:
    return classifier(router_input)

  # Backward-compat fallback for older policy classes.
  router_cfg = getattr(policy, "router_params", {}) or {}
  cmd_threshold = float(
    router_cfg.get(
      "cmd_threshold",
      router_cfg.get("stand_threshold", router_cfg.get("turn_threshold", 0.05)),
    )
  )
  turn_ratio = float(
    router_cfg.get(
      "turn_ratio",
      router_cfg.get("turn_ratio_base", CMD_REGIME_TURN_RATIO_BASE),
    )
  )
  return classify_cmd_regime(
    router_input,
    cmd_threshold=cmd_threshold,
    turn_ratio=turn_ratio,
  )


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Project points to 2D with PCA via SVD.

  Returns:
    x_2d: (N, 2)
    mean: (D,)
    components: (D, 2)
  """
  if x.ndim != 2:
    raise ValueError(f"Expected a 2D array for PCA, got shape {x.shape}.")
  mean = x.mean(axis=0, keepdims=True)
  x_centered = x - mean
  _, _, vh = np.linalg.svd(x_centered, full_matrices=False)
  components = vh[:2].T
  x_2d = x_centered @ components
  return x_2d, mean.reshape(-1), components
