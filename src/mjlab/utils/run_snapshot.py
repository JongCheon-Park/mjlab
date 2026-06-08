"""Run-level snapshot utilities for reproducibility / rollback.

Writes alongside the rsl-rl checkpoint directory:
- ``source/<file>`` : copies of the source files most likely to affect the
  policy (task config, MoE actor, custom reward functions, etc.). Lets a
  future user restore the *exact* code state by `cp -r source/ src/mjlab/`.
- ``summary.md`` : human-readable digest of the reward weights, curriculum
  stages and key parameters. Useful for telling runs apart at a glance.

The git ``mjlab.diff`` already saved by rsl-rl captures the commit hash and
unstaged diff, so the on-disk artifacts together (commit + diff + source/
+ summary.md + params/) are sufficient to fully reconstruct a run.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mjlab import MJLAB_SRC_PATH

# Files snapshotted into ``log_dir/source/``. Paths are relative to
# ``src/mjlab/``. Missing files are silently skipped.
_SNAPSHOT_FILES = (
  # Velocity task — p1
  "tasks/velocity/config/p1/__init__.py",
  "tasks/velocity/config/p1/env_cfgs.py",
  "tasks/velocity/config/p1/rl_cfg.py",
  "tasks/velocity/config/p1/symmetry.py",
  # Velocity task — common base
  "tasks/velocity/velocity_env_cfg.py",
  "tasks/velocity/mdp/rewards.py",
  "tasks/velocity/mdp/human_walking_rewards.py",
  "tasks/velocity/mdp/curriculums.py",
  "tasks/velocity/mdp/observations.py",
  # MoE actor
  "rl/moe_model.py",
  "rl/config.py",
  # Robot constants (KIMM P1)
  "asset_zoo/robots/p1/kimm_p1_constants_wo_4bar.py",
  "asset_zoo/robots/p1/kimm_p1_actuators.py",
)


def _try_copy(src: Path, dst: Path) -> bool:
  if not src.is_file():
    return False
  dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(src, dst)
  return True


def save_source_snapshot(log_dir: Path) -> list[str]:
  """Copy the curated set of source files into ``log_dir/source/``.

  Returns the list of files actually copied (relative paths).
  """
  out_root = log_dir / "source"
  saved: list[str] = []
  for rel in _SNAPSHOT_FILES:
    src = MJLAB_SRC_PATH / rel
    dst = out_root / rel
    if _try_copy(src, dst):
      saved.append(rel)
  return saved


def _format_rewards_table(env_cfg_dict: dict[str, Any]) -> str:
  rewards = env_cfg_dict.get("rewards", {}) or {}
  rows = []
  for name, term in rewards.items():
    if not isinstance(term, dict):
      continue
    w = term.get("weight", 0.0)
    func = term.get("func", "")
    # Surface common scalar params.
    params = term.get("params", {}) or {}
    interesting: list[str] = []
    for key in (
      "target_height",
      "target_air_time",
      "target_double_support_time",
      "std",
      "low_std",
      "sigma_sym",
      "command_threshold",
      "min_first_swing_air_time",
    ):
      if key in params and not isinstance(params[key], (dict, list, tuple)):
        interesting.append(f"{key}={params[key]}")
    rows.append((name, float(w), str(func), ", ".join(interesting)))

  positive = sorted([r for r in rows if r[1] > 0], key=lambda r: -r[1])
  zero = [r for r in rows if r[1] == 0.0]
  negative = sorted([r for r in rows if r[1] < 0], key=lambda r: r[1])

  def _render(group_rows: list[tuple], header: str) -> str:
    if not group_rows:
      return ""
    lines = [f"\n### {header}\n", "| reward | weight | extra |", "| --- | ---: | --- |"]
    for name, w, _, extras in group_rows:
      lines.append(f"| `{name}` | {w:g} | {extras} |")
    return "\n".join(lines) + "\n"

  return (
    _render(positive, "Positive rewards")
    + _render(negative, "Negative rewards (penalties)")
    + _render(zero, "Zero-weight (registered but inactive)")
  )


def _format_curriculum(env_cfg_dict: dict[str, Any]) -> str:
  curriculum = env_cfg_dict.get("curriculum", {}) or {}
  if not curriculum:
    return ""
  lines = ["\n## Curriculum stages\n"]
  for name, term in curriculum.items():
    if not isinstance(term, dict):
      continue
    params = term.get("params", {}) or {}
    lines.append(f"\n### `{name}`\n")
    for key in ("weight_stages", "stages", "velocity_stages"):
      if key in params and isinstance(params[key], list):
        lines.append(f"`{key}`:\n")
        lines.append("```")
        for stage in params[key]:
          lines.append(str(stage))
        lines.append("```\n")
        break
  return "\n".join(lines)


def save_summary(
  log_dir: Path,
  task_id: str,
  env_cfg: Any,
  agent_cfg: Any,
  saved_sources: list[str],
) -> None:
  """Write a human-readable ``summary.md`` next to ``params/``."""
  try:
    env_dict = asdict(env_cfg) if not isinstance(env_cfg, dict) else env_cfg
  except TypeError:
    env_dict = {}
  try:
    agent_dict = asdict(agent_cfg) if not isinstance(agent_cfg, dict) else agent_cfg
  except TypeError:
    agent_dict = {}

  rewards_section = _format_rewards_table(env_dict)
  curriculum_section = _format_curriculum(env_dict)
  experiment = agent_dict.get("experiment_name", "")
  algorithm = agent_dict.get("algorithm", {}) or {}
  actor = agent_dict.get("actor", {}) or {}
  critic = agent_dict.get("critic", {}) or {}

  body = f"""# Run snapshot

- **task_id**: `{task_id}`
- **experiment_name**: `{experiment}`
- **log_dir**: `{log_dir}`

For exact reproduction:
- `params/env.yaml` — full env config (rewards, observations, events, ...)
- `params/agent.yaml` — RL config (PPO + actor + critic)
- `git/mjlab.diff` — commit hash + unstaged diff
- `source/` — snapshot of the {len(saved_sources)} task/reward-related Python files

To roll back to this exact code:
```bash
cp -r {log_dir}/source/* src/mjlab/
git checkout $(grep -A1 'git commit' {log_dir}/git/mjlab.diff | tail -1)
git apply {log_dir}/git/mjlab.diff  # if uncommitted changes were captured
```

## RL config (summary)

- actor class: `{actor.get("class_name", "?")}`
- critic hidden_dims: `{critic.get("hidden_dims", "?")}`
- learning_rate: `{algorithm.get("learning_rate", "?")}`
- entropy_coef: `{algorithm.get("entropy_coef", "?")}`
- clip_param: `{algorithm.get("clip_param", "?")}`
- num_steps_per_env: `{agent_dict.get("num_steps_per_env", "?")}`
- max_iterations: `{agent_dict.get("max_iterations", "?")}`
- save_interval: `{agent_dict.get("save_interval", "?")}`

## Reward stack
{rewards_section}{curriculum_section}

## Snapshotted source files

"""
  for rel in saved_sources:
    body += f"- `source/{rel}`\n"

  out = log_dir / "summary.md"
  out.write_text(body)


def snapshot_run(
  log_dir: Path,
  task_id: str,
  env_cfg: Any,
  agent_cfg: Any,
) -> None:
  """Save source snapshot + summary alongside an active training run."""
  try:
    saved = save_source_snapshot(log_dir)
    save_summary(log_dir, task_id, env_cfg, agent_cfg, saved)
  except Exception as exc:
    # Snapshot must not block training.
    print(f"[WARN] run_snapshot failed: {exc!r}")
