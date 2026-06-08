"""Print a compact reward / termination / curriculum summary for a TB run.

Usage:
    uv run python scripts/check_rewards.py <log_dir>
    uv run python scripts/check_rewards.py  # uses latest p1_velocity_moe run
"""

from __future__ import annotations

import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_ROOT = Path("/home/park/mjlab/logs/rsl_rl")
DEFAULT_EXPERIMENTS = ("p1_velocity_moe", "p1_velocity")


def _latest_run() -> Path:
  candidates = []
  for exp in DEFAULT_EXPERIMENTS:
    exp_dir = LOG_ROOT / exp
    if not exp_dir.is_dir():
      continue
    for run in exp_dir.iterdir():
      if run.is_dir() and any(run.iterdir()):
        candidates.append(run)
  if not candidates:
    raise SystemExit(
      f"No runs with content under {LOG_ROOT}/<{','.join(DEFAULT_EXPERIMENTS)}>"
    )
  return max(candidates, key=lambda p: p.stat().st_mtime)


def _load(logdir: Path) -> EventAccumulator:
  ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
  ea.Reload()
  return ea


def _last(ea: EventAccumulator, tag: str) -> tuple[int, float] | None:
  if tag not in ea.Tags().get("scalars", []):
    return None
  events = ea.Scalars(tag)
  if not events:
    return None
  return events[-1].step, events[-1].value


def main() -> None:
  logdir = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest_run()
  ea = _load(logdir)
  tags = ea.Tags().get("scalars", [])
  print(f"=== {logdir.name} ===")

  reward_tags = [t for t in tags if t.startswith("Episode_Reward/")]
  rows = []
  for t in reward_tags:
    events = ea.Scalars(t)
    if len(events) < 2:
      continue
    rows.append((t.replace("Episode_Reward/", ""), events[0].value, events[-1].value))
  rows.sort(key=lambda r: r[2], reverse=True)

  if not rows:
    print("(no Episode_Reward yet — early iters)")
    return

  last_step = ea.Scalars(reward_tags[0])[-1].step
  print(f"iter = {last_step}\n")

  print(f"{'reward':<40s} {'first':>10s} {'last':>10s} {'Δ':>10s}")
  print("-" * 72)
  for name, first, last in rows:
    delta = last - first
    print(f"  {name:<38s} {first:>10.4f} {last:>10.4f} {delta:>+10.4f}")

  print()
  for tag in ("Train/mean_reward", "Train/mean_episode_length"):
    r = _last(ea, tag)
    if r is not None:
      print(f"  {tag:<35s} {r[1]:>10.2f}")

  print("\nterminations:")
  for t in tags:
    if t.startswith("Episode_Termination/"):
      r = _last(ea, t)
      if r is not None:
        print(f"  {t.replace('Episode_Termination/', ''):<35s} {r[1]:>10.4f}")

  print("\ncurriculums (active):")
  for t in tags:
    if t.startswith("Curriculum/"):
      r = _last(ea, t)
      if r is not None:
        print(f"  {t.replace('Curriculum/', ''):<55s} {r[1]:>10.4f}")


if __name__ == "__main__":
  main()
