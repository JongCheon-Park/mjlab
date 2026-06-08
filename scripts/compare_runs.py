"""Compare two run snapshots (summary.md / params / source).

Usage:
    uv run python scripts/compare_runs.py <run_dir_A> <run_dir_B>

Prints:
- env.yaml diff (reward weights / params)
- source/ file diffs
- which run has lateral_step_lead, what pose stds, etc.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path


def _read(path: Path) -> list[str]:
  if not path.is_file():
    return []
  return path.read_text().splitlines(keepends=True)


def _diff_text(a_path: Path, b_path: Path, label: str) -> None:
  a = _read(a_path)
  b = _read(b_path)
  if not a and not b:
    return
  diff = list(
    difflib.unified_diff(a, b, fromfile=f"A/{label}", tofile=f"B/{label}", n=2)
  )
  if not diff:
    return
  print(f"\n--- {label} differs ({len(diff)} lines) ---")
  print("".join(diff))


def main() -> None:
  if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)
  a = Path(sys.argv[1]).resolve()
  b = Path(sys.argv[2]).resolve()
  print(f"A = {a}")
  print(f"B = {b}")
  print()

  # env.yaml (most impactful diff for reward/cfg differences)
  _diff_text(a / "params" / "env.yaml", b / "params" / "env.yaml", "params/env.yaml")

  # agent.yaml (actor/critic/PPO differences)
  _diff_text(
    a / "params" / "agent.yaml", b / "params" / "agent.yaml", "params/agent.yaml"
  )

  # source/ snapshots
  a_src = a / "source"
  b_src = b / "source"
  if a_src.is_dir() and b_src.is_dir():
    rels = sorted(
      {str(p.relative_to(a_src)) for p in a_src.rglob("*.py")}
      | {str(p.relative_to(b_src)) for p in b_src.rglob("*.py")}
    )
    for rel in rels:
      _diff_text(a_src / rel, b_src / rel, f"source/{rel}")

  # Git commit hash from mjlab.diff
  for tag, run in (("A", a), ("B", b)):
    diff_file = run / "git" / "mjlab.diff"
    if not diff_file.is_file():
      continue
    head = diff_file.read_text().split("\n", 3)[:3]
    commit = next((ln for ln in head if ln and ln[0:1].isalnum()), "")
    print(f"\n{tag} commit: {commit}")


if __name__ == "__main__":
  main()
