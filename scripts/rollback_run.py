"""Restore source files from a run snapshot into src/mjlab/.

Usage:
    uv run python scripts/rollback_run.py <run_dir> [--dry-run]

This copies every file under ``<run_dir>/source/`` back into ``src/mjlab/``,
overwriting the current working tree. Use ``--dry-run`` to preview which
files would be modified.

Note: this restores the *files snapshotted at the time of training*. To
also restore other unstaged changes captured at training time, apply
``<run_dir>/git/mjlab.diff`` afterward.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from mjlab import MJLAB_SRC_PATH


def main() -> None:
  args = sys.argv[1:]
  if not args or args[0] in {"-h", "--help"}:
    print(__doc__)
    sys.exit(0)

  dry = "--dry-run" in args
  run_dir = Path([a for a in args if not a.startswith("--")][0]).resolve()
  source_root = run_dir / "source"
  if not source_root.is_dir():
    print(f"[ERROR] no `source/` snapshot under {run_dir}")
    sys.exit(1)

  files = sorted(source_root.rglob("*.py"))
  if not files:
    print(f"[ERROR] no .py files under {source_root}")
    sys.exit(1)

  print(f"Restoring from: {source_root}")
  print(f"Into:           {MJLAB_SRC_PATH}")
  print(f"{'DRY RUN — no files will be written' if dry else 'Applying'}")
  print()

  for src in files:
    rel = src.relative_to(source_root)
    dst = MJLAB_SRC_PATH / rel
    if dst.is_file() and dst.read_bytes() == src.read_bytes():
      action = "unchanged"
    else:
      action = "WOULD WRITE" if dry else "wrote"
      if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"  [{action:>12s}] src/mjlab/{rel}")

  print()
  if dry:
    print("Re-run without --dry-run to apply.")
  else:
    print("Done. Consider:  git diff   to inspect changes.")


if __name__ == "__main__":
  main()
