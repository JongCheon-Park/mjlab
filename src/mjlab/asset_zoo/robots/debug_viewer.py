"""No-op stubs for KIMM-internal viewer helpers.

The KIMM P1 constants reference these helpers from a private fork. The
training/play code paths in mjlab do not use them, so we provide trivial
stubs here to keep ``import`` working without pulling in the private
viewer code.
"""

from __future__ import annotations

from typing import Any

import mujoco


def apply_fixed_base_viewer_defaults(
  spec: mujoco.MjSpec, *args: Any, **kwargs: Any
) -> None:
  """No-op placeholder."""
  del spec, args, kwargs


def apply_joint_limits_to_viewer_controls(
  spec: mujoco.MjSpec, *args: Any, **kwargs: Any
) -> None:
  """No-op placeholder."""
  del spec, args, kwargs


def resolve_viewer_actuator_gains(*args: Any, **kwargs: Any) -> dict[str, Any]:
  """Return empty gain dict."""
  del args, kwargs
  return {}
