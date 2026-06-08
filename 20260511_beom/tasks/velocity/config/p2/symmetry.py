"""Sagittal-plane mirror symmetry for KIMM P2 velocity training.

P2 shares the same left/right naming conventions and joint-axis rules as P1
(hip_roll/-X, hip_yaw/-Z, ankle_roll/-X, shoulder_roll/X, shoulder_yaw/Z,
wrist_yaw/Z, waist_yaw/-Z), so MirrorFn is reused directly.
"""

from mjlab.tasks.velocity.config.p1.symmetry import MirrorFn

__all__ = ["MirrorFn"]
