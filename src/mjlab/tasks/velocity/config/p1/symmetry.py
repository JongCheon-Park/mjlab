"""Sagittal-plane mirror symmetry helpers for KIMM P1 velocity training.

Provides ``MirrorFn``, the ``data_augmentation_func`` for rsl_rl's
``symmetry_cfg``.  The function reflects observations and actions through the
xz (sagittal) plane — equivalent to swapping the robot's left and right sides.

Mirror rules
------------
* Angular velocity (body frame):  (wx, wy, wz) → (-wx, wy, -wz)
* Projected gravity (body frame):  (gx, gy, gz) → (gx, -gy, gz)
* Velocity command:                (vx, vy, ωz) → (vx, -vy, -ωz)
* Joint state / actions:
    - left ↔ right partner swap
    - sign flip for roll (x-axis) and yaw (z-axis) joints
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

# ---------------------------------------------------------------------------
# Joint sign-flip pattern
# ---------------------------------------------------------------------------

# Joints whose rotation axis has a component that flips sign under y → -y:
#   x-axis joints (roll): sign = -1
#   z-axis joints (yaw):  sign = -1
#   y-axis joints (pitch, knee, elbow, wrist_pitch): sign = +1  (no flip)
_SIGN_FLIP_RE = re.compile(
  r".*(hip_roll|hip_yaw|ankle_roll"
  r"|shoulder_roll|shoulder_yaw"
  r"|wrist_yaw|wrist_roll"
  r"|waist_yaw).*"
)


# ---------------------------------------------------------------------------
# Helper: build joint permutation and sign vector
# ---------------------------------------------------------------------------


def _joint_mirror(
  joint_names: list[str],
) -> tuple[list[int], list[float]]:
  """Return ``(perm, signs)`` for a left ↔ right sagittal mirror.

  ``perm[i]`` is the *source* index for output position *i*.
  ``signs[i]`` is ±1 for the corresponding rotation-axis sign change.
  """
  name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
  perm: list[int] = []
  signs: list[float] = []

  for i, name in enumerate(joint_names):
    # Swap left/right prefix to find the mirror partner.
    if name.startswith("left_"):
      partner = "right_" + name[5:]
    elif name.startswith("right_"):
      partner = "left_" + name[6:]
    else:
      partner = name  # e.g. waist_yaw_joint — no positional swap

    perm.append(name_to_idx.get(partner, i))
    signs.append(-1.0 if _SIGN_FLIP_RE.match(name) else 1.0)

  return perm, signs


# ---------------------------------------------------------------------------
# MirrorFn
# ---------------------------------------------------------------------------


class MirrorFn:
  """Sagittal-plane data-augmentation callable for rsl_rl ``symmetry_cfg``.

  Permutation/sign tensors are built **lazily** on the first call, so the
  object survives ``dataclasses.asdict()`` deep-copy (which resets all
  instance state to ``None``).

  rsl_rl PPO calls this with three distinct signatures:

  1. Data augmentation (both provided):
       ``obs_aug, act_aug = fn(env, obs, actions)``
     Returns ``[2B, ...]`` by concatenating original + mirrored.

  2. Mirror-loss obs-only:
       ``obs_aug, None = fn(env, obs, None)``

  3. Mirror-loss action-only:
       ``None, act_aug = fn(env, None, actions)``
  """

  def __init__(self) -> None:
    self._obs_perm: torch.Tensor | None = None
    self._obs_signs: torch.Tensor | None = None
    self._act_perm: torch.Tensor | None = None
    self._act_signs: torch.Tensor | None = None

  # ------------------------------------------------------------------
  # Lazy build
  # ------------------------------------------------------------------

  def _get_joint_names_for_term(
    self,
    robot: Any,
    term_cfg: Any,
  ) -> list[str]:
    """Resolve the ordered joint names observed by a joint obs term."""
    params = term_cfg.params or {}
    asset_cfg = params.get("asset_cfg")
    if asset_cfg is None:
      return list(robot.joint_names)
    jids = asset_cfg.joint_ids
    if isinstance(jids, slice):
      return list(robot.joint_names)
    return [robot.joint_names[i] for i in jids]

  def _build(self, env: Any) -> None:
    # env is RslRlVecEnvWrapper (rsl_rl VecEnv); unwrap to reach ManagerBasedRlEnv.
    base_env = env.unwrapped
    obs_manager = base_env.observation_manager
    term_names: list[str] = obs_manager._group_obs_term_names["actor"]
    term_dims: list[tuple[int, ...]] = obs_manager._group_obs_term_dim["actor"]
    term_cfgs: list[Any] = obs_manager._group_obs_term_cfgs["actor"]

    robot = base_env.scene["robot"]

    obs_perm: list[int] = []
    obs_signs: list[float] = []
    offset = 0

    for name, dim_tuple, tcfg in zip(term_names, term_dims, term_cfgs, strict=False):
      flat_dim = int(np.prod(dim_tuple))
      # Each term stores history-flattened dim: flat_dim = single_dim * H.
      # Use per-term history_length (group-level override is already applied
      # to term_cfg.history_length by the obs manager during _prepare_terms).
      H = max(tcfg.history_length, 1)
      single_dim = flat_dim // H

      # Build single-step permutation and sign pattern.
      local_perm = list(range(single_dim))
      local_signs = [1.0] * single_dim

      if name == "base_ang_vel":
        # Body-frame angular velocity: (wx, wy, wz) → (-wx, wy, -wz)
        # Roll and yaw rates flip; pitch rate is symmetric.
        local_signs = [-1.0, 1.0, -1.0]

      elif name == "projected_gravity":
        # Gravity in body frame: (gx, gy, gz) → (gx, -gy, gz)
        # When robot tilts right gy<0; mirrored robot tilts left gy>0.
        local_signs = [1.0, -1.0, 1.0]

      elif name == "command":
        # Velocity command: (vx, vy, ωz) → (vx, -vy, -ωz)
        local_signs = [1.0, -1.0, -1.0]

      elif name in ("joint_pos", "joint_vel"):
        jnt_names = self._get_joint_names_for_term(robot, tcfg)
        local_perm, local_signs = _joint_mirror(jnt_names)

      elif name == "actions":
        action_term = base_env.action_manager.get_term("joint_pos")
        jnt_names = list(action_term._target_names)
        local_perm, local_signs = _joint_mirror(jnt_names)

      # else: unrecognised term → identity (no permutation, no sign flip)

      # Expand for history: term-major layout
      # [all_dims@t=0, all_dims@t=1, ..., all_dims@t=H-1]
      for h in range(H):
        base = offset + h * single_dim
        for j in range(single_dim):
          obs_perm.append(base + local_perm[j])
          obs_signs.append(local_signs[j])

      offset += flat_dim

    device = base_env.device
    self._obs_perm = torch.tensor(obs_perm, device=device, dtype=torch.long)
    self._obs_signs = torch.tensor(obs_signs, device=device, dtype=torch.float32)

    # Action permutation — ordered by the joint_pos action term.
    action_term = base_env.action_manager.get_term("joint_pos")
    act_jnt_names = list(action_term._target_names)
    act_perm, act_signs = _joint_mirror(act_jnt_names)
    self._act_perm = torch.tensor(act_perm, device=device, dtype=torch.long)
    self._act_signs = torch.tensor(act_signs, device=device, dtype=torch.float32)

  # ------------------------------------------------------------------
  # Callable interface
  # ------------------------------------------------------------------

  def __call__(
    self,
    env: Any,
    obs: Any,  # tensordict.TensorDict | None
    actions: torch.Tensor | None,
  ) -> tuple[Any, torch.Tensor | None]:
    if self._obs_perm is None:
      self._build(env)

    obs_out = None
    if obs is not None:
      actor_obs: torch.Tensor = obs["actor"]  # [B, obs_dim]
      mirrored_actor = actor_obs[:, self._obs_perm] * self._obs_signs
      # Build a new TensorDict: actor gets proper mirror; other groups
      # (e.g. critic) get plain duplication.
      obs_out = TensorDict(
        {
          key: (
            torch.cat([actor_obs, mirrored_actor], dim=0)
            if key == "actor"
            else torch.cat([obs[key], obs[key]], dim=0)
          )
          for key in obs.keys()
        },
        batch_size=[actor_obs.shape[0] * 2],
        device=obs.device,
      )

    act_out = None
    if actions is not None:
      assert self._act_perm is not None and self._act_signs is not None
      mirrored_acts = actions[:, self._act_perm] * self._act_signs
      act_out = torch.cat([actions, mirrored_acts], dim=0)

    return obs_out, act_out
