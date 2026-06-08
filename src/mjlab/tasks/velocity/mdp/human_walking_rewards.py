"""Human-inspired walking reward functions for humanoid velocity tasks.

These rewards complement the existing biped/gait reward set with three
explicit human-walking-imitation components recommended in recent humanoid
RL literature (Human-Inspired Adaptive Gait Learning, SRA-GAIL, Statistical
Reward Shaping, SKATER):

- :func:`bilateral_symmetry`: encourages anti-phase hip pitch / knee swing
  and mirror-phase hip roll between left and right legs, with adaptive
  attenuation during yaw commands.
- :func:`joint_power`: penalizes |torque · joint_vel| as an energy /
  cost-of-transport proxy.
- :func:`alive_bonus`: constant survival reward to stabilize early PPO.

Joint name patterns follow the standard humanoid convention used by P1 / G1
(left_hip_pitch_joint, left_knee_joint, left_ankle_pitch_joint, ...).
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg


def _resolve_joint_idx(env: ManagerBasedRlEnv, asset_name: str, joint_name: str) -> int:
  """Resolve a single joint index by exact name. Caches per (asset, joint)."""
  asset = env.scene[asset_name]
  # Entity exposes ``joint_names`` as a property (tuple[str, ...]).
  names = list(asset.joint_names)
  if joint_name not in names:
    raise KeyError(
      f"Joint {joint_name!r} not found on entity {asset_name!r}. "
      f"Available joints: {names[:6]}..."
    )
  return names.index(joint_name)


def _resolve_knee_joint_idx(env: ManagerBasedRlEnv, asset_name: str, side: str) -> int:
  """Resolve the knee joint index for V4 (knee_pitch_joint) or P1/Diden
  (knee_joint). Tries the explicit name first, falls back to the
  alternative spelling."""
  asset = env.scene[asset_name]
  names = list(asset.joint_names)
  for candidate in (f"{side}_knee_joint", f"{side}_knee_pitch_joint"):
    if candidate in names:
      return names.index(candidate)
  raise KeyError(
    f"No knee joint found for {side!r} on entity {asset_name!r}. "
    f"Tried 'knee_joint' and 'knee_pitch_joint'. Available: {names[:6]}..."
  )


def bilateral_symmetry(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str = "twist",
  sigma_sym: float = 0.3,
  w_hp: float = 1.0,
  w_hr: float = 0.5,
  w_knee_vel: float = 0.3,
  w_ankle: float = 0.5,
  turn_threshold: float = 1.0,
  hip_roll_mirror_sign: float = 1.0,
) -> torch.Tensor:
  """Reward bilateral leg symmetry following human walking phase relations.

  Targets per joint pair:
  - hip_pitch:  anti-phase   → (q_L + q_R) ≈ 0
  - hip_roll:   mirror       → (q_L - mirror · q_R) ≈ 0
  - knee:       anti-phase   → (q̇_L + q̇_R) ≈ 0 (velocity-based)
  - ankle_pitch: anti-phase  → (q_L + q_R) ≈ 0

  The reward is multiplied by ``(1 - |ω_z_cmd| / turn_threshold)`` clamped to
  [0, 1] so that turning commands relax the symmetry requirement (turning
  inherently breaks bilateral symmetry).

  Args:
    asset_cfg: must point to the robot entity.
    command_name: command name (default "twist") to source yaw cmd from.
    sigma_sym: spread of the exp; smaller = stricter.
    w_hp, w_hr, w_knee_vel, w_ankle: per-pair weights in the error sum.
    turn_threshold: yaw-cmd magnitude at which symmetry weight reaches 0.
    hip_roll_mirror_sign: +1 if left/right hip_roll have same sign for
      symmetric motion (typical); -1 if axes are mirrored in MJCF.

  Returns:
    Tensor (n_envs,) in [0, 1].
  """
  asset_name = asset_cfg.name
  i_lhp = _resolve_joint_idx(env, asset_name, "left_hip_pitch_joint")
  i_rhp = _resolve_joint_idx(env, asset_name, "right_hip_pitch_joint")
  i_lhr = _resolve_joint_idx(env, asset_name, "left_hip_roll_joint")
  i_rhr = _resolve_joint_idx(env, asset_name, "right_hip_roll_joint")
  i_lkn = _resolve_knee_joint_idx(env, asset_name, "left")
  i_rkn = _resolve_knee_joint_idx(env, asset_name, "right")
  i_lap = _resolve_joint_idx(env, asset_name, "left_ankle_pitch_joint")
  i_rap = _resolve_joint_idx(env, asset_name, "right_ankle_pitch_joint")

  asset = env.scene[asset_name]
  q = asset.data.joint_pos
  qd = asset.data.joint_vel

  sym_hp = torch.square(q[:, i_lhp] + q[:, i_rhp])
  sym_hr = torch.square(q[:, i_lhr] - hip_roll_mirror_sign * q[:, i_rhr])
  sym_knee = torch.square(qd[:, i_lkn] + qd[:, i_rkn])
  sym_ankle = torch.square(q[:, i_lap] + q[:, i_rap])

  error = w_hp * sym_hp + w_hr * sym_hr + w_knee_vel * sym_knee + w_ankle * sym_ankle

  yaw_cmd = env.command_manager.get_command(command_name)[:, 2]
  turning_factor = torch.clamp(
    1.0 - torch.abs(yaw_cmd) / max(turn_threshold, 1e-6), min=0.0, max=1.0
  )

  return turning_factor * torch.exp(-error / (sigma_sym**2))


def joint_power(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Energy proxy: sum of |τ · q̇| over selected joints.

  Use a small negative weight (e.g. -5e-4). Direct cost-of-transport (CoT)
  metric should be computed in evaluation, not training, since it requires
  travelled distance.
  """
  asset = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  # mjlab exposes per-actuator force on ``asset.data.actuator_force``. We use
  # the actuator-space torque and joint velocities to approximate mechanical
  # power. For shared-actuator joints the mapping is 1:1 on legs.
  torque = asset.data.actuator_force
  qd = asset.data.joint_vel
  # Restrict to selected joints if provided.
  if joint_ids is not None and len(joint_ids) > 0:
    # actuator_force aligns with controllable joints; resolve via local ids.
    n_act = torque.shape[1]
    valid_ids = [i for i in joint_ids if i < n_act]
    if valid_ids:
      torque = torque[:, valid_ids]
      qd = qd[:, valid_ids]
  return torch.sum(torch.abs(torque * qd), dim=1)


def alive_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Constant survival bonus. Useful in early training when sparse positive
  signal is hard to come by.
  """
  return torch.ones(env.num_envs, device=env.device)


def forward_straight_gait(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str = "twist",
  forward_threshold: float = 0.2,
  hip_yaw_sigma: float = 0.10,
  ankle_roll_sigma: float = 0.08,
) -> torch.Tensor:
  """Bonus for clean straight forward gait (toes forward, ankles flat).

  Soft-gated by command direction: the reward is multiplied by a linear
  ramp factor that is 1.0 when v_y = yaw_cmd = 0 and decays to 0 as
  either reaches ``forward_threshold``. This way "mostly forward"
  commands still receive partial gradient instead of the original binary
  on/off (which was only active ~14% of samples in stage 0).

  In the active regime, reward small hip_yaw and ankle_roll deviations
  from default. When command has lateral/yaw component, ramp drops the
  reward toward 0 so the policy is free to use those joints — i.e.
  constraints are *shaped via reward* not enforced via tight pose std.

  Returns: Tensor (n_envs,) in [0, 1].
  """
  cmd = env.command_manager.get_command(command_name)
  v_y = cmd[:, 1]
  yaw_cmd = cmd[:, 2]
  thr = max(forward_threshold, 1e-6)
  forward_factor_y = torch.clamp(1.0 - torch.abs(v_y) / thr, min=0.0, max=1.0)
  forward_factor_yaw = torch.clamp(1.0 - torch.abs(yaw_cmd) / thr, min=0.0, max=1.0)
  forward_factor = forward_factor_y * forward_factor_yaw

  i_lhy = _resolve_joint_idx(env, asset_cfg.name, "left_hip_yaw_joint")
  i_rhy = _resolve_joint_idx(env, asset_cfg.name, "right_hip_yaw_joint")
  i_lar = _resolve_joint_idx(env, asset_cfg.name, "left_ankle_roll_joint")
  i_rar = _resolve_joint_idx(env, asset_cfg.name, "right_ankle_roll_joint")

  asset = env.scene[asset_cfg.name]
  q = asset.data.joint_pos

  hip_yaw_err = torch.square(q[:, i_lhy]) + torch.square(q[:, i_rhy])
  ankle_roll_err = torch.square(q[:, i_lar]) + torch.square(q[:, i_rar])
  err = hip_yaw_err / (hip_yaw_sigma**2) + ankle_roll_err / (ankle_roll_sigma**2)
  return forward_factor * torch.exp(-err)


def lateral_step_lead(
  env: ManagerBasedRlEnv,
  contact_sensor_name: str = "feet_ground_contact",
  command_name: str = "twist",
  lateral_threshold: float = 0.1,
  left_link_name: str = "left_ankle_roll_link",
  right_link_name: str = "right_ankle_roll_link",
) -> torch.Tensor:
  """Reward proper side-step pattern: when a lateral cmd (v_y) is active,
  the LEADING foot (the one on the side of motion) should be in the air
  while the trailing foot supports the body — i.e., a real lateral step
  rather than a body-tilt slide.

  Behavior:
    v_y >  lateral_threshold (move left):
      reward = 1 if left foot in air AND right foot in contact else 0
    v_y < -lateral_threshold (move right):
      reward = 1 if right foot in air AND left foot in contact else 0
    |v_y| <= lateral_threshold:
      reward = 0 (no lateral cmd active)

  This is a hard binary signal — combine with a small weight (e.g., 0.3)
  to nudge the policy without forcing a single rigid stepping pattern.
  """
  cmd = env.command_manager.get_command(command_name)
  v_y = cmd[:, 1]

  sensor = env.scene[contact_sensor_name]
  names = sensor.primary_names
  if left_link_name not in names or right_link_name not in names:
    return torch.zeros(env.num_envs, device=env.device)
  i_left = names.index(left_link_name)
  i_right = names.index(right_link_name)
  found = sensor.data.found
  assert found is not None
  contact_left = found[:, i_left] > 0
  contact_right = found[:, i_right] > 0

  is_lateral = torch.abs(v_y) > lateral_threshold
  move_left = v_y > 0
  correct_lead = torch.where(
    move_left,
    (~contact_left) & contact_right,  # leading = left, lift it
    contact_left & (~contact_right),  # leading = right, lift it
  )
  return (is_lateral & correct_lead).to(torch.float32)
