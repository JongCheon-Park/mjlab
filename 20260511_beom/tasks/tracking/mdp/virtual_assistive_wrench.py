"""VirtualAssistiveWrench: assistive wrench curriculum for dynamic motions.

Applies a PD-based virtual wrench to the robot's root body to assist with
dynamic motion tracking. The assistance gain (beta) is automatically scheduled
per-bin based on tracking performance, decaying to zero as learning progresses.

Reference: Sleiman et al. - ZEST (2026)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import matrix_from_quat, quat_box_minus

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Pure compute functions (easily testable, no env dependency)
# ---------------------------------------------------------------------------


def compute_pd_wrench(
  *,
  p_cur: torch.Tensor,
  v_cur: torch.Tensor,
  q_cur: torch.Tensor,
  w_cur: torch.Tensor,
  p_ref: torch.Tensor,
  v_ref: torch.Tensor,
  q_ref: torch.Tensor,
  w_ref: torch.Tensor,
  M: float,
  I: torch.Tensor,
  kp_pos: float,
  kd_pos: float,
  kp_rot: float,
  kd_rot: float,
  gravity: torch.Tensor | None = None,
  v_dot_ref: torch.Tensor | None = None,
  omega_dot_ref: torch.Tensor | None = None,
  add_coriolis: bool = False,
  gravity_moment: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute assistive wrench in world frame (ZEST eq. 13a/13b).

  Args:
    p_cur: Current root position (N, 3).
    v_cur: Current root linear velocity (N, 3).
    q_cur: Current root quaternion (w,x,y,z) (N, 4).
    w_cur: Current root angular velocity in world frame (N, 3).
    p_ref: Reference root position (N, 3).
    v_ref: Reference root linear velocity (N, 3).
    q_ref: Reference root quaternion (w,x,y,z) (N, 4).
    w_ref: Reference root angular velocity in world frame (N, 3).
    M: Total robot mass (scalar).
    I: Nominal root inertia matrix in world frame (N, 3, 3).
    kp_pos: Position proportional gain.
    kd_pos: Position derivative gain.
    kp_rot: Rotation proportional gain.
    kd_rot: Rotation derivative gain.
    gravity: Gravity vector in world frame (3,) or (N, 3).
    v_dot_ref: Reference linear acceleration (N, 3). If provided, adds the
      feedforward term from ZEST eq. (13a).
    omega_dot_ref: Reference angular acceleration (N, 3). If provided, adds
      the feedforward term from ZEST eq. (13b).
    add_coriolis: If True, adds the omega x (I @ omega) Coriolis term from
      ZEST eq. (13b).
    gravity_moment: Precomputed gravity moment r_{b,com} x Mg (N, 3). If
      provided, subtracts it from tau per ZEST eq. (13b).

  Returns:
    F: Force in world frame (N, 3).
    tau: Torque in world frame (N, 3).
  """
  # Linear force: F = M * (v_dot_ref + kp * pos_err + kd * vel_err - g)
  # ZEST eq (13a).
  pos_err = p_ref - p_cur
  vel_err = v_ref - v_cur
  pd_lin = kp_pos * pos_err + kd_pos * vel_err
  if v_dot_ref is not None:
    pd_lin = pd_lin + v_dot_ref
  if gravity is not None:
    pd_lin = pd_lin - gravity
  F = M * pd_lin

  # Rotation error via SO(3) log map: rot_err = log(q_ref * q_cur^{-1})
  rot_err = quat_box_minus(q_ref, q_cur)  # (N, 3)

  # Angular velocity error.
  ang_vel_err = w_ref - w_cur

  # Torque: tau = I @ (omega_dot_ref + kp * rot_err + kd * ang_vel_err)
  # ZEST eq (13b).
  pd_signal = kp_rot * rot_err + kd_rot * ang_vel_err  # (N, 3)
  if omega_dot_ref is not None:
    pd_signal = pd_signal + omega_dot_ref
  tau = torch.bmm(I, pd_signal.unsqueeze(-1)).squeeze(-1)  # (N, 3)

  # Coriolis: omega x (I @ omega)  — ZEST eq (13b).
  if add_coriolis:
    Iw = torch.bmm(I, w_cur.unsqueeze(-1)).squeeze(-1)  # (N, 3)
    tau = tau + torch.linalg.cross(w_cur, Iw)

  # Gravity moment: -r_{b,com} x Mg  — ZEST eq (13b).
  if gravity_moment is not None:
    tau = tau - gravity_moment

  return F, tau


def compute_beta(
  S_hat: torch.Tensor,
  eta: float,
  beta_max: float,
) -> torch.Tensor:
  """Compute assistive gain from smoothed similarity.

  beta = clip(1 - S_hat / eta, 0, beta_max)

  Args:
    S_hat: Smoothed similarity (any shape).
    eta: Target similarity threshold.
    beta_max: Maximum assistive scale (< 1.0).

  Returns:
    Assistive gain beta (same shape as S_hat).
  """
  return torch.clamp(1.0 - S_hat / eta, min=0.0, max=beta_max)


# ---------------------------------------------------------------------------
# VirtualAssistiveWrench class — integrates with MotionCommand
# ---------------------------------------------------------------------------


@dataclass
class VirtualAssistiveWrenchCfg:
  """Configuration for VirtualAssistiveWrench assistive wrench."""

  entity_name: str = "robot"
  command_name: str = "motion"

  # PD gains for wrench computation.
  kp_pos: float = 50.0
  kd_pos: float = 20.0
  kp_rot: float = 30.0
  kd_rot: float = 10.0

  # Curriculum parameters.
  beta_max: float = 0.4  # Target similarity threshold.
  eta: float = 0.8  # Maximum assistive scale.

  # Acceleration clamps (used in full mode, i.e. when use_PD_only=False).
  acc_lin_clip: float = 50.0  # m/s² — clamp reference linear acceleration
  acc_ang_clip: float = 100.0  # rad/s² — clamp reference angular acceleration

  # If True, apply only PD terms (no feedforward, no Coriolis, no gravity moment).
  use_PD_only: bool = False


class VirtualAssistiveWrench:
  """assistive wrench with automatic per-bin curriculum.

  Usage:
    Attach to a MotionCommand via ``update_cfg()`` in a motion config file.
    The ``MotionCommand`` calls into this class at reset and command-update time.
  """

  def __init__(
    self, cfg: VirtualAssistiveWrenchCfg, env: ManagerBasedRlEnv, motion_cmd
  ) -> None:
    self.cfg = cfg
    self._env = env
    self.device = env.device

    self.robot = env.scene[cfg.entity_name]
    self.motion_cmd = motion_cmd

    # Anchor body = the body where wrench is applied (e.g. torso_link).
    self._anchor_body_local_id = self.motion_cmd.robot_anchor_body_index
    self._total_mass = self._compute_total_mass()
    self._body_masses = self._get_body_masses()  # (num_bodies,)
    self._nominal_inertia = self._compute_nominal_inertia()
    self._gravity = self.robot.data.gravity_vec_w[0]  # (3,) world frame gravity

    # Per-bin curriculum state.
    bin_count = self.motion_cmd.bin_count
    self.beta_table = torch.full((bin_count,), cfg.beta_max, device=self.device)

    # Per-env episode state.
    num_envs = env.num_envs
    self.episode_beta = torch.full((num_envs, 1), cfg.beta_max, device=self.device)

    # Metrics.
    self.motion_cmd.metrics["virtual_assistive_wrench/beta"] = torch.zeros(
      num_envs, device=self.device
    )
    self.motion_cmd.metrics["virtual_assistive_wrench/beta_mean"] = torch.zeros(
      num_envs, device=self.device
    )

  def _compute_total_mass(self) -> float:
    """Compute total mass from model body_mass."""
    body_ids = self.robot.indexing.body_ids
    masses = self._env.sim.model.body_mass[0, body_ids]
    return float(masses.sum().item())

  def _get_body_masses(self) -> torch.Tensor:
    """Get per-body masses as a tensor, shape (num_bodies,)."""
    body_ids = self.robot.indexing.body_ids
    masses = self._env.sim.model.body_mass[0, body_ids]  # (num_bodies,)
    return masses.to(self.device)

  def _compute_nominal_inertia(self) -> torch.Tensor:
    """Get anchor body inertia as 3x3 matrix in body-local frame.

    MuJoCo body_inertia stores the diagonal of the inertia tensor in the
    body's principal-axis frame.  This is intentionally kept in body-local
    frame; apply_wrench() rotates it to world frame each step via
    I_world = R @ I_body @ R^T.

    Shape: (1, 3, 3) for broadcasting.
    """
    anchor_body_id = self.robot.indexing.body_ids[self._anchor_body_local_id]
    inertia_diag = self._env.sim.model.body_inertia[0, anchor_body_id]  # (3,)
    I = torch.diag(inertia_diag).unsqueeze(0)  # (1, 3, 3)
    return I.to(self.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    """Called on episode reset. Sets episode_beta from beta_table."""
    # Get current bin index for each env.
    current_bins = self.motion_cmd.current_bin_index[env_ids]

    # Look up beta for sampled bins.
    self.episode_beta[env_ids, 0] = self.beta_table[current_bins]

    # Update metrics.
    self.motion_cmd.metrics["virtual_assistive_wrench/beta"][env_ids] = (
      self.episode_beta[env_ids, 0]
    )
    self.motion_cmd.metrics["virtual_assistive_wrench/beta_mean"][:] = (
      self.beta_table.mean()
    )

  def apply_wrench(self) -> None:
    """Compute and apply assistive wrench to robot root body. Called per env step."""
    cmd = self.motion_cmd

    # Current root state.
    p_cur = cmd.robot_anchor_pos_w
    v_cur = cmd.robot_anchor_lin_vel_w
    q_cur = cmd.robot_anchor_quat_w
    w_cur = cmd.robot_anchor_ang_vel_w

    # Reference state.
    p_ref = cmd.anchor_pos_w
    v_ref = cmd.anchor_lin_vel_w
    q_ref = cmd.anchor_quat_w
    w_ref = cmd.anchor_ang_vel_w

    # Rotate body-local inertia to world frame: I_world = R @ I_body @ R^T.
    # matrix_from_quat returns the body-to-world rotation matrix (N, 3, 3).
    R = matrix_from_quat(q_cur)  # (N, 3, 3)
    I_body = self._nominal_inertia.expand(self._env.num_envs, -1, -1)  # (N, 3, 3)
    I_world = torch.bmm(torch.bmm(R, I_body), R.transpose(-1, -2))  # (N, 3, 3)

    # Feedforward accelerations (ZEST eq. 13a/13b full form).
    v_dot_ref = None
    omega_dot_ref = None
    if not self.cfg.use_PD_only:
      v_dot_ref = cmd.anchor_lin_acc_w.clamp(
        -self.cfg.acc_lin_clip, self.cfg.acc_lin_clip
      )
      omega_dot_ref = cmd.anchor_ang_acc_w.clamp(
        -self.cfg.acc_ang_clip, self.cfg.acc_ang_clip
      )

    # Gravity moment: -r_{b,com} x Mg — ZEST eq (13b).
    gravity_moment = None
    if not self.cfg.use_PD_only:
      # body_link_pos_w: (N, num_bodies, 3), world-frame body positions.
      body_pos_w = self.robot.data.body_link_pos_w  # (N, num_bodies, 3)
      com_w = (self._body_masses[None, :, None] * body_pos_w).sum(
        dim=1
      ) / self._total_mass  # (N, 3)
      r_b_com = com_w - p_cur  # (N, 3): CoM offset from base in world frame
      gravity_moment = torch.linalg.cross(
        r_b_com, self._total_mass * self._gravity.unsqueeze(0)
      )  # (N, 3): r_{b,com} x Mg

    # Compute wrench (ZEST eq 13a/13b).
    F, tau = compute_pd_wrench(
      p_cur=p_cur,
      v_cur=v_cur,
      q_cur=q_cur,
      w_cur=w_cur,
      p_ref=p_ref,
      v_ref=v_ref,
      q_ref=q_ref,
      w_ref=w_ref,
      M=self._total_mass,
      I=I_world,
      kp_pos=self.cfg.kp_pos,
      kd_pos=self.cfg.kd_pos,
      kp_rot=self.cfg.kp_rot,
      kd_rot=self.cfg.kd_rot,
      gravity=self._gravity,
      v_dot_ref=v_dot_ref,
      omega_dot_ref=omega_dot_ref,
      add_coriolis=not self.cfg.use_PD_only,
      gravity_moment=gravity_moment,
    )

    # Scale by episode beta (ZEST eq: w_e = beta_e * [F; tau]).
    F_scaled = self.episode_beta * F
    tau_scaled = self.episode_beta * tau

    # Apply to anchor body in world frame.
    self.robot.write_external_wrench_to_sim(
      F_scaled.unsqueeze(1),  # (N, 1, 3)
      tau_scaled.unsqueeze(1),  # (N, 1, 3)
      body_ids=[self._anchor_body_local_id],
    )

  def update_curriculum(self) -> None:
    """Update per-bin beta table from termination-based failure rate.

    Uses MotionCommand.bin_failed_count / bin_total_count for proper
    per-bin failure rate in [0, 1]. Both are EMA-smoothed.
    Implements ZEST S6 §1: "f[i,b] maintained by adaptive RSI".
    """
    cmd = self.motion_cmd

    # ZEST S6 §1: S_hat = 1 - f, beta = clip(1 - S_hat/eta, 0, beta_max)
    # bin_failure_rate is initialized to 1.0 (assume failure) and updated
    # via direct EMA: f ← (1-α)*f + α*(failed/total) per visited bin.
    S_hat = 1.0 - cmd.bin_failure_rate
    self.beta_table = compute_beta(S_hat, self.cfg.eta, self.cfg.beta_max)
