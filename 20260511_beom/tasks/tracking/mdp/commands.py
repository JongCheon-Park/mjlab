from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from mjlab.tasks.tracking.mdp.virtual_assistive_wrench import VirtualAssistiveWrenchCfg

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from collections.abc import Callable
  from typing import Any

  import viser

  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class MotionLoader:
  def __init__(
    self, motion_file: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    data = np.load(motion_file)
    self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
    self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
    self._body_pos_w = torch.tensor(
      data["body_pos_w"], dtype=torch.float32, device=device
    )
    self._body_quat_w = torch.tensor(
      data["body_quat_w"], dtype=torch.float32, device=device
    )
    self._body_lin_vel_w = torch.tensor(
      data["body_lin_vel_w"], dtype=torch.float32, device=device
    )
    self._body_ang_vel_w = torch.tensor(
      data["body_ang_vel_w"], dtype=torch.float32, device=device
    )
    self._body_indexes = body_indexes
    self.body_pos_w = self._body_pos_w[:, self._body_indexes]
    self.body_quat_w = self._body_quat_w[:, self._body_indexes]
    self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
    self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]
    self.time_step_total = self.joint_pos.shape[0]
    self.fps = float(data["fps"][0])

    # Reference accelerations via forward finite difference: a[t] = (v[t+1]-v[t])*fps.
    # Last frame duplicates second-to-last to maintain length.
    self._body_lin_acc_w = self._finite_diff(self._body_lin_vel_w, self.fps)
    self._body_ang_acc_w = self._finite_diff(self._body_ang_vel_w, self.fps)
    self.body_lin_acc_w = self._body_lin_acc_w[:, self._body_indexes]
    self.body_ang_acc_w = self._body_ang_acc_w[:, self._body_indexes]

  @staticmethod
  def _finite_diff(vel: torch.Tensor, fps: float) -> torch.Tensor:
    """Forward finite difference along time axis (dim 0). Shape: (T, ...)."""
    acc = torch.empty_like(vel)
    acc[:-1] = (vel[1:] - vel[:-1]) * fps
    acc[-1] = acc[-2]
    return acc


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = MotionLoader(
      self.cfg.motion_file, self.body_indexes, device=self.device
    )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.obs_pre_shift_steps = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._all_env_ids = torch.arange(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._active_bin_index = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1
    self.bin_failure_rate = torch.ones(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_total = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    # VirtualAssistiveWrench (lazy init on first reset).
    self._virtual_assistive_wrench = None

    # Ghost model created lazily on first visualization
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  @property
  def command(self) -> torch.Tensor:
    return self.get_command()

  def _resolve_time_steps(
    self, step_offset: int = 0, use_pre_shift: bool = False
  ) -> torch.Tensor:
    time_steps = self.time_steps
    if use_pre_shift:
      time_steps = time_steps + self.obs_pre_shift_steps
    if step_offset != 0:
      time_steps = time_steps + step_offset
    return torch.clamp(time_steps, 0, self.motion.time_step_total - 1)

  def get_command(
    self, step_offset: int = 0, use_pre_shift: bool = False
  ) -> torch.Tensor:
    time_steps = self._resolve_time_steps(
      step_offset=step_offset, use_pre_shift=use_pre_shift
    )
    return torch.cat(
      [self.motion.joint_pos[time_steps], self.motion.joint_vel[time_steps]], dim=1
    )

  def get_future_command(
    self, future_len: int, use_pre_shift: bool = False
  ) -> torch.Tensor:
    if future_len < 0:
      raise ValueError(f"future_len must be >= 0, got {future_len}")
    # Convention:
    #   - future_len == 0 -> return a single frame at base_idx (backward-compatible)
    #   - future_len == N (>0) -> return exactly N frames:
    #       [base_idx + 0, ..., base_idx + (N - 1)]
    # where base_idx is the observation-aligned index.
    if future_len == 0:
      return self.get_command(use_pre_shift=use_pre_shift)
    return torch.cat(
      [
        self.get_command(step_offset=offset, use_pre_shift=use_pre_shift)
        for offset in range(future_len)
      ],
      dim=1,
    )

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self.get_anchor_pos_w()

  def get_anchor_pos_w(
    self, step_offset: int = 0, use_pre_shift: bool = False
  ) -> torch.Tensor:
    time_steps = self._resolve_time_steps(
      step_offset=step_offset, use_pre_shift=use_pre_shift
    )
    return (
      self.motion.body_pos_w[time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.get_anchor_quat_w()

  def get_anchor_quat_w(
    self, step_offset: int = 0, use_pre_shift: bool = False
  ) -> torch.Tensor:
    time_steps = self._resolve_time_steps(
      step_offset=step_offset, use_pre_shift=use_pre_shift
    )
    return self.motion.body_quat_w[time_steps, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_lin_acc_w(self) -> torch.Tensor:
    return self.motion.body_lin_acc_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_ang_acc_w(self) -> torch.Tensor:
    return self.motion.body_ang_acc_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def current_bin_index(self) -> torch.Tensor:
    """Map current time_steps to bin indices. Shape (num_envs,)."""
    return torch.clamp(
      (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
      0,
      self.bin_count - 1,
    )

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    episode_failed = self._env.termination_manager.terminated[env_ids]
    terminal_bins = self._active_bin_index[env_ids]
    # Accumulate (not overwrite) bin counts across multiple call sites per step.
    # _adaptive_sampling can be called from both CommandTerm.reset (env
    # termination/timeout) and _update_command (motion overflow) within the
    # same env.step(). Using += ensures termination data from the reset path
    # is not lost when the overflow path fires afterwards.
    self._current_bin_total += torch.bincount(terminal_bins, minlength=self.bin_count)
    if torch.any(episode_failed):
      fail_bins = terminal_bins[episode_failed]
      self._current_bin_failed += torch.bincount(fail_bins, minlength=self.bin_count)

    # Sample using Softmax (Logit) approach + Uniform floor (ZEST S5).
    tau = getattr(self.cfg, "adaptive_temperature", 0.1)
    logits = self.bin_failure_rate / tau
    p_softmax = torch.softmax(logits, dim=0)
    eps = self.cfg.adaptive_uniform_ratio
    sampling_probabilities = (1.0 - eps) * p_softmax + eps / float(self.bin_count)

    sampling_probabilities = torch.nn.functional.pad(
      sampling_probabilities.unsqueeze(0).unsqueeze(0),
      (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
      mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
      sampling_probabilities, self.kernel.view(1, 1, -1)
    ).view(-1)

    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    sampled_bins = torch.multinomial(
      sampling_probabilities, len(env_ids), replacement=True
    )
    self.time_steps[env_ids] = torch.clamp(
      (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
      / self.bin_count
      * self.motion.time_step_total,
      max=self.motion.time_step_total - 1,
    ).long()

    # Update metrics.
    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    pmax, imax = sampling_probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

  def _uniform_sampling(self, env_ids: torch.Tensor):
    self.time_steps[env_ids] = torch.randint(
      0, self.motion.time_step_total, (len(env_ids),), device=self.device
    )
    self.metrics["sampling_entropy"][:] = 1.0  # Maximum entropy for uniform.
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5  # No specific bin preference.

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Clip joint positions and write root + joint state to sim."""
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    # Update VirtualAssistiveWrench curriculum before resampling.
    if self._virtual_assistive_wrench is not None:
      self._virtual_assistive_wrench.update_curriculum()

    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    # Sync active bin tracking for any sampling mode
    self._active_bin_index[env_ids] = self.current_bin_index[env_ids]

    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori = quat_mul(orientations_delta, root_ori)
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel += rand_samples[:, :3]
    root_ang_vel += rand_samples[:, 3:]

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )

    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )

    # Reset VirtualAssistiveWrench episode beta.
    if self._virtual_assistive_wrench is not None:
      self._virtual_assistive_wrench.reset(env_ids)

  def _sample_obs_pre_shift_steps(self, env_ids: torch.Tensor) -> None:
    num_envs = env_ids.numel()
    if num_envs == 0:
      return
    low_s, high_s = self.cfg.pre_shift_range_s
    apply_prob = self.cfg.pre_shift_apply_prob
    if low_s < 0.0 or high_s < low_s:
      raise ValueError(
        f"Invalid pre_shift_range_s={self.cfg.pre_shift_range_s}. Expected 0 <= min <= max."
      )
    if apply_prob < 0.0 or apply_prob > 1.0:
      raise ValueError(
        f"Invalid pre_shift_apply_prob={apply_prob}. Expected value in [0, 1]."
      )
    if high_s <= 0.0 or apply_prob <= 0.0:
      self.obs_pre_shift_steps[env_ids] = 0
      return
    fps = self.motion.fps
    max_shift_steps = int(round(high_s * fps))
    if max_shift_steps <= 0:
      self.obs_pre_shift_steps[env_ids] = 0
      return

    # Apply pre-shift decision is step-wide, not env-wise.
    if apply_prob < 1.0 and bool(
      (torch.rand((), device=self.device) >= apply_prob).item()
    ):
      self.obs_pre_shift_steps[env_ids] = 0
      return

    # Gate is on and positive shift is representable.
    min_shift_steps = 1

    # Fast path for fixed pre-shift range.
    if low_s == high_s:
      fixed_steps = int(round(low_s * fps))
      self.obs_pre_shift_steps[env_ids] = max(min_shift_steps, fixed_steps)
      return

    sampled_s = sample_uniform(
      lower=low_s,
      upper=high_s,
      size=(num_envs,),
      device=self.device,
    )
    sampled_steps = torch.round(sampled_s * fps).long()
    sampled_steps.clamp_(min=min_shift_steps, max=max_shift_steps)
    self.obs_pre_shift_steps[env_ids] = sampled_steps

  def sample_obs_pre_shift_steps(self, env_ids: torch.Tensor | None = None) -> None:
    target_env_ids = self._all_env_ids if env_ids is None else env_ids
    self._sample_obs_pre_shift_steps(target_env_ids)

  def _update_command(self):
    self.time_steps += 1

    # Check for natural bin progression (success)
    new_bin_index = self.current_bin_index
    advanced_envs = torch.where(new_bin_index > self._active_bin_index)[0]
    if len(advanced_envs) > 0:
      advanced_bins = self._active_bin_index[advanced_envs]
      self._current_bin_total += torch.bincount(advanced_bins, minlength=self.bin_count)
      self._active_bin_index[advanced_envs] = new_bin_index[advanced_envs]

    env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    # Observation pre-shift is sampled at every step (env-wise),
    # while reward/termination references remain aligned to current timestep.
    self.sample_obs_pre_shift_steps()

  def update_relative_body_poses(self) -> None:
    """Recompute ``body_pos_relative_w`` and ``body_quat_relative_w``.

    Called after ``reset_to_frame`` so that termination checks that
    compare relative body positions see the correct state.
    """
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

  def _update_command(self):
    self.time_steps += 1
    env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    self.update_relative_body_poses()

    if self.cfg.sampling_mode == "adaptive":
      has_data = self._current_bin_total > 0
      current_f_rate = self._current_bin_failed / self._current_bin_total.clamp(min=1)
      self.bin_failure_rate = torch.where(
        has_data,
        self.cfg.adaptive_alpha * current_f_rate
        + (1 - self.cfg.adaptive_alpha) * self.bin_failure_rate,
        self.bin_failure_rate,
      )
      self._current_bin_failed.zero_()
      self._current_bin_total.zero_()

    # Apply VirtualAssistiveWrench wrench.
    if self._virtual_assistive_wrench is not None:
      self._virtual_assistive_wrench.apply_wrench()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        # Build a ghost model with only visual geoms visible. Collision geoms (nonzero
        # contype/conaffinity) get alpha=0 so the viewer's alpha filter excludes them.
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        for gi in range(self._ghost_model.ngeom):
          if (
            self._ghost_model.geom_contype[gi] != 0
            or self._ghost_model.geom_conaffinity[gi] != 0
          ):
            self._ghost_model.geom_rgba[gi, 3] = 0
          else:
            self._ghost_model.geom_rgba[gi] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

        visualizer.add_ghost_mesh(
          qpos,
          model=self._ghost_model,
          label=f"ghost_{batch}",
        )

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create motion scrubber controls in the Viser viewer."""
    max_frame = int(self.motion.time_step_total) - 1

    with server.gui.add_folder(name.capitalize()):
      scrubber = server.gui.add_slider(
        "Frame",
        min=0,
        max=max_frame,
        step=1,
        initial_value=0,
      )

      @scrubber.on_update
      def _(_) -> None:
        idx = get_env_idx()
        self.time_steps[idx] = int(scrubber.value)
        if on_change is not None:
          on_change()

      all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
      start_btn = server.gui.add_button("Start Here")

      @start_btn.on_click
      def _(_) -> None:
        if request_action is not None:
          request_action(
            "CUSTOM",
            {"type": "gui_reset", "all_envs": all_envs_cb.value},
          )

    self._scrubber_handles = (scrubber, all_envs_cb, start_btn)
    self._set_scrubber_disabled(True)

  def _set_scrubber_disabled(self, disabled: bool) -> None:
    """Enable or disable the motion scrubber GUI controls."""
    for handle in self._scrubber_handles:
      handle.disabled = disabled

  def on_viewer_pause(self, paused: bool) -> None:
    if hasattr(self, "_scrubber_handles"):
      self._set_scrubber_disabled(not paused)

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    if not hasattr(self, "_scrubber_handles"):
      return False
    frame = int(self._scrubber_handles[0].value)
    self.reset_to_frame(env_ids, frame)
    self.update_relative_body_poses()
    return True

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    """Reset to exact reference state at a specific frame.

    Like ``_resample_command`` but deterministic: no random
    perturbations to pose, velocity, or joint positions.
    """
    self.time_steps[env_ids] = frame
    self._write_reference_state_to_sim(
      env_ids,
      self.body_pos_w[env_ids, 0],
      self.body_quat_w[env_ids, 0],
      self.body_lin_vel_w[env_ids, 0],
      self.body_ang_vel_w[env_ids, 0],
      self.joint_pos[env_ids],
      self.joint_vel[env_ids],
    )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  motion_file: str
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  adaptive_temperature: float = 0.1
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  pre_shift_range_s: tuple[float, float] = (0.0, 0.0)
  pre_shift_apply_prob: float = 1.0

  # Optional VirtualAssistiveWrench
  virtual_assistive_wrench: VirtualAssistiveWrenchCfg | None = None
  """Set to a VirtualAssistiveWrenchCfg instance to enable assistive wrench curriculum."""

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    cmd = MotionCommand(self, env)
    if self.virtual_assistive_wrench is not None and self.sampling_mode == "adaptive":
      from mjlab.tasks.tracking.mdp.virtual_assistive_wrench import (
        VirtualAssistiveWrench,
      )

      cmd._virtual_assistive_wrench = VirtualAssistiveWrench(
        self.virtual_assistive_wrench, env, cmd
      )
    return cmd
