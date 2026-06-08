"""MoE actor for velocity-tracking tasks with rule-based hard routing.

A shared trunk and one expert per velocity command axis (lin_vel_x,
lin_vel_y, ang_vel_z) run in parallel on the same observation. At every
forward pass a rule-based gate picks the expert whose command axis has
the largest absolute (normalized) value, and only that expert
contributes to the merged feature. A small head then maps the merged
feature to the action distribution input.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories
from tensordict import TensorDict

from mjlab.rl.config import RslRlModelCfg


@dataclass
class MoEActorModelCfg(RslRlModelCfg):
  """Config for the rule-routed MoE actor.

  Routing modes (chosen automatically by num_experts vs cmd dim):
    - num_experts == cmd_dim: each expert specializes in one cmd axis
      (selected by argmax(|cmd|)).
    - num_experts == cmd_dim + 1: extra last expert is the "standing"
      expert, selected when max(|cmd|) < standing_threshold.
  """

  shared_dims: tuple[int, ...] = (256, 128)
  expert_dims: tuple[int, ...] = (192, 128)
  head_dims: tuple[int, ...] = (128,)
  num_experts: int = 3
  merge: str = "concat"
  cmd_start: int = -3
  cmd_end: int | None = None
  standing_threshold: float = 0.0
  class_name: str = "mjlab.rl.moe_model:MoEActorModel"


class MoEActorModel(nn.Module):
  """MoE actor with rule-based hard routing over velocity-axis experts."""

  is_recurrent: bool = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict[str, Any] | None = None,
    shared_dims: tuple[int, ...] = (256, 128),
    expert_dims: tuple[int, ...] = (192, 128),
    head_dims: tuple[int, ...] = (128,),
    num_experts: int = 3,
    merge: str = "concat",
    cmd_start: int = -3,
    cmd_end: int | None = None,
    standing_threshold: float = 0.0,
  ) -> None:
    del hidden_dims
    super().__init__()

    self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)

    self.obs_normalization = obs_normalization
    if obs_normalization:
      self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
    else:
      self.obs_normalizer = nn.Identity()

    if distribution_cfg is not None:
      dist_class: type[Distribution] = resolve_callable(  # type: ignore[assignment]
        distribution_cfg.pop("class_name")
      )
      self.distribution: Distribution | None = dist_class(
        output_dim, **distribution_cfg
      )
      head_output_dim = self.distribution.input_dim
    else:
      self.distribution = None
      head_output_dim = output_dim

    if merge not in ("concat", "sum"):
      raise ValueError(f"merge must be 'concat' or 'sum', got {merge!r}")
    if merge == "sum" and shared_dims[-1] != expert_dims[-1]:
      raise ValueError(
        "merge='sum' requires shared_dims[-1] == expert_dims[-1] "
        f"(got {shared_dims[-1]} vs {expert_dims[-1]})"
      )

    self.num_experts = num_experts
    # Resolve to non-negative indices so export paths don't depend on negative
    # slicing semantics.
    self.cmd_start = cmd_start if cmd_start >= 0 else self.obs_dim + cmd_start
    self.cmd_end = cmd_end if cmd_end is not None else self.obs_dim
    self.merge = merge
    # Standing expert: if num_experts == cmd_dim + 1, the last expert is
    # activated when all cmd components are below standing_threshold.
    self.standing_threshold = float(standing_threshold)
    self._cmd_dim = self.cmd_end - self.cmd_start
    self._has_standing_expert = num_experts == self._cmd_dim + 1

    self.shared = MLP(self.obs_dim, shared_dims[-1], shared_dims[:-1], activation)
    self.experts = nn.ModuleList(
      [
        MLP(self.obs_dim, expert_dims[-1], expert_dims[:-1], activation)
        for _ in range(num_experts)
      ]
    )
    merged_dim = (
      shared_dims[-1] + expert_dims[-1] if merge == "concat" else shared_dims[-1]
    )
    self.head = MLP(merged_dim, head_output_dim, head_dims, activation)

    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.head)

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    obs_list = [obs[g] for g in self.obs_groups]
    latent = torch.cat(obs_list, dim=-1)  # type: ignore[arg-type]
    return self.obs_normalizer(latent)

  def _select_expert(self, latent: torch.Tensor) -> torch.Tensor:
    cmd = latent[..., self.cmd_start : self.cmd_end]
    cmd_idx = cmd.abs().argmax(dim=-1)
    if self._has_standing_expert and self.standing_threshold > 0.0:
      is_standing = cmd.abs().max(dim=-1).values < self.standing_threshold
      standing_idx = torch.full_like(cmd_idx, self._cmd_dim)
      return torch.where(is_standing, standing_idx, cmd_idx)
    return cmd_idx

  def _features(self, latent: torch.Tensor) -> torch.Tensor:
    shared_out = self.shared(latent)
    expert_outs = torch.stack([e(latent) for e in self.experts], dim=1)
    expert_idx = self._select_expert(latent)
    mask = F.one_hot(expert_idx, num_classes=self.num_experts).to(expert_outs.dtype)
    expert_out = (mask.unsqueeze(-1) * expert_outs).sum(dim=1)
    if self.merge == "concat":
      merged = torch.cat([shared_out, expert_out], dim=-1)
    else:
      merged = shared_out + expert_out
    return self.head(merged)

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    if masks is not None and not self.is_recurrent:
      obs = unpad_trajectories(obs, masks)  # type: ignore[assignment]
    latent = self.get_latent(obs, masks, hidden_state)
    head_out = self._features(latent)
    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(head_out)
        return self.distribution.sample()
      return self.distribution.deterministic_output(head_out)
    return head_out

  def reset(
    self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None
  ) -> None:
    pass

  def get_hidden_state(self) -> HiddenState:
    return None

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    pass

  @property
  def output_mean(self) -> torch.Tensor:
    return self.distribution.mean  # type: ignore[union-attr]

  @property
  def output_std(self) -> torch.Tensor:
    return self.distribution.std  # type: ignore[union-attr]

  @property
  def output_entropy(self) -> torch.Tensor:
    return self.distribution.entropy  # type: ignore[union-attr]

  @property
  def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
    return self.distribution.params  # type: ignore[union-attr]

  def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    return self.distribution.log_prob(outputs)  # type: ignore[union-attr]

  def get_kl_divergence(
    self,
    old_params: tuple[torch.Tensor, ...],
    new_params: tuple[torch.Tensor, ...],
  ) -> torch.Tensor:
    return self.distribution.kl_divergence(old_params, new_params)  # type: ignore[union-attr]

  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      obs_list = [obs[g] for g in self.obs_groups]
      flat = torch.cat(obs_list, dim=-1)  # type: ignore[arg-type]
      self.obs_normalizer.update(flat)  # type: ignore[operator]

  def as_jit(self) -> nn.Module:
    return _TorchMoE(self)

  def as_onnx(self, verbose: bool) -> nn.Module:
    return _OnnxMoE(self, verbose)

  def _get_obs_dim(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
  ) -> tuple[list[str], int]:
    active = obs_groups[obs_set]
    total = 0
    for g in active:
      if len(obs[g].shape) != 2:
        raise ValueError(
          f"MoEActorModel only supports 1D observations, got shape "
          f"{obs[g].shape} for '{g}'."
        )
      total += obs[g].shape[-1]
    return active, total


class _ExportableMoE(nn.Module):
  """Pre-concatenated, deterministic forward for JIT/ONNX export."""

  is_recurrent: bool = False

  def __init__(self, model: MoEActorModel) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.shared = copy.deepcopy(model.shared)
    self.experts = nn.ModuleList(copy.deepcopy(e) for e in model.experts)
    self.head = copy.deepcopy(model.head)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.input_size = model.obs_dim
    self.num_experts = model.num_experts
    self.cmd_start = model.cmd_start
    self.cmd_end = model.cmd_end
    self.merge = model.merge
    self.standing_threshold = model.standing_threshold
    self.cmd_dim = model._cmd_dim
    self.has_standing_expert = model._has_standing_expert

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    latent = self.obs_normalizer(x)
    shared_out = self.shared(latent)
    expert_outs = torch.stack([e(latent) for e in self.experts], dim=1)
    cmd = latent[..., self.cmd_start : self.cmd_end]
    cmd_idx = cmd.abs().argmax(dim=-1)
    if self.has_standing_expert and self.standing_threshold > 0.0:
      is_standing = cmd.abs().max(dim=-1).values < self.standing_threshold
      standing_idx = torch.full_like(cmd_idx, self.cmd_dim)
      expert_idx = torch.where(is_standing, standing_idx, cmd_idx)
    else:
      expert_idx = cmd_idx
    mask = F.one_hot(expert_idx, num_classes=self.num_experts).to(expert_outs.dtype)
    expert_out = (mask.unsqueeze(-1) * expert_outs).sum(dim=1)
    merged = (
      torch.cat([shared_out, expert_out], dim=-1)
      if self.merge == "concat"
      else shared_out + expert_out
    )
    return self.deterministic_output(self.head(merged))


class _TorchMoE(_ExportableMoE):
  @torch.jit.export
  def reset(self) -> None:
    pass


class _OnnxMoE(_ExportableMoE):
  def __init__(self, model: MoEActorModel, verbose: bool) -> None:
    super().__init__(model)
    self.verbose = verbose

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
