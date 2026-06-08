"""Utilities for reading actuator torques as physical torque channels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator import BuiltinPdActuator
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(frozen=True)
class TorqueChannelSelection:
  """Mapping from selected actuator controls to physical torque channels.

  ``BuiltinPdActuator`` is represented by two MuJoCo actuator elements per joint
  (position + velocity), but downstream rewards/observations should see one
  physical joint-torque channel. Regular actuators keep their one-control/one-channel
  layout.
  """

  num_channels: int
  selected_id_to_col: dict[int, int]
  regular_local_ctrl_ids: torch.Tensor
  regular_global_ctrl_ids: torch.Tensor
  regular_cols: torch.Tensor
  builtin_pd_joint_ids: torch.Tensor
  builtin_pd_global_joint_ids: torch.Tensor
  builtin_pd_cols: torch.Tensor


def _selected_local_ctrl_ids(
  asset: Entity,
  asset_cfg: SceneEntityCfg,
  device: str | torch.device,
) -> torch.Tensor:
  if isinstance(asset_cfg.actuator_ids, slice):
    num_actuators = asset.data.actuator_force.shape[1]
    return torch.arange(num_actuators, device=device, dtype=torch.long)[
      asset_cfg.actuator_ids
    ]
  return torch.as_tensor(asset_cfg.actuator_ids, device=device, dtype=torch.long)


def _empty_long(device: str | torch.device) -> torch.Tensor:
  return torch.empty(0, device=device, dtype=torch.long)


def resolve_torque_channels(
  asset: Entity,
  asset_cfg: SceneEntityCfg,
  device: str | torch.device,
) -> TorqueChannelSelection:
  """Resolve selected controls into physical torque channels.

  For a ``BuiltinPdActuator`` target, either the position or velocity MuJoCo
  element selects the same joint-space channel. If both halves are selected, the
  channel appears once, at the first selected half's position.
  """

  selected_ids = _selected_local_ctrl_ids(asset, asset_cfg, device)

  actuators = getattr(asset, "actuators", ())
  if not isinstance(actuators, (list, tuple)):
    actuators = ()

  pd_ctrl_to_joint: dict[int, tuple[tuple[int, int], int, int]] = {}
  for actuator_index, actuator in enumerate(actuators):
    if not isinstance(actuator, BuiltinPdActuator):
      continue
    n = actuator.num_targets
    target_ids = actuator.target_ids.to(device=device, dtype=torch.long)
    global_joint_ids = asset.indexing.joint_ids[target_ids].to(
      device=device, dtype=torch.long
    )
    ctrl_ids = actuator.ctrl_ids.to(device=device, dtype=torch.long)
    for ctrl_col, ctrl_id in enumerate(ctrl_ids.tolist()):
      target_col = ctrl_col if ctrl_col < n else ctrl_col - n
      key = (actuator_index, target_col)
      pd_ctrl_to_joint[ctrl_id] = (
        key,
        int(target_ids[target_col].item()),
        int(global_joint_ids[target_col].item()),
      )

  selected_id_to_col: dict[int, int] = {}
  regular_local_ctrl_ids: list[int] = []
  regular_global_ctrl_ids: list[int] = []
  regular_cols: list[int] = []
  builtin_pd_joint_ids: list[int] = []
  builtin_pd_global_joint_ids: list[int] = []
  builtin_pd_cols: list[int] = []
  seen_pd_targets: set[tuple[int, int]] = set()

  global_ctrl_ids = getattr(getattr(asset.data, "indexing", None), "ctrl_ids", None)
  if not isinstance(global_ctrl_ids, torch.Tensor):
    global_ctrl_ids = torch.arange(
      asset.data.actuator_force.shape[1], device=device, dtype=torch.long
    )
  else:
    global_ctrl_ids = global_ctrl_ids.to(device=device, dtype=torch.long)
  out_col = 0
  for local_ctrl_id in selected_ids.tolist():
    pd_entry = pd_ctrl_to_joint.get(local_ctrl_id)
    if pd_entry is not None:
      key, joint_id, global_joint_id = pd_entry
      if key in seen_pd_targets:
        continue
      seen_pd_targets.add(key)
      builtin_pd_joint_ids.append(joint_id)
      builtin_pd_global_joint_ids.append(global_joint_id)
      builtin_pd_cols.append(out_col)
      out_col += 1
      continue

    selected_id_to_col[local_ctrl_id] = out_col
    regular_local_ctrl_ids.append(local_ctrl_id)
    regular_global_ctrl_ids.append(int(global_ctrl_ids[local_ctrl_id].item()))
    regular_cols.append(out_col)
    out_col += 1

  return TorqueChannelSelection(
    num_channels=out_col,
    selected_id_to_col=selected_id_to_col,
    regular_local_ctrl_ids=(
      torch.tensor(regular_local_ctrl_ids, device=device, dtype=torch.long)
      if regular_local_ctrl_ids
      else _empty_long(device)
    ),
    regular_global_ctrl_ids=(
      torch.tensor(regular_global_ctrl_ids, device=device, dtype=torch.long)
      if regular_global_ctrl_ids
      else _empty_long(device)
    ),
    regular_cols=(
      torch.tensor(regular_cols, device=device, dtype=torch.long)
      if regular_cols
      else _empty_long(device)
    ),
    builtin_pd_joint_ids=(
      torch.tensor(builtin_pd_joint_ids, device=device, dtype=torch.long)
      if builtin_pd_joint_ids
      else _empty_long(device)
    ),
    builtin_pd_global_joint_ids=(
      torch.tensor(builtin_pd_global_joint_ids, device=device, dtype=torch.long)
      if builtin_pd_global_joint_ids
      else _empty_long(device)
    ),
    builtin_pd_cols=(
      torch.tensor(builtin_pd_cols, device=device, dtype=torch.long)
      if builtin_pd_cols
      else _empty_long(device)
    ),
  )


def _upper_limit(
  field: torch.Tensor,
  ids: torch.Tensor,
  *,
  num_envs: int,
) -> torch.Tensor:
  if ids.numel() == 0:
    return torch.empty(num_envs, 0, device=ids.device, dtype=field.dtype)
  if field.ndim == 3:
    return torch.abs(field[:, ids, 1])
  if field.ndim == 2:
    return torch.abs(field[ids, 1]).unsqueeze(0).expand(num_envs, -1)
  raise ValueError(
    f"Expected range field to have 2 or 3 dims, got shape {tuple(field.shape)}."
  )


def _num_envs(env: ManagerBasedRlEnv, asset: Entity) -> int:
  num_envs = getattr(env, "num_envs", None)
  if isinstance(num_envs, int):
    return num_envs
  return int(asset.data.actuator_force.shape[0])


def read_torque(
  env: ManagerBasedRlEnv,
  asset: Entity,
  selection: TorqueChannelSelection,
) -> torch.Tensor:
  """Read physical torque for a selection."""

  device = asset.data.actuator_force.device
  dtype = asset.data.actuator_force.dtype
  num_envs = _num_envs(env, asset)
  torque = torch.zeros((num_envs, selection.num_channels), device=device, dtype=dtype)

  if selection.regular_cols.numel() > 0:
    torque[:, selection.regular_cols] = asset.data.actuator_force[
      :, selection.regular_local_ctrl_ids
    ]

  if selection.builtin_pd_cols.numel() > 0:
    torque[:, selection.builtin_pd_cols] = asset.data.qfrc_actuator[
      :, selection.builtin_pd_joint_ids
    ]

  return torque


def read_torque_and_limits(
  env: ManagerBasedRlEnv,
  asset: Entity,
  selection: TorqueChannelSelection,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Read physical torque and matching effort limit for a selection."""

  device = asset.data.actuator_force.device
  dtype = asset.data.actuator_force.dtype
  num_envs = _num_envs(env, asset)
  torque = read_torque(env, asset, selection)
  effort_limit = torch.zeros_like(torque)

  if selection.regular_cols.numel() > 0:
    limit = _upper_limit(
      env.sim.model.actuator_forcerange,
      selection.regular_global_ctrl_ids,
      num_envs=num_envs,
    ).to(device=device, dtype=dtype)
    effort_limit[:, selection.regular_cols] = limit

  if selection.builtin_pd_cols.numel() > 0:
    limit = _upper_limit(
      env.sim.model.jnt_actfrcrange,
      selection.builtin_pd_global_joint_ids,
      num_envs=num_envs,
    ).to(device=device, dtype=dtype)
    effort_limit[:, selection.builtin_pd_cols] = limit

  return torque, effort_limit
