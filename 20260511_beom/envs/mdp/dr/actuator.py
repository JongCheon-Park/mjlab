"""Domain randomization functions for actuators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.actuator import (
  BuiltinPositionActuator,
  FourbarPdActuator,
  IdealPdActuator,
)
from mjlab.actuator.actuator import Actuator, TransmissionType
from mjlab.actuator.xml_actuator import XmlActuator
from mjlab.entity import Entity
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ._core import _DEFAULT_ASSET_CFG
from ._types import resolve_distribution

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _resolve_selected_actuators(
  asset: Entity,
  asset_cfg: SceneEntityCfg,
) -> list[Actuator]:
  """Return actuators selected by ``asset_cfg.actuator_ids`` as a list."""
  if isinstance(asset_cfg.actuator_ids, list):
    return [asset.actuators[i] for i in asset_cfg.actuator_ids]

  selected = asset.actuators[asset_cfg.actuator_ids]
  if isinstance(selected, list):
    return selected
  return [selected]


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def pd_gains(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform"] = "uniform",
  operation: Literal["scale", "abs"] = "scale",
) -> None:
  """Randomize PD stiffness and damping gains.

  Args:
    env: The environment.
    env_ids: Environment IDs to randomize. If None, randomizes all.
    kp_range: (min, max) for proportional gain randomization.
    kd_range: (min, max) for derivative gain randomization.
    asset_cfg: Asset configuration specifying which entity and actuators.
    distribution: Distribution type ("uniform" or "log_uniform").
    operation: "scale" multiplies default gains by sampled values, "abs" sets
      absolute values.
  """
  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  actuators = _resolve_selected_actuators(asset, asset_cfg)

  for actuator in actuators:
    ctrl_ids = actuator.global_ctrl_ids

    dist = resolve_distribution(distribution)
    kp_samples = dist.sample(
      torch.tensor(kp_range[0], device=env.device),
      torch.tensor(kp_range[1], device=env.device),
      (len(env_ids), len(ctrl_ids)),
      env.device,
    )
    kd_samples = dist.sample(
      torch.tensor(kd_range[0], device=env.device),
      torch.tensor(kd_range[1], device=env.device),
      (len(env_ids), len(ctrl_ids)),
      env.device,
    )

    if isinstance(actuator, BuiltinPositionActuator) or (
      isinstance(actuator, XmlActuator) and actuator.command_field == "position"
    ):
      if operation == "scale":
        default_gainprm = env.sim.get_default_field("actuator_gainprm")
        default_biasprm = env.sim.get_default_field("actuator_biasprm")
        env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = (
          default_gainprm[ctrl_ids, 0] * kp_samples
        )
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = (
          default_biasprm[ctrl_ids, 1] * kp_samples
        )
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = (
          default_biasprm[ctrl_ids, 2] * kd_samples
        )
      elif operation == "abs":
        env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = kp_samples
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = -kp_samples
        env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = -kd_samples

    elif isinstance(actuator, (IdealPdActuator, FourbarPdActuator)):
      assert actuator.stiffness is not None
      assert actuator.damping is not None
      if operation == "scale":
        assert actuator.default_stiffness is not None
        assert actuator.default_damping is not None
        actuator.set_gains(
          env_ids,
          kp=actuator.default_stiffness[env_ids] * kp_samples,
          kd=actuator.default_damping[env_ids] * kd_samples,
        )
      elif operation == "abs":
        actuator.set_gains(env_ids, kp=kp_samples, kd=kd_samples)

    else:
      raise TypeError(
        f"pd_gains only supports BuiltinPositionActuator, "
        f"XmlActuator (position), IdealPdActuator, and FourbarPdActuator "
        f"(with optional delay), "
        f"got {type(actuator).__name__}"
      )


def fourbar_motor_armature(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ratio_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Randomize motor armature (I_m) for FourbarPdActuator.

  Scales the per-env motor armature by a uniform factor. This propagates
  to D_o(q), h_0, and M_o computations automatically.

  Args:
    env: The environment.
    env_ids: Environment IDs to randomize. If None, randomizes all.
    ratio_range: (min, max) scale factors applied to default motor armature.
    asset_cfg: Asset configuration specifying which entity and actuators.
  """
  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  actuators = _resolve_selected_actuators(asset, asset_cfg)

  for actuator in actuators:
    if not isinstance(actuator, FourbarPdActuator):
      continue
    assert actuator.motor_armature is not None
    assert actuator.default_motor_armature is not None
    scale = torch.empty(len(env_ids), device=env.device).uniform_(
      ratio_range[0], ratio_range[1]
    )
    actuator.set_motor_armature(
      env_ids,
      actuator.default_motor_armature[env_ids] * scale,
    )


@requires_model_fields("actuator_forcerange")
def effort_limits(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  effort_limit_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform"] = "uniform",
  operation: Literal["scale", "abs"] = "scale",
) -> None:
  """Randomize actuator effort limits.

  Args:
    env: The environment.
    env_ids: Environment IDs to randomize. If None, randomizes all.
    effort_limit_range: (min, max) for effort limit randomization.
    asset_cfg: Asset configuration specifying which entity and actuators.
    distribution: Distribution type ("uniform" or "log_uniform").
    operation: "scale" multiplies default limits, "abs" sets absolute values.
  """
  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  actuators = _resolve_selected_actuators(asset, asset_cfg)

  for actuator in actuators:
    ctrl_ids = actuator.global_ctrl_ids
    num_actuators = len(ctrl_ids)

    dist = resolve_distribution(distribution)
    effort_samples = dist.sample(
      torch.tensor(effort_limit_range[0], device=env.device),
      torch.tensor(effort_limit_range[1], device=env.device),
      (len(env_ids), num_actuators),
      env.device,
    )

    if isinstance(actuator, BuiltinPositionActuator) or (
      isinstance(actuator, XmlActuator) and actuator.command_field == "position"
    ):
      if operation == "scale":
        default_forcerange = env.sim.get_default_field("actuator_forcerange")
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 0] = (
          default_forcerange[ctrl_ids, 0] * effort_samples
        )
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 1] = (
          default_forcerange[ctrl_ids, 1] * effort_samples
        )
      elif operation == "abs":
        env.sim.model.actuator_forcerange[
          env_ids[:, None], ctrl_ids, 0
        ] = -effort_samples
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 1] = (
          effort_samples
        )

    elif isinstance(actuator, (IdealPdActuator, FourbarPdActuator)):
      assert actuator.force_limit is not None
      if operation == "scale":
        assert actuator.default_force_limit is not None
        actuator.set_effort_limit(
          env_ids,
          effort_limit=actuator.default_force_limit[env_ids] * effort_samples,
        )
      elif operation == "abs":
        actuator.set_effort_limit(env_ids, effort_limit=effort_samples)

    else:
      raise TypeError(
        f"effort_limits only supports BuiltinPositionActuator, "
        f"XmlActuator (position), IdealPdActuator, and FourbarPdActuator "
        f"(with optional delay), "
        f"got {type(actuator).__name__}"
      )


class actuator_rfi:
  """Inject random joint torques at every simulation substep.

  This term is intended for ``mode="startup"``. On first call it monkeypatches
  ``env.sim.step`` and injects i.i.d. random torques into ``qfrc_applied`` right
  before every physics step.

  The term is scoped to selected actuators and only supports JOINT transmission.
  """

  @dataclass
  class _JointActuatorRfiUnit:
    ctrl_ids: torch.Tensor
    dof_ids: torch.Tensor
    dof_cols: torch.Tensor
    gear: torch.Tensor | None = None

    def compute_joint_injection(self, term: actuator_rfi) -> torch.Tensor:
      native_low, native_high = term._get_actuator_forcerange(self.ctrl_ids)
      native_base = term._get_actuator_force(self.ctrl_ids)
      sample = term._sample_injection(native_low, native_high)
      sample = term._clamp_sample_to_limits(
        sample,
        low=native_low,
        high=native_high,
        base=native_base,
      )
      if self.gear is None or self.gear.shape != sample.shape:
        self.gear = term._get_actuator_gear(self.ctrl_ids, dtype=sample.dtype)
      sample.mul_(self.gear)
      return sample

  @dataclass
  class _FourbarActuatorRfiUnit:
    actuator: FourbarPdActuator
    dof_ids: torch.Tensor
    dof_cols: torch.Tensor

    def compute_joint_injection(self, term: actuator_rfi) -> torch.Tensor:
      assert self.actuator.force_limit is not None
      native_high = self.actuator.force_limit.to(device=term._device, dtype=torch.float)
      if native_high.shape[0] == 1 and term._num_envs > 1:
        native_high = native_high.expand(term._num_envs, -1)
      native_low = -native_high

      baseline = self.actuator.last_motor_torque
      if baseline is None:
        native_base = torch.zeros_like(native_high)
      else:
        native_base = baseline.to(device=term._device, dtype=torch.float)
        if native_base.shape[0] == 1 and term._num_envs > 1:
          native_base = native_base.expand(term._num_envs, -1)

      sample = term._sample_injection(native_low, native_high)
      sample = term._clamp_sample_to_limits(
        sample,
        low=native_low,
        high=native_high,
        base=native_base,
      )
      return self.actuator.project_motor_torque_to_joint(sample)

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    params = dict(cfg.params)

    self._env = env
    self._num_envs = env.num_envs
    self._device = env.device
    if "ratio_range" not in params:
      raise ValueError("actuator_rfi requires 'ratio_range' in params.")
    ratio_range = params.pop("ratio_range")
    asset_cfg = params.pop("asset_cfg", _DEFAULT_ASSET_CFG)
    if params:
      raise ValueError(
        f"Unexpected params for actuator_rfi: {sorted(params.keys())}. "
        "Supported params are 'ratio_range' and optional 'asset_cfg'."
      )

    self._ratio_low, self._ratio_high = self._resolve_ratio_range(ratio_range)
    self._asset_cfg = asset_cfg
    self._asset: Entity = env.scene[self._asset_cfg.name]
    self._dof_ids, self._units = self._resolve_actuator_units(
      self._asset, self._asset_cfg
    )
    self._last_injection = torch.zeros(
      (self._num_envs, len(self._dof_ids)),
      dtype=torch.float,
      device=self._device,
    )
    self._current_injection = torch.zeros_like(self._last_injection)
    self._delta_injection = torch.zeros_like(self._last_injection)
    cfg.params = {}

    self._installed = False
    self._orig_step: Callable[[], None] | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
  ) -> None:
    del env, env_ids
    if self._installed:
      return

    self._orig_step = self._env.sim.step

    def _wrapped_step() -> None:
      self._inject_once()
      assert self._orig_step is not None
      self._orig_step()

    self._env.sim.step = _wrapped_step
    self._installed = True

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._last_injection[env_ids] = 0.0
    self._current_injection[env_ids] = 0.0
    self._delta_injection[env_ids] = 0.0

  def _inject_once(self) -> None:
    if len(self._dof_ids) == 0:
      return

    sample = self._current_injection
    sample.zero_()
    for unit in self._units:
      sample.index_add_(1, unit.dof_cols, unit.compute_joint_injection(self))

    self._delta_injection.copy_(sample)
    self._delta_injection.sub_(self._last_injection)
    self._env.sim.data.qfrc_applied.index_add_(
      1,
      self._dof_ids,
      self._delta_injection,
    )
    self._last_injection.copy_(sample)

  def _sample_injection(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    ratio = torch.rand(
      low.shape,
      device=self._device,
      dtype=torch.float,
    )
    ratio = ratio * (self._ratio_high - self._ratio_low) + self._ratio_low
    max_effort = torch.maximum(high, -low)
    return ratio * max_effort

  @staticmethod
  def _resolve_ratio_range(
    ratio_range: tuple[float, float] | list[float] | None,
  ) -> tuple[float, float]:
    if ratio_range is None:
      raise ValueError("actuator_rfi requires 'ratio_range' in params.")
    if not isinstance(ratio_range, (tuple, list)) or len(ratio_range) != 2:
      raise ValueError(
        f"'ratio_range' must be a tuple/list of length 2, got {ratio_range!r}."
      )
    low = float(ratio_range[0])
    high = float(ratio_range[1])
    if low > high:
      raise ValueError(f"Invalid ratio_range: low ({low}) > high ({high}).")
    return low, high

  def _resolve_actuator_units(
    self, asset: Entity, asset_cfg: SceneEntityCfg
  ) -> tuple[
    torch.Tensor,
    list[_JointActuatorRfiUnit | _FourbarActuatorRfiUnit],
  ]:
    actuators = _resolve_selected_actuators(asset, asset_cfg)

    units: list[
      actuator_rfi._JointActuatorRfiUnit | actuator_rfi._FourbarActuatorRfiUnit
    ] = []
    all_dof_ids: set[int] = set()
    for actuator in actuators:
      if actuator.transmission_type != TransmissionType.JOINT:
        raise ValueError(
          "actuator_rfi supports only JOINT transmission actuators. "
          f"Got transmission_type={actuator.transmission_type}."
        )

      joint_ids = actuator.target_ids.to(device=self._device, dtype=torch.long)
      dof_ids = asset.indexing.joint_v_adr[joint_ids].to(dtype=torch.long)
      if len(torch.unique(dof_ids)) != len(dof_ids):
        raise ValueError(
          "actuator_rfi expects unique JOINT actuator targets per actuator group, "
          f"got duplicated dof ids {dof_ids.tolist()}."
        )

      if isinstance(actuator, FourbarPdActuator):
        unit = actuator_rfi._FourbarActuatorRfiUnit(
          actuator=actuator,
          dof_ids=dof_ids,
          dof_cols=torch.empty(0, dtype=torch.long, device=self._device),
        )
      else:
        ctrl_ids = actuator.global_ctrl_ids.to(device=self._device, dtype=torch.long)
        if len(dof_ids) != len(ctrl_ids):
          raise ValueError(
            "actuator_rfi expects one-to-one mapping between JOINT actuator targets "
            f"and actuator controls, got {len(dof_ids)} targets and {len(ctrl_ids)} controls."
          )
        unit = actuator_rfi._JointActuatorRfiUnit(
          ctrl_ids=ctrl_ids,
          dof_ids=dof_ids,
          dof_cols=torch.empty(0, dtype=torch.long, device=self._device),
        )

      units.append(unit)
      all_dof_ids.update(dof_ids.tolist())

    if not all_dof_ids:
      return torch.empty(0, dtype=torch.long, device=self._device), []

    sorted_dof_ids = sorted(all_dof_ids)
    dof_ids_tensor = torch.tensor(sorted_dof_ids, dtype=torch.long, device=self._device)
    dof_to_col = {dof_id: col for col, dof_id in enumerate(sorted_dof_ids)}
    for unit in units:
      unit.dof_cols = torch.tensor(
        [dof_to_col[dof_id] for dof_id in unit.dof_ids.tolist()],
        dtype=torch.long,
        device=self._device,
      )

    return dof_ids_tensor, units

  def _get_actuator_forcerange(
    self,
    ctrl_ids: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    forcerange = self._env.sim.model.actuator_forcerange
    if forcerange.ndim == 2:
      forcerange = forcerange.unsqueeze(0).expand(self._num_envs, -1, -1)
    elif forcerange.shape[0] == 1 and self._num_envs > 1:
      forcerange = forcerange.expand(self._num_envs, -1, -1)

    ctrl_range = forcerange[:, ctrl_ids, :]
    low = ctrl_range[:, :, 0].to(device=self._device, dtype=torch.float)
    high = ctrl_range[:, :, 1].to(device=self._device, dtype=torch.float)
    return low, high

  def _get_actuator_gear(
    self,
    ctrl_ids: torch.Tensor,
    *,
    dtype: torch.dtype,
  ) -> torch.Tensor:
    actuator_gear = getattr(self._env.sim.model, "actuator_gear", None)
    if actuator_gear is None:
      gear = torch.ones(
        (self._num_envs, len(ctrl_ids)),
        device=self._device,
        dtype=dtype,
      )
    elif actuator_gear.ndim == 2:
      gear = actuator_gear[ctrl_ids, 0].to(device=self._device, dtype=dtype)
      gear = gear.unsqueeze(0).expand(self._num_envs, -1)
    else:
      gear = actuator_gear[:, ctrl_ids, 0].to(device=self._device, dtype=dtype)
      if gear.shape[0] == 1 and self._num_envs > 1:
        gear = gear.expand(self._num_envs, -1)
    return gear

  def _get_actuator_force(self, ctrl_ids: torch.Tensor) -> torch.Tensor:
    actuator_force = getattr(self._env.sim.data, "actuator_force", None)
    if actuator_force is None:
      baseline_force = torch.zeros(
        (self._num_envs, len(ctrl_ids)),
        device=self._device,
        dtype=torch.float,
      )
    else:
      baseline_force = actuator_force.to(device=self._device, dtype=torch.float)
      if baseline_force.ndim == 1:
        baseline_force = baseline_force.unsqueeze(0).expand(self._num_envs, -1)
      baseline_force = baseline_force[:, ctrl_ids]

    return baseline_force

  def _clamp_sample_to_limits(
    self,
    sample: torch.Tensor,
    *,
    low: torch.Tensor,
    high: torch.Tensor,
    base: torch.Tensor,
  ) -> torch.Tensor:
    if sample.numel() == 0:
      return sample

    low = low - base
    high = high - base

    invalid = low > high
    if invalid.any():
      midpoint = 0.5 * (low + high)
      low = torch.where(invalid, midpoint, low)
      high = torch.where(invalid, midpoint, high)

    return torch.clamp(sample, min=low, max=high)


class actuator_motor_bandwidth:
  """Inject actuator-force corrections that emulate finite motor bandwidth.

  This term reuses the same step-hook and qfrc_applied delta pattern as
  :class:`actuator_rfi`, but replaces random injection with a first-order low-pass
  filtered version of the actuator's native force:

  ``qfrc_applied += filtered_force - raw_force``

  For regular joint actuators, ``raw_force`` is read from ``data.actuator_force``.
  Since this hook runs before the native MuJoCo step, built-in/implicit actuators
  see the previous physics step's actuator force; this intentionally trades exact
  same-step LPF equivalence for compatibility with implicit actuators. For
  :class:`FourbarPdActuator`, the filter is applied in motor-torque space via
  ``last_motor_torque`` before projecting the correction back to joint space.
  """

  @dataclass
  class _JointActuatorBandwidthUnit:
    ctrl_ids: torch.Tensor
    dof_ids: torch.Tensor
    dof_cols: torch.Tensor
    filtered_force: torch.Tensor | None = None
    correction: torch.Tensor | None = None
    gear: torch.Tensor | None = None
    alpha: torch.Tensor | None = None

    def compute_joint_injection(self, term: actuator_motor_bandwidth) -> torch.Tensor:
      raw_force = term._get_actuator_force(self.ctrl_ids)
      correction = self._compute_correction(raw_force, term._alpha_for(self, raw_force))
      if self.gear is None or self.gear.shape != correction.shape:
        self.gear = term._get_actuator_gear(self.ctrl_ids, dtype=correction.dtype)
      correction.mul_(self.gear)
      return correction

    def reset(
      self, term: actuator_motor_bandwidth, env_ids: torch.Tensor | slice
    ) -> None:
      if self.filtered_force is not None:
        self.filtered_force[env_ids] = 0.0
      term._resample_alpha(self, env_ids)

    def _compute_correction(
      self,
      raw_force: torch.Tensor,
      alpha: float | torch.Tensor,
    ) -> torch.Tensor:
      if self.filtered_force is None:
        self.filtered_force = torch.zeros_like(raw_force)
      if self.correction is None:
        self.correction = torch.empty_like(raw_force)

      self.correction.copy_(raw_force)
      self.correction.sub_(self.filtered_force)
      self.correction.mul_(alpha)
      self.filtered_force.add_(self.correction)
      self.correction.copy_(self.filtered_force)
      self.correction.sub_(raw_force)
      return self.correction

  @dataclass
  class _FourbarActuatorBandwidthUnit:
    actuator: FourbarPdActuator
    dof_ids: torch.Tensor
    dof_cols: torch.Tensor
    filtered_motor_torque: torch.Tensor | None = None
    correction: torch.Tensor | None = None
    alpha: torch.Tensor | None = None

    def compute_joint_injection(self, term: actuator_motor_bandwidth) -> torch.Tensor:
      raw_torque = self.actuator.last_motor_torque
      if raw_torque is None:
        force_limit = self.actuator.force_limit
        if force_limit is None:
          return torch.zeros(
            (term._num_envs, len(self.dof_ids)),
            device=term._device,
            dtype=torch.float,
          )
        raw_torque = torch.zeros_like(force_limit)
      raw_torque = raw_torque.to(device=term._device, dtype=torch.float)
      if raw_torque.shape[0] == 1 and term._num_envs > 1:
        raw_torque = raw_torque.expand(term._num_envs, -1)

      correction = self._compute_correction(
        raw_torque, term._alpha_for(self, raw_torque)
      )
      return self.actuator.project_motor_torque_to_joint(correction)

    def reset(
      self, term: actuator_motor_bandwidth, env_ids: torch.Tensor | slice
    ) -> None:
      if self.filtered_motor_torque is not None:
        self.filtered_motor_torque[env_ids] = 0.0
      term._resample_alpha(self, env_ids)

    def _compute_correction(
      self,
      raw_torque: torch.Tensor,
      alpha: float | torch.Tensor,
    ) -> torch.Tensor:
      if self.filtered_motor_torque is None:
        self.filtered_motor_torque = torch.zeros_like(raw_torque)
      if self.correction is None:
        self.correction = torch.empty_like(raw_torque)

      self.correction.copy_(raw_torque)
      self.correction.sub_(self.filtered_motor_torque)
      self.correction.mul_(alpha)
      self.filtered_motor_torque.add_(self.correction)
      self.correction.copy_(self.filtered_motor_torque)
      self.correction.sub_(raw_torque)
      return self.correction

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    params = dict(cfg.params)

    self._env = env
    self._num_envs = env.num_envs
    self._device = env.device
    self._asset_cfg = params.pop("asset_cfg", _DEFAULT_ASSET_CFG)
    self._enabled = bool(params.pop("enabled", True))
    time_constant_s = params.pop("time_constant_s", None)
    cutoff_hz = params.pop("cutoff_hz", None)
    time_constant_s_range = params.pop("time_constant_s_range", None)
    cutoff_hz_range = params.pop("cutoff_hz_range", None)
    if params:
      raise ValueError(
        f"Unexpected params for actuator_motor_bandwidth: {sorted(params.keys())}. "
        "Supported params are 'time_constant_s', 'cutoff_hz', "
        "'time_constant_s_range', 'cutoff_hz_range', and optional 'asset_cfg' "
        "or 'enabled'."
      )
    bandwidth_params = (
      time_constant_s,
      cutoff_hz,
      time_constant_s_range,
      cutoff_hz_range,
    )
    bandwidth_param_count = sum(value is not None for value in bandwidth_params)
    if bandwidth_param_count > 1:
      raise ValueError(
        "actuator_motor_bandwidth requires exactly one of 'time_constant_s', "
        "'cutoff_hz', 'time_constant_s_range', or 'cutoff_hz_range'."
      )
    if self._enabled and bandwidth_param_count == 0:
      raise ValueError(
        "actuator_motor_bandwidth requires a bandwidth parameter when enabled. "
        "Set enabled=False to install the hook before a curriculum supplies one."
      )

    self._dt = self._resolve_dt(env)
    self._time_constant_s_range = None
    self._cutoff_hz_range = None
    self._alpha = None
    if bandwidth_param_count == 0:
      pass
    elif time_constant_s_range is not None:
      self._time_constant_s_range = self._resolve_range(
        "time_constant_s_range",
        time_constant_s_range,
        lower_bound=0.0,
        strict_lower=False,
      )
      self._alpha = None
    elif cutoff_hz_range is not None:
      self._cutoff_hz_range = self._resolve_range(
        "cutoff_hz_range",
        cutoff_hz_range,
        lower_bound=0.0,
        strict_lower=True,
      )
      self._alpha = None
    else:
      time_constant = self._resolve_time_constant(
        time_constant_s=time_constant_s,
        cutoff_hz=cutoff_hz,
      )
      self._alpha = self._alpha_from_time_constant(time_constant)

    self._asset: Entity = env.scene[self._asset_cfg.name]
    self._dof_ids, self._units = self._resolve_actuator_units(
      self._asset, self._asset_cfg
    )
    self._last_injection = torch.zeros(
      (self._num_envs, len(self._dof_ids)),
      dtype=torch.float,
      device=self._device,
    )
    self._current_injection = torch.zeros_like(self._last_injection)
    self._delta_injection = torch.zeros_like(self._last_injection)
    self._has_active_injection = False
    # Class-based event terms are invoked later with ``term_cfg.params`` as
    # kwargs. This term consumes params at construction, so clear them after
    # caching to match actuator_rfi and avoid duplicate __call__ kwargs.
    cfg.params = {}

    self._installed = False
    self._orig_step: Callable[[], None] | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
  ) -> None:
    del env, env_ids
    if self._installed:
      return

    self._orig_step = self._env.sim.step

    def _wrapped_step() -> None:
      self._inject_once()
      assert self._orig_step is not None
      self._orig_step()

    self._env.sim.step = _wrapped_step
    self._installed = True

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._last_injection[env_ids] = 0.0
    self._current_injection[env_ids] = 0.0
    self._delta_injection[env_ids] = 0.0
    for unit in self._units:
      unit.reset(self, env_ids)

  @property
  def enabled(self) -> bool:
    """Whether this bandwidth correction currently injects qfrc corrections."""
    return self._enabled

  @property
  def cutoff_hz_range(self) -> tuple[float, float] | None:
    """Current sampled-cutoff range, when configured in cutoff-Hz mode."""
    return self._cutoff_hz_range

  def set_enabled(self, enabled: bool) -> None:
    """Enable or disable qfrc correction without removing the installed hook."""
    enabled = bool(enabled)
    if enabled and not self._has_bandwidth_config():
      raise ValueError(
        "Cannot enable actuator_motor_bandwidth without a bandwidth "
        "configuration. Set cutoff_hz_range, time_constant_s_range, cutoff_hz, "
        "or time_constant_s first."
      )
    self._enabled = enabled

  def set_cutoff_hz_range(
    self,
    cutoff_hz_range: tuple[float, float],
    *,
    invalidate_alpha: bool = True,
  ) -> None:
    """Switch the active bandwidth randomization to a cutoff-Hz range."""
    self._cutoff_hz_range = self._resolve_range(
      "cutoff_hz_range",
      cutoff_hz_range,
      lower_bound=0.0,
      strict_lower=True,
    )
    self._time_constant_s_range = None
    self._alpha = None
    if invalidate_alpha:
      self._invalidate_unit_alpha()

  def _inject_once(self) -> None:
    if len(self._dof_ids) == 0:
      return

    if not self._enabled:
      self._clear_active_injection()
      return

    sample = self._current_injection
    sample.zero_()
    for unit in self._units:
      sample.index_add_(1, unit.dof_cols, unit.compute_joint_injection(self))

    self._delta_injection.copy_(sample)
    self._delta_injection.sub_(self._last_injection)
    self._env.sim.data.qfrc_applied.index_add_(
      1,
      self._dof_ids,
      self._delta_injection,
    )
    self._last_injection.copy_(sample)
    self._has_active_injection = True

  def _clear_active_injection(self) -> None:
    if not self._has_active_injection:
      return
    self._delta_injection.copy_(self._last_injection)
    self._delta_injection.neg_()
    self._env.sim.data.qfrc_applied.index_add_(
      1,
      self._dof_ids,
      self._delta_injection,
    )
    self._last_injection.zero_()
    self._current_injection.zero_()
    self._delta_injection.zero_()
    self._has_active_injection = False

  def _invalidate_unit_alpha(self) -> None:
    for unit in self._units:
      unit.alpha = None

  def _has_bandwidth_config(self) -> bool:
    return (
      self._alpha is not None
      or self._time_constant_s_range is not None
      or self._cutoff_hz_range is not None
    )

  @staticmethod
  def _resolve_dt(env: ManagerBasedRlEnv) -> float:
    dt = getattr(env, "physics_dt", None)
    if dt is None:
      sim = getattr(env, "sim", None)
      dt = getattr(sim, "dt", None)
    if dt is None:
      raise ValueError(
        "actuator_motor_bandwidth requires env.physics_dt or env.sim.dt."
      )
    dt = float(dt)
    if dt <= 0.0:
      raise ValueError(f"Simulation dt must be positive, got {dt}.")
    return dt

  @staticmethod
  def _resolve_time_constant(
    *,
    time_constant_s: float | None,
    cutoff_hz: float | None,
  ) -> float:
    if cutoff_hz is not None:
      cutoff_hz = float(cutoff_hz)
      if cutoff_hz <= 0.0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}.")
      return 1.0 / (2.0 * torch.pi * cutoff_hz)

    assert time_constant_s is not None
    time_constant_s = float(time_constant_s)
    if time_constant_s < 0.0:
      raise ValueError(f"time_constant_s must be non-negative, got {time_constant_s}.")
    return time_constant_s

  @staticmethod
  def _resolve_range(
    name: str,
    value: tuple[float, float] | list[float],
    *,
    lower_bound: float,
    strict_lower: bool,
  ) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
      raise ValueError(f"'{name}' must be a tuple/list of length 2, got {value!r}.")
    low = float(value[0])
    high = float(value[1])
    invalid_low = low <= lower_bound if strict_lower else low < lower_bound
    if invalid_low:
      comparator = ">" if strict_lower else ">="
      raise ValueError(f"{name}[0] must be {comparator} {lower_bound}, got {low}.")
    if low > high:
      raise ValueError(f"Invalid {name}: low ({low}) > high ({high}).")
    return low, high

  def _alpha_from_time_constant(
    self,
    time_constant_s: float | torch.Tensor,
  ) -> float | torch.Tensor:
    return self._dt / (time_constant_s + self._dt + 1e-8)

  def _alpha_for(
    self,
    unit: _JointActuatorBandwidthUnit | _FourbarActuatorBandwidthUnit,
    reference: torch.Tensor,
  ) -> float | torch.Tensor:
    if self._alpha is not None:
      return self._alpha

    if not self._has_bandwidth_config():
      raise ValueError(
        "actuator_motor_bandwidth is enabled without a bandwidth configuration."
      )
    if unit.alpha is None or unit.alpha.shape != reference.shape:
      unit.alpha = self._sample_alpha(
        reference.shape, reference.device, reference.dtype
      )
    return unit.alpha

  def _resample_alpha(
    self,
    unit: _JointActuatorBandwidthUnit | _FourbarActuatorBandwidthUnit,
    env_ids: torch.Tensor | slice,
  ) -> None:
    if self._alpha is not None or unit.alpha is None:
      return
    unit.alpha[env_ids] = self._sample_alpha(
      unit.alpha[env_ids].shape,
      unit.alpha.device,
      unit.alpha.dtype,
    )

  def _sample_alpha(
    self,
    shape: torch.Size | tuple[int, ...],
    device: torch.device | str,
    dtype: torch.dtype,
  ) -> torch.Tensor:
    if self._cutoff_hz_range is not None:
      low, high = self._cutoff_hz_range
      cutoff_hz = torch.rand(shape, device=device, dtype=dtype) * (high - low) + low
      time_constant_s = 1.0 / (2.0 * torch.pi * cutoff_hz)
    else:
      assert self._time_constant_s_range is not None
      low, high = self._time_constant_s_range
      time_constant_s = (
        torch.rand(shape, device=device, dtype=dtype) * (high - low) + low
      )
    return self._dt / (time_constant_s + self._dt + 1e-8)

  _get_actuator_force = actuator_rfi._get_actuator_force
  _get_actuator_gear = actuator_rfi._get_actuator_gear

  def _resolve_actuator_units(
    self, asset: Entity, asset_cfg: SceneEntityCfg
  ) -> tuple[
    torch.Tensor,
    list[_JointActuatorBandwidthUnit | _FourbarActuatorBandwidthUnit],
  ]:
    actuators = _resolve_selected_actuators(asset, asset_cfg)

    units: list[
      actuator_motor_bandwidth._JointActuatorBandwidthUnit
      | actuator_motor_bandwidth._FourbarActuatorBandwidthUnit
    ] = []
    all_dof_ids: set[int] = set()
    for actuator in actuators:
      if actuator.transmission_type != TransmissionType.JOINT:
        raise ValueError(
          "actuator_motor_bandwidth supports only JOINT transmission actuators. "
          f"Got transmission_type={actuator.transmission_type}."
        )

      joint_ids = actuator.target_ids.to(device=self._device, dtype=torch.long)
      dof_ids = asset.indexing.joint_v_adr[joint_ids].to(dtype=torch.long)
      if len(torch.unique(dof_ids)) != len(dof_ids):
        raise ValueError(
          "actuator_motor_bandwidth expects unique JOINT actuator targets per "
          f"actuator group, got duplicated dof ids {dof_ids.tolist()}."
        )

      if isinstance(actuator, FourbarPdActuator):
        unit = actuator_motor_bandwidth._FourbarActuatorBandwidthUnit(
          actuator=actuator,
          dof_ids=dof_ids,
          dof_cols=torch.empty(0, dtype=torch.long, device=self._device),
        )
      else:
        ctrl_ids = actuator.global_ctrl_ids.to(device=self._device, dtype=torch.long)
        if len(dof_ids) != len(ctrl_ids):
          raise ValueError(
            "actuator_motor_bandwidth expects one-to-one mapping between JOINT "
            f"actuator targets and actuator controls, got {len(dof_ids)} targets "
            f"and {len(ctrl_ids)} controls."
          )
        unit = actuator_motor_bandwidth._JointActuatorBandwidthUnit(
          ctrl_ids=ctrl_ids,
          dof_ids=dof_ids,
          dof_cols=torch.empty(0, dtype=torch.long, device=self._device),
        )

      units.append(unit)
      all_dof_ids.update(dof_ids.tolist())

    if not all_dof_ids:
      return torch.empty(0, dtype=torch.long, device=self._device), []

    sorted_dof_ids = sorted(all_dof_ids)
    dof_ids_tensor = torch.tensor(sorted_dof_ids, dtype=torch.long, device=self._device)
    dof_to_col = {dof_id: col for col, dof_id in enumerate(sorted_dof_ids)}
    for unit in units:
      unit.dof_cols = torch.tensor(
        [dof_to_col[dof_id] for dof_id in unit.dof_ids.tolist()],
        dtype=torch.long,
        device=self._device,
      )

    return dof_ids_tensor, units
