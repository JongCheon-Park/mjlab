# Author  : jojaebeom@kimm.re.kr
# Version : 0.0.1
# Date    : 2026-02-27

"""KIMM P1 actuator registry (KRO100/KRO80/AK80/AK70).

Structure:
- MotorSpec: motor-domain source specs.
- ReducerStageSpec: reducer ratio and stage inertia.
- ModuleSpec: output-domain limits with auto-derivation helpers.

Policy:
- ReducerStageSpec is used for ratio/inertia and armature calculation.
- Output limits/speeds can be pinned via ModuleSpec override fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping

ModuleName = Literal["KRO100", "KRO80", "AK80", "AK70"]
LimitMode = Literal["rated", "peak"]
VelocityMode = Literal["rated", "no_load"]
ModuleMappingValue = ModuleName | tuple[ModuleName, float]
P1ActuatorMode = Literal["implicit", "explicit"]

__all__ = [
  "ModuleName",
  "LimitMode",
  "VelocityMode",
  "P1ActuatorMode",
  "P1_MODULE_MAP",
  "ActuatorParams",
  "MotorSpec",
  "ReducerStageSpec",
  "ModuleSpec",
  "KIMM_Actuator",
  "KIMM_ACTUATORS",
  "rpm_to_rad_s",
  "compute_pd_from_armature",
  "build_actuator_params",
]


def rpm_to_rad_s(rpm: float) -> float:
  return rpm * 2.0 * math.pi / 60.0


def compute_pd_from_armature(
  armature_kgm2: float,
  *,
  hz: float,
  damping_ratio: float,
) -> tuple[float, float]:
  wn = 2.0 * math.pi * float(hz)
  j = float(armature_kgm2)
  kp = j * (wn**2)
  kd = 2.0 * float(damping_ratio) * j * wn
  return kp, kd


@dataclass(frozen=True)
class ActuatorParams:
  effort_limit_nm: float
  velocity_limit_rad_s: float
  armature_kgm2: float
  stiffness: float
  damping: float


@dataclass(frozen=True)
class MotorSpec:
  rated_torque_nm: float | None = None
  peak_torque_nm: float | None = None
  rated_speed_rpm: float | None = None
  no_load_speed_rpm: float | None = None
  rotor_inertia_gcm2: float | None = None


@dataclass(frozen=True)
class ReducerStageSpec:
  ratio: float
  inertia_kgm2: float = 0.0


@dataclass(frozen=True)
class ModuleSpec:
  name: ModuleName

  # Input sources.
  motor: MotorSpec | None = None
  stages: tuple[ReducerStageSpec, ...] = ()
  total_ratio: float | None = None

  # Direct output-domain overrides (mainly for module-only specs like AK series).
  rated_torque_nm_override: float | None = None
  peak_torque_nm_override: float | None = None
  rated_speed_rpm_override: float | None = None
  output_no_load_speed_rpm_override: float | None = None
  output_armature_kgm2_override: float | None = None

  drivetrain_efficiency: float = 1.0

  def __post_init__(self) -> None:
    if not self.stages and self.total_ratio is None:
      raise ValueError(f"[{self.name}] provide stages or total_ratio")

    if self.total_ratio is not None and self.stages:
      product = 1.0
      for stage in self.stages:
        product *= float(stage.ratio)
      if not math.isclose(
        float(self.total_ratio), product, rel_tol=1e-9, abs_tol=1e-12
      ):
        raise ValueError(
          f"[{self.name}] total_ratio({self.total_ratio}) != product(stages)({product})"
        )

    has_all_direct = (
      self.rated_torque_nm_override is not None
      and self.peak_torque_nm_override is not None
      and self.rated_speed_rpm_override is not None
      and self.output_no_load_speed_rpm_override is not None
      and self.output_armature_kgm2_override is not None
    )

    if self.motor is None and not has_all_direct:
      raise ValueError(
        f"[{self.name}] needs motor spec or full direct output overrides"
      )

  @property
  def resolved_total_ratio(self) -> float:
    if self.total_ratio is not None:
      return float(self.total_ratio)
    ratio = 1.0
    for stage in self.stages:
      ratio *= float(stage.ratio)
    return ratio

  @property
  def rated_torque_nm(self) -> float:
    if self.rated_torque_nm_override is not None:
      return float(self.rated_torque_nm_override)
    if self.motor is None:
      raise ValueError(f"[{self.name}] no motor info for rated torque")
    if self.motor.rated_torque_nm is None:
      raise ValueError(f"[{self.name}] motor.rated_torque_nm is missing")
    return float(self.motor.rated_torque_nm) * self.resolved_total_ratio

  @property
  def peak_torque_nm(self) -> float:
    if self.peak_torque_nm_override is not None:
      return float(self.peak_torque_nm_override)
    if self.motor is None:
      raise ValueError(f"[{self.name}] no motor info for peak torque")
    if self.motor.peak_torque_nm is None:
      raise ValueError(f"[{self.name}] motor.peak_torque_nm is missing")
    return float(self.motor.peak_torque_nm) * self.resolved_total_ratio

  @property
  def rated_speed_rpm(self) -> float:
    if self.rated_speed_rpm_override is not None:
      return float(self.rated_speed_rpm_override)
    if self.motor is None:
      raise ValueError(f"[{self.name}] no motor info for rated speed")
    if self.motor.rated_speed_rpm is None:
      raise ValueError(f"[{self.name}] motor.rated_speed_rpm is missing")
    return float(self.motor.rated_speed_rpm) / self.resolved_total_ratio

  @property
  def no_load_speed_rpm(self) -> float:
    if self.output_no_load_speed_rpm_override is not None:
      return float(self.output_no_load_speed_rpm_override)
    if self.motor is None:
      raise ValueError(f"[{self.name}] no motor info for no-load speed")
    if self.motor.no_load_speed_rpm is None:
      raise ValueError(f"[{self.name}] motor.no_load_speed_rpm is missing")
    return float(self.motor.no_load_speed_rpm) / self.resolved_total_ratio

  @property
  def output_armature_kgm2(self) -> float:
    if self.output_armature_kgm2_override is not None:
      return float(self.output_armature_kgm2_override)
    if self.motor is None:
      raise ValueError(f"[{self.name}] no motor info for armature")
    if self.motor.rotor_inertia_gcm2 is None:
      raise ValueError(f"[{self.name}] motor.rotor_inertia_gcm2 is missing")

    ratio = self.resolved_total_ratio
    # 1 g*cm^2 = 1e-7 kg*m^2
    rotor_inertia_kgm2 = float(self.motor.rotor_inertia_gcm2) * 1e-7
    out = rotor_inertia_kgm2 * (ratio**2)

    remaining = ratio
    for stage in self.stages:
      remaining /= float(stage.ratio)
      out += float(stage.inertia_kgm2) * (remaining**2)
    return out

  def torque_limit_nm(
    self,
    mode: LimitMode,
    drivetrain_efficiency: float | None = None,
  ) -> float:
    efficiency = self.drivetrain_efficiency
    if drivetrain_efficiency is not None:
      efficiency = float(drivetrain_efficiency)
    if mode == "peak":
      return self.peak_torque_nm * efficiency
    if mode == "rated":
      return self.rated_torque_nm * efficiency
    raise ValueError(f"Unknown torque mode: {mode}")

  def velocity_limit_rad_s(self, mode: VelocityMode) -> float:
    if mode == "rated":
      return rpm_to_rad_s(self.rated_speed_rpm)
    if mode == "no_load":
      return rpm_to_rad_s(self.no_load_speed_rpm)
    raise ValueError(f"Unknown velocity mode: {mode}")


class KIMM_Actuator:
  def __init__(self) -> None:
    kro100_motor = MotorSpec(
      rated_torque_nm=4.0,
      peak_torque_nm=12.0,
      rated_speed_rpm=2000.0,
      no_load_speed_rpm=2550.0,
      rotor_inertia_gcm2=8700.0,
    )
    kro80_motor = MotorSpec(
      rated_torque_nm=1.3,
      peak_torque_nm=4.0,
      rated_speed_rpm=3600.0,
      no_load_speed_rpm=5040.0,
      rotor_inertia_gcm2=2612.0,
    )

    # AK80/AK70: only rotor inertia is needed in MotorSpec because
    # output torque/speed values are pinned by module overrides.
    ak80_motor = MotorSpec(rotor_inertia_gcm2=564.5)
    ak70_motor = MotorSpec(rotor_inertia_gcm2=753.4788)

    self._modules: dict[ModuleName, ModuleSpec] = {
      "KRO100": ModuleSpec(
        name="KRO100",
        motor=kro100_motor,
        stages=(
          ReducerStageSpec(ratio=4.0, inertia_kgm2=0.0),
          ReducerStageSpec(ratio=4.0, inertia_kgm2=0.0),
        ),
      ),
      "KRO80": ModuleSpec(
        name="KRO80",
        motor=kro80_motor,
        stages=(
          ReducerStageSpec(ratio=6.0, inertia_kgm2=0.0),
          ReducerStageSpec(ratio=5.0, inertia_kgm2=0.0),
        ),
      ),
      "AK80": ModuleSpec(
        name="AK80",
        motor=ak80_motor,
        stages=(ReducerStageSpec(ratio=64.0, inertia_kgm2=0.0),),
        rated_torque_nm_override=48.0,
        peak_torque_nm_override=120.0,
        rated_speed_rpm_override=48.0,
        output_no_load_speed_rpm_override=60.0,
      ),
      "AK70": ModuleSpec(
        name="AK70",
        motor=ak70_motor,
        stages=(ReducerStageSpec(ratio=10.0, inertia_kgm2=0.0),),
        rated_torque_nm_override=8.3,
        peak_torque_nm_override=24.8,
        rated_speed_rpm_override=310.0,
        output_no_load_speed_rpm_override=480.0,
      ),
    }

  def available(self) -> list[ModuleName]:
    return list(self._modules.keys())

  def get(self, name: ModuleName) -> ModuleSpec:
    return self._modules[name]


KIMM_ACTUATORS = KIMM_Actuator()


# P1 actuator-module mapping (joint group -> actuator module).
P1_MODULE_MAP: dict[str, ModuleMappingValue] = {
  "KRO100": "KRO100",
  "KRO80": "KRO80",
  "KRO80x2": ("KRO80", 2.0),  # parallel ankle
  "AK80": "AK80",
  "AK70": "AK70",
}


def build_actuator_params(
  joint_module: Mapping[str, ModuleMappingValue],
  *,
  effort_mode: LimitMode,
  velocity_mode: VelocityMode,
  drivetrain_efficiency: float | None = None,
  pd_hz: float | None = None,
  pd_damping_ratio: float = 1.0,
) -> dict[str, ActuatorParams]:
  joint_params: dict[str, ActuatorParams] = {}

  for joint_name, mapping in joint_module.items():
    if isinstance(mapping, tuple):
      module_name, multiplier = mapping
    else:
      module_name, multiplier = mapping, 1.0

    spec = KIMM_ACTUATORS.get(module_name)
    effort = (
      spec.torque_limit_nm(
        effort_mode,
        drivetrain_efficiency=drivetrain_efficiency,
      )
      * multiplier
    )
    velocity = spec.velocity_limit_rad_s(velocity_mode)
    armature = spec.output_armature_kgm2 * multiplier
    stiffness = 0.0
    damping = 0.0

    if pd_hz is not None:
      stiffness, damping = compute_pd_from_armature(
        armature,
        hz=pd_hz,
        damping_ratio=pd_damping_ratio,
      )

    joint_params[joint_name] = ActuatorParams(
      effort_limit_nm=effort,
      velocity_limit_rad_s=velocity,
      armature_kgm2=armature,
      stiffness=stiffness,
      damping=damping,
    )

  return joint_params


if __name__ == "__main__":
  registry = KIMM_Actuator()
  print("KIMM actuator summary")
  # Headers
  print(
    f"{'name':<8s} "
    f"{'ratio':>6s} "
    f"{'torque(R/P)':>13s} "
    f"{'speed(R/NL)':>14s} "
    f"{'armature':>12s}"
  )
  for name in sorted(registry.available()):  # type: ignore[type-var]
    mod = registry.get(name)  # type: ignore[arg-type]
    print(
      f"{name:<8s} "
      f"{mod.resolved_total_ratio:6.1f} "
      f"{mod.rated_torque_nm:6.1f}/{mod.peak_torque_nm:6.1f} "
      f"{mod.velocity_limit_rad_s('rated'):6.1f}/{mod.velocity_limit_rad_s('no_load'):6.1f} "
      f"{mod.output_armature_kgm2:12.8f}"
    )
