# unitree_rl_lab/assets/robots/kimm_actuators.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""KIMM actuator module registry (KRO100 / KRO80) and datasheet-based limits.

This module is intentionally robot-config agnostic.
Robot-specific JOINT_MODULE mapping and how you apply limits belong in kimm.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Mapping, Optional, Tuple, TypedDict

__all__ = [
  "ModuleName",
  "LimitMode",
  "VelocityMode",
  "rpm_to_rad_s",
  "rad_s_to_rpm",
  "compute_pd_from_armature",
  "JointParams",
  "build_joint_params",
  "MotorSpec",
  "ReducerStageSpec",
  "ModuleSpec",
  "KIMM_Actuator",
  "KIMM_ACTUATORS",
]
# AK80, AK70
ModuleName = Literal["KRO100", "KRO80"]
LimitMode = Literal["rated", "peak"]
VelocityMode = Literal["rated", "no_load"]


def rpm_to_rad_s(rpm: float) -> float:
  return rpm * 2.0 * math.pi / 60.0


def rad_s_to_rpm(rad_s: float) -> float:
  return rad_s * 60.0 / (2.0 * math.pi)


def compute_pd_from_armature(
  armature_kgm2: float,
  *,
  hz: float = 10.0,
  damping_ratio: float = 1.0,
) -> Tuple[float, float]:
  """Compute (stiffness, damping) from armature using a 2nd-order target.

  Unitree convention:
    wn = 2*pi*hz
    kp = J * wn^2
    kd = 2*zeta*J*wn

  Damping ratio (zeta) suggestions:
    - 0.7 : under-damped (faster, more overshoot)
    - 1.0 : ~critical (balanced)
    - 2.0 : over-damped (more stable, sluggish)
  """
  wn = 2.0 * math.pi * float(hz)
  J = float(armature_kgm2)
  kp = J * (wn**2)
  kd = 2.0 * float(damping_ratio) * J * wn
  return float(kp), float(kd)


class JointParams(TypedDict):
  effort_limit_nm: Dict[str, float]
  velocity_limit_rad_s: Dict[str, float]
  armature_kgm2: Dict[str, float]
  stiffness: Dict[str, float]
  damping: Dict[str, float]


def build_joint_params(
  joint_module: Mapping[str, ModuleSpec],
  *,
  effort_mode: LimitMode,
  velocity_mode: VelocityMode,
  default_effort_nm: float | None = None,
  default_velocity_rad_s: float | None = None,
  default_armature_kgm2: float | None = None,
  pd_hz: float | None = None,
  pd_damping_ratio: float = 2.0,  # zeta
) -> JointParams:
  """Build per-joint dicts for effort/velocity/armature in a consistent way.

  - Uses module registry (KIMM_ACTUATORS) for each mapped joint.
  - For joints not present in `joint_module`, you can optionally provide defaults.
  """
  eff: Dict[str, float] = {}
  vel: Dict[str, float] = {}
  arm: Dict[str, float] = {}
  kp: Dict[str, float] = {}
  kd: Dict[str, float] = {}

  for j, spec in joint_module.items():
    if isinstance(spec, tuple):
      mod_name, mult = spec
    else:
      mod_name, mult = spec, 1.0

    mod = KIMM_ACTUATORS.get(mod_name)

    eff[j] = float(mod.torque_limit_nm(effort_mode)) * float(mult)
    vel[j] = float(mod.velocity_limit_rad_s(velocity_mode))
    # armature: output-equivalent, Unitree convention (kg*m^2)
    # NOTE: internal gear inertias unknown -> stage_inertias_kgm2=[0,0] => rotor-only reflected
    arm[j] = float(mod.output_armature_kgm2) * float(mult)

    # Optional: derive PD gains from armature at target natural frequency
    if pd_hz is not None:
      kp[j], kd[j] = compute_pd_from_armature(
        arm[j],
        hz=float(pd_hz),
        damping_ratio=float(pd_damping_ratio),
      )

  return {
    "effort_limit_nm": eff,
    "velocity_limit_rad_s": vel,
    "armature_kgm2": arm,
    "stiffness": kp,
    "damping": kd,
  }


@dataclass(frozen=True)
class MotorSpec:
  """Raw motor spec (motor shaft domain)."""

  name: str

  # Electrical / constants
  rated_voltage_v: float
  kv_rpm_per_v: float
  kt_nm_per_a: float
  ke_v_per_krpm: float
  phase_resistance_ohm: float
  phase_inductance_h: float
  pole_pairs: int
  winding: str  # "star" / "delta"

  # Operating points
  rated_torque_nm: float
  peak_torque_nm: float
  rated_speed_rpm: float
  no_load_speed_rpm: float
  rated_current_a: float
  peak_current_a: float

  # Mass/inertia (as given)
  weight_kg: float
  inertia_gcm2: float
  cogging_torque_nmm: float

  @property
  def inertia_kgm2(self) -> float:
    # 1 g*cm^2 = 1e-7 kg*m^2
    return self.inertia_gcm2 * 1e-7


@dataclass(frozen=True)
class ReducerStageSpec:
  """Planetary stage spec (optional metadata)."""

  stage: int
  ratio: float
  module: float
  planet_gears: int
  rated_torque_nm: float
  peak_torque_nm: float
  rated_speed_rpm: float


@dataclass(frozen=True)
class ModuleSpec:
  """Actuator module expressed in output domain (joint-side).

  Note: Keep compatibility helpers used by kimm.py:
    - torque_limit_nm(mode)
    - velocity_limit_rad_s(mode)

  IMPORTANT:
  stage inertias are NOT known for KIMM right now.
  We set them to 0.0 by default in the registry, meaning:
      armature == rotor inertia reflected to output (rotor-only model).
  """

  name: ModuleName
  motor: MotorSpec
  stages: List[ReducerStageSpec]

  # Packaging
  outer_diameter_mm: float
  thickness_mm: float
  mass_kg: float

  # Output-domain specs (what you use for joint/actuator limits)
  output_total_ratio: float  # motor_speed / output_speed
  output_rated_torque_nm: float
  output_peak_torque_nm: float
  output_rated_speed_rpm: float

  # Optional: if datasheet gives output no-load, else computed from motor no-load / ratio
  output_no_load_speed_rpm: Optional[float] = None

  # Optional: efficiency derating (not provided -> 1.0)
  drivetrain_efficiency: float = 1.0

  # [J1(after stage1), J2(after stage2/output), ...]
  stage_inertias_kgm2: List[float] = None

  # Precomputed output-equivalent armature (kg*m^2) used by Isaac/PhysX.
  output_armature_kgm2: float = 0.0

  # ---- Common derived helpers ----
  @property
  def total_ratio(self) -> float:
    """Backwards-compat alias (older code used total_ratio)."""
    return self.output_total_ratio

  @property
  def output_rated_speed_rad_s(self) -> float:
    return rpm_to_rad_s(self.output_rated_speed_rpm)

  def compute_output_armature_kgm2(self) -> float:
    """Compute output-equivalent armature (kg*m^2).

    Unitree convention:
      armature = J2 + J1*i2^2 + Jr*(i1*i2)^2   (2-stage example)

    Generalized for N stages:
      Jr reflected by (prod ratios)^2
      stage inertia after stage k reflected by (prod remaining ratios)^2
    """
    ratios = [s.ratio for s in self.stages]
    stage_J = self.stage_inertias_kgm2 or [0.0] * len(ratios)

    if len(stage_J) != len(ratios):
      raise ValueError(
        f"[{self.name}] stage_inertias_kgm2 length mismatch: "
        f"{len(stage_J)} vs stages {len(ratios)}"
      )

    total_ratio = 1.0
    for r in ratios:
      total_ratio *= float(r)

    # rotor reflected to output
    J_out = float(self.motor.inertia_kgm2) * (total_ratio**2)

    # stage inertias: Jk is after stage k (1-indexed), so reflect by remaining ratios
    remaining = total_ratio
    for Jk, rk in zip(stage_J, ratios, strict=False):
      remaining /= float(rk)
      J_out += float(Jk) * (remaining**2)

    return float(J_out)

  def output_no_load_speed_rpm_value(self) -> float:
    if self.output_no_load_speed_rpm is not None:
      return self.output_no_load_speed_rpm
    return self.motor.no_load_speed_rpm / self.output_total_ratio

  def output_no_load_speed_rad_s(self) -> float:
    return rpm_to_rad_s(self.output_no_load_speed_rpm_value())

  # ---- Compatibility with existing kimm.py (minimal refactor) ----
  def torque_limit_nm(self, mode: LimitMode) -> float:
    """Return output torque limit (rated or peak), optionally derated by efficiency."""
    if mode == "peak":
      return self.output_peak_torque_nm * self.drivetrain_efficiency
    if mode == "rated":
      return self.output_rated_torque_nm * self.drivetrain_efficiency
    raise ValueError(f"Unknown torque limit mode: {mode}")

  def velocity_limit_rad_s(self, mode: VelocityMode) -> float:
    """Return output velocity limit (rated or no-load)."""
    if mode == "rated":
      return rpm_to_rad_s(self.output_rated_speed_rpm)
    if mode == "no_load":
      return self.output_no_load_speed_rad_s()
    raise ValueError(f"Unknown velocity mode: {mode}")

  def sanity_check(self, *, rel_tol: float = 0.05) -> None:
    """Basic consistency checks vs motor*ratio (efficiency not applied)."""
    t_rated_calc = self.motor.rated_torque_nm * self.output_total_ratio
    t_peak_calc = self.motor.peak_torque_nm * self.output_total_ratio
    w_rated_calc = self.motor.rated_speed_rpm / self.output_total_ratio

    def close(a: float, b: float, rel: float = rel_tol) -> bool:
      denom = max(abs(a), abs(b), 1e-9)
      return abs(a - b) / denom <= rel

    if not close(self.output_rated_torque_nm, t_rated_calc):
      raise ValueError(
        f"[{self.name}] output_rated_torque mismatch: "
        f"{self.output_rated_torque_nm} vs motor*ratio {t_rated_calc}"
      )
    if not close(self.output_peak_torque_nm, t_peak_calc):
      raise ValueError(
        f"[{self.name}] output_peak_torque mismatch: "
        f"{self.output_peak_torque_nm} vs motor*ratio {t_peak_calc}"
      )
    if not close(self.output_rated_speed_rpm, w_rated_calc):
      raise ValueError(
        f"[{self.name}] output_rated_speed mismatch: "
        f"{self.output_rated_speed_rpm} vs motor/ratio {w_rated_calc}"
      )


class KIMM_Actuator:
  """Registry/loader for KRO actuator modules (KRO100, KRO80)."""

  def __init__(self) -> None:
    self._modules: Dict[ModuleName, ModuleSpec] = self._build_registry()
    for m in self._modules.values():
      m.sanity_check()

  def available(self) -> List[ModuleName]:
    return list(self._modules.keys())

  def get(self, name: ModuleName) -> ModuleSpec:
    return self._modules[name]

  @staticmethod
  def _build_registry() -> Dict[ModuleName, ModuleSpec]:
    # ---- Motor specs (from your sheet) ----
    ro100_kv55_std = MotorSpec(
      name="RO100 KV55 Standard",
      rated_voltage_v=48.0,
      kv_rpm_per_v=55.0,
      kt_nm_per_a=0.2,
      ke_v_per_krpm=18.67,
      phase_resistance_ohm=0.143,
      phase_inductance_h=137e-6,
      pole_pairs=21,
      winding="star",
      rated_torque_nm=4.0,
      peak_torque_nm=12.0,
      rated_speed_rpm=2000.0,
      no_load_speed_rpm=2550.0,
      rated_current_a=20.0,
      peak_current_a=62.0,
      weight_kg=0.710,
      inertia_gcm2=8700.0,
      cogging_torque_nmm=55.0,
    )

    ro80_kv105_std = MotorSpec(
      name="RO80 KV105 Standard",
      rated_voltage_v=48.0,
      kv_rpm_per_v=105.0,
      kt_nm_per_a=0.087,
      ke_v_per_krpm=9.07,
      phase_resistance_ohm=0.120,
      phase_inductance_h=103e-6,
      pole_pairs=21,
      winding="delta",
      rated_torque_nm=1.3,
      peak_torque_nm=4.0,
      rated_speed_rpm=3600.0,
      no_load_speed_rpm=5040.0,
      rated_current_a=15.0,
      peak_current_a=50.0,
      weight_kg=0.352,
      inertia_gcm2=2612.0,
      cogging_torque_nmm=24.0,
    )

    # ---- Reducer stage metadata (optional) ----
    kro100_stage1 = ReducerStageSpec(
      stage=1,
      ratio=4.0,
      module=1.0,
      planet_gears=3,
      rated_torque_nm=16.0,
      peak_torque_nm=48.0,
      rated_speed_rpm=500.0,
    )
    kro100_stage2 = ReducerStageSpec(
      stage=2,
      ratio=4.0,
      module=1.0,
      planet_gears=3,
      rated_torque_nm=64.0,
      peak_torque_nm=192.0,
      rated_speed_rpm=125.0,
    )

    kro80_stage1 = ReducerStageSpec(
      stage=1,
      ratio=6.0,
      module=0.5,
      planet_gears=3,
      rated_torque_nm=7.8,
      peak_torque_nm=24.0,
      rated_speed_rpm=600.0,
    )
    kro80_stage2 = ReducerStageSpec(
      stage=2,
      ratio=5.0,
      module=0.8,
      planet_gears=4,
      rated_torque_nm=39.0,
      peak_torque_nm=120.0,
      rated_speed_rpm=120.0,
    )

    # ---- Module specs (output domain) ----
    # ---- Module specs (output domain) ----
    # NOTE (VERY IMPORTANT):
    # We do NOT know the internal gear/shaft inertias yet.
    # So we intentionally set stage_inertias_kgm2 = [0.0, 0.0] as placeholders.
    # This makes armature a "rotor-only reflected inertia" model.
    kro100_stage_inertias = [0.0, 0.0]
    kro80_stage_inertias = [0.0, 0.0]

    kro100 = ModuleSpec(
      name="KRO100",
      motor=ro100_kv55_std,
      stages=[kro100_stage1, kro100_stage2],
      outer_diameter_mm=127.5,
      thickness_mm=67.0,
      mass_kg=2.10,
      output_total_ratio=16.0,
      output_rated_torque_nm=64.0,
      output_peak_torque_nm=192.0,
      output_rated_speed_rpm=125.0,
      output_no_load_speed_rpm=None,
      drivetrain_efficiency=1.0,
      stage_inertias_kgm2=kro100_stage_inertias,
      output_armature_kgm2=0.0,  # will be filled just below
    )
    # Fill armature (output-equivalent, kg*m^2)
    object.__setattr__(
      kro100, "output_armature_kgm2", kro100.compute_output_armature_kgm2()
    )

    kro80 = ModuleSpec(
      name="KRO80",
      motor=ro80_kv105_std,
      stages=[kro80_stage1, kro80_stage2],
      outer_diameter_mm=109.0,
      thickness_mm=61.6,
      mass_kg=1.46,
      output_total_ratio=30.0,
      output_rated_torque_nm=39.0,
      output_peak_torque_nm=120.0,
      output_rated_speed_rpm=120.0,
      output_no_load_speed_rpm=None,
      drivetrain_efficiency=1.0,
      stage_inertias_kgm2=kro80_stage_inertias,
      output_armature_kgm2=0.0,  # will be filled just below
    )
    object.__setattr__(
      kro80, "output_armature_kgm2", kro80.compute_output_armature_kgm2()
    )

    return {"KRO100": kro100, "KRO80": kro80}


# Shared singleton (so robot cfg files don't rebuild registry repeatedly)
KIMM_ACTUATORS = KIMM_Actuator()


if __name__ == "__main__":
  reg = KIMM_Actuator()

  def _print_motor(ms: MotorSpec) -> None:
    print("  [Motor]")
    print(f"    name            : {ms.name}")
    print(f"    rated_voltage   : {ms.rated_voltage_v:.1f} V")
    print(f"    kv              : {ms.kv_rpm_per_v:.1f} rpm/V")
    print(f"    kt              : {ms.kt_nm_per_a:.3f} N·m/A")
    print(f"    ke              : {ms.ke_v_per_krpm:.2f} V/krpm")
    print(f"    R_phase         : {ms.phase_resistance_ohm:.3f} Ω")
    print(f"    L_phase         : {ms.phase_inductance_h * 1e6:.1f} µH")
    print(f"    pole_pairs      : {ms.pole_pairs:d}")
    print(f"    winding         : {ms.winding}")
    print(f"    rated_torque    : {ms.rated_torque_nm:.2f} N·m")
    print(f"    peak_torque     : {ms.peak_torque_nm:.2f} N·m")
    print(f"    rated_speed     : {ms.rated_speed_rpm:.0f} rpm")
    print(f"    no_load_speed   : {ms.no_load_speed_rpm:.0f} rpm")
    print(f"    rated_current   : {ms.rated_current_a:.1f} A")
    print(f"    peak_current    : {ms.peak_current_a:.1f} A")
    print(f"    weight          : {ms.weight_kg:.3f} kg")
    print(
      f"    inertia         : {ms.inertia_gcm2:.1f} g·cm² ({ms.inertia_kgm2:.6f} kg·m²)"
    )
    print(f"    cogging_torque  : {ms.cogging_torque_nmm:.1f} N·mm")

  def _print_module(m: ModuleSpec) -> None:
    print(f"\n[{m.name}]")
    print("  [Module Output]")
    print(f"    total_ratio     : {m.output_total_ratio:.1f} (motor/output)")
    print(f"    rated_torque    : {m.output_rated_torque_nm:.2f} N·m")
    print(f"    peak_torque     : {m.output_peak_torque_nm:.2f} N·m")
    print(
      f"    rated_speed     : {m.output_rated_speed_rpm:.1f} rpm ({m.output_rated_speed_rad_s:.2f} rad/s)"
    )
    print(
      f"    armature        : {m.output_armature_kgm2:.8f} kg·m² (output-equivalent)"
    )
    print(
      f"    stage_inertias  : {m.stage_inertias_kgm2}  # IMPORTANT: unknown -> placeholders (rotor-only model)"
    )
    nl_rpm = m.output_no_load_speed_rpm_value()
    print(
      f"    no_load_speed   : {nl_rpm:.1f} rpm ({m.output_no_load_speed_rad_s():.2f} rad/s)"
    )
    print(f"    eff(derating)   : {m.drivetrain_efficiency:.2f}")
    print("  [Packaging]")
    print(f"    OD              : {m.outer_diameter_mm:.1f} mm")
    print(f"    thickness       : {m.thickness_mm:.1f} mm")
    print(f"    mass            : {m.mass_kg:.3f} kg")
    if m.stages:
      print("  [Reducer Stages]")
      for s in m.stages:
        print(
          f"    stage {s.stage}: ratio {s.ratio:.2f}, "
          f"rated {s.rated_torque_nm:.1f} N·m, peak {s.peak_torque_nm:.1f} N·m, "
          f"rated_speed {s.rated_speed_rpm:.0f} rpm"
        )
    _print_motor(m.motor)

    # quick derived checks users usually care about
    print("  [Quick Derived]")
    print(f"    output torque_limit(rated): {m.torque_limit_nm('rated'):.2f} N·m")
    print(f"    output torque_limit(peak) : {m.torque_limit_nm('peak'):.2f} N·m")
    print(
      f"    output vel_limit(rated)   : {m.velocity_limit_rad_s('rated'):.2f} rad/s"
    )
    print(
      f"    output vel_limit(no_load) : {m.velocity_limit_rad_s('no_load'):.2f} rad/s"
    )
    print(f"    output armature         : {m.output_armature_kgm2:.8f} kg·m²")

  # stable order
  for name in sorted(reg.available()):
    _print_module(reg.get(name))
