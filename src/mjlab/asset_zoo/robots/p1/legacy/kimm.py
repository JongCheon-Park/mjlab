# unitree_rl_lab/assets/robots/kimm.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for KIMM kimm_bot robot (URDF-based).

Goal:
- Policy action: waist + legs only (lower body).
- Upper body (everything except waist+legs): hold at 0 rad using PD actuators.

This version:
- Lower-body limits (effort/velocity) are derived compactly from actuator module specs (KRO100/KRO80).
- You maintain only JOINT_MODULE mapping per joint.
"""

from __future__ import annotations

import os
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass
from unitree_rl_lab.assets.robots.kimm_actuators import (
  LimitMode,
  ModuleSpec,
  build_joint_params,
)

# Choose how to derive limits from module specs
# - effort_limit_sim: "peak"
# - velocity_limit_sim: "no_load"
LOWER_EFFORT_MODE: LimitMode = "peak"  # "peak" or "rated"
LOWER_VELOCITY_MODE: Literal["rated", "no_load"] = "no_load"

# ------------------------------------------------------------------------------------
# Action scale factor
# ------------------------------------------------------------------------------------
ACTION_SCALE = 0.5

# ------------------------------------------------------------------------------------
# Paths (edit if needed)
# ------------------------------------------------------------------------------------
KIMM_BOT_ASSET_DIR = "/home/park2204/Jaebeom/pixi-IsaacLab-Unitree/KIMM/robots"
KIMM_BOT_URDF = os.path.join(KIMM_BOT_ASSET_DIR, "kimm_bot_3.urdf")
KIMM_BOT_MESH_DIR = os.path.join(KIMM_BOT_ASSET_DIR, "meshes")


# ------------------------------------------------------------------------------------
# Joint names (from URDF)
# ------------------------------------------------------------------------------------
# Lower body (policy-controlled)
WAIST_YAW = "Waist_yaw_joint"

R_HIP_PITCH = "right_hip_pitch_joint"
R_HIP_ROLL = "right_hip_roll_joint"
R_HIP_YAW = "right_hip_yaw_joint"
R_KNEE = "right_knee_joint"
R_ANK_PITCH = "right_ankle_pitch_joint"
R_ANK_ROLL = "right_ankle_roll_joint"

L_HIP_PITCH = "left_hip_pitch_joint"
L_HIP_ROLL = "left_hip_roll_joint"
L_HIP_YAW = "left_hip_yaw_joint"
L_KNEE = "left_knee_joint"
L_ANK_PITCH = "left_ankle_pitch_joint"
L_ANK_ROLL = "left_ankle_roll_joint"

LOWER_BODY_JOINTS: list[str] = [
  WAIST_YAW,
  R_HIP_PITCH,
  R_HIP_ROLL,
  R_HIP_YAW,
  R_KNEE,
  R_ANK_PITCH,
  R_ANK_ROLL,
  L_HIP_PITCH,
  L_HIP_ROLL,
  L_HIP_YAW,
  L_KNEE,
  L_ANK_PITCH,
  L_ANK_ROLL,
]

# Upper body (NOT in policy action, but PD-hold at 0)
UPPER_BODY_JOINTS: list[str] = [
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_pitch_joint",
  "right_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_wrist_roll_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_pitch_joint",
  "left_wrist_yaw_joint",
  "left_wrist_pitch_joint",
  "left_wrist_roll_joint",
]

ALL_JOINTS: list[str] = LOWER_BODY_JOINTS + UPPER_BODY_JOINTS

KIMM_BOT_POLICY_ACTION_JOINTS: list[str] = list(LOWER_BODY_JOINTS)
KIMM_BOT_SDK_JOINT_ORDER: list[str] = list(ALL_JOINTS)


# ------------------------------------------------------------------------------------
# Joint -> module mapping (ONLY THING TO MAINTAIN for lower body)
# ------------------------------------------------------------------------------------
# Rule-of-thumb based on your previous 64Nm vs 39Nm split:
# - KRO100: hip_pitch, knee, waist yaw
# - KRO80 : hip_roll, hip_yaw, ankles
JOINT_MODULE: dict[str, ModuleSpec] = {
  WAIST_YAW: "KRO100",
  # right
  R_HIP_PITCH: "KRO100",
  R_KNEE: "KRO100",
  R_HIP_ROLL: "KRO80",
  R_HIP_YAW: "KRO80",
  R_ANK_PITCH: ("KRO80", 2.0),  # Parallel
  R_ANK_ROLL: ("KRO80", 2.0),  # Parallel
  # left
  L_HIP_PITCH: "KRO100",
  L_KNEE: "KRO100",
  L_HIP_ROLL: "KRO80",
  L_HIP_YAW: "KRO80",
  L_ANK_PITCH: ("KRO80", 2.0),  # Parallel
  L_ANK_ROLL: ("KRO80", 2.0),  # Parallel
}


# ------------------------------------------------------------------------------------
# Unitree-style base classes
# ------------------------------------------------------------------------------------
@configclass
class KimmArticulationCfg(ArticulationCfg):
  """Configuration for KIMM articulations."""

  joint_sdk_names: list[str] = None
  soft_joint_pos_limit_factor = 0.9


@configclass
class KimmUrdfFileCfg(sim_utils.UrdfFileCfg):
  """URDF import settings for KIMM."""

  fix_base: bool = False
  activate_contact_sensors: bool = True
  replace_cylinders_with_capsules = True

  # Let actuators define PD gains
  joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
      stiffness=0.0, damping=0.0
    )
  )

  articulation_props = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=True,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=4,
  )
  rigid_props = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=1.0,
  )

  def replace_asset(self, meshes_dir: str, urdf_path: str):
    """Build /tmp robot_description with symlinked meshes + urdf."""
    tmp_root = "/tmp/IsaacLab/kimm_bot"
    tmp_meshes_dir = os.path.join(tmp_root, "meshes")
    os.makedirs(tmp_root, exist_ok=True)

    # replace meshes symlink
    if os.path.islink(tmp_meshes_dir) or os.path.exists(tmp_meshes_dir):
      try:
        os.remove(tmp_meshes_dir)
      except IsADirectoryError:
        import shutil

        shutil.rmtree(tmp_meshes_dir)
    os.symlink(meshes_dir, tmp_meshes_dir)

    # replace urdf symlink
    self.asset_path = os.path.join(tmp_root, "robot.urdf")
    if os.path.islink(self.asset_path) or os.path.exists(self.asset_path):
      os.remove(self.asset_path)
    os.symlink(urdf_path, self.asset_path)


# ------------------------------------------------------------------------------------
# Auto PD gains from armature @ 10Hz
# ------------------------------------------------------------------------------------
PD_HZ = 10.0

# Damping ratio suggestions (zeta):
#   0.7 : under-damped (faster, overshoot)
#   1.0 : ~critical (balanced default)
#   2.0 : over-damped (stable, sluggish)
PD_DAMPING_RATIO = 2.0


# ------------------------------------------------------------------------------------
# Limits + Armature (derived from module specs via a single consistent API)
# ------------------------------------------------------------------------------------
_lower = build_joint_params(
  JOINT_MODULE,
  effort_mode=LOWER_EFFORT_MODE,
  velocity_mode=LOWER_VELOCITY_MODE,
  pd_hz=PD_HZ,
  pd_damping_ratio=PD_DAMPING_RATIO,
)

EFFORT_LIMIT: dict[str, float] = {}
VELOCITY_LIMIT: dict[str, float] = {}
ARMATURE_PER_JOINT: dict[str, float] = {}
STIFFNESS: dict[str, float] = {}
DAMPING: dict[str, float] = {}

# lower-body (module-mapped)
EFFORT_LIMIT.update(_lower["effort_limit_nm"])
VELOCITY_LIMIT.update(_lower["velocity_limit_rad_s"])
ARMATURE_PER_JOINT.update(_lower["armature_kgm2"])
STIFFNESS.update(_lower["stiffness"])
DAMPING.update(_lower["damping"])

# upper-body defaults (not module-mapped)
for j in UPPER_BODY_JOINTS:
  EFFORT_LIMIT.setdefault(j, 30.0)
  VELOCITY_LIMIT.setdefault(j, 10.0)
  ARMATURE_PER_JOINT.setdefault(j, 0.01)
  STIFFNESS.setdefault(j, 50)
  DAMPING.setdefault(j, 1.0)

# ------------------------------------------------------------------------------------
# Fixed PD overrides (manual tuning)
# ------------------------------------------------------------------------------------
# After auto-PD initialization, overwrite selected joints with hand-tuned gains.
# FIXED_KP = {
#     # lower
#     WAIST_YAW: 100.0,

#     R_HIP_YAW: 100.0,
#     R_HIP_ROLL: 100.0,
#     R_HIP_PITCH: 100.0,
#     R_KNEE: 100.0,
#     R_ANK_PITCH: 100.0,
#     R_ANK_ROLL: 100.0,

#     L_HIP_YAW: 100.0,
#     L_HIP_ROLL: 100.0,
#     L_HIP_PITCH: 100.0,
#     L_KNEE: 100.0,
#     L_ANK_PITCH: 100.0,
#     L_ANK_ROLL: 100.0,
# }

# FIXED_KD = {
#     # lower
#     WAIST_YAW: 1.0,

#     R_HIP_YAW: 1.0,
#     R_HIP_ROLL: 1.0,
#     R_HIP_PITCH: 1.0,
#     R_KNEE: 1.0,
#     R_ANK_PITCH: 1.0,
#     R_ANK_ROLL: 1.0,

#     L_HIP_YAW: 1.0,
#     L_HIP_ROLL: 1.0,
#     L_HIP_PITCH: 1.0,
#     L_KNEE: 1.0,
#     L_ANK_PITCH: 1.0,
#     L_ANK_ROLL: 1.0,
# }

# Safety check: ensure all joints have PD entries
missing_kp = sorted(set(ALL_JOINTS) - set(STIFFNESS.keys()))
missing_kd = sorted(set(ALL_JOINTS) - set(DAMPING.keys()))
if missing_kp:
  raise RuntimeError(f"Missing stiffness for joints: {missing_kp}")
if missing_kd:
  raise RuntimeError(f"Missing damping for joints: {missing_kd}")


# ------------------------------------------------------------------------------------
# Initial pose
# ------------------------------------------------------------------------------------
INIT_JOINT_POS: dict[str, float] = {name: 0.0 for name in ALL_JOINTS}

INIT_JOINT_POS.update(
  {
    # waist
    WAIST_YAW: 0.0,
    # hips
    L_HIP_ROLL: 0.0,
    R_HIP_ROLL: 0.0,
    L_HIP_YAW: 0.0,
    R_HIP_YAW: 0.0,
    L_HIP_PITCH: 0.15,
    R_HIP_PITCH: -0.15,
    # knees
    L_KNEE: 0.25,
    R_KNEE: 0.25,
    L_ANK_PITCH: -0.10,
    R_ANK_PITCH: -0.10,
    # ankles roll neutral
    L_ANK_ROLL: -0.05,
    R_ANK_ROLL: 0.05,
  }
)


# ------------------------------------------------------------------------------------
# Actuator builders
# ------------------------------------------------------------------------------------
def _make_implicit_actuator(joints: list[str]) -> ImplicitActuatorCfg:
  return ImplicitActuatorCfg(
    joint_names_expr=joints,
    effort_limit_sim={j: EFFORT_LIMIT[j] for j in joints},
    velocity_limit_sim={j: VELOCITY_LIMIT[j] for j in joints},
    stiffness={j: STIFFNESS[j] for j in joints},
    damping={j: DAMPING[j] for j in joints},
    armature={j: ARMATURE_PER_JOINT[j] for j in joints},
    # armature=0.01
  )


# ------------------------------------------------------------------------------------
# Main robot cfg
# ------------------------------------------------------------------------------------
LEG_JOINTS = [
  R_HIP_PITCH,
  R_HIP_ROLL,
  R_HIP_YAW,
  R_KNEE,
  R_ANK_PITCH,
  R_ANK_ROLL,
  L_HIP_PITCH,
  L_HIP_ROLL,
  L_HIP_YAW,
  L_KNEE,
  L_ANK_PITCH,
  L_ANK_ROLL,
]

KIMM_BOT_CFG = KimmArticulationCfg(
  spawn=KimmUrdfFileCfg(
    asset_path=KIMM_BOT_URDF,
  ),
  init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.88),
    joint_pos=INIT_JOINT_POS,
    joint_vel={".*": 0.0},
  ),
  soft_joint_pos_limit_factor=0.95,
  actuators={
    # Lower body (policy-controlled)
    "waist_yaw": _make_implicit_actuator([WAIST_YAW]),
    "legs": _make_implicit_actuator(LEG_JOINTS),
    # Upper body: NOT in policy action, just hold at 0 (PD)
    "upper_body_hold": _make_implicit_actuator(UPPER_BODY_JOINTS),
  },
  joint_sdk_names=KIMM_BOT_SDK_JOINT_ORDER,
)


# ------------------------------------------------------------------------------------
# Action scale (Unitree-style): ACTION_SCALE * effort / stiffness
# ------------------------------------------------------------------------------------
def _as_dict(val, names: list[str]) -> dict[str, float]:
  if isinstance(val, dict):
    return {k: float(v) for k, v in val.items()}
  return {n: float(val) for n in names}


# Scales for all actuated joints (lower + upper)
KIMM_BOT_ACTION_SCALE_ALL: dict[str, float] = {}

for actuator in KIMM_BOT_CFG.actuators.values():
  names = list(actuator.joint_names_expr)
  e = _as_dict(actuator.effort_limit_sim, names)
  s = _as_dict(actuator.stiffness, names)
  for n in names:
    if n in e and n in s and s[n] != 0.0:
      KIMM_BOT_ACTION_SCALE_ALL[n] = ACTION_SCALE * e[n] / s[n]

# Scales for policy joints only (what you should use in your task)
KIMM_BOT_ACTION_SCALE: dict[str, float] = {
  j: KIMM_BOT_ACTION_SCALE_ALL[j]
  for j in KIMM_BOT_POLICY_ACTION_JOINTS
  if j in KIMM_BOT_ACTION_SCALE_ALL
}
