"""Diden Walker constants for MJLAB training."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

DIDEN_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "diden" / "xmls" / "diden.xml"
)
assert DIDEN_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(DIDEN_XML))


##
# Actuator config.
#
# The original MJCF has no <actuator> block. We use BuiltinPositionActuatorCfg
# (implicit PD via mujoco position actuators) grouped by joint role. Numbers
# below are placeholders chosen from typical humanoid scales — adjust once
# real motor specs are available.
##

# Hip pitch/yaw and knee carry most of the leg load → larger torque/stiffness.
DIDEN_ACTUATOR_HIP_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
  ),
  stiffness=200.0,
  damping=10.0,
  effort_limit=120.0,
  armature=0.01,
)

# Hip roll resists lateral sway; modest torque is enough.
DIDEN_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint",),
  stiffness=150.0,
  damping=8.0,
  effort_limit=80.0,
  armature=0.01,
)

# Ankle pitch/roll — lower torque, tighter control for foot orientation.
DIDEN_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
  ),
  stiffness=80.0,
  damping=4.0,
  effort_limit=40.0,
  armature=0.005,
)

# Waist pitch — kept stiff; usually held at default by JointPositionHoldAction.
DIDEN_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch_joint",),
  stiffness=200.0,
  damping=10.0,
  effort_limit=100.0,
  armature=0.01,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.85),
  joint_pos={
    ".*_hip_pitch_joint": 0.0,
    ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0,
    ".*_knee_joint": 0.0,
    ".*_ankle_pitch_joint": 0.0,
    ".*_ankle_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

# Slightly squatted keyframe — common humanoid initial pose so the robot
# doesn't start with locked knees.
# Natural standing pelvis height with KNEES_BENT joints settles to ~0.85 m
# (measured: zero-action settle by step 100). Spawn slightly above so
# MuJoCo lands cleanly without bouncing up against gravity.
KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.87),
  joint_pos={
    ".*_hip_pitch_joint": -0.20,
    ".*_knee_joint": 0.45,
    ".*_ankle_pitch_joint": -0.25,
    "waist_pitch_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision.*",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision.*": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

##
# Final config.
##

DIDEN_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    DIDEN_ACTUATOR_HIP_KNEE,
    DIDEN_ACTUATOR_HIP_ROLL,
    DIDEN_ACTUATOR_ANKLE,
    DIDEN_ACTUATOR_WAIST,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_diden_robot_cfg() -> EntityCfg:
  """Get a fresh Diden Walker robot configuration instance."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=DIDEN_ARTICULATION,
  )


DIDEN_ACTION_SCALE: dict[str, float] = {}
for a in DIDEN_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    DIDEN_ACTION_SCALE[n] = 0.25 * e / s
