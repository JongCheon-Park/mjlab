"""KIMM P1 constants for MJLAB training."""

from pathlib import Path

import mujoco
from mjlab.asset_zoo.robots.kimm_p1.kimm_p1_actuators import KIMM_ACTUATORS

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

P1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "kimm_p1" / "xmls" / "kimm_p1.xml"
)
assert P1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(P1_XML))


##
# Actuator config.
##

# Get module specs from the KIMM actuator registry.
_KRO100 = KIMM_ACTUATORS.get("KRO100")
_KRO80 = KIMM_ACTUATORS.get("KRO80")

# Armature values (output-equivalent, rotor-only reflected inertia).
ARMATURE_KRO100 = _KRO100.output_armature_kgm2
ARMATURE_KRO80 = _KRO80.output_armature_kgm2
ARMATURE_KRO80x2 = ARMATURE_KRO80 * 2  # parallel ankle

# Upper-body arm actuator (placeholder values, not module-mapped).
ARMATURE_ARM = 0.1

# PD gains from a 2nd-order target model using the joint effective inertia J.
#
# Assumed (single-DOF) joint dynamics:      J * qdd = tau
# PD torque (tracking a constant target):   tau = kp*(q_des - q) - kd*qdot
# Error dynamics becomes:                   J*edd + kd*ed + kp*e = 0
#
# Match to standard 2nd-order form:
#   edd + 2*zeta*wn*ed + wn^2*e = 0
# -> kp = J * wn^2
# -> kd = 2 * zeta * J * wn
#
# NOTE:
# - Here we use "armature" as J (kg*m^2). In MuJoCo, armature adds to the joint inertia.
# - wn is angular natural frequency in rad/s (NOT Hz).
# - If you want a 10 Hz natural frequency fn, then wn = 2*pi*fn.

# FN_HZ = 4.0                    # desired natural frequency in Hz (cycles/s)
# WN_RAD_S = 2.0 * 3.1415926535 * FN_HZ   # angular natural frequency wn in rad/s
# ZETA = 2.0                      # damping ratio (over-damped)

FN_HZ = 4  # desired natural frequency in Hz (cycles/s)
WN_RAD_S = 2.0 * 3.1415926535 * FN_HZ  # angular natural frequency wn in rad/s
ZETA = 0.7  # damping ratio (over-damped)
# Stiffness (kp): N*m/rad
STIFFNESS_KRO100 = ARMATURE_KRO100 * (WN_RAD_S**2)
STIFFNESS_KRO80 = ARMATURE_KRO80 * (WN_RAD_S**2)
STIFFNESS_KRO80x2 = ARMATURE_KRO80x2 * (WN_RAD_S**2)
STIFFNESS_ARM = ARMATURE_ARM * (WN_RAD_S**2)

# Damping (kd): N*m*s/rad
DAMPING_KRO100 = 2.0 * ZETA * ARMATURE_KRO100 * WN_RAD_S
DAMPING_KRO80 = 2.0 * ZETA * ARMATURE_KRO80 * WN_RAD_S
DAMPING_KRO80x2 = 2.0 * ZETA * ARMATURE_KRO80x2 * WN_RAD_S
DAMPING_ARM = 2.0 * ZETA * ARMATURE_ARM * WN_RAD_S

# Effort limits from module specs (peak torque).
mode = "peak"
EFFORT_KRO100 = _KRO100.torque_limit_nm(mode)
EFFORT_KRO80 = _KRO80.torque_limit_nm(mode)
EFFORT_KRO80x2 = EFFORT_KRO80 * 2  # parallel ankle
EFFORT_ARM = 50.0  # upper-body placeholder

# Actuator groups mapped to joint regex patterns.
P1_ACTUATOR_KRO100 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "waist_yaw_joint",
    ".*_hip_pitch_joint",
    ".*_knee_joint",
  ),
  stiffness=STIFFNESS_KRO100,
  damping=DAMPING_KRO100,
  effort_limit=EFFORT_KRO100,
  armature=ARMATURE_KRO100,
)

P1_ACTUATOR_KRO80 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
  ),
  stiffness=STIFFNESS_KRO80,
  damping=DAMPING_KRO80,
  effort_limit=EFFORT_KRO80,
  armature=ARMATURE_KRO80,
)

P1_ACTUATOR_KRO80x2 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
  ),
  stiffness=STIFFNESS_KRO80x2,
  damping=DAMPING_KRO80x2,
  effort_limit=EFFORT_KRO80x2,
  armature=ARMATURE_KRO80x2,
)

P1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_yaw_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=STIFFNESS_ARM,
  damping=DAMPING_ARM,
  effort_limit=EFFORT_ARM,
  armature=ARMATURE_ARM,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.835),
  joint_pos={
    ".*_hip_pitch_joint": 0.0,
    ".*_knee_joint": 0.0,
    ".*_ankle_pitch_joint": 0.0,
    ".*_shoulder_pitch_joint": 0.0,
    ".*_elbow_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.83),
  joint_pos={
    "left_hip_pitch_joint": 0.15,
    "right_hip_pitch_joint": -0.15,
    ".*_knee_joint": 0.25,
    "left_ankle_pitch_joint": -0.10,
    "right_ankle_pitch_joint": -0.10,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)


##
# Final config.
##

P1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    P1_ACTUATOR_KRO100,
    P1_ACTUATOR_KRO80,
    P1_ACTUATOR_KRO80x2,
    P1_ACTUATOR_ARM,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_p1_robot_cfg() -> EntityCfg:
  """Get a fresh KIMM P1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=P1_ARTICULATION,
  )


P1_ACTION_SCALE: dict[str, float] = {}
for a in P1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    P1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  cfg = get_p1_robot_cfg()

  # Modify spec for viewer: fix base and add ground.
  original_spec_fn = cfg.spec_fn

  def get_viewer_spec() -> mujoco.MjSpec:
    spec = original_spec_fn()
    # Fix base by removing freejoint and setting global position.
    try:
      base = spec.body("base_link")
      base.pos = cfg.init_state.pos  # Use the height from init_state.
      for j in list(base.joints):
        if j.type == mujoco.mjtJoint.mjJNT_FREE or j.name == "floating_base_joint":
          spec.delete(j)
    except Exception as e:
      print(f"Base fix failed: {e}")

    # Add ground plane, light, and sky.
    spec.worldbody.add_geom(
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=[50, 50, 0.05],
      rgba=[0.1, 0.1, 0.1, 1],
    )
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
    return spec

  cfg.spec_fn = get_viewer_spec
  robot = Entity(cfg)

  model = robot.spec.compile()

  print(f"\n{'=' * 100}")
  print(f"Natural Frequency (Hz): {WN_RAD_S / (2 * 3.1415926535):.1f}")
  print(f"Damping Ratio: {ZETA:.1f}")
  print(f"Total Actuators: {model.nu}")
  print(
    f"{'ID':<3} | {'Name':<25} | {'Stiffness':<10} | {'Damping':<10} | {'Effort':<10} | {'Armature':<10} | {'Method':<10}"
  )
  print(
    f"{'-' * 3} | {'-' * 25} | {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10}"
  )

  for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)

    # Extract params
    stiff = model.actuator_gainprm[i, 0]
    damping = -model.actuator_biasprm[i, 2]  # Damping is negative in biasprm
    effort = (
      model.actuator_forcerange[i, 1]
      if model.actuator_forcelimited[i]
      else float("inf")
    )

    # Get joint armature from transmission target
    trnid = model.actuator_trnid[i, 0]
    joint_armature = model.dof_armature[model.jnt_dofadr[trnid]]

    # Check method (position vs motor)
    method = (
      "Pos" if model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED else "Motor"
    )

    print(
      f"{i:<3} | {name:<25} | {stiff:<10.1f} | {damping:<10.1f} | {effort:<10.1f} | {joint_armature:<10.4f} | {method:<10}"
    )
  print(f"{'=' * 100}\n")

  viewer.launch(model)
