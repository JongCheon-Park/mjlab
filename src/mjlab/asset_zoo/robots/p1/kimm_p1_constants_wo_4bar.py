# Author  : jojaebeom@kimm.re.kr
# Version : 0.0.1
# Date    : 2026-02-27

"""KIMM P1 constants for MJLAB training."""

import re
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import (
  BuiltinPositionActuatorCfg,
  IdealPdActuatorCfg,
)
from mjlab.asset_zoo.robots.debug_viewer import (
  apply_fixed_base_viewer_defaults,
  apply_joint_limits_to_viewer_controls,
)
from mjlab.asset_zoo.robots.p1.kimm_p1_actuators import (
  P1_MODULE_MAP,
  ActuatorParams,
  P1ActuatorMode,
  build_actuator_params,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


def build_pd_actuator_cfg(
  target_names_expr: tuple[str, ...],
  params: ActuatorParams,
  actuator_mode: P1ActuatorMode,
) -> BuiltinPositionActuatorCfg | IdealPdActuatorCfg:
  """Build a PD actuator config in implicit or explicit mode."""
  if actuator_mode == "implicit":
    return BuiltinPositionActuatorCfg(
      target_names_expr=target_names_expr,
      stiffness=params.stiffness,
      damping=params.damping,
      effort_limit=params.effort_limit_nm,
      armature=params.armature_kgm2,
    )
  if actuator_mode == "explicit":
    return IdealPdActuatorCfg(
      target_names_expr=target_names_expr,
      stiffness=params.stiffness,
      damping=params.damping,
      effort_limit=params.effort_limit_nm,
      armature=params.armature_kgm2,
    )
  raise ValueError(
    f"Invalid actuator_mode={actuator_mode!r}. Expected one of: 'implicit', 'explicit'."
  )


##
# MJCF and assets.
##

P1_XML: Path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "p1" / "xmls" / "kimm_p1.xml"
assert P1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(P1_XML))


##
# Actuator config.
##

FN_HZ = 5.0
ZETA = 0.7
P1_ACTUATOR_MODE: P1ActuatorMode = "implicit"  # "implicit" | "explicit"

_P1_PARAMS = build_actuator_params(
  P1_MODULE_MAP,
  effort_mode="rated",  # peak / rated
  velocity_mode="rated",  # no-load / rated
  drivetrain_efficiency=1.0,
  pd_hz=FN_HZ,
  pd_damping_ratio=ZETA,
)

P1_KRO100: ActuatorParams = _P1_PARAMS["KRO100"]
P1_KRO80: ActuatorParams = _P1_PARAMS["KRO80"]
P1_KRO80x2: ActuatorParams = _P1_PARAMS["KRO80x2"]
P1_AK80: ActuatorParams = _P1_PARAMS["AK80"]
P1_AK70: ActuatorParams = _P1_PARAMS["AK70"]

if P1_ACTUATOR_MODE == "implicit":
  _P1_PD_ACTUATOR_CFG_CLS = BuiltinPositionActuatorCfg
elif P1_ACTUATOR_MODE == "explicit":
  _P1_PD_ACTUATOR_CFG_CLS = IdealPdActuatorCfg
else:
  raise ValueError(
    f"Invalid P1_ACTUATOR_MODE={P1_ACTUATOR_MODE!r}. "
    "Expected one of: 'implicit', 'explicit'."
  )

# Actuator groups mapped to joint regex patterns.
P1_ACTUATOR_KRO100 = _P1_PD_ACTUATOR_CFG_CLS(
  target_names_expr=(
    "waist_yaw_joint",
    ".*_hip_pitch_joint",
    ".*_knee_joint",
  ),
  stiffness=P1_KRO100.stiffness,
  damping=P1_KRO100.damping,
  effort_limit=P1_KRO100.effort_limit_nm,
  armature=P1_KRO100.armature_kgm2,
)

P1_ACTUATOR_KRO80 = _P1_PD_ACTUATOR_CFG_CLS(
  target_names_expr=(
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
  ),
  stiffness=P1_KRO80.stiffness,
  damping=P1_KRO80.damping,
  effort_limit=P1_KRO80.effort_limit_nm,
  armature=P1_KRO80.armature_kgm2,
)

P1_ACTUATOR_KRO80x2 = _P1_PD_ACTUATOR_CFG_CLS(
  target_names_expr=(
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
  ),
  stiffness=P1_KRO80x2.stiffness,
  damping=P1_KRO80x2.damping,
  effort_limit=P1_KRO80x2.effort_limit_nm,
  armature=P1_KRO80x2.armature_kgm2,
)

P1_ACTUATOR_AK80 = _P1_PD_ACTUATOR_CFG_CLS(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
  ),
  stiffness=P1_AK80.stiffness,
  damping=P1_AK80.damping,
  effort_limit=P1_AK80.effort_limit_nm,
  armature=P1_AK80.armature_kgm2,
)

P1_ACTUATOR_AK70 = _P1_PD_ACTUATOR_CFG_CLS(
  target_names_expr=(
    ".*_wrist_yaw_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=P1_AK70.stiffness,
  damping=P1_AK70.damping,
  effort_limit=P1_AK70.effort_limit_nm,
  armature=P1_AK70.armature_kgm2,
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
    "left_shoulder_roll_joint": 0.05,
    "right_shoulder_roll_joint": -0.05,
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
    P1_ACTUATOR_AK80,
    P1_ACTUATOR_AK70,
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
  assert isinstance(a, BuiltinPositionActuatorCfg | IdealPdActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    P1_ACTION_SCALE[n] = 0.25 * e / s


def get_p1_viewer_spec() -> mujoco.MjSpec:
  """Get a debug viewer spec with fixed base and white background."""
  cfg = get_p1_robot_cfg()
  spec = get_spec()
  apply_fixed_base_viewer_defaults(
    spec,
    base_body_name="base_link",
    base_pos=cfg.init_state.pos,
  )
  return spec


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  cfg = get_p1_robot_cfg()
  cfg.spec_fn = get_p1_viewer_spec
  robot = Entity(cfg)
  apply_joint_limits_to_viewer_controls(robot.spec)

  model = robot.spec.compile()

  # For explicit mode: kp/kd live in the cfg objects, not in the MuJoCo model.
  # Build a lookup: actuator_name -> (stiffness, damping, effort_limit).
  _explicit_params: dict[str, tuple[float, float, float]] = {}
  if P1_ACTUATOR_MODE == "explicit":
    _all_act_cfgs = (
      P1_ACTUATOR_KRO100,
      P1_ACTUATOR_KRO80,
      P1_ACTUATOR_KRO80x2,
      P1_ACTUATOR_AK80,
      P1_ACTUATOR_AK70,
    )
    for _j in range(model.nu):
      _act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, _j)
      _matched = False
      for _act_cfg in _all_act_cfgs:
        if _matched:
          break
        for _pattern in _act_cfg.target_names_expr:
          if re.fullmatch(_pattern, _act_name):
            _eff = _act_cfg.effort_limit
            _explicit_params[_act_name] = (
              _act_cfg.stiffness,
              _act_cfg.damping,
              float("inf") if _eff is None else _eff,
            )
            _matched = True
            break

  print(f"\n{'=' * 100}")
  print(f"ActuatorMode: {P1_ACTUATOR_MODE}")
  print(f"Natural Frequency (Hz): {FN_HZ:.1f}")
  print(f"Damping Ratio: {ZETA:.1f}")
  print(f"Total Actuators: {model.nu}")
  print(
    f"{'ID':<3} | {'Name':<30} | {'Stiffness':<10} | {'Damping':<10} | {'Effort':<10} | {'Scale':<10} | {'Armature':<10} | {'Method':<10}"
  )
  print(
    f"{'-' * 3} | {'-' * 30} | {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10}"
  )

  for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)

    # Get joint armature from transmission target (valid for both modes).
    trnid = model.actuator_trnid[i, 0]
    joint_armature = model.dof_armature[model.jnt_dofadr[trnid]]

    if P1_ACTUATOR_MODE == "implicit":
      # <position> actuator: kp/kd are baked into the MuJoCo model.
      stiff = model.actuator_gainprm[i, 0]
      damping = -model.actuator_biasprm[i, 2]
      effort = (
        model.actuator_forcerange[i, 1]
        if model.actuator_forcelimited[i]
        else float("inf")
      )
      method = "Implicit"
    else:
      # <motor> actuator: kp/kd are handled in Python; read from cfg.
      stiff, damping, effort = _explicit_params[name]
      method = "Explicit"

    # Use P1_ACTION_SCALE for the scale lookup
    scale = 0.0
    for pattern, val in P1_ACTION_SCALE.items():
      if re.fullmatch(pattern, name):
        scale = val
        break

    print(
      f"{i:<3} | {name:<30} | {stiff:<10.1f} | {damping:<10.1f} | {effort:<10.1f} | {scale:<10.4f} | {joint_armature:<10.4f} | {method:<10}"
    )
  print(f"{'=' * 125}\n")

  viewer.launch(model)
