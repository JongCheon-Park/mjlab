# Author  : jojaebeom@kimm.re.kr
# Version : 0.0.2
# Date    : 2026-04-09

"""KIMM P1 constants for MJLAB training."""

import re
from copy import deepcopy
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import (
  BuiltinPositionActuatorCfg,
  FourbarAnkleGroupCfg,
  FourbarPdActuatorCfg,
  IdealPdActuatorCfg,
)
from mjlab.asset_zoo.robots.debug_viewer import (
  apply_fixed_base_viewer_defaults,
  apply_joint_limits_to_viewer_controls,
  resolve_viewer_actuator_gains,
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
ZETA = 2.0
P1_ACTUATOR_MODE: P1ActuatorMode = "implicit"

_P1_PARAMS = build_actuator_params(
  P1_MODULE_MAP,
  effort_mode="rated",  # peak / rated
  velocity_mode="rated",  # no-load / rated
  drivetrain_efficiency=1.5,
  pd_hz=FN_HZ,
  pd_damping_ratio=ZETA,
)

P1_KRO100: ActuatorParams = _P1_PARAMS["KRO100"]
P1_KRO80: ActuatorParams = _P1_PARAMS["KRO80"]
P1_KRO80x2: ActuatorParams = _P1_PARAMS["KRO80x2"]
P1_AK80: ActuatorParams = _P1_PARAMS["AK80"]
P1_AK70: ActuatorParams = _P1_PARAMS["AK70"]

# Actuator groups mapped to joint regex patterns.
P1_ACTUATOR_KRO100 = build_pd_actuator_cfg(
  (
    "waist_yaw_joint",
    ".*_hip_pitch_joint",
    ".*_knee_joint",
  ),
  P1_KRO100,
  P1_ACTUATOR_MODE,
)

P1_ACTUATOR_KRO80 = build_pd_actuator_cfg(
  (
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
  ),
  P1_KRO80,
  P1_ACTUATOR_MODE,
)

P1_ACTUATOR_ANKLE = FourbarPdActuatorCfg(
  target_names_expr=(),
  stiffness=P1_KRO80.stiffness,
  damping=P1_KRO80.damping,
  effort_limit=P1_KRO80.effort_limit_nm,
  motor_armature=P1_KRO80.armature_kgm2,
  robot="p1",
  mode="dynamic",
  ankles=(
    FourbarAnkleGroupCfg(
      side="left",
      joint_names=(
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
      ),
    ),
    FourbarAnkleGroupCfg(
      side="right",
      joint_names=(
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
      ),
    ),
  ),
)

P1_ACTUATOR_AK80 = build_pd_actuator_cfg(
  (
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
  ),
  P1_AK80,
  P1_ACTUATOR_MODE,
)

P1_ACTUATOR_AK70 = build_pd_actuator_cfg(
  (
    ".*_wrist_yaw_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_roll_joint",
  ),
  P1_AK70,
  P1_ACTUATOR_MODE,
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
    "left_hip_roll_joint": 0.04,
    "right_hip_roll_joint": -0.04,
    "left_hip_pitch_joint": 0.15,
    "right_hip_pitch_joint": -0.15,
    ".*_knee_joint": 0.25,
    "left_ankle_pitch_joint": -0.10,
    "right_ankle_pitch_joint": -0.10,
    "left_ankle_roll_joint": -0.04,
    "right_ankle_roll_joint": 0.04,
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
    P1_ACTUATOR_ANKLE,
    P1_ACTUATOR_AK80,
    P1_ACTUATOR_AK70,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_p1_robot_cfg() -> EntityCfg:
  """Get a fresh KIMM P1 robot configuration instance.

  Returns deep-copied nested configs so task-specific mutations do not leak
  between train/play variants registered in the task registry.
  """
  return EntityCfg(
    init_state=deepcopy(KNEES_BENT_KEYFRAME),
    collisions=(deepcopy(FULL_COLLISION),),
    spec_fn=get_spec,
    articulation=deepcopy(P1_ARTICULATION),
  )


P1_ACTION_SCALE: dict[str, float] = {}
for actuator_cfg in P1_ARTICULATION.actuators:
  effort_limit = actuator_cfg.effort_limit
  stiffness = actuator_cfg.stiffness
  assert effort_limit is not None
  for pattern in actuator_cfg.target_names_expr:
    P1_ACTION_SCALE[pattern] = 0.25 * effort_limit / stiffness


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


def _get_p1_explicit_actuator_params(
  model: mujoco.MjModel,
) -> dict[str, tuple[float, float, float, str]]:
  """Build actuator summary params from Python configs, including Fourbar."""
  all_act_cfgs = (
    P1_ACTUATOR_KRO100,
    P1_ACTUATOR_KRO80,
    P1_ACTUATOR_ANKLE,
    P1_ACTUATOR_AK80,
    P1_ACTUATOR_AK70,
  )
  explicit_params: dict[str, tuple[float, float, float, str]] = {}
  for actuator_id in range(model.nu):
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    for act_cfg in all_act_cfgs:
      matched = False
      for pattern in act_cfg.target_names_expr:
        if re.fullmatch(pattern, actuator_name):
          effort = act_cfg.effort_limit
          method = (
            "Fourbar"
            if isinstance(act_cfg, FourbarPdActuatorCfg)
            else "Explicit"
            if P1_ACTUATOR_MODE == "explicit"
            else "Implicit"
          )
          explicit_params[actuator_name] = (
            act_cfg.stiffness,
            act_cfg.damping,
            float("inf") if effort is None else effort,
            method,
          )
          matched = True
          break
      if matched:
        break
  return explicit_params


def get_p1_actuator_summary_rows(model: mujoco.MjModel) -> list[dict[str, float | str]]:
  """Return formatted actuator summary rows for debug printing."""
  explicit_params = _get_p1_explicit_actuator_params(model)
  summary_rows: list[dict[str, float | str]] = []
  for actuator_id in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    trnid = model.actuator_trnid[actuator_id, 0]
    joint_armature = model.dof_armature[model.jnt_dofadr[trnid]]
    stiff, damping, effort, method = explicit_params[name]

    scale = 0.0
    for pattern, value in P1_ACTION_SCALE.items():
      if re.fullmatch(pattern, name):
        scale = value
        break

    summary_rows.append(
      {
        "id": actuator_id,
        "name": name,
        "stiffness": stiff,
        "damping": damping,
        "effort": effort,
        "scale": scale,
        "armature": joint_armature,
        "method": method,
      }
    )
  return summary_rows


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  cfg = get_p1_robot_cfg()
  cfg.spec_fn = get_p1_viewer_spec
  robot = Entity(cfg)
  apply_joint_limits_to_viewer_controls(
    robot.spec,
    actuator_gains=resolve_viewer_actuator_gains(
      robot.spec, cfg.articulation.actuators
    ),
  )

  model = robot.spec.compile()

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

  for row in get_p1_actuator_summary_rows(model):
    print(
      f"{int(row['id']):<3} | {str(row['name']):<30} | {float(row['stiffness']):<10.1f}"
      f" | {float(row['damping']):<10.1f} | {float(row['effort']):<10.1f}"
      f" | {float(row['scale']):<10.4f} | {float(row['armature']):<10.4f}"
      f" | {str(row['method']):<10}"
    )
  print(f"{'=' * 125}\n")

  viewer.launch(model)
