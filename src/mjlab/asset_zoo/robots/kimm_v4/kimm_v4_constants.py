# Author  : jojaebeom@kimm.re.kr
# Version : 0.0.1
# Date    : 2026-05-20

"""KIMM V4 constants for MJLAB training."""

import re
from copy import deepcopy
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import (
  BuiltinPdActuatorCfg,
  IdealPdActuatorCfg,
)
from mjlab.asset_zoo.robots.debug_viewer import (
  apply_fixed_base_viewer_defaults,
  apply_joint_limits_to_viewer_controls,
  resolve_viewer_actuator_gains,
)
from mjlab.asset_zoo.robots.kimm_v4.kimm_v4_actuators import (
  V4_MODULE_MAP,
  ActuatorParams,
  V4ActuatorMode,
  build_actuator_params,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


def build_pd_actuator_cfg(
  target_names_expr: tuple[str, ...],
  params: ActuatorParams,
  actuator_mode: V4ActuatorMode,
) -> BuiltinPdActuatorCfg | IdealPdActuatorCfg:
  """Build a PD actuator config in implicit or explicit mode."""
  if actuator_mode == "implicit":
    return BuiltinPdActuatorCfg(
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

V4_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "kimm_v4" / "xmls" / "kimm_v4.xml"
)
assert V4_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(V4_XML))


##
# Actuator config.
##

FN_HZ = 5.0
ZETA = 2.0
V4_ACTUATOR_MODE: V4ActuatorMode = "implicit"

_X12_PARAMS = build_actuator_params(
  V4_MODULE_MAP,
  effort_mode="rated",
  velocity_mode="rated",
  drivetrain_efficiency=2.0,
  pd_hz=FN_HZ,
  pd_damping_ratio=ZETA,
)
_X8_PARAMS = build_actuator_params(
  V4_MODULE_MAP,
  effort_mode="rated",
  velocity_mode="rated",
  drivetrain_efficiency=2.0,
  pd_hz=FN_HZ,
  pd_damping_ratio=ZETA,
)

V4_X8: ActuatorParams = _X8_PARAMS["X8"]
V4_X12: ActuatorParams = _X12_PARAMS["X12"]

# The URDF-derived actuator classes in the source V4 MJCF use the RMD X12 class
# for hip/knee pitch and the RMD X8 class for waist, hip roll/yaw, and ankles.
# This shared variant intentionally models ankle pitch/roll as plain PD joints
# without fourbar motor-space coupling.
V4_ACTUATOR_X12 = build_pd_actuator_cfg(
  (
    ".*_hip_pitch_joint",
    ".*_knee_joint",
  ),
  V4_X12,
  V4_ACTUATOR_MODE,
)

V4_ACTUATOR_X8 = build_pd_actuator_cfg(
  (
    "waist_yaw_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
  ),
  V4_X8,
  V4_ACTUATOR_MODE,
)


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.84),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.84),
  joint_pos={
    "waist_yaw_joint": 0.0,
    "right_hip_pitch_joint": -0.0872664601,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.1745329201,
    "right_ankle_pitch_joint": -0.0872664601,
    "right_ankle_roll_joint": 0.0,
    "left_hip_pitch_joint": 0.0872664601,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.1745329201,
    "left_ankle_pitch_joint": -0.0872664601,
    "left_ankle_roll_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-8]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-8]_collision$": 1},
  friction={r"^(left|right)_foot[1-8]_collision$": (0.6,)},
)


##
# Final config.
##

V4_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    V4_ACTUATOR_X12,
    V4_ACTUATOR_X8,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_v4_robot_cfg() -> EntityCfg:
  """Get a fresh KIMM V4 robot configuration instance.

  Returns deep-copied nested configs so task-specific mutations do not leak
  between train/play variants registered in the task registry.
  """
  return EntityCfg(
    init_state=deepcopy(KNEES_BENT_KEYFRAME),
    collisions=(deepcopy(FULL_COLLISION),),
    spec_fn=get_spec,
    articulation=deepcopy(V4_ARTICULATION),
  )


V4_ACTION_SCALE: dict[str, float] = {}
for actuator_cfg in V4_ARTICULATION.actuators:
  assert isinstance(actuator_cfg, (BuiltinPdActuatorCfg, IdealPdActuatorCfg))
  effort_limit = actuator_cfg.effort_limit
  stiffness = actuator_cfg.stiffness
  assert effort_limit is not None
  for pattern in actuator_cfg.target_names_expr:
    V4_ACTION_SCALE[pattern] = 0.25 * effort_limit / stiffness


def get_v4_viewer_spec() -> mujoco.MjSpec:
  """Get a debug viewer spec with fixed base and white background."""
  cfg = get_v4_robot_cfg()
  spec = get_spec()
  apply_fixed_base_viewer_defaults(
    spec,
    base_body_name="torso_link",
    base_pos=cfg.init_state.pos,
  )
  return spec


def _get_v4_explicit_actuator_params(
  model: mujoco.MjModel,
) -> dict[str, tuple[float, float, float, str]]:
  """Build actuator summary params from Python configs."""
  all_act_cfgs = (V4_ACTUATOR_X12, V4_ACTUATOR_X8)
  explicit_params: dict[str, tuple[float, float, float, str]] = {}
  for actuator_id in range(model.nu):
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    target_name = _summary_target_name(actuator_name)
    if target_name is None:
      continue
    for act_cfg in all_act_cfgs:
      matched = False
      for pattern in act_cfg.target_names_expr:
        if re.fullmatch(pattern, target_name):
          effort = act_cfg.effort_limit
          method = "Explicit" if V4_ACTUATOR_MODE == "explicit" else "Implicit"
          explicit_params[target_name] = (
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


def _summary_target_name(actuator_name: str) -> str | None:
  """Map MuJoCo actuator element names to one summary row per controlled target."""
  if actuator_name.endswith("_pd_pos"):
    return actuator_name.removesuffix("_pd_pos")
  if actuator_name.endswith("_pd_vel"):
    return None
  return actuator_name


def get_v4_actuator_summary_rows(model: mujoco.MjModel) -> list[dict[str, float | str]]:
  """Return formatted actuator summary rows for debug printing."""
  explicit_params = _get_v4_explicit_actuator_params(model)
  summary_rows: list[dict[str, float | str]] = []
  for actuator_id in range(model.nu):
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    name = _summary_target_name(actuator_name)
    if name is None:
      continue
    trnid = model.actuator_trnid[actuator_id, 0]
    joint_armature = float(model.dof_armature[model.jnt_dofadr[trnid]])
    stiff, damping, effort, method = explicit_params[name]

    scale = 0.0
    for pattern, value in V4_ACTION_SCALE.items():
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

  cfg = get_v4_robot_cfg()
  assert cfg.articulation is not None
  cfg.spec_fn = get_v4_viewer_spec
  robot = Entity(cfg)
  apply_joint_limits_to_viewer_controls(
    robot.spec,
    actuator_gains=resolve_viewer_actuator_gains(
      robot.spec, cfg.articulation.actuators
    ),
  )

  model = robot.spec.compile()

  print(f"\n{'=' * 100}")
  print(f"ActuatorMode: {V4_ACTUATOR_MODE}")
  print(f"Natural Frequency (Hz): {FN_HZ:.1f}")
  print(f"Damping Ratio: {ZETA:.1f}")
  print(f"Total Actuators: {model.nu}")
  print(
    f"{'ID':<3} | {'Name':<30} | {'Stiffness':<10} | {'Damping':<10} | {'Effort':<10} | {'Scale':<10} | {'Armature':<10} | {'Method':<10}"
  )
  print(
    f"{'-' * 3} | {'-' * 30} | {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10}"
  )

  for row in get_v4_actuator_summary_rows(model):
    print(
      f"{int(row['id']):<3} | {str(row['name']):<30} | {float(row['stiffness']):<10.1f}"
      f" | {float(row['damping']):<10.1f} | {float(row['effort']):<10.1f}"
      f" | {float(row['scale']):<10.4f} | {float(row['armature']):<10.4f}"
      f" | {str(row['method']):<10}"
    )
  print(f"{'=' * 125}\n")

  viewer.launch(model)
