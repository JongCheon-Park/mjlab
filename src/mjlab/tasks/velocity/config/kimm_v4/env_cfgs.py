"""KIMM V4 velocity environment — minimal common walking reward set.

표준 보행 RL reward만 활성 (~9개). 학습 안 되면 reward 문제 아니라
robot/PD/scale 문제로 좁혀짐.

활성 reward:
  - track_linear_velocity (5.0)    : x/y velocity 추종
  - track_angular_velocity (5.0)   : yaw rate 추종
  - upright (1.0)                  : torso 꼿꼿
  - base_height (1.0)              : 키 유지
  - pose (1.0)                     : default 자세 retention
  - biped_air_time (1.0)           : 발 들기 (single-support)
  - action_rate_l2 (-0.02)         : action smoothness
  - dof_pos_limits (-1.0)          : 관절 limit
  - self_collisions (-1.0)         : 자기충돌

비활성 (weight=0):
  - bilateral_symmetry, energy_power, alive_bonus,
  - forward_straight_gait, lateral_step_lead,
  - heel/toe pattern × 8,
  - hip_roll_torque_l2, ankle_roll_vel_l2,
  - foot_clearance, foot_swing_height, foot_slip, foot_flat,
  - thigh_swing_target, soft_landing, body_ang_vel, angular_momentum,
  - biped_first_swing_foot, biped_double_support_time, biped_standing_stability,
  - joint_effort_limit, joint_torque_rate

V4-specific 유지:
  - knee_joint patterns (P1 스타일)
  - target_height = 0.75 (deep KNEES_BENT)
  - PD: X12 Kp=509, X8 Kp=250
  - Action scale: X12 = 0.25 rad
"""

import math
import re

from mjlab.asset_zoo.robots import V4_ACTION_SCALE, get_v4_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import (
  JointPositionActionCfg,
  JointPositionHoldActionCfg,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

LEG_JOINT_PATTERNS = (
  r".*_hip_.*_joint",
  r".*_knee_joint",
  r".*_ankle_.*_joint",
)
WAIST_JOINT_PATTERNS = (r"waist_yaw_joint",)


def _build_joint_patterns(include_waist: bool) -> tuple[str, ...]:
  patterns = list(LEG_JOINT_PATTERNS)
  if include_waist:
    patterns.extend(WAIST_JOINT_PATTERNS)
  return tuple(patterns)


def _build_hold_joint_patterns(control_waist: bool) -> tuple[str, ...]:
  if control_waist:
    return ()
  return WAIST_JOINT_PATTERNS


def _filter_scale_dict(scale, patterns):
  return {k: v for k, v in scale.items() if any(re.fullmatch(p, k) for p in patterns)}


def _filter_posture_std(std, patterns):
  if not any("waist" in p for p in patterns):
    return {k: v for k, v in std.items() if "waist" not in k}
  return std


def v4_rough_env_cfg(
  play: bool = False,
  observe_waist: bool = True,
  control_waist: bool = True,
) -> ManagerBasedRlEnvCfg:
  """KIMM V4 rough-terrain — minimal walking reward set (9 active)."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 45

  cfg.scene.entities = {"robot": get_v4_robot_cfg()}

  # --- Disable all DR ---
  for ev_name in (
    "push_robot",
    "foot_friction",
    "encoder_bias",
    "encoder_bias_ankle_roll",
    "torso_pseudo_inertia",
    "link_pseudo_inertia",
    "joint_armature",
    "joint_friction",
    "joint_damping",
    "actuator_rfi",
    "fourbar_motor_armature",
  ):
    cfg.events.pop(ev_name, None)

  # --- Sensors ---
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "base_link"

  site_names = ("left_foot", "right_foot")

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
    history_length=0,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="torso_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="torso_link", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # --- Action / observation filtering ---
  action_patterns = _build_joint_patterns(control_waist)
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = action_patterns
  joint_pos_action.scale = _filter_scale_dict(V4_ACTION_SCALE, action_patterns)

  hold_patterns = _build_hold_joint_patterns(control_waist)
  if hold_patterns:
    cfg.actions["joint_pos_hold"] = JointPositionHoldActionCfg(
      entity_name="robot",
      actuator_names=hold_patterns,
    )
  else:
    cfg.actions.pop("joint_pos_hold", None)

  obs_patterns = _build_joint_patterns(observe_waist)
  obs_asset_cfg = SceneEntityCfg("robot", joint_names=obs_patterns)
  for group_name in ("actor", "critic"):
    for term_name in ("joint_pos", "joint_vel"):
      if term_name in cfg.observations[group_name].terms:
        term = cfg.observations[group_name].terms[term_name]
        if term.params is None:
          term.params = {}
        term.params["asset_cfg"] = obs_asset_cfg

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.20

  # ===========================================================
  # === Minimal walking reward set (9 active) ===
  # ===========================================================

  # 1. track_linear_velocity (5.0) - V4 needs strong tracking signal
  cfg.rewards["track_linear_velocity"].weight = 5.0

  # 2. track_angular_velocity (5.0)
  cfg.rewards["track_angular_velocity"].weight = 5.0

  # 3. upright (1.0) - pitch 좁게 (forward lean 방지), roll 1° 엄격.
  # std_x = sin(5°): pitch 2.5°에서 78%, 5°에서 37% → 앞으로 기울임 페널티
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["upright"].params["weight_xy"] = (1.0, 1.0)
  cfg.rewards["upright"].params["std_xy"] = (
    math.sin(math.radians(5.0)),  # pitch: forward lean 방지
    math.sin(math.radians(1.0)),  # roll: 엄격
  )
  cfg.rewards["upright"].weight = 1.0

  # 4. base_height (1.0) - V4 standing torso z = 0.84
  cfg.rewards["base_height"].params["target_height"] = 0.84
  cfg.rewards["base_height"].weight = 1.0

  # 5. pose (1.0) - per-joint std (★ 유일한 변경 vs v4_t14).
  # hip_roll/ankle_roll 좁게 → 다리 벌림 / 발목 옆 흔들림 방지.
  # hip_pitch/knee 넓게 → stride 자유.
  cfg.rewards["pose"].weight = 1.0
  cfg.rewards["pose"].params["std_standing"] = _filter_posture_std(
    {
      r".*hip_pitch.*": 0.15,
      r".*hip_roll.*": 0.05,
      r".*hip_yaw.*": 0.05,
      r".*knee.*": 0.20,
      r".*ankle_pitch.*": 0.10,
      r".*ankle_roll.*": 0.10,
      r".*waist_yaw.*": 0.05,
    },
    action_patterns,
  )
  cfg.rewards["pose"].params["std_walking"] = _filter_posture_std(
    {
      r".*hip_pitch.*": 0.40,
      r".*hip_roll.*": 0.15,  # ← 다리 벌림 페널티
      r".*hip_yaw.*": 0.20,
      r".*knee.*": 0.70,
      r".*ankle_pitch.*": 0.30,
      r".*ankle_roll.*": 0.10,  # ← 발목 옆 페널티
      r".*waist_yaw.*": 0.20,
    },
    action_patterns,
  )
  cfg.rewards["pose"].params["std_running"] = _filter_posture_std(
    {
      r".*hip_pitch.*": 0.50,
      r".*hip_roll.*": 0.20,
      r".*hip_yaw.*": 0.20,
      r".*knee.*": 0.80,
      r".*ankle_pitch.*": 0.35,
      r".*ankle_roll.*": 0.15,
      r".*waist_yaw.*": 0.30,
    },
    action_patterns,
  )
  cfg.rewards["pose"].params["std_turning"] = _filter_posture_std(
    {
      r".*hip_pitch.*": 0.40,
      r".*hip_roll.*": 0.20,
      r".*hip_yaw.*": 0.60,  # turning은 hip_yaw 자유
      r".*knee.*": 0.60,
      r".*ankle_pitch.*": 0.25,
      r".*ankle_roll.*": 0.10,
      r".*waist_yaw.*": 0.20,
    },
    action_patterns,
  )
  cfg.rewards["pose"].params["asset_cfg"].joint_names = action_patterns

  # 6. biped_air_time (1.0, v4_t14 baseline) - 발 들기 보상.
  cfg.rewards["biped_air_time"].weight = 1.0

  # 7. action_rate_l2 (-0.02) - smoothness
  cfg.rewards["action_rate_l2"].weight = -0.02

  # 8. dof_pos_limits (-1.0) - 관절 limit
  cfg.rewards["dof_pos_limits"].weight = -1.0

  # 9. self_collisions (-1.0) - 자기충돌
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # 10. foot_clearance (-1.0) - swing 시 발이 0.15m target에 안 가면 페널티.
  # forward-lean+발 뒤 방지: 발이 지면 근처 머물면 페널티 → 제대로 들기 유도.
  cfg.rewards["foot_clearance"].params["target_height"] = 0.15
  cfg.rewards["foot_clearance"].weight = -1.0

  # 11. foot_swing_height (-1.0) - landing 시 peak < 0.15m 페널티.
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.15
  cfg.rewards["foot_swing_height"].weight = -1.0

  # ===========================================================
  # === 나머지 모두 OFF (weight=0) ===
  # ===========================================================
  for off_name in (
    "body_ang_vel",
    "angular_momentum",
    "joint_effort_limit",
    "joint_torque_rate",
    "biped_first_swing_foot",
    "biped_double_support_time",
    "biped_standing_stability",
    "foot_slip",
    "foot_flat",
    "thigh_swing_target",
    "soft_landing",
    "heelstrike_foot_pitch_pattern",
    "toeoff_foot_pitch_pattern",
    "heelstrike_knee_pattern",
    "toeoff_knee_pattern",
    "heelstrike_hip_pitch_pattern",
    "toeoff_hip_pitch_pattern",
    "heelstrike_hip_roll_pattern",
    "toeoff_hip_roll_pattern",
    "heelstrike_shoulder_pitch_pattern",
    "toeoff_shoulder_pitch_pattern",
    "toeoff_elbow_pattern",
  ):
    if off_name in cfg.rewards:
      cfg.rewards[off_name].weight = 0.0

  # V4 has no arms.
  cfg.rewards.pop("pose_upper_fixed", None)
  cfg.rewards.pop("wrist_joint_pos", None)

  # Foot sensor wiring (rewards weight=0 but asset_cfg still needs to be valid).
  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # --- Play mode ---
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def v4_flat_env_cfg(
  play: bool = False,
  observe_waist: bool = True,
  control_waist: bool = True,
) -> ManagerBasedRlEnvCfg:
  """KIMM V4 flat-terrain — minimal walking reward set."""
  cfg = v4_rough_env_cfg(
    play=play,
    observe_waist=observe_waist,
    control_waist=control_waist,
  )

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  for group in ("actor", "critic"):
    cfg.observations[group].terms.pop("height_scan", None)

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    cfg.curriculum.pop("command_vel", None)
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
