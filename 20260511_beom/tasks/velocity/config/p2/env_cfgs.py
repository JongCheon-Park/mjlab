"""KIMM P2 velocity environment configurations."""

import math
import re

from mjlab.asset_zoo.robots.kimm_p2.kimm_p2_constants import P2_ACTUATOR_MODE

from mjlab.asset_zoo.robots import P2_ACTION_SCALE, get_p2_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import (
  JointPositionActionCfg,
  JointPositionHoldActionCfg,
)
from mjlab.managers.curriculum_manager import CurriculumTermCfg
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
ARM_JOINT_PATTERNS = (
  r".*_shoulder_.*_joint",
  r".*_elbow_joint",
  r".*_wrist_.*_joint",
)

P2_EXPLICIT_TIMESTEP = 0.001
P2_EXPLICIT_DECIMATION = 20


def _build_joint_patterns(
  include_waist: bool,
  include_arms: bool,
) -> tuple[str, ...]:
  """Build a tuple of joint name regex patterns based on flags."""
  if include_waist and include_arms:
    return (".*",)
  patterns = list(LEG_JOINT_PATTERNS)
  if include_waist:
    patterns.extend(WAIST_JOINT_PATTERNS)
  if include_arms:
    patterns.extend(ARM_JOINT_PATTERNS)
  return tuple(patterns)


def _build_hold_joint_patterns(
  control_waist: bool,
  control_arms: bool,
) -> tuple[str, ...]:
  """Build joint patterns for actuator groups held at default pose."""
  patterns = []
  if not control_waist:
    patterns.extend(WAIST_JOINT_PATTERNS)
  if not control_arms:
    patterns.extend(ARM_JOINT_PATTERNS)
  return tuple(patterns)


def _filter_scale_dict(
  scale: dict[str, float],
  patterns: tuple[str, ...],
) -> dict[str, float]:
  """Filter the action scale dict to only include keys matching patterns."""
  if patterns == (".*",):
    return scale
  return {k: v for k, v in scale.items() if any(re.fullmatch(p, k) for p in patterns)}


def _filter_posture_std(
  std: dict[str, float],
  patterns: tuple[str, ...],
) -> dict[str, float]:
  """Filter posture std dict to remove entries for disabled joints."""
  if patterns == (".*",):
    return std
  waist_keys = {r".*waist_yaw.*"}
  arm_keys = {
    r".*shoulder_pitch.*",
    r".*shoulder_roll.*",
    r".*shoulder_yaw.*",
    r".*elbow.*",
    r".*wrist.*",
  }
  excluded = set()
  if not any(re.fullmatch(r"waist_yaw_joint", p) for p in patterns):
    excluded |= waist_keys
  if not any("shoulder" in p for p in patterns):
    excluded |= arm_keys
  return {k: v for k, v in std.items() if k not in excluded}


def kimm_p2_rough_env_cfg(
  play: bool = False,
  observe_waist: bool = True,
  observe_arms: bool = True,
  control_waist: bool = True,
  control_arms: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create KIMM P2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  if P2_ACTUATOR_MODE == "explicit":
    cfg.sim.mujoco.timestep = P2_EXPLICIT_TIMESTEP
    cfg.decimation = P2_EXPLICIT_DECIMATION

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 45

  cfg.scene.entities = {"robot": get_p2_robot_cfg()}

  # Set raycast sensor frame to P2 base_link (root body).
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=site_name, entity="robot") for site_name in site_names
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
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
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

  # --- Action space filtering ---
  action_patterns = _build_joint_patterns(control_waist, control_arms)
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = action_patterns
  joint_pos_action.scale = _filter_scale_dict(P2_ACTION_SCALE, action_patterns)

  hold_patterns = _build_hold_joint_patterns(control_waist, control_arms)
  if hold_patterns:
    cfg.actions["joint_pos_hold"] = JointPositionHoldActionCfg(
      entity_name="robot",
      actuator_names=hold_patterns,
    )
  else:
    cfg.actions.pop("joint_pos_hold", None)

  # --- Observation filtering ---
  obs_patterns = _build_joint_patterns(observe_waist, observe_arms)
  if obs_patterns != (".*",):
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
  twist_cmd.viz.z_offset = 1.15
  # twist_cmd.drag_wrench = mdp.VelocityDragWrenchCfg(
  #   asset_cfg=SceneEntityCfg("robot", body_names=("base_link",)),
  #   enable_prob=0.5,
  #   drag_force_range=(0.0, 10.0),
  #   drag_damping_range=(1.0, 10.0),
  #   start_steps=20_000 * 24,
  #   force_clip_xy=250.0,
  # )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

  cfg.events["torso_pseudo_inertia"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["torso_pseudo_inertia"].params["alpha_range"] = (0.1, 0.13)
  cfg.events["torso_pseudo_inertia"].params["t1_range"] = (-0.025, 0.0)
  cfg.events["link_pseudo_inertia"].params["asset_cfg"].body_names = (
    ".*_hip_yaw_link",
    ".*_knee_link",
    ".*_ankle_roll_link",
  )
  cfg.events["link_pseudo_inertia"].params["alpha_range"] = (0, 0.05)

  # Actuator randomization
  # (0=KRO100, 1=KRO80, 2=ANKLE, 3=AK80, 4=AK70).
  dr_actuator_ids = [0, 1, 2]
  if control_arms:
    dr_actuator_ids.extend([3, 4])
  # cfg.events["pd_gains"].params["asset_cfg"].actuator_ids = dr_actuator_ids
  cfg.events["actuator_rfi"].params["asset_cfg"].actuator_ids = dr_actuator_ids
  cfg.events["actuator_motor_bandwidth"] = EventTermCfg(
    mode="startup",
    func=dr.actuator_motor_bandwidth,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_ids=dr_actuator_ids),
      "enabled": False,
    },
  )

  robot_cfg = cfg.scene.entities["robot"]
  assert robot_cfg.articulation is not None
  for actuator_id in dr_actuator_ids:
    actuator_cfg = robot_cfg.articulation.actuators[actuator_id]
    actuator_cfg.delay_min_lag = 0
    actuator_cfg.delay_max_lag = 4
    # Upstream-style direct actuator delay: slow-varying stochastic jitter.
    actuator_cfg.delay_hold_prob = 0.5
    actuator_cfg.delay_update_period = 20
    actuator_cfg.delay_per_env_phase = False

  # Exclude ankle DOFs from joint_armature DR — FourbarPdActuator manages its own
  # motor armature via D_o(q) = I_m * diag(J_inv^T @ J_inv).
  cfg.events["joint_armature"].params["asset_cfg"].joint_names = (r"^(?!.*ankle).*",)
  # Motor armature randomization for fourbar ankle actuator (index 2).
  cfg.events["fourbar_motor_armature"] = EventTermCfg(
    mode="startup",
    func=dr.fourbar_motor_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_ids=[2]),
      "ratio_range": (1.0, 2.0),
    },
  )
  cfg.rewards["pose"].weight = 2.0
  cfg.rewards["pose"].params["std_standing"] = _filter_posture_std(
    {
      # Lower body.
      r".*hip_pitch.*": 0.15,
      r".*hip_roll.*": 0.05,
      r".*hip_yaw.*": 0.05,
      r".*knee.*": 0.2,
      r".*ankle_pitch.*": 0.1,
      r".*ankle_roll.*": 0.1,
      # Waist.
      r".*waist_yaw.*": 0.05,
      # Arms.
      r".*shoulder_pitch.*": 0.05,
      r".*shoulder_roll.*": 0.05,
      r".*shoulder_yaw.*": 0.05,
      r".*elbow.*": 0.05,
      r".*wrist.*": 0.05,
    },
    action_patterns,
  )
  cfg.rewards["pose"].params["weight_standing"] = 5.0
  cfg.rewards["pose"].params["std_turning"] = _filter_posture_std(
    {
      # Lower body.
      r".*hip_pitch.*": 0.4,
      r".*hip_roll.*": 0.2,
      r".*hip_yaw.*": 0.3,
      r".*knee.*": 0.4,
      r".*ankle_pitch.*": 0.25,
      r".*ankle_roll.*": 0.1,
      # Waist.
      r".*waist_yaw.*": 0.2,
      # Arms.
      r".*shoulder_pitch.*": 0.15,
      r".*shoulder_roll.*": 0.15,
      r".*shoulder_yaw.*": 0.1,
      r".*elbow.*": 0.15,
      r".*wrist.*": 0.3,
    },
    action_patterns,
  )
  cfg.rewards["pose"].params["std_walking"] = _filter_posture_std(
    {
      # Lower body.
      r".*hip_pitch.*": 0.3,
      r".*hip_roll.*": 0.15,
      r".*hip_yaw.*": 0.15,
      r".*knee.*": 0.6,
      r".*ankle_pitch.*": 0.25,
      r".*ankle_roll.*": 0.1,
      # Waist.
      r".*waist_yaw.*": 0.2,
      # Arms.
      r".*shoulder_pitch.*": 0.4,
      r".*shoulder_roll.*": 0.15,
      r".*shoulder_yaw.*": 0.1,
      r".*elbow.*": 0.15,
      r".*wrist.*": 0.3,
    },
    action_patterns,
  )
  cfg.rewards["pose"].params["std_running"] = _filter_posture_std(
    {
      # Lower body.
      r".*hip_pitch.*": 0.5,
      r".*hip_roll.*": 0.2,
      r".*hip_yaw.*": 0.2,
      r".*knee.*": 0.8,
      r".*ankle_pitch.*": 0.35,
      r".*ankle_roll.*": 0.15,
      # Waist.
      r".*waist_yaw.*": 0.3,
      # Arms.
      r".*shoulder_pitch.*": 0.5,
      r".*shoulder_roll.*": 0.2,
      r".*shoulder_yaw.*": 0.15,
      r".*elbow.*": 0.35,
      r".*wrist.*": 0.3,
    },
    action_patterns,
  )

  # Ensure the pose reward only applies to the controlled joints to avoid tensor size mismatches.
  # (e.g. std_standing matching 27 joints while std_walking matching 12).
  cfg.rewards["pose"].params["asset_cfg"].joint_names = action_patterns
  if not control_arms:
    cfg.rewards.pop("pose_upper_fixed", None)

  # Extra wrist posture reward: keep compliant wrist joints near default angle.
  if control_arms:
    cfg.rewards["wrist_joint_pos"] = RewardTermCfg(
      func=mdp.posture,
      weight=0.5,
      params={
        "std": {r".*_wrist_.*_joint": 0.5},
        "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_wrist_.*_joint",)),
      },
    )

  # Upright reward
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["upright"].params["weight_xy"] = (1.0, 5.0)
  cfg.rewards["upright"].params["std_xy"] = (
    math.sin(math.radians(10.0)),
    math.sin(math.radians(1.0)),
  )

  # cfg.rewards["startup_pitch_upright"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.rewards["thigh_swing_target"].params["asset_cfg"].body_names = (
    ".*_hip_roll_link",
  )
  cfg.rewards["thigh_swing_target"].params["axis_signs"] = {
    r"left_hip_roll_link": 1.0,
    r"right_hip_roll_link": -1.0,
  }

  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.05

  ######
  cfg.rewards["track_linear_velocity"].weight = 6.0
  cfg.rewards["track_angular_velocity"].weight = 3.0

  cfg.rewards["joint_torque_rate"].weight = -1.0e-1

  ######

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  feet_target_height = 0.1
  cfg.rewards["foot_clearance"].params["target_height"] = feet_target_height
  cfg.rewards["foot_swing_height"].params["target_height"] = feet_target_height

  cfg.rewards["foot_flat"].params["target_height"] = feet_target_height * 0.5

  for reward_name in [
    "foot_flat",
    "heelstrike_foot_pitch_pattern",
    "toeoff_foot_pitch_pattern",
  ]:
    cfg.rewards[reward_name].params["asset_cfg"].body_names = (".*ankle_roll.*",)
  for reward_name in ["heelstrike_knee_pattern", "toeoff_knee_pattern"]:
    cfg.rewards[reward_name].params["asset_cfg"].joint_names = (r".*_knee_joint",)
  for reward_name in ["heelstrike_hip_pitch_pattern", "toeoff_hip_pitch_pattern"]:
    cfg.rewards[reward_name].params["asset_cfg"].joint_names = (r".*_hip_pitch_joint",)
  for reward_name in ["heelstrike_hip_roll_pattern", "toeoff_hip_roll_pattern"]:
    cfg.rewards[reward_name].params["asset_cfg"].joint_names = (r".*_hip_roll_joint",)

  cfg.rewards["heelstrike_foot_pitch_pattern"].weight = 0.1
  cfg.rewards["toeoff_foot_pitch_pattern"].weight = 0.1
  cfg.rewards["heelstrike_knee_pattern"].weight = 0.1
  cfg.rewards["toeoff_knee_pattern"].weight = 0.1
  cfg.rewards["heelstrike_hip_pitch_pattern"].weight = 0.1
  cfg.rewards["toeoff_hip_pitch_pattern"].weight = 0.1
  cfg.rewards["heelstrike_hip_roll_pattern"].weight = 0.1
  cfg.rewards["toeoff_hip_roll_pattern"].weight = 0.1

  cfg.rewards["toeoff_knee_pattern"].params["axis_signs"] = {
    r"right_knee_joint": 1.0,
    r"left_knee_joint": 1.0,
  }
  cfg.rewards["heelstrike_knee_pattern"].params["axis_signs"] = {
    r"right_knee_joint": 1.0,
    r"left_knee_joint": 1.0,
  }
  cfg.rewards["toeoff_hip_pitch_pattern"].params["axis_signs"] = {
    r"right_hip_pitch_joint": -1.0,
    r"left_hip_pitch_joint": 1.0,
  }
  cfg.rewards["heelstrike_hip_pitch_pattern"].params["axis_signs"] = {
    r"right_hip_pitch_joint": -1.0,
    r"left_hip_pitch_joint": 1.0,
  }
  cfg.rewards["heelstrike_hip_roll_pattern"].params["axis_signs"] = {
    r"right_hip_roll_joint": -1.0,
    r"left_hip_roll_joint": 1.0,
  }
  cfg.rewards["toeoff_hip_roll_pattern"].params["axis_signs"] = {
    r"right_hip_roll_joint": -1.0,
    r"left_hip_roll_joint": 1.0,
  }

  cfg.rewards["toeoff_hip_roll_pattern"].params["target_angle_deg"] = 2.5

  # Base height target adjusted for P2 standing height (~0.83m).
  cfg.rewards["base_height"].params["target_height"] = 0.84

  cfg.rewards["biped_standing_stability"].weight = 1.0

  cfg.curriculum["biped_double_support_time_weight"] = CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
      "reward_name": "biped_double_support_time",
      "weight_stages": [
        {"step": 0, "weight": 0.0},
        {"step": 1_000 * 24, "weight": 1.0},
      ],
    },
  )

  cfg.curriculum["biped_first_swing_foot_weight"] = CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
      "reward_name": "biped_first_swing_foot",
      "weight_stages": [
        {"step": 0, "weight": 0.0},
        {"step": 500 * 24, "weight": 2.0},
      ],
    },
  )

  cfg.rewards["biped_double_support_time"].params["target_double_support_time"] = 0.1
  cfg.rewards["biped_air_time"].params["target_air_time"] = 0.4

  cfg.curriculum["biped_air_time_weight"] = CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
      "reward_name": "biped_air_time",
      "weight_stages": [
        {"step": 0, "weight": 0.0},
        {"step": 500 * 24, "weight": 30.0},
        {"step": 750 * 24, "weight": 20.0},
        {"step": 1_000 * 24, "weight": 10.0},
      ],
    },
  )
  cfg.curriculum["biped_air_time_post_landing_mask_steps"] = CurriculumTermCfg(
    func=envs_mdp.reward_curriculum,
    params={
      "reward_name": "biped_air_time",
      "stages": [
        {
          "step": 0,
          "params": {
            "post_landing_mask_steps_low_speed": 0,
            "post_landing_mask_steps_high_speed": 0,
          },
        },
        {
          "step": 2_500 * 24,
          "params": {
            "post_landing_mask_steps_low_speed": 10,
            "post_landing_mask_steps_high_speed": 5,
          },
        },
      ],
    },
  )

  # cfg.curriculum["soft_landing_weight"] = CurriculumTermCfg(
  #   func=mdp.reward_weight,
  #   params={
  #     "reward_name": "soft_landing",
  #     "weight_stages": [
  #       {"step": 0,            "weight": -0.001,},
  #       {"step": 2_500 * 24,   "weight": -0.005},
  #       {"step": 5_000 * 24,   "weight": -0.01},
  #     ],
  #   },
  # )

  cfg.rewards["hip_roll_torque_l2"] = RewardTermCfg(
    func=mdp.joint_torques_l2,
    weight=-1.0e-4,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        actuator_names=(r".*_hip_roll_joint",),
      ),
    },
  )

  cfg.rewards["joint_effort_limit"].params["soft_ratio"] = 0.7
  cfg.rewards["joint_effort_limit"].params["power"] = 2.0

  cfg.curriculum["joint_effort_limit_weight"] = CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
      "reward_name": "joint_effort_limit",
      "weight_stages": [
        {"step": 0, "weight": -1.0e-3},
        {"step": 1_000 * 24, "weight": -2.0e-3},
        {"step": 2_000 * 24, "weight": -4.0e-3},
        # {"step": 5_000 * 24,   "weight": -0.005},
      ],
    },
  )

  # cfg.curriculum["track_linear_velocity_std"] = CurriculumTermCfg(
  #   func=envs_mdp.reward_curriculum,
  #   params={
  #     "reward_name": "track_linear_velocity",
  #     "stages": [
  #       {"step": 0, "params": {"std": math.sqrt(0.5)}},
  #       {"step": 2_500 * 24, "params": {"std": math.sqrt(0.3)}},
  #     ],
  #   },
  # )
  cfg.curriculum["track_angular_velocity_low_std"] = CurriculumTermCfg(
    func=envs_mdp.reward_curriculum,
    params={
      "reward_name": "track_angular_velocity",
      "stages": [
        {"step": 0, "params": {"low_std": math.sqrt(0.5)}},
        {"step": 5_000 * 24, "params": {"low_std": math.sqrt(0.25)}},
      ],
    },
  )

  cfg.curriculum["actuator_motor_bandwidth"] = CurriculumTermCfg(
    func=envs_mdp.actuator_motor_bandwidth_curriculum,
    params={
      "event_name": "actuator_motor_bandwidth",
      "stages": [
        {"step": 0, "enabled": False},
        {"step": 5_000 * 24, "enabled": True, "cutoff_hz_range": (15.0, 30.0)},
        {"step": 10_000 * 24, "cutoff_hz_range": (10.0, 30.0)},
        {"step": 15_000 * 24, "cutoff_hz_range": (5.0, 30.0)},
        {"step": 20_000 * 24, "cutoff_hz_range": (2.5, 30.0)},
        {"step": 30_000 * 24, "cutoff_hz_range": (2.5, 20.0)},
        {"step": 40_000 * 24, "cutoff_hz_range": (2.5, 10.0)},
      ],
    },
  )

  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (-0.5, 0.5),
      "lin_vel_y": (-0.5, 0.5),
      "ang_vel_z": (-0.5, 0.5),
    },
    {
      "step": 2_500 * 24,
      "lin_vel_x": (-0.75, 0.75),
      "lin_vel_y": (-0.75, 0.75),
      "ang_vel_z": (-1.0, 1.0),
    },
    {
      "step": 5_000 * 24,
      "lin_vel_x": (-1.0, 1.0),
      "lin_vel_y": (-1.0, 1.0),
      "ang_vel_z": (-1.0, 1.0),
    },
    {
      "step": 10_000 * 24,
      "lin_vel_x": (-1.25, 1.25),
      "lin_vel_y": (-1.0, 1.0),
      "ang_vel_z": (-1.25, 1.25),
    },
    {
      "step": 20_000 * 24,
      "lin_vel_x": (-1.25, 1.25),
      "lin_vel_y": (-1.0, 1.0),
      "ang_vel_z": (-1.5, 1.5),
    },
  ]

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("actuator_rfi", None)
    cfg.events.pop("actuator_motor_bandwidth", None)
    cfg.curriculum.pop("actuator_motor_bandwidth", None)
    robot_cfg = cfg.scene.entities["robot"]
    assert robot_cfg.articulation is not None
    for actuator_id in dr_actuator_ids:
      actuator_cfg = robot_cfg.articulation.actuators[actuator_id]
      actuator_cfg.delay_min_lag = 0
      actuator_cfg.delay_max_lag = 0
      actuator_cfg.delay_hold_prob = 0.0
      actuator_cfg.delay_update_period = 0
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


def kimm_p2_flat_env_cfg(
  play: bool = False,
  observe_waist: bool = True,
  observe_arms: bool = True,
  control_waist: bool = True,
  control_arms: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create KIMM P2 flat terrain velocity configuration."""
  cfg = kimm_p2_rough_env_cfg(
    play=play,
    observe_waist=observe_waist,
    observe_arms=observe_arms,
    control_waist=control_waist,
    control_arms=control_arms,
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
  del cfg.observations["critic"].terms["height_scan"]

  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  if play:
    assert "command_vel" in cfg.curriculum
    del cfg.curriculum["command_vel"]

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
