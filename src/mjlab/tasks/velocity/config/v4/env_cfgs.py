"""KIMM V4 velocity environment configurations."""

import math

from mjlab.asset_zoo.robots import (
  V4_ACTION_SCALE,
  get_v4_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
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
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg, ms2steps


def kimm_v4_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create KIMM V4 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 120

  cfg.scene.entities = {"robot": get_v4_robot_cfg()}

  # Set raycast sensor frame to V4 freejoint/root body.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "torso_link"

  site_names = ("right_foot", "left_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 9)
  )

  # Wire foot height scan to per-foot sites.
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
  feet_center_cfg = ContactSensorCfg(
    name="feet_center_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r"^(left|right)_foot[45]_collision$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="maxforce",
    num_slots=1,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="torso_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="torso_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    feet_center_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = V4_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

  cfg.events["torso_pseudo_inertia"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["torso_pseudo_inertia"].params["t1_range"] = (-0.025, 0.05)
  cfg.events["link_pseudo_inertia"].params["alpha_range"] = (-0.05, 0.05)
  cfg.events["link_pseudo_inertia"].params["asset_cfg"].body_names = (
    ".*_hip_.*_link",
    ".*_knee_link",
    ".*_ankle_roll_link",
  )

  # V4 has 2 actuator config groups: X12 hip/knee pitch and X8 waist/roll/yaw/ankles.
  dr_actuator_ids = [0, 1]
  cfg.events["pd_gains"].params["asset_cfg"].actuator_ids = dr_actuator_ids
  cfg.events["actuator_rfi"].params["asset_cfg"].actuator_ids = dr_actuator_ids

  # Rationale for std values:
  # - V4 has waist yaw plus lower-body joints only.
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_turning"] = {
    # Lower body.
    r".*hip_pitch.*": 0.4,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.6,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # Waist.
    r".*waist_yaw.*": 0.3,
  }

  cfg.rewards.pop("pose_upper_fixed", None)
  cfg.rewards["waist_yaw_fixed"] = RewardTermCfg(
    func=mdp.variable_posture,
    weight=1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=("waist_yaw_joint",)),
      "command_name": "twist",
      "std_standing": {".*": 0.15},
      "std_turning": {".*": 0.15},
      "std_walking": {".*": 0.15},
      "std_running": {".*": 0.15},
      "weight_standing": 1.0,
      "weight_turning": 1.0,
      "weight_walking": 1.0,
      "weight_running": 1.0,
      "walking_threshold": 0.05,
      "running_threshold": 1.5,
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

  cfg.rewards["thigh_swing_pattern"].params["asset_cfg"].body_names = (
    ".*_hip_roll_link",
  )
  cfg.rewards["thigh_swing_pattern"].params["axis_signs"] = {
    r"left_hip_roll_link": 1.0,
    r"right_hip_roll_link": -1.0,
  }  # inward
  cfg.rewards["shank_swing_pattern"].params["asset_cfg"].body_names = (".*_knee_link",)
  cfg.rewards["shank_swing_pattern"].params["axis_signs"] = {
    r"left_knee_link": 1.0,
    r"right_knee_link": -1.0,
  }  # inward

  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.05

  ######
  cfg.rewards["track_linear_velocity"].weight = 2.0
  cfg.rewards["track_angular_velocity"].weight = 5.0

  for reward_name in [
    "foot_clearance",
    "foot_slip",
    "foot_force_momentum",
    "foot_lateral_distance",
  ]:  # noqa: E501
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  feet_target_height = 0.1
  cfg.rewards["foot_clearance"].params["target_height"] = feet_target_height
  cfg.rewards["foot_swing_height"].params["target_height"] = feet_target_height

  cfg.rewards["knee_height"].weight = 1.0
  cfg.rewards["knee_height"].params["asset_cfg"].body_names = (".*_knee_link",)
  cfg.rewards["knee_height"].params["reference_asset_cfg"].body_names = ("base_link",)

  for reward_name in ["heelstrike_foot_pitch_pattern", "toeoff_foot_pitch_pattern"]:
    cfg.rewards[reward_name].params["asset_cfg"].body_names = (".*ankle_roll.*",)
  # for reward_name in ["heelstrike_ankle_pitch_pattern", "toeoff_ankle_pitch_pattern"]:
  #   cfg.rewards[reward_name].params["asset_cfg"].joint_names = (r".*_ankle_pitch_joint",)
  for reward_name in ["heelstrike_knee_pattern", "toeoff_knee_pattern"]:
    cfg.rewards[reward_name].params["asset_cfg"].joint_names = (r".*_knee_joint",)
  for reward_name in ["heelstrike_hip_pitch_pattern", "toeoff_hip_pitch_pattern"]:
    cfg.rewards[reward_name].params["asset_cfg"].joint_names = (r".*_hip_pitch_joint",)
  for reward_name in ["heelstrike_hip_roll_pattern", "toeoff_hip_roll_pattern"]:
    cfg.rewards[reward_name].params["asset_cfg"].joint_names = (r".*_hip_roll_joint",)
  for reward_name in ["heelstrike_hip_yaw_pattern", "toeoff_hip_yaw_pattern"]:
    cfg.rewards[reward_name].params["asset_cfg"].joint_names = (r".*_hip_yaw_joint",)

  for reward_name in [
    "heelstrike_elbow_pattern",
    "toeoff_elbow_pattern",
    "heelstrike_shoulder_pitch_pattern",
    "toeoff_shoulder_pitch_pattern",
  ]:
    cfg.rewards.pop(reward_name, None)

  cfg.rewards["toeoff_knee_pattern"].params["axis_signs"] = {
    r"right_knee_joint": 1.0,
    r"left_knee_joint": 1.0,
  }
  cfg.rewards["heelstrike_knee_pattern"].params["axis_signs"] = {
    r"right_knee_joint": 1.0,
    r"left_knee_joint": 1.0,
  }
  # cfg.rewards["heelstrike_ankle_pitch_pattern"].params["axis_signs"] = {r"right_ankle_pitch_joint": 1.0, r"left_ankle_pitch_joint": 1.0,}
  # cfg.rewards["toeoff_ankle_pitch_pattern"].params["axis_signs"] = {r"right_ankle_pitch_joint": 1.0, r"left_ankle_pitch_joint": 1.0,}
  cfg.rewards["toeoff_hip_pitch_pattern"].params["axis_signs"] = {
    r"right_hip_pitch_joint": -1.0,
    r"left_hip_pitch_joint": 1.0,
  }
  cfg.rewards["heelstrike_hip_pitch_pattern"].params["axis_signs"] = {
    r"right_hip_pitch_joint": -1.0,
    r"left_hip_pitch_joint": 1.0,
  }
  cfg.rewards["heelstrike_hip_roll_pattern"].params["axis_signs"] = {
    r"right_hip_roll_joint": 1.0,
    r"left_hip_roll_joint": -1.0,
  }
  cfg.rewards["toeoff_hip_roll_pattern"].params["axis_signs"] = {
    r"right_hip_roll_joint": 1.0,
    r"left_hip_roll_joint": -1.0,
  }
  cfg.rewards["heelstrike_hip_yaw_pattern"].params["axis_signs"] = {
    r"right_hip_yaw_joint": 1.0,
    r"left_hip_yaw_joint": 1.0,
  }
  cfg.rewards["toeoff_hip_yaw_pattern"].params["axis_signs"] = {
    r"right_hip_yaw_joint": 1.0,
    r"left_hip_yaw_joint": 1.0,
  }

  # cfg.rewards["heelstrike_foot_pitch_pattern"].weight = 0.0
  # cfg.rewards["toeoff_foot_pitch_pattern"].weight = 0.0
  # cfg.rewards["heelstrike_knee_pattern"].weight = 0.0
  # cfg.rewards["toeoff_knee_pattern"].weight = 0.0
  # cfg.rewards["heelstrike_hip_pitch_pattern"].weight = 0.0
  # cfg.rewards["toeoff_hip_pitch_pattern"].weight = 0.0
  # cfg.rewards["heelstrike_hip_roll_pattern"].weight = 0.0
  # cfg.rewards["toeoff_hip_roll_pattern"].weight = 0.0

  cfg.rewards["base_height"].params["target_height"] = 0.84

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

  # cfg.rewards["biped_double_support_time"].params["target_double_support_time"] = 0.125
  # cfg.rewards["biped_air_time"].params["target_air_time"] = 0.6

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
            "post_landing_mask_steps_low_speed": ms2steps(100),
            "post_landing_mask_steps_high_speed": ms2steps(40),
          },
        },
      ],
    },
  )

  cfg.curriculum["foot_force_momentum_weight"] = CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
      "reward_name": "foot_force_momentum",
      "weight_stages": [
        {"step": 0, "weight": 0.0},
        {"step": 2_500 * 24, "weight": -0.1},
      ],
    },
  )

  cfg.curriculum["foot_landing_weight"] = CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
      "reward_name": "foot_landing",
      "weight_stages": [
        {"step": 0, "weight": -1.0e-3},
        {"step": 5_000 * 24, "weight": -1.0e-2},
        {"step": 10_000 * 24, "weight": -1.0e-1},
      ],
    },
  )

  cfg.curriculum["track_linear_velocity_weight"] = CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
      "reward_name": "track_linear_velocity",
      "weight_stages": [
        {"step": 0, "weight": 3.0},
        {"step": 5_000 * 24, "weight": 4.0},
        {"step": 10_000 * 24, "weight": 5.0},
        {"step": 15_000 * 24, "weight": 6.0},
        {"step": 20_000 * 24, "weight": 7.0},
      ],
    },
  )
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

  # cfg.curriculum["actuator_rfi_range"] = CurriculumTermCfg(
  #   func=envs_mdp.actuator_rfi_curriculum,
  #   params={
  #     "event_name": "actuator_rfi",
  #     "stages": [
  #       {"step": 0, "params": {"ratio_range": (-0.0, 0.0)}},
  #       {"step": 5_000 * 24, "params": {"ratio_range": (-0.05, 0.05)}},
  #     ],
  #   },
  # )

  vx, vy, wz = "lin_vel_x", "lin_vel_y", "ang_vel_z"
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, vx: (0.0, 0.5), vy: (-0.5, 0.5), wz: (-0.5, 0.5)},
    {"step": 2_500, vx: (-0.5, 0.5), vy: (-0.5, 0.5), wz: (-0.5, 0.5)},
    {"step": 5_000 * 24, vx: (-0.75, 0.75), vy: (-0.75, 0.75), wz: (-1.0, 1.0)},
    {"step": 10_000 * 24, vx: (-1.0, 1.0), vy: (-1.0, 1.0), wz: (-1.0, 1.0)},
    {"step": 15_000 * 24, vx: (-1.0, 1.5), vy: (-1.0, 1.0), wz: (-1.0, 1.0)},
    {"step": 20_000 * 24, vx: (-1.0, 2.0), vy: (-1.0, 1.0), wz: (-1.0, 1.0)},
    {"step": 20_000 * 24, vx: (-1.0, 2.0), vy: (-1.0, 1.0), wz: (-1.5, 1.5)},
  ]

  cfg.curriculum["command_vel"].params["motion_mix_stages"] = [
    {"step": 0, "rel_standing_envs": 1.0, "rel_turn_in_place_envs": 0.0},
    {"step": 250 * 24, "rel_standing_envs": 0.1, "rel_turn_in_place_envs": 0.1},
  ]

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("actuator_rfi", None)
    cfg.curriculum.pop("actuator_rfi_range", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
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


def kimm_v4_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create KIMM V4 flat terrain velocity configuration."""
  cfg = kimm_v4_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  # del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-2.0, 3.0)
    twist_cmd.ranges.ang_vel_z = (-2.0, 2.0)

  return cfg
