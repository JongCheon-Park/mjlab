"""Unitree G1 velocity environment configurations."""

import math

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
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


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
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
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["torso_pseudo_inertia"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["torso_pseudo_inertia"].params["t1_range"] = (0.01, 0.05)
  cfg.events["link_pseudo_inertia"].params["alpha_range"] = (-0.05, 0.05)
  cfg.events["link_pseudo_inertia"].params["asset_cfg"].body_names = (
    # ".*_hip_.*_link",
    ".*_knee_link",
    ".*_wrist_yaw_link",
    # ".*_ankle_roll_link",
  )

  # # For Shoes
  # cfg.events["foot_pseudo_inertia"] = EventTermCfg(
  #   mode="startup",
  #   func=dr.pseudo_inertia,
  #   params={
  #     "asset_cfg": SceneEntityCfg("robot", body_names=(".*_ankle_roll_link")),
  #     "alpha_range": (0.1, 0.3),
  #     "t1_range": (-0.015, 0.03),
  #     "t2_range": (-0.015, 0.015),
  #     "t3_range": (-0.03, 0.015),
  #   },
  # )

  # cfg.events["encoder_bias"].params["asset_cfg"].joint_names = ("^((?!ankle_.*_joint).)*$",)
  # cfg.events["encoder_bias_ankle_pitch"] = EventTermCfg(
  #   mode="startup",
  #   func=dr.encoder_bias,
  #   params={
  #     "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_ankle_pitch_joint",)),
  #     "bias_range": (-0.05, 0.05),
  #   },
  # )

  # PD gains randomization: G1 has 6 actuator groups.
  # 0: Arms (G1_ACTUATOR_5020)
  # 1: Hip Pitch/Yaw (G1_ACTUATOR_7520_14)
  # 2: Hip Roll / Knee (G1_ACTUATOR_7520_22)
  # 3: Wrist (G1_ACTUATOR_4010)
  # 4: Waist (G1_ACTUATOR_WAIST)
  # 5: Ankle (G1_ACTUATOR_ANKLE)
  # cfg.events["pd_gains"].params["asset_cfg"].actuator_ids = [0, 1, 2, 3, 4, 5]
  cfg.events["actuator_rfi"].params["asset_cfg"].actuator_ids = [0, 1, 2, 3, 4, 5]

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
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
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
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
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.4,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.4,
    r".*wrist.*": 0.3,
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
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.5,
    r".*wrist.*": 0.3,
  }

  # Upright reward
  # cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["upright"].params["asset_cfg"].body_names = (
    "torso_link",
  )  # "pelvis", "torso_link"
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
  cfg.rewards["track_linear_velocity"].weight = 2.5
  cfg.rewards["track_angular_velocity"].weight = 5.0

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

  cfg.rewards["toeoff_elbow_pattern"].params["asset_cfg"].joint_names = (
    r".*_elbow_joint",
  )
  for reward_name in [
    "heelstrike_shoulder_pitch_pattern",
    "toeoff_shoulder_pitch_pattern",
  ]:
    cfg.rewards[reward_name].params["asset_cfg"].joint_names = (
      r".*_shoulder_pitch_joint",
    )

  cfg.rewards["heelstrike_foot_pitch_pattern"].weight = 0.1
  cfg.rewards["toeoff_foot_pitch_pattern"].weight = 0.1
  cfg.rewards["heelstrike_knee_pattern"].weight = 0.1
  cfg.rewards["toeoff_knee_pattern"].weight = 0.1
  cfg.rewards["heelstrike_hip_pitch_pattern"].weight = 0.1
  cfg.rewards["toeoff_hip_pitch_pattern"].weight = 0.1
  cfg.rewards["heelstrike_hip_roll_pattern"].weight = 0.1
  cfg.rewards["toeoff_hip_roll_pattern"].weight = 0.1

  cfg.rewards["toeoff_elbow_pattern"].weight = 0.1
  cfg.rewards["heelstrike_shoulder_pitch_pattern"].weight = 0.1
  cfg.rewards["toeoff_shoulder_pitch_pattern"].weight = 0.1

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
    r"left_hip_pitch_joint": -1.0,
  }
  cfg.rewards["heelstrike_hip_pitch_pattern"].params["axis_signs"] = {
    r"right_hip_pitch_joint": -1.0,
    r"left_hip_pitch_joint": -1.0,
  }
  cfg.rewards["heelstrike_hip_roll_pattern"].params["axis_signs"] = {
    r"right_hip_roll_joint": 1.0,
    r"left_hip_roll_joint": -1.0,
  }
  cfg.rewards["toeoff_hip_roll_pattern"].params["axis_signs"] = {
    r"right_hip_roll_joint": 1.0,
    r"left_hip_roll_joint": -1.0,
  }

  cfg.rewards["toeoff_elbow_pattern"].params["axis_signs"] = {
    r"right_elbow_joint": 1.0,
    r"left_elbow_joint": 1.0,
  }
  cfg.rewards["heelstrike_shoulder_pitch_pattern"].params["axis_signs"] = {
    r"right_shoulder_pitch_joint": -1.0,
    r"left_shoulder_pitch_joint": -1.0,
  }
  cfg.rewards["toeoff_shoulder_pitch_pattern"].params["axis_signs"] = {
    r"right_shoulder_pitch_joint": -1.0,
    r"left_shoulder_pitch_joint": -1.0,
  }

  cfg.rewards["biped_standing_stability"].weight = 1.0
  cfg.rewards["biped_standing_stability"].params["standing_action_rate_weight"] = 0.1

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
  cfg.curriculum["biped_air_time_weight"] = CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
      "reward_name": "biped_air_time",
      "weight_stages": [
        {"step": 0, "weight": 0.0},
        {"step": 500 * 24, "weight": 5.0},
        {"step": 750 * 24, "weight": 2.5},
        {"step": 1_000 * 24, "weight": 1.0},
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

  # cfg.curriculum["track_angular_velocity_weight"] = CurriculumTermCfg(
  #   func=mdp.reward_weight,
  #   params={
  #     "reward_name": "track_angular_velocity",
  #     "weight_stages": [
  #       {"step": 0,            "weight": 5.0,},
  #       {"step": 15_000 * 24,   "weight": 4.0},
  #     ],
  #   },
  # )

  # cfg.curriculum["track_linear_velocity_std"] = CurriculumTermCfg(
  #   func=envs_mdp.reward_curriculum,
  #   params={
  #     "reward_name": "track_linear_velocity",
  #     "stages": [
  #       {"step": 0, "params": {"std": math.sqrt(0.3)}},
  #       {"step": 2_500 * 24, "params": {"std": math.sqrt(0.1)}},
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

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("actuator_rfi", None)
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


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

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


def unitree_g1_add_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat velocity config with ADD discriminator observations."""
  cfg = unitree_g1_flat_env_cfg(play=play)

  # Add detailed self-collision sensor for discriminator (per-body).
  self_collision_detailed_cfg = ContactSensorCfg(
    name="self_collision_detailed",
    primary=ContactMatch(mode="body", pattern=".*_link|pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (*cfg.scene.sensors, self_collision_detailed_cfg)

  site_names = ("left_foot", "right_foot")

  discriminator_terms = {
    "lin_vel_error": ObservationTermCfg(
      func=mdp.lin_vel_error,
      params={"command_name": "twist"},
    ),
    "ang_vel_error": ObservationTermCfg(
      func=mdp.ang_vel_error,
      params={"command_name": "twist"},
    ),
    "orientation_error": ObservationTermCfg(
      func=mdp.orientation_error,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis", "torso_link"))
      },
    ),
    "base_height_error": ObservationTermCfg(
      func=mdp.base_height_error,
      params={"target_height": 0.78},
    ),
    "action_rate_slack": ObservationTermCfg(
      func=mdp.action_rate_slack,
      params={"threshold": 0.5},
    ),
    "joint_limit_slack": ObservationTermCfg(
      func=mdp.joint_limit_slack,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collision_slack": ObservationTermCfg(
      func=mdp.self_collision_slack,
      params={"sensor_name": "self_collision_detailed"},
    ),
    "variable_posture_error": ObservationTermCfg(
      func=mdp.variable_posture_error,
      params={
        "std_standing": cfg.rewards["pose"].params["std_standing"],
        "std_turning": cfg.rewards["pose"].params["std_turning"],
        "std_walking": cfg.rewards["pose"].params["std_walking"],
        "std_running": cfg.rewards["pose"].params["std_running"],
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
      },
    ),
    "foot_clearance_error": ObservationTermCfg(
      func=mdp.foot_clearance_error,
      params={
        "target_height": 0.1,
        "vel_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),
    "foot_slip_slack": ObservationTermCfg(
      func=mdp.foot_slip_slack,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold": 0.025,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),
    "foot_flat_error": ObservationTermCfg(
      func=mdp.foot_flat_error,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_ratio": 0.4,
        "asset_cfg": SceneEntityCfg(
          "robot",
          body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        ),
      },
    ),
    "body_ang_vel_error": ObservationTermCfg(
      func=mdp.body_ang_vel_error,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
    ),
    "angular_momentum_error": ObservationTermCfg(
      func=mdp.angular_momentum_error,
      params={"sensor_name": "robot/root_angmom"},
    ),
  }
  cfg.observations["discriminator"] = ObservationGroupCfg(
    terms=discriminator_terms,
    concatenate_terms=True,
    enable_corruption=False,
  )

  return cfg
