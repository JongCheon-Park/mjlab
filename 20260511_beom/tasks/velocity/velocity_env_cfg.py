"""Velocity task configuration.

This module provides a factory function to create a base velocity task config.
Robot-specific configurations call the factory and customize as needed.
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base velocity tracking task configuration."""

  ##
  # Sensors
  ##

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),  # Set per-robot.
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
  )

  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=(),  # Set per-robot: frame and pattern.
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
    viz=TerrainHeightSensorCfg.VizCfg(
      show_rays=True,
      hit_color=(1.0, 0.0, 1.0, 0.8),  # Magenta rays.
      hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
    ),
  )

  ##
  # Observations
  ##

  actor_terms = {
    # "base_lin_vel": ObservationTermCfg(
    #   func=mdp.builtin_sensor,
    #   params={"sensor_name": "robot/imu_lin_vel"},
    #   noise=Unoise(n_min=-0.5, n_max=0.5),
    # ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    # "height_scan": ObservationTermCfg(
    #   func=envs_mdp.height_scan,
    #   params={"sensor_name": "terrain_scan"},
    #   noise=Unoise(n_min=-0.1, n_max=0.1),
    #   scale=1 / terrain_scan.max_distance,
    # ),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=1 / terrain_scan.max_distance,
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"sensor_name": "foot_height_scan"},
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=5,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=5,
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "mean_action_acc": MetricsTermCfg(
      func=mdp.mean_action_acc,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      # Motion Probabilities (Sum <= 1.0, remainder is Move)
      rel_standing_envs=0.1,  # 10% Stand (Zero Vel)
      rel_turn_in_place_envs=0.1,  # 10% Turn (Zero Linear, Variable Angular)
      rel_longitudinal_envs=0.4,  # 40% of move envs use x-axis-only commands.
      rel_lateral_envs=0.2,  # 20% of move envs use y-axis-only commands.
      # Speed Level Probabilities (Independent of Motion)
      rel_slow_envs=0.3,  # 30% Slow (Low Vel Range) - Applied to Turn & Move
      rel_fast_envs=0.2,  # 20% Fast (High Vel Range) - Applied to Turn & Move
      # Additional Modifiers
      rel_heading_envs=0.3,  # 30% Heading Tracking (Overrides Angular Vel)
      slow_threshold=0.25,  # 25% of max range (lin & ang)
      fast_threshold=0.75,  # 75% of max range (lin & ang)
      turn_threshold=1.0,  # Caps yaw rate for move commands only; turn-in-place keeps full ang_vel_z range.
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.5, 0.5),
        heading=(-math.pi, math.pi),
      ),
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.01, 0.05),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.1, 0.1),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=("^((?!ankle_roll_joint).)*$",)
        ),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "encoder_bias_ankle_roll": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_ankle_roll_joint",)),
        "bias_range": (-0.03, 0.03),
      },
    ),
    # alpha scales mass & inertia by e^(2α):
    #   ±0.01 → ±2%,  ±0.05 → ±10%,  ±0.1 → ±22%
    "torso_pseudo_inertia": EventTermCfg(
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "alpha_range": (-0.05, 0.05),
        "t1_range": (-0.025, 0.05),
        "t2_range": (-0.05, 0.05),
        "t3_range": (-0.05, 0.05),
      },
    ),
    "link_pseudo_inertia": EventTermCfg(
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "alpha_range": (-0.025, 0.025),
        "t_range": (-0.01, 0.01),
      },
    ),
    "joint_armature": EventTermCfg(
      mode="startup",
      func=dr.joint_armature,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "scale",
        "ranges": (0.9, 1.1),
      },
    ),
    "joint_friction": EventTermCfg(
      mode="startup",
      func=dr.joint_friction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=(".*_hip_.*", ".*_knee_.*", ".*_ankle_.*")
        ),
        "operation": "abs",
        "ranges": (0.05, 1.0),
      },
    ),
    "joint_damping": EventTermCfg(
      mode="startup",
      func=dr.joint_damping,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "abs",
        "ranges": (0.05, 1.0),
      },
    ),
    "actuator_rfi": EventTermCfg(
      mode="startup",
      func=dr.actuator_rfi,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_ids=[]),
        "ratio_range": (-0.05, 0.05),
      },
    ),
    # "pd_gains": EventTermCfg(
    #   mode="reset",
    #   func=dr.pd_gains,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot", actuator_ids=[]),
    #     "kp_range": (0.925, 1.0),
    #     "kd_range": (0.925, 1.0),
    #     "operation": "scale",
    #   },
    # ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=2.0,
      params={
        "command_name": "twist",
        "std": math.sqrt(0.1),
        "std_walking": math.sqrt(0.15),
        "std_running": math.sqrt(0.3),
        "walking_threshold": 0.5,
        "running_threshold": 1.5,
      },
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=2.0,
      params={
        "command_name": "twist",
        "low_std": math.sqrt(0.25),
        "high_std": math.sqrt(0.5),
        "speed_threshold": 0.5,
      },
    ),
    "upright": RewardTermCfg(
      func=mdp.flat_orientation_multi,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
        "weights": [1.0],
        "weight_xy": (1.0, 1.0),
        "std_xy": (math.sin(math.radians(5.0)), math.sin(math.radians(1.0))),
      },
    ),
    # "upright": RewardTermCfg(
    #   func=mdp.flat_orientation,
    #   weight=1.0,
    #   params={
    #     "std": math.sqrt(0.2),
    #     "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
    #   },
    # ),
    "base_height": RewardTermCfg(
      func=mdp.base_height_l2,
      weight=1.0,
      params={
        "target_height": 0.78,
        "command_name": "twist",
        "std": 0.03,
        "std_walking": 0.05,
        "std_running": 0.07,
        "walking_threshold": 0.5,
        "running_threshold": 1.0,
      },
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": {},  # Set per-robot.
        "std_turning": {},  # Optional per-robot; defaults to std_walking.
        "std_walking": {},  # Set per-robot.
        "std_running": {},  # Set per-robot.
        "weight_standing": 5.0,
        "weight_turning": 1.0,
        "weight_walking": 1.0,
        "weight_running": 1.0,
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    "pose_upper_fixed": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            r".*shoulder_roll.*",
            r".*shoulder_yaw.*",
            r".*wrist.*",
          ),
        ),
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
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=0.0,  # Override per-robot
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=0.0,  # Override per-robot
      params={"sensor_name": "robot/root_angmom"},
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    # "joint_acc": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_effort_limit": RewardTermCfg(
      weight=-1.0e-2,
      func=mdp.joint_effort_limits,
      params={"soft_ratio": 0.9, "power": 1.0},
    ),
    "joint_torque_rate": RewardTermCfg(func=mdp.joint_torque_rate_l2, weight=-5.0e-2),
    # "joint_torque": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1.0e-7),
    # "air_time": RewardTermCfg(
    #   func=mdp.feet_air_time,
    #   weight=0.0,  # Override per-robot.
    #   params={
    #     "sensor_name": "feet_ground_contact",
    #     "threshold_min": 0.05,
    #     "threshold_max": 0.5,
    #     "command_name": "twist",
    #     "command_threshold": 0.5,
    #   },
    # ),
    # "gait_cycle": RewardTermCfg(
    #   func=mdp.gait_cycle,
    #   weight=0.0,  # Enable and tune per-task.
    #   params={
    #     "sensor_name": "feet_ground_contact",
    #     "command_name": "twist",
    #     "phase_target": 0.5,
    #   },
    # ),
    "biped_air_time": RewardTermCfg(
      func=mdp.biped_air_time,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "target_air_time": 0.5,
        "std": math.sqrt(0.1),
        "std_turning": math.sqrt(0.4),
        "std_walking": math.sqrt(0.2),
        "std_running": math.sqrt(0.3),
        "command_name": "twist",
        "command_threshold": 0.05,
        "walking_threshold": 0.5,
        "running_threshold": 1.0,
        # Speed-branch landing cooldown.
        # - < walking_threshold: post_landing_mask_steps_low_speed
        # - walking_threshold, running_threshold: post_landing_mask_steps_high_speed
        # - >= running_threshold: disabled
        "post_landing_mask_steps_low_speed": 0,
        "post_landing_mask_steps_high_speed": 0,
        "symmetry_penalty_weight": 0.5,
        "symmetry_ema_alpha": 0.05,
      },
    ),
    "biped_first_swing_foot": RewardTermCfg(
      func=mdp.biped_first_swing_foot,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "wrong_swing_penalty": 0.5,
        "timeout_steps": 15,  # 0.3 sec, 1 step = 0.02s in the default velocity env.
        "no_swing_penalty": 1.0,
        "min_first_swing_air_time": 0.2,
        "load_transfer_weight": 0.5,
      },
    ),
    "biped_double_support_time": RewardTermCfg(
      func=mdp.biped_double_support_time,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "target_double_support_time": 0.125,
        "std": math.sqrt(0.01),
        "std_walking": math.sqrt(0.04),
        "min_first_swing_air_time": 0.2,
        "command_name": "twist",
        "command_threshold": 0.05,
        "walking_threshold": 0.5,
        "running_threshold": 1.0,
      },
    ),
    "biped_standing_stability": RewardTermCfg(
      func=mdp.biped_standing_stability,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "standing_weight": 2.0,
        "standing_air_penalty_weight": 0.5,
        "standing_action_rate_weight": 0.2,
      },
    ),
    # "toe_off_push": RewardTermCfg(
    #   func=mdp.toe_off_push,
    #   weight=0.0,
    #   params={
    #     "sensor_name": "feet_ground_contact",
    #     "command_name": "twist",
    #     "command_threshold": 0.05,
    #     "push_window_steps": 2,
    #     "eps": 1.0e-6,
    #     "max_reward": 1.0,
    #   },
    # ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-5.0,
      params={
        "target_height": 0.1,
        "height_sensor_name": "foot_height_scan",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height,
      weight=-1.0,
      params={
        "sensor_name": "feet_ground_contact",
        "height_sensor_name": "foot_height_scan",
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_flat": RewardTermCfg(
      func=mdp.feet_orientation_flat,
      weight=0.1,
      params={
        "std": math.sqrt(0.2),
        "sensor_name": "feet_ground_contact",
        "stance_force_ratio": 0.4,
        "target_height": 0.05,
        "settle_steps": 5,
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "heelstrike_foot_pitch_pattern": RewardTermCfg(
      func=mdp.foot_pitch_pattern,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "mode": "heel",
        "target_angle_deg": 15.0,  # 목표 발 기울기 (deg)
        "std": math.sin(math.radians(15.0)),  # 허용 발 기울기 (sin of angle)
        "running_threshold": 1.0,  # heel-strike 패턴 보상 비활성화 (m/s)
        "window_steps": 10,  # 직전 0.2s 자세 버퍼
        "temporal_std_steps": 5.0,  # legacy event-only window; post reward uses current state
        "post_reward_steps": 5,  # event step + 4 post-event control steps
        "post_reward_std_steps": 2,  # meaningful through ~3 steps, then decays sharply
        "min_air_steps": 10,  # 직전 공중 구간이 너무 짧은 landing 무시
        "symmetry_penalty_weight": 1.0,
        "symmetry_ema_alpha": 0.2,
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "toeoff_foot_pitch_pattern": RewardTermCfg(
      func=mdp.foot_pitch_pattern,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "mode": "toe",
        "target_angle_deg": 20.0,  # 목표 발 기울기 (deg)
        "std": math.sin(math.radians(20.0)),  # 허용 발 기울기 (sin of angle)
        "window_steps": 10,  # 직전 0.2s 자세 버퍼
        "temporal_std_steps": 5.0,  # legacy event-only window; post reward uses current state
        "post_reward_steps": 5,  # event step + 4 post-event control steps
        "post_reward_std_steps": 2,  # meaningful through ~3 steps, then decays sharply
        "min_contact_steps": 2,  # 직전 접촉 구간이 매우 짧은 liftoff만 무시
        "min_air_steps": 2,  # standing→첫 발 떼기 무시
        "symmetry_penalty_weight": 1.0,
        "symmetry_ema_alpha": 0.2,
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    # joint_gait_pattern sign convention (after axis_signs correction):
    #   positive target_angle_deg = flexion (knee bend / hip forward swing)
    #   negative target_angle_deg = extension (hip backward push)
    # Only active for forward commands; backward is left to free learning.
    "heelstrike_knee_pattern": RewardTermCfg(
      func=mdp.joint_gait_pattern,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "mode": "heel",
        "target_angle_deg": 10.0,
        "std": math.radians(10.0),
        "std_running": math.radians(5.0),
        "running_threshold": 1.0,
        "window_steps": 10,  # 직전 0.2s 관절각 버퍼
        "temporal_std_steps": 5.0,  # legacy event-only window; post reward uses current state
        "post_reward_steps": 5,  # event step + 4 post-event control steps
        "post_reward_std_steps": 2,  # meaningful through ~3 steps, then decays sharply
        "min_air_steps": 10,  # 직전 공중 구간이 너무 짧은 landing 무시
        "axis_signs": 1.0,
        "symmetry_penalty_weight": 1.0,
        "symmetry_ema_alpha": 0.2,
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),  # Set per-robot.
      },
    ),
    "toeoff_knee_pattern": RewardTermCfg(
      func=mdp.joint_gait_pattern,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "mode": "toe",
        "target_angle_deg": 30.0,
        "std": math.radians(30.0),
        "std_running": math.radians(25.0),
        "running_threshold": 1.0,
        "window_steps": 10,  # 직전 0.2s 관절각 버퍼
        "temporal_std_steps": 5.0,  # legacy event-only window; post reward uses current state
        "post_reward_steps": 5,  # event step + 4 post-event control steps
        "post_reward_std_steps": 2,  # meaningful through ~3 steps, then decays sharply
        "min_contact_steps": 2,  # 직전 접촉 구간이 매우 짧은 liftoff만 무시
        "min_air_steps": 2,  # standing→첫 발 떼기 무시
        "axis_signs": 1.0,
        "symmetry_penalty_weight": 1.0,
        "symmetry_ema_alpha": 0.2,
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),  # Set per-robot.
      },
    ),
    "heelstrike_hip_pitch_pattern": RewardTermCfg(
      func=mdp.joint_gait_pattern,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "mode": "heel",
        "target_angle_deg": 30.0,
        "std": math.radians(30.0),
        "std_running": math.radians(25.0),
        "running_threshold": 1.0,
        "window_steps": 10,  # 직전 0.2s 관절각 버퍼
        "temporal_std_steps": 5.0,  # legacy event-only window; post reward uses current state
        "post_reward_steps": 5,  # event step + 4 post-event control steps
        "post_reward_std_steps": 2,  # meaningful through ~3 steps, then decays sharply
        "min_air_steps": 10,  # 직전 공중 구간이 너무 짧은 landing 무시
        "axis_signs": 1.0,
        "symmetry_penalty_weight": 1.0,
        "symmetry_ema_alpha": 0.2,
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),  # Set per-robot.
      },
    ),
    "toeoff_hip_pitch_pattern": RewardTermCfg(
      func=mdp.joint_gait_pattern,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "mode": "toe",
        "target_angle_deg": -10.0,
        "std": math.radians(10.0),
        "std_running": math.radians(5.0),
        "running_threshold": 1.0,
        "window_steps": 10,  # 직전 0.2s 관절각 버퍼
        "temporal_std_steps": 5.0,  # legacy event-only window; post reward uses current state
        "post_reward_steps": 5,  # event step + 4 post-event control steps
        "post_reward_std_steps": 2,  # meaningful through ~3 steps, then decays sharply
        "min_contact_steps": 2,  # 직전 접촉 구간이 매우 짧은 liftoff만 무시
        "min_air_steps": 2,  # standing→첫 발 떼기 무시
        "axis_signs": 1.0,
        "symmetry_penalty_weight": 1.0,
        "symmetry_ema_alpha": 0.2,
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),  # Set per-robot.
      },
    ),
    "heelstrike_hip_roll_pattern": RewardTermCfg(
      func=mdp.joint_gait_pattern,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "mode": "heel",
        "target_angle_deg": 5.0,
        "std": math.radians(2.5),
        "std_running": math.radians(5.0),
        "running_threshold": 1.0,
        "window_steps": 10,
        "temporal_std_steps": 5.0,
        "post_reward_steps": 5,
        "post_reward_std_steps": 2,
        "min_air_steps": 10,
        "command_threshold": 0.05,
        "forward_only": True,
        "axis_signs": 1.0,
        "symmetry_penalty_weight": 1.0,
        "symmetry_ema_alpha": 0.2,
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),
      },
    ),
    "toeoff_hip_roll_pattern": RewardTermCfg(
      func=mdp.joint_gait_pattern,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "mode": "toe",
        "target_angle_deg": 0.0,
        "std": math.radians(2.5),
        "std_running": math.radians(5.0),
        "running_threshold": 1.0,
        "window_steps": 10,
        "temporal_std_steps": 5.0,
        "post_reward_steps": 5,
        "post_reward_std_steps": 2,
        "min_contact_steps": 2,
        "min_air_steps": 2,
        "command_threshold": 0.05,
        "forward_only": True,
        "axis_signs": 1.0,
        "symmetry_penalty_weight": 1.0,
        "symmetry_ema_alpha": 0.2,
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),
      },
    ),
    "heelstrike_shoulder_pitch_pattern": RewardTermCfg(
      func=mdp.joint_gait_pattern,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "mode": "heel",
        "target_angle_deg": -15.0,
        "std": math.radians(15.0),
        "std_running": math.radians(10.0),
        "running_threshold": 1.0,
        "window_steps": 10,
        "temporal_std_steps": 5.0,
        "post_reward_steps": 5,
        "post_reward_std_steps": 2,
        "min_air_steps": 10,
        "axis_signs": 1.0,
        "symmetry_penalty_weight": 1.0,
        "symmetry_ema_alpha": 0.2,
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),
      },
    ),
    "toeoff_shoulder_pitch_pattern": RewardTermCfg(
      func=mdp.joint_gait_pattern,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "mode": "toe",
        "target_angle_deg": 5.0,
        "std": math.radians(5.0),
        "std_running": math.radians(2.5),
        "running_threshold": 1.0,
        "window_steps": 10,
        "temporal_std_steps": 5.0,
        "post_reward_steps": 5,
        "post_reward_std_steps": 2,
        "min_contact_steps": 2,
        "min_air_steps": 2,
        "axis_signs": 1.0,
        "symmetry_penalty_weight": 1.0,
        "symmetry_ema_alpha": 0.2,
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),
      },
    ),
    "toeoff_elbow_pattern": RewardTermCfg(
      func=mdp.joint_gait_pattern,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "mode": "toe",
        "target_angle_deg": -10.0,
        "std": math.radians(30.0),
        "std_running": math.radians(20.0),
        "running_threshold": 1.0,
        "window_steps": 25,
        "temporal_std_steps": 20.0,
        "post_reward_steps": 5,
        "post_reward_std_steps": 2,
        "min_contact_steps": 2,
        "min_air_steps": 2,
        "axis_signs": 1.0,
        "symmetry_penalty_weight": 1.0,
        "symmetry_ema_alpha": 0.2,
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),
      },
    ),
    "thigh_swing_target": RewardTermCfg(
      func=mdp.thigh_swing_target,
      weight=0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "lin_vel_y_threshold": 0.05,
        "yaw_threshold": 0.5,
        "target_angle_deg": 2.5,  # inward thigh swing target (positive is hip adduction)
        "std": math.sin(math.radians(2.5)),
        "symmetry_penalty_weight": 1.0,
        "symmetry_ema_alpha": 0.05,
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    # "startup_pitch_upright": RewardTermCfg(
    #   func=mdp.startup_pitch_upright,
    #   weight=-1.0,
    #   params={
    #     "command_name": "twist",
    #     "command_threshold": 0.05,
    #     "pre_window_steps": 5,
    #     "post_window_steps": 5,
    #     "std": math.sin(math.radians(5)),
    #     "axis": 0,
    #     "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
    #   },
    # ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-3,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-2.0,
      params={"sensor_name": "self_collision"},
    ),
    # "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-1.0),
    # "stand_still": RewardTermCfg(
    #   func=mdp.stand_still,
    #   weight=-1.0,
    #   params={
    #     "command_name": "twist",
    #     "command_threshold": 0.1,
    #     "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #   },
    # ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "base_height_low": TerminationTermCfg(
      func=mdp.root_height_below_minimum,
      params={"minimum_height": 0.25},
    ),
    "nan_detected": TerminationTermCfg(func=mdp.nan_detection),
    "out_of_terrain_bounds": TerminationTermCfg(
      func=mdp.out_of_terrain_bounds,
      time_out=True,
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {
            "step": 0,
            "lin_vel_x": (-0.5, 0.5),
            "lin_vel_y": (-0.5, 0.5),
            "ang_vel_z": (-0.5, 0.5),
          },
          {
            "step": 2_500 * 24,
            "lin_vel_x": (-1.0, 1.0),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_z": (-1.0, 1.0),
          },
          {
            "step": 5_000 * 24,
            "lin_vel_x": (-1.5, 2.0),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_z": (-1.5, 1.5),
          },
          {
            "step": 10_000 * 24,
            "lin_vel_x": (-1.5, 3.0),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_z": (-2.0, 2.0),
          },
          {
            "step": 20_000 * 24,
            "lin_vel_x": (-1.5, 3.5),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_z": (-2.0, 2.0),
          },
        ],
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
      ),
      sensors=(terrain_scan, foot_height_scan),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
