"""KIMM p2 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  P2_ACTION_SCALE,
  get_p2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import (
  discriminator_obs,
  make_tracking_env_cfg,
)

P2_EXPLICIT_TIMESTEP = 0.001
P2_EXPLICIT_DECIMATION = 20


def kimm_p2_flat_tracking_env_cfg(
  has_state_estimation: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create KIMM p2 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  # if P2ActuatorMode == "explicit":
  #   cfg.sim.mujoco.timestep = P2_EXPLICIT_TIMESTEP
  #   cfg.decimation = P2_EXPLICIT_DECIMATION

  # Add Wrist Joint Tracking Reward
  cfg.rewards["wrist_joint_pos"] = mdp.RewardTermCfg(
    func=mdp.motion_joint_pos_error_exp,
    weight=0.5,
    params={"command_name": "motion", "std": 1.0, "joint_names": (".*_wrist_.*",)},
  )
  cfg.rewards["wrist_joint_vel"] = mdp.RewardTermCfg(
    func=mdp.motion_joint_vel_error_exp,
    weight=0.5,
    params={"command_name": "motion", "std": 6.28, "joint_names": (".*_wrist_.*",)},
  )

  cfg.scene.entities = {"robot": get_p2_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = P2_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "base_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["torso_pseudo_inertia"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["link_pseudo_inertia"].params["asset_cfg"].body_names = (
    # ".*_hip_.*_link",
    ".*_knee_link",
    # ".*_wrist_yaw_link",
    ".*_ankle_roll_link",
  )

  # PD gains randomization: p2 has 5 actuator groups
  # (0=KRO100, 1=KRO80, 2=ANKLE, 3=AK80, 4=AK70).
  cfg.events["pd_gains"].params["asset_cfg"].actuator_ids = [0, 1, 2, 3, 4]

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg


def kimm_p2_tracking_add_env_cfg(
  play: bool = False,
  has_state_estimation: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Configuration for p2 Tracking-ADD task with discriminator observations."""

  # 1. Reuse Flat Tracking Config
  cfg = kimm_p2_flat_tracking_env_cfg(
    play=play,
    has_state_estimation=has_state_estimation,
  )

  # Add detailed self-collision sensor for discriminator (per-body).
  self_collision_detailed_cfg = ContactSensorCfg(
    name="self_collision_detailed",
    primary=ContactMatch(mode="body", pattern=".*_link|base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (*cfg.scene.sensors, self_collision_detailed_cfg)

  # 2. Add Discriminator Observations
  discriminator_terms = discriminator_obs()
  # wrist (p2 specific)
  discriminator_terms["wrist_joint_pos_error"] = ObservationTermCfg(
    func=mdp.joint_pos_error,
    params={"command_name": "motion", "joint_names": (".*_wrist_.*",)},
  )
  discriminator_terms["wrist_joint_vel_error"] = ObservationTermCfg(
    func=mdp.joint_vel_error,
    params={"command_name": "motion", "joint_names": (".*_wrist_.*",)},
  )

  cfg.observations["discriminator"] = ObservationGroupCfg(
    terms=discriminator_terms,
    concatenate_terms=True,
    enable_corruption=False,
  )

  return cfg
