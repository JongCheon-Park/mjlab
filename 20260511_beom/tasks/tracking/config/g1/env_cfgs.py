"""Unitree G1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import (
  discriminator_obs,
  make_tracking_env_cfg,
)


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
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
    ".*_wrist_yaw_link",
    ".*_ankle_roll_link",
  )

  # PD gains randomization: G1 has 6 actuator groups
  # 0: Arms (G1_ACTUATOR_5020)
  # 1: Hip Pitch/Yaw (G1_ACTUATOR_7520_14)
  # 2: Hip Roll / Knee (G1_ACTUATOR_7520_22)
  # 3: Wrist (G1_ACTUATOR_4010)
  # 4: Waist (G1_ACTUATOR_WAIST)
  # 5: Ankle (G1_ACTUATOR_ANKLE)
  cfg.events["pd_gains"].params["asset_cfg"].actuator_ids = [0, 1, 2, 3, 4, 5]

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


def unitree_g1_tracking_add_env_cfg(
  play: bool = False,
  has_state_estimation: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Configuration for G1 Tracking-ADD task with discriminator observations."""

  # Reuse Flat Tracking Config
  cfg = unitree_g1_flat_tracking_env_cfg(
    play=play,
    has_state_estimation=has_state_estimation,
  )

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

  # Add Discriminator Observations
  discriminator_terms = discriminator_obs()

  cfg.observations["discriminator"] = ObservationGroupCfg(
    terms=discriminator_terms,
    concatenate_terms=True,
    enable_corruption=False,
  )

  return cfg
