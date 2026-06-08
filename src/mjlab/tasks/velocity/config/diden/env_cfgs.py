"""Diden Walker velocity environment configurations.

Reward stack ported from P4's working CMoE-aligned setup (CMoE paper Table II
+ a few small additions: alive_bonus, upright bonus). Diden differs from P4
in two ways that make this porting simpler than for P4:

  - Joint axes are all L/R aligned (no URDF anti-alignment quirk), so the
    KNEES_BENT keyframe uses symmetric same-sign defaults via regex.
  - Pelvis is the floating root (legs and torso_link both branch from it),
    so the self_collision sensor's "pelvis subtree" naturally covers the
    entire robot — no fix needed there.

Reward weights are kept identical to P4's tuned values so the two robots
share the same learning recipe.
"""

import re

from mjlab.asset_zoo.robots import (
  DIDEN_ACTION_SCALE,
  get_diden_robot_cfg,
)
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
from mjlab.tasks.velocity.mdp import human_walking_rewards as hi_rew
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

LEG_JOINT_PATTERNS = (
  r".*_hip_.*_joint",
  r".*_knee_joint",
  r".*_ankle_.*_joint",
)
WAIST_JOINT_PATTERNS = (r"waist_pitch_joint",)


def _build_joint_patterns(include_waist: bool) -> tuple[str, ...]:
  patterns: list[str] = list(LEG_JOINT_PATTERNS)
  if include_waist:
    patterns.extend(WAIST_JOINT_PATTERNS)
  return tuple(patterns)


def _build_hold_joint_patterns(control_waist: bool) -> tuple[str, ...]:
  if control_waist:
    return ()
  return WAIST_JOINT_PATTERNS


def _filter_scale_dict(
  scale: dict[str, float],
  patterns: tuple[str, ...],
) -> dict[str, float]:
  return {k: v for k, v in scale.items() if any(re.fullmatch(p, k) for p in patterns)}


def diden_rough_env_cfg(
  play: bool = False,
  observe_waist: bool = True,
  control_waist: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Diden Walker rough-terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 45

  cfg.scene.entities = {"robot": get_diden_robot_cfg()}

  # Discard the default reward stack and build a CMoE-aligned one from scratch.
  cfg.rewards.clear()

  # --- Sensors ---
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(f"{side}_foot1_collision" for side in ("left", "right"))

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
  # Self-collision: pelvis is the floating root for Diden, so its subtree
  # covers every body (legs + torso). Internal robot contacts all visible.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
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

  # --- Action / observation ---
  action_patterns = _build_joint_patterns(control_waist)
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = action_patterns
  joint_pos_action.scale = _filter_scale_dict(DIDEN_ACTION_SCALE, action_patterns)

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
  twist_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  if "torso_pseudo_inertia" in cfg.events:
    cfg.events["torso_pseudo_inertia"].params["asset_cfg"].body_names = ("torso_link",)

  # --- CMoE Table II reward stack (with P4's tuning) ---

  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=mdp.track_linear_velocity,
    weight=2.0,
    params={
      "command_name": "twist",
      "std": 0.5,
      "std_walking": 0.5,
      "std_running": 0.5,
    },
  )

  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=mdp.track_angular_velocity,
    weight=2.0,
    params={
      "command_name": "twist",
      "std": 0.5,
    },
  )

  cfg.rewards["body_ang_vel"] = RewardTermCfg(
    func=mdp.body_angular_velocity_penalty,
    weight=-0.05,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
  )

  # Orientation penalty (CMoE Table II value, -2.0).
  cfg.rewards["flat_orientation"] = RewardTermCfg(
    func=envs_mdp.flat_orientation_l2,
    weight=-2.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
  )

  # Positive upright bonus (carrot). Not in CMoE but helps PPO discover
  # upright tracking gait. Small weight (+0.5) so it doesn't override
  # CMoE's reward balance.
  cfg.rewards["upright"] = RewardTermCfg(
    func=mdp.upright,
    weight=0.5,
    params={
      "std": 0.4472,  # sqrt(0.2)
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )

  # Target 0.85 matches measured natural standing pelvis height under
  # KNEES_BENT pose. CMoE's -15 weight was softened to -5 on an
  # uncalibrated robot to avoid dominating the learning signal.
  cfg.rewards["base_height_l2"] = RewardTermCfg(
    func=mdp.base_height_l2,
    weight=-5.0,
    params={"target_height": 0.85},
  )

  # Self-collision: reduced from CMoE's -15 to -2 so a single contact
  # doesn't dominate the per-step reward. Termination still handles
  # truly catastrophic events.
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-2.0,
    params={"sensor_name": "self_collision", "force_threshold": 0.1},
  )

  cfg.rewards["biped_air_time"] = RewardTermCfg(
    func=mdp.biped_air_time,
    weight=1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.05,
      "target_air_time": 0.5,
      "std": 0.3162,
    },
  )

  cfg.rewards["hip_dof_error"] = RewardTermCfg(
    func=envs_mdp.posture,
    weight=0.5,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_hip_.*_joint",)),
      "std": {".*": 0.3},
    },
  )

  cfg.rewards["joint_acc_l2"] = RewardTermCfg(
    func=envs_mdp.joint_acc_l2, weight=-2.5e-7
  )
  cfg.rewards["joint_vel_l2"] = RewardTermCfg(
    func=envs_mdp.joint_vel_l2, weight=-5.0e-4
  )
  cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    func=envs_mdp.joint_torques_l2, weight=-1.0e-5
  )
  cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=envs_mdp.action_rate_l2, weight=-0.3
  )

  cfg.rewards["dof_pos_limits"] = RewardTermCfg(
    func=envs_mdp.joint_pos_limits, weight=-2.0
  )

  cfg.rewards["joint_effort_limit"] = RewardTermCfg(
    func=mdp.joint_effort_limits,
    weight=-1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "soft_ratio": 0.9,
      "power": 1.0,
    },
  )

  cfg.rewards["alive_bonus"] = RewardTermCfg(
    func=hi_rew.alive_bonus,
    weight=0.2,
    params={},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
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


def diden_flat_env_cfg(
  play: bool = False,
  observe_waist: bool = True,
  control_waist: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Diden Walker flat-terrain velocity configuration."""
  cfg = diden_rough_env_cfg(
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
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
