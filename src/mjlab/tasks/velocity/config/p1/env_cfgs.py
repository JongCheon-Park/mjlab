"""KIMM P1 velocity environment configurations."""

import math
import re

from mjlab.asset_zoo.robots import (
  P1_ACTION_SCALE,
  get_p1_robot_cfg,
)
from mjlab.asset_zoo.robots.p1.kimm_p1_constants_wo_4bar import (
  P1_ACTUATOR_MODE as P1ActuatorMode,
)
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

# Joint name patterns for each body group.
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

P1_EXPLICIT_TIMESTEP = 0.001
P1_EXPLICIT_DECIMATION = 20


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
  # Keep entries whose regex would match at least one joint in the patterns.
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


def kimm_p1_rough_env_cfg(
  play: bool = False,
  observe_waist: bool = True,
  observe_arms: bool = True,
  control_waist: bool = True,
  control_arms: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create KIMM P1 rough terrain velocity configuration.

  Args:
    play: Whether to apply play mode overrides.
    observe_waist: Include waist joint in observations.
    observe_arms: Include arm joints in observations.
    control_waist: Include waist joint in action space.
    control_arms: Include arm joints in action space.
  """
  cfg = make_velocity_env_cfg()

  if P1ActuatorMode == "explicit":
    cfg.sim.mujoco.timestep = P1_EXPLICIT_TIMESTEP
    cfg.decimation = P1_EXPLICIT_DECIMATION

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 45

  cfg.scene.entities = {"robot": get_p1_robot_cfg()}

  # Set raycast sensor frame to P1 base_link (root body).
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
  joint_pos_action.scale = _filter_scale_dict(P1_ACTION_SCALE, action_patterns)

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
      r".*hip_yaw.*": 0.6,
      r".*knee.*": 0.6,
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
      # t5: DoF는 풀어주고 reward로 유도 (user feedback). pose는 multi-
      # direction 학습에 충분히 자유롭게. 직진 자세 강제는 forward_straight_gait
      # reward (cmd가 forward-dominant일 때만 활성)로 분리.
      r".*hip_pitch.*": 0.4,
      r".*hip_roll.*": 0.15,
      # hip_yaw 0.45 → 0.20: was loosened to 0.45 earlier to allow
      # swing-leg hip_yaw motion during yaw turn, but the trade-off
      # let the policy abuse hip_yaw during forward / side-step too.
      # Tightened back. Yaw-turn needs are now served by std_turning
      # (0.6) which kicks in when the command is yaw-dominant.
      r".*hip_yaw.*": 0.20,
      r".*knee.*": 0.7,
      r".*ankle_pitch.*": 0.30,
      r".*ankle_roll.*": 0.10,
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
  cfg.rewards["upright"].params["weights"] = [1.0]
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

  # --- Anti-standing-local-minimum fixes (2026-05-11) ---
  # Previous P1-MoE run plateaued at "stand still and exploit exp-tail
  # tracking reward". Relax penalties that block stride motion and weaken the
  # over-strong upright pull.
  cfg.rewards["dof_pos_limits"].weight = -1.0  # was -5.0; allow more joint range

  # --- Anti-reward-collapse fixes (2026-05-11 r3) ---
  # The previous run (15-36-15) peaked at iter 700 with mean_reward=145, then
  # decayed to 55 by iter 1688 — classic over-optimization of one reward at
  # the expense of total reward. The walking emerged (biped_air_time=7.96)
  # but the policy walked TOO violently:
  #   - action_rate_l2 = -3.82/sec (huge)
  #   - angular_momentum = -2.16/sec
  #   - biped_standing_stability = -2.05/sec (couldn't stand still under cmd=0)
  # Inspired by Human-Inspired Adaptive Gait paper (anti-phase hip pitch,
  # in-phase hip roll) and SKATER (mild penalties, strong symmetry):
  # 1) reduce action_rate_l2 further to allow smooth stride
  # 2) keep upright restoration; cut angular_momentum hard (it eats walking)
  # 3) bilateral_symmetry reward (registered below) carries the rhythm signal
  #    rather than relying solely on penalty walls
  cfg.rewards["action_rate_l2"].weight = -0.02  # was -0.05; was -0.1 originally
  cfg.rewards["upright"].weight = 1.0  # restored from 0.5
  cfg.rewards["body_ang_vel"].weight = -0.1  # was -0.2; less aggressive
  cfg.rewards["angular_momentum"].weight = -0.03  # was -0.1; was eating walking
  # `biped_standing_stability` accumulates negative during walking cmd → halve
  # it so it doesn't fight stride.
  cfg.rewards["biped_standing_stability"].weight = 0.5  # was 1.0

  # --- Human-inspired natural-gait additions (HI paper inspired) ---
  # Add explicit bilateral symmetry, energy cost, and survival bonus.
  # See docs/p1_human_walking_reward_design.md §3.5 + §6.
  from mjlab.managers.reward_manager import RewardTermCfg as _RTC
  from mjlab.tasks.velocity.mdp import human_walking_rewards as hi_rew

  # t5: 0.8 (t4)에서 0.6으로 약간 완화. 너무 강하면 turning에서도 대칭 강제하려
  # 해서 turning_factor 감쇠해도 부담. forward_straight_gait가 직진 직진성을
  # 별도로 보강하므로 bilateral_symmetry는 약간 풀어줘도 충분.
  cfg.rewards["bilateral_symmetry"] = _RTC(
    func=hi_rew.bilateral_symmetry,
    weight=0.6,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "command_name": "twist",
      "sigma_sym": 0.4,
      "w_hp": 1.0,  # hip pitch anti-phase (HI: strongest correlation)
      "w_hr": 0.5,  # hip roll mirror
      "w_knee_vel": 0.3,  # knee anti-phase (velocity-based)
      "w_ankle": 0.5,  # ankle pitch anti-phase
      # 1.0 → 0.4: yaw_cmd 0.3 같은 저속 회전에서도 symmetry reward가
      # 0.7 factor로 켜져 turn 자세를 막아서 회전 학습이 죽었음.
      # 0.4 threshold면 |yaw_cmd|>0.4부터 완전 해제, 그 미만 구간만
      # 점진 감쇠 → 직진/저속회전에선 켜지고 회전 명령엔 빠르게 해제.
      "turn_threshold": 0.4,
      "hip_roll_mirror_sign": 1.0,
    },
  )

  cfg.rewards["energy_power"] = _RTC(
    func=hi_rew.joint_power,
    weight=-5.0e-4,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          r".*_hip_.*_joint",
          r".*_knee_joint",
          r".*_ankle_.*_joint",
        ),
      ),
    },
  )

  cfg.rewards["alive_bonus"] = _RTC(
    func=hi_rew.alive_bonus,
    weight=0.2,
    params={},
  )

  # t5: forward-direction quality reward. Activates ONLY when cmd is
  # forward-dominant (|v_y| < 0.1, |yaw| < 0.1). Rewards small hip_yaw and
  # ankle_roll so the policy learns clean straight gait WITHOUT freezing
  # those DoFs globally — turning / side-stepping still free to use them.
  cfg.rewards["forward_straight_gait"] = _RTC(
    func=hi_rew.forward_straight_gait,
    weight=0.5,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "command_name": "twist",
      # 0.1 → 0.3: threshold 0.1이면 cmd drift 0.1 넘어가는 순간 reward
      # factor=0. test run에서 vx 명령 시 vy/yaw drift가 0.2~0.3까지
      # 보였는데 그 구간에서 직진 강제 gradient가 전혀 없었음. 0.3으로
      # 확장하면 약한 drift에서도 점진적으로 reward가 살아남.
      "forward_threshold": 0.3,
      "hip_yaw_sigma": 0.10,
      "ankle_roll_sigma": 0.08,
    },
  )

  # --- Anti-tilt side-step (r3.2) ---
  # When v_y cmd is active, the previous policy tilted body sideways and
  # slid rather than properly side-stepping. Add an explicit reward for the
  # LEADING leg being in the air during lateral motion, and tighten upright
  # y-axis so body lean is more expensive than proper foot work.
  # t5: lateral_step_lead 재활성 (multi-direction 학습 복원).
  cfg.rewards["lateral_step_lead"] = _RTC(
    func=hi_rew.lateral_step_lead,
    weight=0.3,
    params={
      "contact_sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "lateral_threshold": 0.1,
      "left_link_name": "left_ankle_roll_link",
      "right_link_name": "right_ankle_roll_link",
    },
  )
  # t3 hybrid: t1 had sin(1°) (loose) → body tilt during side-step possible.
  # t2 had sin(0.7°) (very tight) → side-step OK but may have over-penalized
  # natural forward sway. Middle ground: sin(0.85°).
  cfg.rewards["upright"].params["std_xy"] = (
    math.sin(math.radians(10.0)),  # x (pitch) unchanged
    math.sin(math.radians(0.85)),  # y (roll) — between t1 sin(1°) and t2 sin(0.7°)
  )

  # --- Anti-mincing (잔발걸음) — r3.1 ---
  # Default min_first_swing_air_time=0.2s allowed the policy to satisfy the
  # "first swing" criterion with very short hops, opening a path to a fast
  # mincing/shuffle gait. HI paper measured human normal walking swing time
  # at 0.40–0.50s, so we raise the floor to 0.35s — any first swing shorter
  # than that incurs penalty, forcing real strides rather than micro-steps.
  # t3 hybrid: between t1 (0.20, too loose — mincing allowed) and
  # t2 (0.35, possibly too restrictive for forward cadence).
  cfg.rewards["biped_first_swing_foot"].params["min_first_swing_air_time"] = 0.30
  cfg.rewards["biped_double_support_time"].params["min_first_swing_air_time"] = 0.30

  ######
  cfg.rewards["track_linear_velocity"].weight = 5.0
  cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)
  # std_walking/std_running: relaxed back from the earlier drift-fix
  # values (√0.08, √0.3) which made the tracking exp too sharp.
  # At cmd vx=1.0 with vx_actual=0 (in-place stepping), the old
  # std_running gave reward ≈ 5 × exp(-1/0.3) ≈ 0.18 — too small to
  # compete with biped_air_time +15. New values give meaningful
  # reward (~+1.5) for partial tracking so PPO finds the forward-
  # motion gradient instead of stalling at in-place stepping.
  cfg.rewards["track_linear_velocity"].params["std_walking"] = math.sqrt(0.20)
  cfg.rewards["track_linear_velocity"].params["std_running"] = math.sqrt(0.5)
  cfg.rewards["track_linear_velocity"].params["walking_threshold"] = 0.5
  cfg.rewards["track_linear_velocity"].params["running_threshold"] = 1.0
  # Equalized with track_linear_velocity weight (5.0). Previously 2.5
  # left yaw tracking at half the gradient strength of linear, so PPO
  # consistently deprioritized yaw learning. Same weight makes yaw
  # tracking competitive in the reward landscape.
  cfg.rewards["track_angular_velocity"].weight = 5.0
  cfg.rewards["track_angular_velocity"].params["low_std"] = math.sqrt(0.25)
  cfg.rewards["track_angular_velocity"].params["high_std"] = math.sqrt(0.5)
  cfg.rewards["track_angular_velocity"].params["speed_threshold"] = 0.5

  # cfg.rewards["joint_torque_rate"].weight = -1.0e-1

  ######

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  feet_target_height = 0.15
  cfg.rewards["foot_clearance"].params["target_height"] = feet_target_height
  # foot_clearance/swing_height: reduced from -5 each → -2 each. Their
  # combined -10 max was comparable to total tracking reward (+7.5),
  # so the policy was pushed to either avoid lifting feet (slide gait)
  # or hyper-fixate on foot-height accuracy at the expense of tracking.
  # -2 keeps the foot-height shaping meaningful without dominating.
  cfg.rewards["foot_clearance"].weight = -2.0
  cfg.rewards["foot_swing_height"].params["target_height"] = feet_target_height
  cfg.rewards["foot_swing_height"].weight = -2.0
  # cfg.rewards["foot_slip"].weight = -1.0

  cfg.rewards["foot_flat"].params["target_height"] = feet_target_height * 0.8
  # t4 (r2) self-audit fix: base weight 0.1 was too small — flat-foot landing
  # signal got drowned by upright (6/sec) / biped_air_time (15/sec). Boost
  # 4× so foot-pitch is meaningfully rewarded during stance.
  cfg.rewards["foot_flat"].weight = 0.4

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

  # Base height target adjusted for P1 standing height (~0.83m).
  cfg.rewards["base_height"].params["target_height"] = 0.83

  cfg.rewards["biped_standing_stability"].weight = 0.5  # r3 fix: was 1.0

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
  # Air-time reward — weight tuned down from 30 to 15 (r3 fix).
  # The aggressive 30-weight pushed the policy to walk too violently,
  # accumulating large action_rate / angular_momentum penalties that
  # eventually overtook the air_time gains (mean_reward decayed 145 → 55).
  # Bilateral symmetry reward (registered below) now carries part of the
  # rhythm signal, allowing a milder air_time weight.
  # biped_air_time weight curriculum: previously 0 → 15 → 15 → 10 which
  # dominated track_linear_velocity (weight 5). Net effect: policy got
  # ~+15 from gait reward by stepping in place regardless of actual
  # forward velocity, and tracking signal at high cmd vx was near zero
  # (exp(-1/0.15) ≈ 0). Reducing the air-time weight ceiling to ~5 makes
  # forward translation worth the action cost. Gait still emerges
  # because the reward is positive over a wide stride range.
  cfg.curriculum["biped_air_time_weight"] = CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
      "reward_name": "biped_air_time",
      "weight_stages": [
        {"step": 0, "weight": 0.0},
        {"step": 250 * 24, "weight": 5.0},
        {"step": 1_000 * 24, "weight": 5.0},
        {"step": 3_000 * 24, "weight": 3.0},
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
          "step": 1_000 * 24,
          "params": {
            "post_landing_mask_steps_low_speed": 10,
            "post_landing_mask_steps_high_speed": 5,
          },
        },
      ],
    },
  )

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

  # Swing-phase ankle_roll wiggle 억제. 사이드보행 시 swing leg가 발목
  # roll을 불필요하게 흔드는 현상 (test run diagnosis). action_rate_l2는
  # 글로벌이라 ankle만 타겟하지 못해 신호가 묻혔음. ankle_roll 속도
  # 자체에 직접 페널티.
  cfg.rewards["ankle_roll_vel_l2"] = RewardTermCfg(
    func=mdp.joint_vel_l2,
    weight=-2.0e-2,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(r".*_ankle_roll_joint",),
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
        # {"step": 1_000 * 24,   "weight": -2.0e-3},
        # {"step": 2_000 * 24,   "weight": -4.0e-3},
        # {"step": 5_000 * 24,   "weight": -0.005},
      ],
    },
  )

  # t3 hybrid: t1 used σ=0.20 from start which produced precise forward
  # gait (user feedback: t1's forward is good). t2's looser σ=0.28 made
  # forward sloppier. Stay tight from the start like t1, then tighten further.
  cfg.curriculum["track_linear_velocity_std"] = CurriculumTermCfg(
    func=envs_mdp.reward_curriculum,
    params={
      "reward_name": "track_linear_velocity",
      "stages": [
        {"step": 0, "params": {"std": math.sqrt(0.04)}},  # σ ≈ 0.20 (t1)
        {"step": 5_000 * 24, "params": {"std": math.sqrt(0.03)}},  # σ ≈ 0.17
        {"step": 15_000 * 24, "params": {"std": math.sqrt(0.022)}},  # σ ≈ 0.15
      ],
    },
  )
  # low_std curriculum was too tight at final stage (σ=0.16), making
  # the tracking landscape essentially binary — only near-perfect yaw
  # gives any reward, anything else ≈ 0. PPO never learns to turn
  # because there is no gradient from random exploration. Keep stage 0
  # loose, then only modestly tighten so partial yaw tracking still
  # gives meaningful reward (~0.4-0.6) for the policy to climb.
  cfg.curriculum["track_angular_velocity_low_std"] = CurriculumTermCfg(
    func=envs_mdp.reward_curriculum,
    params={
      "reward_name": "track_angular_velocity",
      "stages": [
        {"step": 0, "params": {"low_std": math.sqrt(0.25)}},  # σ ≈ 0.50
        {"step": 5_000 * 24, "params": {"low_std": math.sqrt(0.15)}},  # σ ≈ 0.39
        {"step": 15_000 * 24, "params": {"low_std": math.sqrt(0.10)}},  # σ ≈ 0.32
      ],
    },
  )

  # t5: cmd_y / cmd_yaw 복원. 자유도 잠그지 말고 reward로 forward 정밀도
  # 유도 (user feedback). multi-direction 학습이 일반화·robust 함.
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
      "ang_vel_z": (-1.5, 1.5),
    },
    {
      "step": 10_000 * 24,
      "lin_vel_x": (-1.25, 1.25),
      "lin_vel_y": (-1.0, 1.0),
      "ang_vel_z": (-1.5, 1.5),
    },
    {
      "step": 20_000 * 24,
      "lin_vel_x": (-1.25, 1.25),
      "lin_vel_y": (-1.0, 1.0),
      "ang_vel_z": (-2.0, 2.0),
    },
  ]

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("actuator_rfi", None)
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


def kimm_p1_flat_env_cfg(
  play: bool = False,
  observe_waist: bool = True,
  observe_arms: bool = True,
  control_waist: bool = True,
  control_arms: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create KIMM P1 flat terrain velocity configuration."""
  cfg = kimm_p1_rough_env_cfg(
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

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum.
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  if play:
    # Disable command curriculum.
    assert "command_vel" in cfg.curriculum
    del cfg.curriculum["command_vel"]

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
