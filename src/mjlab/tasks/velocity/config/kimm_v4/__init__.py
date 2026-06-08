"""KIMM V4 velocity task registration.

Single fixed config:
  - MoE: SHARED + 4 routed experts (vx, vy, yaw, STANDING) = 5 networks
    (2 active per step: shared + 1 selected expert)
  - Reward: G1-style minimal + foot-lift (biped_air_time, biped_standing_stability)
  - PD: P1-aligned (X12 Kp=509), V4-specific (X8 Kp=250 for standing balance)
  - Pose: KNEES_BENT deep crouch (hip ±15°, knee +35°, ankle -15°)
  - Action scale: X12 0.25 rad (walking exploration), X8 G1-formula
  - All logs go to logs/rsl_rl/v4_velocity_wholebody_moe/
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import v4_flat_env_cfg
from .rl_cfg import v4_wholebody_moe_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-V4",
  env_cfg=v4_flat_env_cfg(
    observe_waist=True,
    control_waist=True,
  ),
  play_env_cfg=v4_flat_env_cfg(
    play=True,
    observe_waist=True,
    control_waist=True,
  ),
  rl_cfg=v4_wholebody_moe_ppo_runner_cfg(
    experiment_name="v4_velocity_wholebody_moe",
    num_experts=4,  # vx + vy + yaw + STANDING
    standing_threshold=0.05,  # |cmd|_max < 0.05 m/s → STANDING expert
  ),
  runner_cls=VelocityOnPolicyRunner,
)
