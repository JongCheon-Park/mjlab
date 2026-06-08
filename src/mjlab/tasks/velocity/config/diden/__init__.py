from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import diden_flat_env_cfg
from .rl_cfg import diden_moe_ppo_runner_cfg, diden_ppo_runner_cfg

# Legs-only variant (waist_pitch held at default). Recommended starting point.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Diden",
  env_cfg=diden_flat_env_cfg(
    observe_waist=True,
    control_waist=False,
  ),
  play_env_cfg=diden_flat_env_cfg(
    play=True,
    observe_waist=True,
    control_waist=False,
  ),
  rl_cfg=diden_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Diden-MoE",
  env_cfg=diden_flat_env_cfg(
    observe_waist=True,
    control_waist=False,
  ),
  play_env_cfg=diden_flat_env_cfg(
    play=True,
    observe_waist=True,
    control_waist=False,
  ),
  rl_cfg=diden_moe_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Whole-body variant (waist_pitch is controlled).
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Diden-Wholebody",
  env_cfg=diden_flat_env_cfg(
    observe_waist=True,
    control_waist=True,
  ),
  play_env_cfg=diden_flat_env_cfg(
    play=True,
    observe_waist=True,
    control_waist=True,
  ),
  rl_cfg=diden_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
