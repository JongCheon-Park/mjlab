from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import kimm_p1_flat_env_cfg
from .rl_cfg import kimm_p1_ppo_runner_cfg

# register_mjlab_task(
#   task_id="Mjlab-Velocity-Flat-KIMM-P1",
#   env_cfg=kimm_p1_flat_env_cfg(),
#   play_env_cfg=kimm_p1_flat_env_cfg(play=True),
#   rl_cfg=kimm_p1_ppo_runner_cfg(),
#   runner_cls=VelocityOnPolicyRunner,
# )

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-KIMM-P1",
  env_cfg=kimm_p1_flat_env_cfg(
    observe_waist=True,
    observe_arms=False,
    control_waist=False,
    control_arms=False,
  ),
  play_env_cfg=kimm_p1_flat_env_cfg(
    play=True,
    observe_waist=True,
    observe_arms=False,
    control_waist=False,
    control_arms=False,
  ),
  rl_cfg=kimm_p1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-KIMM-P1-Wholebody",
  env_cfg=kimm_p1_flat_env_cfg(
    observe_waist=True,
    observe_arms=True,
    control_waist=True,
    control_arms=True,
  ),
  play_env_cfg=kimm_p1_flat_env_cfg(
    play=True,
    observe_waist=True,
    observe_arms=True,
    control_waist=True,
    control_arms=True,
  ),
  rl_cfg=kimm_p1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
