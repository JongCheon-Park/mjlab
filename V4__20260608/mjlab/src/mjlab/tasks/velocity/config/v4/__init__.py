from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  kimm_v4_flat_env_cfg,
  kimm_v4_rough_env_cfg,
)
from .rl_cfg import kimm_v4_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-KIMM-V4",
  env_cfg=kimm_v4_rough_env_cfg(),
  play_env_cfg=kimm_v4_rough_env_cfg(play=True),
  rl_cfg=kimm_v4_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-KIMM-V4",
  env_cfg=kimm_v4_flat_env_cfg(),
  play_env_cfg=kimm_v4_flat_env_cfg(play=True),
  rl_cfg=kimm_v4_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
