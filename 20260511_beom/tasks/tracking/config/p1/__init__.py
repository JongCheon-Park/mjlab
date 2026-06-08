from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import kimm_p1_flat_tracking_env_cfg, kimm_p1_tracking_add_env_cfg
from .rl_cfg import (
  kimm_p1_tracking_add_runner_cfg,
  kimm_p1_tracking_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-KIMM-P1",
  env_cfg=kimm_p1_flat_tracking_env_cfg(),
  play_env_cfg=kimm_p1_flat_tracking_env_cfg(play=True),
  rl_cfg=kimm_p1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Add-Flat-KIMM-P1",
  env_cfg=kimm_p1_tracking_add_env_cfg(),
  play_env_cfg=kimm_p1_tracking_add_env_cfg(play=True),
  rl_cfg=kimm_p1_tracking_add_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
