"""KIMM P1 task registration.

Wrapped in try/except — P1 depends on some KIMM-internal reward functions
(`thigh_swing_target`, etc.) that may not be present in all mjlab versions.
If registration fails, P1 tasks won't be available but other tasks (V4 etc.)
still load.
"""

try:
  from mjlab.tasks.registry import register_mjlab_task
  from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

  from .env_cfgs import kimm_p1_flat_env_cfg
  from .rl_cfg import p1_moe_ppo_runner_cfg, p1_ppo_runner_cfg

  register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-P1",
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
    rl_cfg=p1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )

  register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-P1-MoE",
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
    rl_cfg=p1_moe_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )

  register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-P1-Wholebody",
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
    rl_cfg=p1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
except (ImportError, KeyError, AttributeError) as e:
  import warnings

  warnings.warn(
    f"P1 task registration skipped (missing dep): {type(e).__name__}: {e}",
    stacklevel=2,
  )
