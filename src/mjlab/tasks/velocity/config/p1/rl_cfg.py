"""RL configuration for KIMM P1 velocity task.

Provides two actor variants:
- ``p1_ppo_runner_cfg``: baseline MLP actor (matches g1 baseline shape).
- ``p1_moe_ppo_runner_cfg``: MoEActorModel with shared trunk + 3 cmd-axis
  experts (lin_vel_x / lin_vel_y / ang_vel_z) and rule-based hard routing.
Both share the same env / critic / PPO; only the actor block differs.
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.rl.moe_model import MoEActorModelCfg


def p1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Baseline MLP actor + MLP critic for KIMM P1 velocity tracking."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="p1_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def p1_moe_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """MoE actor variant — same env / critic / PPO, swap actor only."""
  cfg = p1_ppo_runner_cfg()
  cfg.experiment_name = "p1_velocity_moe"
  cfg.actor = MoEActorModelCfg(
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    shared_dims=(256, 128),
    expert_dims=(192, 128),
    head_dims=(128,),
    num_experts=3,
    merge="concat",
    cmd_start=-3,
    cmd_end=None,
  )
  return cfg


# Backward-compat aliases (existing tasks referenced these names).
kimm_p1_ppo_runner_cfg = p1_ppo_runner_cfg
kimm_p1_moe_ppo_runner_cfg = p1_moe_ppo_runner_cfg
