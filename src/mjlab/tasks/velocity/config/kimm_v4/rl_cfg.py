"""RL configuration for KIMM V4 velocity task — MoE actor, Wholebody."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.rl.moe_model import MoEActorModelCfg


def v4_wholebody_moe_ppo_runner_cfg(
  experiment_name: str = "v4_velocity_wholebody_moe",
  num_experts: int = 3,
  standing_threshold: float = 0.0,
) -> RslRlOnPolicyRunnerCfg:
  """MoE actor + MLP critic for KIMM V4 wholebody velocity tracking.

  Args:
    num_experts:
      - 3: classic cmd-axis MoE (vx / vy / yaw experts).
      - 4: adds a "standing" expert selected when |cmd|_max <
        standing_threshold. Set standing_threshold > 0 to activate.
    standing_threshold: m/s and rad/s threshold below which the standing
      expert is selected (only meaningful when num_experts=4).
  """
  return RslRlOnPolicyRunnerCfg(
    actor=MoEActorModelCfg(
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
      num_experts=num_experts,
      merge="concat",
      cmd_start=-3,
      cmd_end=None,
      standing_threshold=standing_threshold,
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
    experiment_name=experiment_name,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
