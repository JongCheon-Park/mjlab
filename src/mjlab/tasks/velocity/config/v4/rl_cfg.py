"""RL configuration for KIMM V4 velocity task — MoE actor."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.rl.moe_model import MoEActorModelCfg


def kimm_v4_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for KIMM V4 velocity task.

  Actor: 4-expert MoE
    - 1 SHARED MLP (always-on)
    - 4 routed experts: vx, vy, yaw, STANDING (1 selected per step)
    - Active per step = shared + 1 expert = 2 paths (총 5 networks 등록)
    - Standing routing: |cmd|_max < 0.05 m/s → STANDING expert

  Critic: plain MLP (512, 256, 128).
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
      num_experts=4,
      merge="concat",
      cmd_start=-3,
      cmd_end=None,
      standing_threshold=0.05,
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
    experiment_name="v4_velocity_moe",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
