"""RL configuration for KIMM p2 tracking task."""

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def kimm_p2_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for KIMM p2 tracking task."""
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
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="p2_tracking",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


@dataclass
class RslRlAddAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "ADDAlgorithm"
  # ADD specific args
  disc_normalizer_stop_iter: int = 2500
  disc_batch_size: int = 2048
  disc_replay_samples: int = 1000
  disc_buffer_size: int = 200000
  disc_loss_weight: float = 5.0
  disc_logit_reg: float = 0.01
  disc_weight_decay: float = 0.0001
  disc_reward_scale: float = 0.25
  disc_hidden_dims: tuple[int, ...] = (1024, 512)
  disc_activation: str = "relu"

  # GP Schedule
  disc_grad_penalty: float | tuple[float, ...] = (10.0, 1.0)
  disc_grad_penalty_decay_steps: int = 100

  # Reward Weight Schedule (ADD <-> DeepMimic Transition)
  reward_warmup_steps: int = 0
  disc_reward_weight: float | tuple[float, ...] = (0.0, 0.5)
  task_reward_weight: float | tuple[float, ...] = (1.0, 0.5)
  reward_weights_schedule_steps: int = 2500


def kimm_p2_tracking_add_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for p2 Tracking-ADD task."""
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
    algorithm=RslRlAddAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="p2_tracking_add",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
      "discriminator": ("discriminator",),
    },
  )
