"""RL configuration for KIMM P1 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def kimm_p1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for KIMM P1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      # hidden_dims=(512, 256, 128),
      # activation="elu",
      # obs_normalization=True,
      class_name="MoEMLPModel",
      activation="elu",
      obs_normalization=True,
      # hidden_dims defines the shared observation trunk before routing.
      hidden_dims=(512, 128, 32),  # input_dim -> hidden_dims -> gate/expert_hidden_dims
      # Shared experts are always active. top_k counts routed specialists only,
      # so top_k=1 with one shared expert means two total active experts per sample.
      learnable_shared_specialist_ratio=False,
      shared_specialist_ratio_init=0.5,  # in (0,1): shared ratio; specialist uses (1 - shared)
      # shared_specialist_ratio_lamda=1.0e-4,  # regularization lambda for shared/specialist ratio anchor
      num_shared_experts=1,
      shared_expert_hidden_dims=(
        256,
      ),  # hidden_dims[-1] -> shared_expert_hidden_dims -> moe_output_dim
      num_experts=4,
      top_k=1,
      expert_hidden_dims=(
        128,
      ),  # hidden_dims[-1] -> expert_hidden_dims -> moe_output_dim
      moe_output_dim=64,  # moe_output_dim  -> output_dim (policy action dim)
      # cmd_regime routing (specialist experts only):
      #   v_lin = [vx, vy], ||v_lin|| = sqrt(vx^2 + vy^2)
      #   regime 0: standing    -> |vx| + |vy| + |yaw| < cmd_threshold
      #   regime 1: turning if |yaw| > cmd_threshold and
      #     (||v_lin|| < cmd_threshold or |yaw| > turn_ratio * ||v_lin||)
      #   regime 2: vx-dominant -> not standing/turning and |vx| >= |vy|
      #   regime 3: vy-dominant -> remaining samples
      #
      # NOTE:
      # - command_slice must point to latest (vx, vy, yaw) in flattened actor obs.
      # - In cmd_regime mode, num_experts must be divisible by 4 and top_k is
      #   constrained in the model to select each regime's specialist block.
      router_mode="cmd_regime",
      router_params={
        "command_slice": (232, 235),  # actor obs slice for latest twist (vx, vy, yaw)
        "cmd_threshold": 0.05,  # shared threshold for standing and turning split
        "turn_ratio": 2.0,  # manual turning ratio
        "vx_vy_ratio": 1.0,  # manual vx/vy dominance ratio
        # DINO-style frozen EMA teacher for always-on shared MoE expert weights.
        "shared_expert_ema_momentum": 0.992,
        "shared_expert_ema_final_momentum": 1.0,
        "shared_expert_ema_loss_coef": 0.005,
        "shared_expert_ema_start_step": 2_500,
        # Shared feature regime invariance with gradient reversal:
        # - classifier learns regime from shared features
        # - gradient reversal flips encoder gradients to remove regime information from shared features
        "shared_regime_grl_coeff": 0.1,
        "shared_regime_loss_coef": 1.0,
        "shared_regime_hidden_dims": (
          128,
        ),  # moe_output_dim -> shared_regime_hidden_dims -> 4-way regime classification logits
      },
      # router_mode="learned",
      # router_params={
      #   "gate_hidden_dims": (64,),  # hidden_dims[-1] -> gate_hidden_dims   -> num_experts
      #   "gate_dropout": 0.1,
      #   "aux_loss_coef": 0.1,
      # },
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
      # symmetry_cfg={
      #   "use_data_augmentation": True,
      #   "use_mirror_loss": True,
      #   "mirror_loss_coeff": (0.0, 0.1),
      #   "mirror_loss_schedule_steps": 5000,
      #   "mirror_loss_coeff": 0.05,
      #   "data_augmentation_func": MirrorFn(),
      # },
    ),
    experiment_name="p1_velocity",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
