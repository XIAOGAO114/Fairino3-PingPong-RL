"""RSL-RL PPO configuration for the symmetric dual-arm ping-pong task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 4000
    save_interval = 25
    experiment_name = "fairino3_dual_centerline_v1"

    actor = RslRlMLPModelCfg(
        class_name="test_isaac_dual.tasks.manager_based.fairino3_dual_pingpong.models.DualArmActor",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.4,
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0015,  # raised for right-arm fine-tuning
        num_learning_epochs=5,
        num_mini_batches=32,
        learning_rate=3.0e-4,  # raised for right-arm fine-tuning
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
