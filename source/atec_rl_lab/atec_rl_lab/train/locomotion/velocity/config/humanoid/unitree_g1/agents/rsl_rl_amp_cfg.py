"""RSL-RL agent cfg for Unitree G1 AMP training.

Mirrors `unitree_b2/agents/rsl_rl_ppo_cfg.py` but:
  - obs_groups maps the env's `policy/critic/amp` groups to the algorithmic actor/critic/amp roles
  - algorithm.class_name is set via train_amp.py at run-time (no class_name field on
    RslRlPpoAlgorithmCfg, so we patch the dict)

Note: the `class_name` for AMPPPO is injected by `scripts/rsl_rl/train_amp.py` via
`patch_agent_dict_for_amp`. Per-run AMP hyperparameters (motion files, reward coef,
etc.) likewise come from the train_amp.py CLI args.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeG1AMPRoughRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 200
    experiment_name = "unitree_g1_amp_rough"
    empirical_normalization = False  # rsl-rl >= 4.0.0 uses per-model obs_normalization

    # Map algorithmic roles -> env obs groups exposed by G1AMPObservationsCfg
    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
        "amp": ["amp"],
    }

    # Use the deprecated `policy` field; handle_deprecated_rsl_rl_cfg() will lift it to
    # actor/critic for rsl-rl >= 4.0.0.
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
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
    )


@configclass
class UnitreeG1AMPFlatRunnerCfg(UnitreeG1AMPRoughRunnerCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "unitree_g1_amp_flat"
        self.max_iterations = 10000
