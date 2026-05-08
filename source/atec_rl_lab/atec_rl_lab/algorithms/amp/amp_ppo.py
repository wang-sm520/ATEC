"""AMP-PPO algorithm for rsl-rl-lib 3.0.1.

Subclass of `rsl_rl.algorithms.PPO` that adds an Adversarial Motion Priors discriminator
on top of the standard PPO objective. Implementation notes:

- The env is expected to expose an "amp" observation group (alongside the standard
  "policy"/"critic" groups). On each env step, `transition.observations["amp"]` (set by
  super().act) is the AMP state at time t, and the post-step `obs["amp"]` is the AMP
  state at time t+1. Together they form the (s, s_next) pair the discriminator scores.
- AMP reward replaces the task reward (or blends with it via `task_reward_lerp`). The
  modified reward is then handed to `super().process_env_step`, so PPO's normal
  bootstrapping/value loss/etc. all use the AMP-shaped reward.
- The discriminator update piggy-backs on PPO's mini-batch loop: per PPO mini-batch we
  sample one AMP policy batch (from the FIFO replay buffer) and one AMP expert batch
  (from the motion loader), add `expert_loss + policy_loss + grad_pen` to the total
  loss, and step the same optimizer (matches bxi/TienKung-Lab).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import chain
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from .discriminator import Discriminator
from .motion_loader_g1 import MotionLoaderG1
from .replay_buffer import AMPReplayBuffer


class AMPPPO(PPO):
    """PPO with an Adversarial Motion Priors discriminator."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        # AMP-specific
        amp_obs_dim: int,
        amp_motion_files: Sequence[str] | None = None,
        amp_time_between_frames: float = 0.02,
        # ---- Defaults aligned with bxi/TienKung-Lab BXDof29WalkAgentCfg ----
        amp_reward_coef: float = 0.3,                     # bxi: 0.3
        amp_task_reward_lerp: float = 0.6,                # bxi: 0.6 (60% task / 40% AMP)
        amp_discr_hidden_dims: Sequence[int] = (1024, 512, 256),  # bxi: 3 layers
        amp_replay_buffer_size: int = 100_000,            # bxi: 100000
        amp_normalize_obs: bool = True,
        amp_grad_pen_lambda: float = 10.0,                # bxi: 10
        amp_num_preload_transitions: int = 200_000,       # bxi: 200000
        amp_loss_coef: float = 1.0,                       # bxi (amploss_coef): 1.0
        amp_trunk_weight_decay: float = 1.0e-3,           # bxi: 10e-4 = 1e-3
        amp_head_weight_decay: float = 1.0e-1,            # bxi: 10e-2 = 1e-1
        # PPO kwargs forwarded to super
        **ppo_kwargs: Any,
    ) -> None:
        super().__init__(actor=actor, critic=critic, storage=storage, **ppo_kwargs)

        self.amp_obs_dim = int(amp_obs_dim)
        self.amp_grad_pen_lambda = float(amp_grad_pen_lambda)
        self.amp_loss_coef = float(amp_loss_coef)

        # Discriminator (always built; if no motion files, AMP loss/reward effectively no-op).
        self.discriminator = Discriminator(
            amp_obs_dim=self.amp_obs_dim,
            hidden_dims=tuple(amp_discr_hidden_dims),
            amp_reward_coef=amp_reward_coef,
            task_reward_lerp=amp_task_reward_lerp,
            normalize_obs=amp_normalize_obs,
        ).to(self.device)

        # Motion expert loader (optional — None means we run as plain PPO)
        if amp_motion_files is not None and len(amp_motion_files) > 0:
            self.motion_loader: MotionLoaderG1 | None = MotionLoaderG1(
                device=self.device,
                time_between_frames=amp_time_between_frames,
                motion_files=list(amp_motion_files),
                preload_transitions=True,
                num_preload_transitions=int(amp_num_preload_transitions),
            )
            if self.motion_loader.observation_dim != self.amp_obs_dim:
                raise ValueError(
                    f"motion_loader.observation_dim={self.motion_loader.observation_dim} "
                    f"!= env amp_obs_dim={self.amp_obs_dim}. Check motion file format."
                )
        else:
            self.motion_loader = None
            print("[AMPPPO] No motion files provided; running as plain PPO (AMP losses skipped).")

        # Replay buffer for policy AMP transitions.
        self.amp_replay_buffer = AMPReplayBuffer(
            obs_dim=self.amp_obs_dim, capacity=int(amp_replay_buffer_size), device=self.device
        )

        # Add discriminator params to the existing PPO optimizer with separate weight decay
        # (matches bxi: trunk has lighter wd, output head has heavier wd).
        self.optimizer.add_param_group(
            {"params": list(self.discriminator.trunk_parameters()), "weight_decay": float(amp_trunk_weight_decay)}
        )
        self.optimizer.add_param_group(
            {"params": list(self.discriminator.head_parameters()), "weight_decay": float(amp_head_weight_decay)}
        )

        # Cache last predicted disc stats for logging.
        self._last_amp_metrics: dict[str, float] = {}

    # ---- env step interception ----

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Replace task reward with AMP reward, store (s, s_next) pair, then defer to PPO."""
        # If we have no motion loader, behave exactly like PPO (no AMP shaping, no buffer).
        if self.motion_loader is None:
            super().process_env_step(obs, rewards, dones, extras)
            return

        if "amp" not in self.transition.observations.keys() or "amp" not in obs.keys():
            raise KeyError(
                "AMPPPO requires the env to expose an 'amp' obs group. "
                "transition.observations keys=" + str(list(self.transition.observations.keys()))
            )

        prev_amp = self.transition.observations["amp"].to(self.device).detach()
        curr_amp = obs["amp"].to(self.device).detach()

        # Replace rewards with AMP-shaped reward (no grad).
        amp_reward, _d = self.discriminator.predict_amp_reward(prev_amp, curr_amp, rewards.to(self.device))

        # Store the policy transition for discriminator training.
        self.amp_replay_buffer.insert(prev_amp, curr_amp)

        # Update normalizer with newly observed states (only in train mode).
        if self.discriminator.normalizer is not None and self.discriminator.normalizer.training:
            self.discriminator.normalizer.update(prev_amp)

        super().process_env_step(obs, amp_reward, dones, extras)

    # ---- update with extra AMP loss ----

    def update(self) -> dict[str, float]:
        """PPO update with an additional discriminator loss per mini-batch.

        We re-implement the body of `PPO.update()` and tack on AMP terms inside the
        mini-batch loop. If no motion_loader is available, falls back to vanilla PPO.
        """
        if self.motion_loader is None or self.amp_replay_buffer.is_empty:
            return super().update()

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_amp_expert_loss = 0.0
        mean_amp_policy_loss = 0.0
        mean_amp_grad_pen = 0.0
        mean_amp_d_real = 0.0
        mean_amp_d_fake = 0.0

        # We don't support symmetry/recurrent + AMP combinations here (keep it simple).
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise NotImplementedError("AMPPPO currently only supports feed-forward actors/critics.")
        if self.symmetry is not None:
            raise NotImplementedError("AMPPPO is not compatible with symmetry augmentation in this implementation.")

        ppo_generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        num_total_minibatches = self.num_mini_batches * self.num_learning_epochs
        amp_policy_gen = self.amp_replay_buffer.feed_forward_generator(
            num_total_minibatches, self._amp_minibatch_size()
        )
        amp_expert_gen = self.motion_loader.feed_forward_generator(
            num_total_minibatches, self._amp_minibatch_size()
        )

        for batch, (amp_pol_s, amp_pol_sn), (amp_exp_s, amp_exp_sn) in zip(
            ppo_generator, amp_policy_gen, amp_expert_gen
        ):
            # ---- standard PPO loss recompute ----
            self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy

            # KL-adaptive learning-rate schedule (same as PPO)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif self.desired_kl / 2.0 > kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # ---- AMP discriminator loss ----
            d_policy = self.discriminator.compute_logits(amp_pol_s, amp_pol_sn)
            d_expert = self.discriminator.compute_logits(amp_exp_s, amp_exp_sn)
            expert_l = self.discriminator.expert_loss(d_expert)
            policy_l = self.discriminator.policy_loss(d_policy)
            grad_pen = self.discriminator.compute_grad_pen(amp_exp_s, amp_exp_sn, lambda_=self.amp_grad_pen_lambda)
            amp_total = self.amp_loss_coef * (0.5 * (expert_l + policy_l) + grad_pen)

            loss = ppo_loss + amp_total

            # ---- single backward + step (covers actor, critic, discriminator) ----
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_amp_expert_loss += expert_l.item()
            mean_amp_policy_loss += policy_l.item()
            mean_amp_grad_pen += grad_pen.item()
            mean_amp_d_real += d_expert.detach().mean().item()
            mean_amp_d_fake += d_policy.detach().mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_amp_expert_loss /= num_updates
        mean_amp_policy_loss /= num_updates
        mean_amp_grad_pen /= num_updates
        mean_amp_d_real /= num_updates
        mean_amp_d_fake /= num_updates

        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp_expert": mean_amp_expert_loss,
            "amp_policy": mean_amp_policy_loss,
            "amp_grad_pen": mean_amp_grad_pen,
            "amp_d_real": mean_amp_d_real,
            "amp_d_fake": mean_amp_d_fake,
        }
        self._last_amp_metrics = loss_dict.copy()
        return loss_dict

    # ---- internal ----

    def _amp_minibatch_size(self) -> int:
        """Match the AMP mini-batch size to PPO's (so generators iterate together)."""
        batch_size = self.storage.num_envs * self.storage.num_transitions_per_env
        return max(1, batch_size // self.num_mini_batches)

    # ---- save/load with discriminator ----

    def save(self) -> dict:
        d = super().save()
        d["discriminator_state_dict"] = self.discriminator.state_dict()
        return d

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        ret = super().load(loaded_dict, load_cfg, strict)
        if "discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=strict)
        return ret

    def train_mode(self) -> None:
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.discriminator.eval()

    # ---- factory used by OnPolicyRunner ----

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "AMPPPO":
        """Mirror `PPO.construct_algorithm` but inject AMP-specific kwargs.

        Required cfg["algorithm"] keys (in addition to PPO's):
            class_name: 'atec_rl_lab.algorithms.amp.amp_ppo.AMPPPO' (or registered alias)
            amp_motion_files: list[str]                       (paths to motion JSONs)
            amp_reward_coef, amp_task_reward_lerp, amp_discr_hidden_dims, ...
        """
        # Resolve class callables (mirrors PPO.construct_algorithm)
        alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class = resolve_callable(cfg["critic"].pop("class_name"))

        # Resolve obs groups: actor / critic / amp (latter optional)
        default_sets = ["actor", "critic"]
        if "amp" in obs.keys():
            default_sets.append("amp")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # No RND / symmetry support for AMP path; keep cfg cleaning consistent
        cfg["algorithm"].setdefault("rnd_cfg", None)
        cfg["algorithm"].setdefault("symmetry_cfg", None)

        # Build actor / critic
        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        # Storage (RL mode)
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Inject env-derived AMP params
        if "amp" in obs.keys():
            amp_obs_dim = int(obs["amp"].shape[-1])
        else:
            amp_obs_dim = int(cfg["algorithm"].get("amp_obs_dim", 0))
            if amp_obs_dim == 0:
                raise ValueError(
                    "AMP obs group not in env observations and amp_obs_dim not set in cfg['algorithm']."
                )
        cfg["algorithm"]["amp_obs_dim"] = amp_obs_dim
        cfg["algorithm"].setdefault("amp_time_between_frames", float(getattr(env, "step_dt", 0.02)))

        alg = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])
        return alg
