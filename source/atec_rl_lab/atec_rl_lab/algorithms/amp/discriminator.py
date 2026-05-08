"""AMP discriminator: distinguishes policy (s, s_next) pairs from expert mocap pairs.

Reward formula and grad-penalty regularizer match the bxi/TienKung-Lab implementation, but
the network is built using rsl-rl-lib 3.0.1 primitives (`rsl_rl.modules.MLP`,
`rsl_rl.modules.EmpiricalNormalization`).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from rsl_rl.modules import MLP, EmpiricalNormalization


class Discriminator(nn.Module):
    """Two-input MLP discriminator over (state, next_state) AMP observation pairs.

    Args:
        amp_obs_dim: dimensionality of a single AMP observation (so input to the network
                     is `2 * amp_obs_dim`, the cat of s and s_next).
        hidden_dims: hidden layer sizes for the trunk MLP. Final 1-D logit head is added on top.
        amp_reward_coef: scalar multiplier on the discriminator-derived reward.
        task_reward_lerp: blend coefficient in `[0, 1]` for the task reward; 0 means pure AMP
                          reward, 1 means pure task reward.
        normalize_obs: if True, attach an `EmpiricalNormalization` over amp_obs_dim that is
                       updated outside this module (typically by AMPPPO).
    """

    def __init__(
        self,
        amp_obs_dim: int,
        hidden_dims: Sequence[int] = (1024, 512),
        amp_reward_coef: float = 0.3,
        task_reward_lerp: float = 0.0,
        normalize_obs: bool = True,
    ) -> None:
        super().__init__()
        self.amp_obs_dim = int(amp_obs_dim)
        self.amp_reward_coef = float(amp_reward_coef)
        self.task_reward_lerp = float(task_reward_lerp)

        # Trunk: input is concat(s, s_next), output last hidden dim
        # We use the rsl-rl-lib MLP with output_dim = last hidden, so it ends with linear.
        # Then we add ReLU + a 1-D linear head separately so we can grad-pen on the head output.
        self.trunk = MLP(
            input_dim=2 * self.amp_obs_dim,
            output_dim=int(hidden_dims[-1]),
            hidden_dims=list(hidden_dims[:-1]) if len(hidden_dims) > 1 else list(hidden_dims),
            activation="relu",
            last_activation="relu",  # so trunk output is post-activation features
        )
        self.amp_linear = nn.Linear(int(hidden_dims[-1]), 1)

        if normalize_obs:
            self.normalizer: EmpiricalNormalization | None = EmpiricalNormalization(shape=self.amp_obs_dim, eps=1e-2)
        else:
            self.normalizer = None

    # ---- forward & loss ----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute scalar logits from concatenated (s, s_next).

        Args:
            x: tensor of shape (B, 2 * amp_obs_dim).
        Returns:
            logits of shape (B, 1). No sigmoid applied.
        """
        return self.amp_linear(self.trunk(x))

    def _maybe_normalize(self, state: torch.Tensor, next_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.normalizer is None:
            return state, next_state
        return self.normalizer(state), self.normalizer(next_state)

    def compute_logits(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """Forward over (state, next_state) pair, applying normalizer if attached."""
        s, s_next = self._maybe_normalize(state, next_state)
        return self.forward(torch.cat([s, s_next], dim=-1))

    def compute_grad_pen(
        self, expert_state: torch.Tensor, expert_next_state: torch.Tensor, lambda_: float = 10.0
    ) -> torch.Tensor:
        """R1-style gradient penalty on expert input pairs."""
        s, s_next = self._maybe_normalize(expert_state, expert_next_state)
        expert_data = torch.cat([s, s_next], dim=-1)
        expert_data.requires_grad_(True)
        disc = self.forward(expert_data)
        ones = torch.ones(disc.size(), device=disc.device)
        grad = torch.autograd.grad(
            outputs=disc, inputs=expert_data, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        # Penalty pushes ||grad|| towards 0 (R1)
        grad_pen = lambda_ * (grad.pow(2).sum(dim=-1)).mean()
        return grad_pen

    @torch.no_grad()
    def predict_amp_reward(
        self, state: torch.Tensor, next_state: torch.Tensor, task_reward: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute AMP reward (and raw discriminator output d) from a (s, s_next) pair.

        Reward: `amp_reward_coef * clamp(1 - 0.25 * (d - 1)^2, min=0)` (per Peng et al. 2021 / bxi).

        Returns:
            reward: shape `task_reward.shape` after blending; AMP-only when `task_reward_lerp == 0`.
            d:      raw discriminator logit, shape (B, 1).
        """
        was_training = self.training
        self.eval()
        d = self.compute_logits(state, next_state)
        amp_reward = self.amp_reward_coef * torch.clamp(1.0 - 0.25 * (d - 1.0).pow(2), min=0.0)
        amp_reward = amp_reward.squeeze(-1)  # match task_reward shape (B,)
        if self.task_reward_lerp > 0.0:
            reward = (1.0 - self.task_reward_lerp) * amp_reward + self.task_reward_lerp * task_reward
        else:
            reward = amp_reward
        if was_training:
            self.train()
        return reward, d

    # ---- discriminator training loss helpers ----

    @staticmethod
    def expert_loss(d_expert: torch.Tensor) -> torch.Tensor:
        """LSGAN-style: push expert logits towards +1."""
        return (d_expert - torch.ones_like(d_expert)).pow(2).mean()

    @staticmethod
    def policy_loss(d_policy: torch.Tensor) -> torch.Tensor:
        """LSGAN-style: push policy logits towards -1."""
        return (d_policy + torch.ones_like(d_policy)).pow(2).mean()

    def trunk_parameters(self):
        return self.trunk.parameters()

    def head_parameters(self):
        return self.amp_linear.parameters()


if __name__ == "__main__":
    # Smoke test.
    torch.manual_seed(0)
    amp_obs_dim = 73
    B = 32
    disc = Discriminator(amp_obs_dim=amp_obs_dim, hidden_dims=(64, 32), amp_reward_coef=0.3)
    s = torch.randn(B, amp_obs_dim)
    s_next = torch.randn(B, amp_obs_dim)
    if disc.normalizer is not None:
        disc.normalizer.update(torch.cat([s, s_next], dim=0))

    logits = disc.compute_logits(s, s_next)
    assert logits.shape == (B, 1)

    # AMP reward
    task_r = torch.zeros(B)
    r, d = disc.predict_amp_reward(s, s_next, task_r)
    assert r.shape == (B,)
    assert d.shape == (B, 1)
    print(f"reward range: [{r.min().item():.4f}, {r.max().item():.4f}]")

    # Grad penalty backward
    gp = disc.compute_grad_pen(s, s_next, lambda_=10.0)
    expert_l = disc.expert_loss(disc.compute_logits(s, s_next))
    policy_l = disc.policy_loss(disc.compute_logits(s, s_next))
    loss = expert_l + policy_l + gp
    loss.backward()
    print(f"loss components: expert={expert_l.item():.4f} policy={policy_l.item():.4f} gp={gp.item():.4f}")

    # Buffer test
    from atec_rl_lab.algorithms.amp.replay_buffer import AMPReplayBuffer

    buf = AMPReplayBuffer(obs_dim=amp_obs_dim, capacity=200, device="cpu")
    for _ in range(5):
        buf.insert(torch.randn(64, amp_obs_dim), torch.randn(64, amp_obs_dim))
    assert len(buf) == 200, len(buf)
    for s_b, sn_b in buf.feed_forward_generator(num_mini_batch=2, mini_batch_size=16):
        assert s_b.shape == (16, amp_obs_dim)
        assert sn_b.shape == (16, amp_obs_dim)
    print("Discriminator + ReplayBuffer smoke test passed.")
