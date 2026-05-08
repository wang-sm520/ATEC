"""Fixed-capacity FIFO buffer for AMP (state, next_state) transitions.

The discriminator is trained on (s, s_next) pairs sampled from this buffer (policy data)
versus pairs sampled from a MotionLoaderG1 (expert data). Keeping a buffer larger than a
single rollout improves discriminator stability — bxi defaults to 100k.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch


class AMPReplayBuffer:
    """Pre-allocated ring buffer of (state, next_state) pairs on a single device."""

    def __init__(self, obs_dim: int, capacity: int, device: str | torch.device) -> None:
        if capacity <= 0:
            raise ValueError(f"AMPReplayBuffer capacity must be > 0, got {capacity}")
        self.obs_dim = int(obs_dim)
        self.capacity = int(capacity)
        self.device = device
        self.states = torch.zeros(self.capacity, self.obs_dim, device=device)
        self.next_states = torch.zeros(self.capacity, self.obs_dim, device=device)
        self._write_idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def is_empty(self) -> bool:
        return self._size == 0

    def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        """Append a batch of transitions (one per env-step) to the buffer."""
        if states.shape != next_states.shape:
            raise ValueError(f"shape mismatch: {states.shape} vs {next_states.shape}")
        if states.dim() != 2 or states.shape[1] != self.obs_dim:
            raise ValueError(f"expected (B, {self.obs_dim}) tensor, got {tuple(states.shape)}")

        n = states.shape[0]
        if n > self.capacity:
            # Keep the most recent `capacity` items
            states = states[-self.capacity :]
            next_states = next_states[-self.capacity :]
            n = self.capacity

        end = self._write_idx + n
        if end <= self.capacity:
            self.states[self._write_idx : end].copy_(states.detach())
            self.next_states[self._write_idx : end].copy_(next_states.detach())
        else:
            head = self.capacity - self._write_idx
            self.states[self._write_idx :].copy_(states[:head].detach())
            self.next_states[self._write_idx :].copy_(next_states[:head].detach())
            tail = n - head
            self.states[:tail].copy_(states[head:].detach())
            self.next_states[:tail].copy_(next_states[head:].detach())
        self._write_idx = (self._write_idx + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield `num_mini_batch` random (s, s_next) batches each of size mini_batch_size."""
        if self._size == 0:
            raise RuntimeError("AMPReplayBuffer is empty; collect rollouts before calling generator.")
        for _ in range(num_mini_batch):
            idxs = torch.randint(low=0, high=self._size, size=(mini_batch_size,), device=self.device)
            yield self.states[idxs], self.next_states[idxs]
