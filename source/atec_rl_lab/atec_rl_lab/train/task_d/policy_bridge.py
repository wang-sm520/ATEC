"""Velocity-command policy bridge for the Task D G1 locomotion policy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from atec_rl_lab.train.task_d.types import VelocityCommand


class G1VelocityPolicyBridge:
    BODY_29_IDX = list(range(29))
    ACTION_DIM_BODY = 29

    HISTORY_LEN = 10
    DIM_ANG_VEL = 3
    DIM_CMD = 3
    DIM_GRAVITY = 3
    DIM_JP = 29
    DIM_JV = 29
    DIM_LASTACT = 29
    POLICY_INPUT_DIM = HISTORY_LEN * (
        DIM_ANG_VEL + DIM_CMD + DIM_GRAVITY + DIM_JP + DIM_JV + DIM_LASTACT
    )

    TRAINING_ACTION_SCALE_29 = (
        0.231, 0.231, 0.231, 0.231, 0.213, 0.213,
        0.231, 0.231, 0.231, 0.231, 0.213, 0.213,
        0.154, 0.213, 0.213,
        0.373, 0.373, 0.213, 0.373,
        0.23, 0.23, 0.23,
        0.373, 0.373, 0.213, 0.373,
        0.23, 0.23, 0.23,
    )
    EVAL_ACTION_SCALE = 0.5

    def __init__(self, policy_path: str, device: str = "cuda"):
        self.policy_path = policy_path
        self.device = torch.device(device)
        self.policy: Any | None = None

        self.action_scale_ratio = torch.tensor(
            [scale / self.EVAL_ACTION_SCALE for scale in self.TRAINING_ACTION_SCALE_29],
            device=self.device,
            dtype=torch.float32,
        ).view(1, self.ACTION_DIM_BODY)
        self.reset()

    @property
    def policy_input_dim(self) -> int:
        return self.POLICY_INPUT_DIM

    def _make_buffer(self, dim: int) -> torch.Tensor:
        return torch.zeros((self.HISTORY_LEN, dim), device=self.device, dtype=torch.float32)

    def reset(self) -> None:
        self._buf_ang_vel = self._make_buffer(self.DIM_ANG_VEL)
        self._buf_cmd = self._make_buffer(self.DIM_CMD)
        self._buf_gravity = self._make_buffer(self.DIM_GRAVITY)
        self._buf_jp = self._make_buffer(self.DIM_JP)
        self._buf_jv = self._make_buffer(self.DIM_JV)
        self._buf_lastact = self._make_buffer(self.DIM_LASTACT)

    @staticmethod
    def _push(buf: torch.Tensor, new_row: torch.Tensor) -> None:
        buf[:-1] = buf[1:].clone()
        buf[-1] = new_row.reshape(-1)

    def _load_policy(self) -> Any:
        if self.policy is None:
            self.policy = torch.jit.load(self.policy_path, map_location=self.device)
            self.policy.eval()
        return self.policy

    def _coerce_command(self, velocity_command: VelocityCommand | torch.Tensor | Sequence[float]) -> torch.Tensor:
        if isinstance(velocity_command, VelocityCommand):
            values = (velocity_command.vx, velocity_command.vy, velocity_command.wz)
            return torch.tensor(values, device=self.device, dtype=torch.float32)

        if isinstance(velocity_command, torch.Tensor):
            command = velocity_command.to(device=self.device, dtype=torch.float32).reshape(-1)
        else:
            command = torch.as_tensor(list(velocity_command), device=self.device, dtype=torch.float32).reshape(-1)

        if command.numel() != self.DIM_CMD:
            raise ValueError(f"velocity_command must have 3 values, got {command.numel()}")
        return command

    def _build_policy_input(
        self,
        proprio: torch.Tensor,
        velocity_command: VelocityCommand | torch.Tensor | Sequence[float],
    ) -> tuple[torch.Tensor, int]:
        proprio = proprio.to(device=self.device, dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2 or proprio.shape[0] != 1:
            raise ValueError(f"proprio must have shape (1, N) or (N,), got {tuple(proprio.shape)}")

        full_action_dim = (int(proprio.shape[-1]) - 12) // 3
        if 12 + 3 * full_action_dim != int(proprio.shape[-1]) or full_action_dim < self.ACTION_DIM_BODY:
            raise ValueError(f"invalid proprio shape for ATEC layout: {tuple(proprio.shape)}")

        base_ang_vel = proprio[0, 3:6]
        projected_gravity = proprio[0, 9:12]
        command = self._coerce_command(velocity_command)

        jp_start = 12
        jv_start = jp_start + full_action_dim
        act_start = jv_start + full_action_dim

        joint_pos_body = proprio[0, jp_start:jp_start + self.ACTION_DIM_BODY]
        joint_vel_body = proprio[0, jv_start:jv_start + self.ACTION_DIM_BODY]
        last_action_body = (
            proprio[0, act_start:act_start + self.ACTION_DIM_BODY] / self.action_scale_ratio.reshape(-1)
        )

        self._push(self._buf_ang_vel, base_ang_vel)
        self._push(self._buf_cmd, command)
        self._push(self._buf_gravity, projected_gravity)
        self._push(self._buf_jp, joint_pos_body)
        self._push(self._buf_jv, joint_vel_body)
        self._push(self._buf_lastact, last_action_body)

        policy_input = torch.cat(
            [
                self._buf_ang_vel.reshape(-1),
                self._buf_cmd.reshape(-1),
                self._buf_gravity.reshape(-1),
                self._buf_jp.reshape(-1),
                self._buf_jv.reshape(-1),
                self._buf_lastact.reshape(-1),
            ],
            dim=-1,
        ).unsqueeze(0)
        return policy_input, full_action_dim

    def act(
        self,
        proprio: torch.Tensor,
        velocity_command: VelocityCommand | torch.Tensor | Sequence[float],
    ) -> list[float]:
        policy_input, full_action_dim = self._build_policy_input(proprio, velocity_command)

        with torch.inference_mode():
            action_body = self._load_policy()(policy_input)

        if not isinstance(action_body, torch.Tensor):
            action_body = torch.as_tensor(action_body, device=self.device, dtype=torch.float32)
        if action_body.ndim == 1:
            action_body = action_body.unsqueeze(0)
        action_body = action_body.to(device=self.device, dtype=torch.float32)[:, : self.ACTION_DIM_BODY]
        if action_body.shape[-1] != self.ACTION_DIM_BODY:
            raise ValueError(f"policy returned {action_body.shape[-1]} actions, expected 29")

        action_body = action_body * self.action_scale_ratio
        action_full = torch.zeros((1, full_action_dim), device=self.device, dtype=torch.float32)
        action_full[:, self.BODY_29_IDX] = action_body
        return action_full[0].detach().cpu().tolist()
