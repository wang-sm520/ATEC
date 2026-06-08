"""ATEC Task B G1 baseline: visual search, contact scoring, conservative push.

This file is intentionally self-contained for submission packaging. It does not
import atec_rl_lab modules at runtime; Isaac-only helpers live in scripts/.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - the eval image provides torch.
    torch = None


_DIR = os.path.dirname(os.path.abspath(__file__))
_POLICY_PATH = None
for _name in ("policy_a.pt", "policy.pt"):
    _candidate = os.path.join(_DIR, _name)
    if os.path.exists(_candidate):
        _POLICY_PATH = _candidate
        break


def _wrap_to_pi(angle: float) -> float:
    wrapped = (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if math.isclose(wrapped, -math.pi, abs_tol=1e-12) else wrapped


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _as_float(value: Any) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    def distance_to(self, xy: tuple[float, float]) -> float:
        return math.hypot(float(xy[0]) - self.x, float(xy[1]) - self.y)

    def bearing_to(self, xy: tuple[float, float]) -> float:
        return math.atan2(float(xy[1]) - self.y, float(xy[0]) - self.x)


@dataclass(frozen=True)
class Detection:
    track_id: int
    label: str
    rel_x: float
    rel_y: float
    distance: float
    confidence: float
    world_x: float
    world_y: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class PlannerOutput:
    phase: str
    command: tuple[float, float, float]
    arm_mode: str
    target_world: tuple[float, float] | None = None


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

    def __init__(self, policy_path: str | None, device: str = "cuda"):
        if torch is None:
            raise RuntimeError("torch is required for G1VelocityPolicyBridge")
        self.policy_path = policy_path
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.policy = None
        self.action_scale_ratio = torch.tensor(
            [scale / self.EVAL_ACTION_SCALE for scale in self.TRAINING_ACTION_SCALE_29],
            device=self.device,
            dtype=torch.float32,
        ).view(1, self.ACTION_DIM_BODY)
        self.reset()

    @property
    def policy_input_dim(self) -> int:
        return self.POLICY_INPUT_DIM

    def _make_buffer(self, dim: int):
        return torch.zeros((self.HISTORY_LEN, dim), device=self.device, dtype=torch.float32)

    def reset(self) -> None:
        self._buf_ang_vel = self._make_buffer(self.DIM_ANG_VEL)
        self._buf_cmd = self._make_buffer(self.DIM_CMD)
        self._buf_gravity = self._make_buffer(self.DIM_GRAVITY)
        self._buf_jp = self._make_buffer(self.DIM_JP)
        self._buf_jv = self._make_buffer(self.DIM_JV)
        self._buf_lastact = self._make_buffer(self.DIM_LASTACT)

    @staticmethod
    def _push(buf, row) -> None:
        buf[:-1] = buf[1:].clone()
        buf[-1] = row.reshape(-1)

    def _load_policy(self):
        if self.policy is None:
            if self.policy_path is None:
                raise FileNotFoundError("policy_a.pt or policy.pt is required next to solution_task_b_g1.py")
            self.policy = torch.jit.load(self.policy_path, map_location=self.device)
            self.policy.eval()
        return self.policy

    def _coerce_command(self, command: Sequence[float]):
        out = torch.as_tensor(list(command), device=self.device, dtype=torch.float32).reshape(-1)
        if out.numel() != self.DIM_CMD:
            raise ValueError(f"velocity command must have 3 values, got {out.numel()}")
        return out

    def _build_policy_input(self, proprio, command: Sequence[float]):
        proprio = proprio.to(device=self.device, dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2 or proprio.shape[0] != 1:
            raise ValueError(f"proprio must have shape (1, N) or (N,), got {tuple(proprio.shape)}")

        full_action_dim = (int(proprio.shape[-1]) - 12) // 3
        if 12 + 3 * full_action_dim != int(proprio.shape[-1]):
            raise ValueError(f"invalid ATEC proprio layout: {tuple(proprio.shape)}")
        if full_action_dim < self.ACTION_DIM_BODY:
            raise ValueError(f"G1 policy requires at least 29 action joints, got {full_action_dim}")

        base_ang_vel = proprio[0, 3:6]
        projected_gravity = proprio[0, 9:12]
        cmd = self._coerce_command(command)

        jp_start = 12
        jv_start = jp_start + full_action_dim
        act_start = jv_start + full_action_dim
        joint_pos_body = proprio[0, jp_start:jp_start + self.ACTION_DIM_BODY]
        joint_vel_body = proprio[0, jv_start:jv_start + self.ACTION_DIM_BODY]
        last_action_body = proprio[0, act_start:act_start + self.ACTION_DIM_BODY] / self.action_scale_ratio.reshape(-1)

        self._push(self._buf_ang_vel, base_ang_vel)
        self._push(self._buf_cmd, cmd)
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

    def act(self, proprio, command: Sequence[float]) -> list[float]:
        policy_input, full_action_dim = self._build_policy_input(proprio, command)
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


class DeadReckoningOdometry:
    def __init__(self, dt: float = 0.02, x0: float = -10.0, y0: float = -10.0, yaw0: float = 0.0):
        self.dt = float(dt)
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.yaw0 = float(yaw0)
        self.reset()

    def reset(self) -> Pose2D:
        self.x = self.x0
        self.y = self.y0
        self.yaw = self.yaw0
        self.vx_b = 0.0
        self.vy_b = 0.0
        return self.pose

    @property
    def pose(self) -> Pose2D:
        return Pose2D(self.x, self.y, self.yaw)

    @staticmethod
    def _normalized(v: list[float]) -> list[float]:
        n = math.sqrt(sum(c * c for c in v))
        return [0.0, 0.0, 1.0] if n <= 1e-8 else [c / n for c in v]

    def update(self, proprio_row: Sequence[float]) -> Pose2D:
        lin = [_as_float(proprio_row[i]) for i in range(0, 3)]
        ang = [_as_float(proprio_row[i]) for i in range(3, 6)]
        grav = [_as_float(proprio_row[i]) for i in range(9, 12)]
        up = self._normalized([-grav[0], -grav[1], -grav[2]])
        yaw_rate = sum(a * u for a, u in zip(ang, up))

        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        self.x += (c * lin[0] - s * lin[1]) * self.dt
        self.y += (s * lin[0] + c * lin[1]) * self.dt
        self.yaw = _wrap_to_pi(self.yaw + yaw_rate * self.dt)
        self.vx_b = lin[0]
        self.vy_b = lin[1]
        return self.pose


class AlgSolution:
    """Temporary shell. Later tasks replace this with the full controller."""

    def reset(self, **kwargs) -> None:
        return None

    def predicts(self, obs: dict, current_score: float):
        proprio = obs["proprio"]
        action_dim = (int(proprio.shape[-1]) - 12) // 3 if hasattr(proprio, "shape") else 33
        return {"action": [0.0] * action_dim, "giveup": False}
