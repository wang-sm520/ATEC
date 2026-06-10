"""ATEC Task D submission — self-contained single-file solution.

Pushes the box into x in [-1.4, 0.7] (+14) and advances the robot past x=-1.4 (+2),
for 16 points, reusing the frozen G1 locomotion policy (policy.pt) as a velocity
tracker. NO atec_rl_lab imports: the eval container only ships this file + policy.pt.

Strategy (validated in sim, true score 16):
  warmup -> approach (get behind the box) -> push (drive +x until the box jams against
  the platform, ~x=-1.07) -> advance (sidestep to clear flat ground, cross x=-1.4).
Robot pose is dead-reckoned from the measured base velocity in obs['proprio'];
the box starts at the deterministic world pose (-3, 1.6).
"""

from __future__ import annotations

import math
import os
from typing import Any, Sequence

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
# The bridge expects the Task A AMP actor (960-dim term-major history). Locally it
# lives as policy_a.pt; in the submission container only policy.pt is shipped, so
# copy policy_a.pt -> policy.pt before building. policy_a.pt is preferred when both
# exist (demo's stock policy.pt is a DIFFERENT 1040-dim model and will not work).
_POLICY_PATH = None
for _name in ("policy_a.pt", "policy.pt"):
    _p = os.path.join(_DIR, _name)
    if os.path.exists(_p):
        _POLICY_PATH = _p
        break


# --------------------------------------------------------------------------- #
# Small math helpers
# --------------------------------------------------------------------------- #
def _wrap_to_pi(angle: float) -> float:
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if math.isclose(wrapped, -math.pi, abs_tol=1e-12) else wrapped


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _as_float(value: Any) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


# --------------------------------------------------------------------------- #
# Dead-reckoning odometry from proprioception
# --------------------------------------------------------------------------- #
class _Odometry:
    def __init__(self, dt: float = 0.02, x0: float = -3.0, y0: float = 0.0):
        self.dt, self.x0, self.y0 = dt, x0, y0
        self.reset()

    def reset(self) -> None:
        self.x, self.y, self.yaw = self.x0, self.y0, 0.0
        self.vx_b = self.vy_b = 0.0

    def update(self, proprio_row: Sequence[float]) -> tuple[float, float, float]:
        lin = [_as_float(proprio_row[i]) for i in range(0, 3)]
        ang = [_as_float(proprio_row[i]) for i in range(3, 6)]
        grav = [_as_float(proprio_row[i]) for i in range(9, 12)]
        up = self._normalized([-grav[0], -grav[1], -grav[2]])
        yaw_rate = sum(a * u for a, u in zip(ang, up))
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (c * lin[0] - s * lin[1]) * self.dt
        self.y += (s * lin[0] + c * lin[1]) * self.dt
        self.yaw = _wrap_to_pi(self.yaw + yaw_rate * self.dt)
        self.vx_b, self.vy_b = lin[0], lin[1]
        return self.x, self.y, self.yaw

    @staticmethod
    def _normalized(v):
        n = math.sqrt(sum(c * c for c in v))
        return [0.0, 0.0, 1.0] if n <= 1e-8 else [c / n for c in v]


# --------------------------------------------------------------------------- #
# Velocity-command -> joint action bridge around the frozen G1 locomotion policy
# --------------------------------------------------------------------------- #
class _G1VelocityPolicyBridge:
    ACTION_DIM_BODY = 29
    HISTORY_LEN = 10
    DIMS = (3, 3, 3, 29, 29, 29)  # ang_vel, cmd, gravity, jp, jv, last_act
    TRAINING_ACTION_SCALE_29 = (
        0.231, 0.231, 0.231, 0.231, 0.213, 0.213,
        0.231, 0.231, 0.231, 0.231, 0.213, 0.213,
        0.154, 0.213, 0.213,
        0.373, 0.373, 0.213, 0.373, 0.23, 0.23, 0.23,
        0.373, 0.373, 0.213, 0.373, 0.23, 0.23, 0.23,
    )
    EVAL_ACTION_SCALE = 0.5

    def __init__(self, policy_path: str, device: str = "cuda"):
        self.policy_path = policy_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.policy = None
        self.action_scale_ratio = torch.tensor(
            [s / self.EVAL_ACTION_SCALE for s in self.TRAINING_ACTION_SCALE_29],
            device=self.device, dtype=torch.float32,
        ).view(1, self.ACTION_DIM_BODY)
        self.reset()

    def reset(self) -> None:
        self._buf = [torch.zeros((self.HISTORY_LEN, d), device=self.device, dtype=torch.float32)
                     for d in self.DIMS]

    @staticmethod
    def _push(buf: torch.Tensor, row: torch.Tensor) -> None:
        buf[:-1] = buf[1:].clone()
        buf[-1] = row.reshape(-1)

    def _load(self):
        if self.policy is None:
            self.policy = torch.jit.load(self.policy_path, map_location=self.device).eval()
        return self.policy

    def act(self, proprio: torch.Tensor, cmd: Sequence[float]) -> list[float]:
        proprio = proprio.to(device=self.device, dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        full_dim = (int(proprio.shape[-1]) - 12) // 3
        b = self.ACTION_DIM_BODY
        ang = proprio[0, 3:6]
        grav = proprio[0, 9:12]
        command = torch.tensor(tuple(cmd), device=self.device, dtype=torch.float32)
        jp = proprio[0, 12:12 + b]
        jv = proprio[0, 12 + full_dim:12 + full_dim + b]
        la = proprio[0, 12 + 2 * full_dim:12 + 2 * full_dim + b] / self.action_scale_ratio.reshape(-1)
        for buf, row in zip(self._buf, (ang, command, grav, jp, jv, la)):
            self._push(buf, row)
        policy_input = torch.cat([buf.reshape(-1) for buf in self._buf], dim=-1).unsqueeze(0)
        with torch.inference_mode():
            out = self._load()(policy_input)
        if not isinstance(out, torch.Tensor):
            out = torch.as_tensor(out, device=self.device, dtype=torch.float32)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        action_body = out.to(device=self.device, dtype=torch.float32)[:, :b] * self.action_scale_ratio
        action_full = torch.zeros((1, full_dim), device=self.device, dtype=torch.float32)
        action_full[:, :b] = action_body
        return action_full[0].detach().cpu().tolist()


# --------------------------------------------------------------------------- #
# Closed-loop pushing state machine
# --------------------------------------------------------------------------- #
class _PushController:
    BOX_START = (-3.0, 1.6)
    CONTACT_OFFSET = 0.65
    BOX_TARGET_X = 0.0

    def __init__(self, warmup_steps: int = 20, dt: float = 0.02):
        self.warmup_steps = warmup_steps
        self.odom = _Odometry(dt=dt)
        self.reset()

    def reset(self) -> None:
        self.odom.reset()
        self.phase = "warmup"
        self.box_x, self.box_y = self.BOX_START
        self.step = 0
        self._jam_ref_x = -1e9
        self._jam_ref_step = 0

    def update(self, proprio_row: Sequence[float]) -> tuple[float, float, float]:
        rx, ry, ryaw = self.odom.update(proprio_row)
        if self.phase == "push":
            pred = rx + self.CONTACT_OFFSET
            if pred > self.box_x:
                self.box_x = pred
        self._update_phase(rx, ry, ryaw)
        cmd = self._control(rx, ry, ryaw)
        self.step += 1
        return cmd

    def _update_phase(self, rx, ry, ryaw):
        if self.phase == "warmup":
            if self.step >= self.warmup_steps:
                self.phase = "approach"
        elif self.phase == "approach":
            if (rx < self.box_x - 0.50 and abs(ry - self.box_y) < 0.13
                    and abs(_wrap_to_pi(ryaw)) < 0.13 and rx > self.box_x - 0.85):
                self.phase, self._jam_ref_x, self._jam_ref_step = "push", rx, self.step
        elif self.phase == "push":
            if self.box_x >= self.BOX_TARGET_X:
                self.phase = "advance"
            elif rx - self._jam_ref_x > 0.06:
                self._jam_ref_x, self._jam_ref_step = rx, self.step
            elif self.step - self._jam_ref_step > 120:
                self.phase = "advance"
        # "advance": once the box is secured, keep walking forward (+x) toward the
        # finish in a clear lane — never stop (may walk into the pit; acceptable).

    def _control(self, rx, ry, ryaw) -> tuple[float, float, float]:
        if self.phase == "warmup":
            return 0.0, 0.0, 0.0
        if self.phase == "hold":
            return 0.0, 0.0, _clamp(2.0 * -ryaw, -0.8, 0.8)
        if self.phase == "approach":
            tx = self.box_x - (0.90 if rx > self.box_x - 0.55 else 0.70)
            return self._goto(rx, ry, ryaw, tx, self.box_y, -0.5, 0.9, 0.6)
        if self.phase == "advance":
            # First sidestep to the clear lane (y~0) WITHOUT going +x (the box is
            # ahead), then keep driving +x toward the finish indefinitely.
            if ry > 0.4:
                return self._goto(rx, ry, ryaw, rx, 0.0, -0.3, 0.25, 0.6)
            return self._goto(rx, ry, ryaw, rx + 3.0, 0.0, 0.3, 0.95, 0.4)
        # push
        ey = self.box_y - ry
        return 0.9, _clamp(1.2 * ey, -0.4, 0.4), _clamp(2.0 * -ryaw - 1.0 * ey, -0.8, 0.8)

    @staticmethod
    def _goto(rx, ry, ryaw, tx, ty, vx_lo, vx_hi, vy_abs):
        ex, ey = tx - rx, ty - ry
        c, s = math.cos(ryaw), math.sin(ryaw)
        return (
            _clamp(1.4 * (c * ex + s * ey), vx_lo, vx_hi),
            _clamp(1.4 * (-s * ex + c * ey), -vy_abs, vy_abs),
            _clamp(2.0 * _wrap_to_pi(-ryaw), -1.0, 1.0),
        )


# --------------------------------------------------------------------------- #
# ATEC entry point
# --------------------------------------------------------------------------- #
class AlgSolution:
    def __init__(self):
        self.bridge = _G1VelocityPolicyBridge(policy_path=_POLICY_PATH)
        self.controller = _PushController()

    def reset(self, **kwargs) -> None:
        self.bridge.reset()
        self.controller.reset()

    def predicts(self, obs: dict, current_score: float):
        proprio = obs["proprio"]
        row = proprio[0] if hasattr(proprio, "shape") and len(proprio.shape) >= 2 else proprio
        cmd = self.controller.update(row)
        action = self.bridge.act(proprio, cmd)
        return {"action": action, "giveup": False}
