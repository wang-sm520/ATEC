"""Task B G1 navigation primitives: pose, dead-reckoning odometry, posture guard.

Ported verbatim from solution_task_b_g1.py. Stdlib-only (math/dataclasses/typing);
PostureGuard duck-types torch/numpy/list inputs and does NOT import torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


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


class PostureGuard:
    """监测 projected_gravity 水平分量，判断是否快栽倒。

    Input: either a 1-D proprio row (list/array of length 12+3*N) OR a 2-D
    batch (list-of-rows / 2-D array), in which case row[0] is used.
    Reads ``row[9:12] == projected_gravity`` (gx, gy, gz); upright ≈ (0,0,-1).
    Return contract: ``"ok"`` | ``"recover"``. As a side effect ``check()``
    also stores the result on ``self.state``.
    """
    # ~20° tilt of the upright axis (asin(0.35) ≈ 20.5°); tune during squat debugging.
    TILT_THRESH = 0.35

    def __init__(self, tilt_thresh=None):
        self.tilt_thresh = float(self.TILT_THRESH if tilt_thresh is None else tilt_thresh)
        self.reset()

    def reset(self):
        self.state = "ok"

    def check(self, proprio_row):
        if hasattr(proprio_row, "ndim"):
            row = proprio_row[0] if proprio_row.ndim == 2 else proprio_row
        elif (hasattr(proprio_row, "__len__") and len(proprio_row) > 0
              and hasattr(proprio_row[0], "__len__")):
            row = proprio_row[0]
        else:
            row = proprio_row
        gx = _as_float(row[9]); gy = _as_float(row[10])
        tilt = math.hypot(gx, gy)
        self.state = "recover" if tilt > self.tilt_thresh else "ok"
        return self.state
