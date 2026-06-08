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


class AlgSolution:
    """Temporary shell. Later tasks replace this with the full controller."""

    def reset(self, **kwargs) -> None:
        return None

    def predicts(self, obs: dict, current_score: float):
        proprio = obs["proprio"]
        action_dim = (int(proprio.shape[-1]) - 12) // 3 if hasattr(proprio, "shape") else 33
        return {"action": [0.0] * action_dim, "giveup": False}
