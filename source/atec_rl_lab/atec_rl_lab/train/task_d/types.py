"""Shared lightweight types for Task D closed-loop control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi], keeping the boundary as +pi."""
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if math.isclose(wrapped, -math.pi, abs_tol=1e-12):
        return math.pi
    return wrapped


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0

    def distance_xy(self, other: "Pose2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class ObstacleFrame:
    origin: Pose2D
    confidence: float = 0.0
    valid: bool = True

    def world_to_local(self, pose: Pose2D) -> Pose2D:
        dx = pose.x - self.origin.x
        dy = pose.y - self.origin.y
        c = math.cos(-self.origin.yaw)
        s = math.sin(-self.origin.yaw)
        return Pose2D(
            x=c * dx - s * dy,
            y=s * dx + c * dy,
            yaw=wrap_to_pi(pose.yaw - self.origin.yaw),
        )

    def local_to_world(self, pose: Pose2D) -> Pose2D:
        c = math.cos(self.origin.yaw)
        s = math.sin(self.origin.yaw)
        return Pose2D(
            x=self.origin.x + c * pose.x - s * pose.y,
            y=self.origin.y + s * pose.x + c * pose.y,
            yaw=wrap_to_pi(pose.yaw + self.origin.yaw),
        )


@dataclass(frozen=True)
class VelocityCommand:
    vx: float
    vy: float
    wz: float

    def clamped(
        self,
        vx_range: tuple[float, float] = (-0.5, 1.2),
        vy_abs: float = 0.6,
        wz_abs: float = 1.57,
    ) -> "VelocityCommand":
        return VelocityCommand(
            vx=clamp(self.vx, vx_range[0], vx_range[1]),
            vy=clamp(self.vy, -vy_abs, vy_abs),
            wz=clamp(self.wz, -wz_abs, wz_abs),
        )


class TaskDPhase(str, Enum):
    WARMUP = "WARMUP"
    MOVE_TO_BOX_LANE = "MOVE_TO_BOX_LANE"
    ALIGN_BEHIND_BOX = "ALIGN_BEHIND_BOX"
    PUSH_BOX_TO_BRIDGE = "PUSH_BOX_TO_BRIDGE"
    BACK_OFF_AND_CENTER = "BACK_OFF_AND_CENTER"
    CROSS_ON_BOX = "CROSS_ON_BOX"
    FINISH = "FINISH"


@dataclass(frozen=True)
class RobotState:
    pose_world: Pose2D
    pose_obstacle: Pose2D
    body_vx: float
    body_vy: float
    yaw_rate: float
    step_count: int


@dataclass(frozen=True)
class ObstacleMeasurement:
    frame: ObstacleFrame
    source: str = "lidar"

    @classmethod
    def invalid(cls, source: str = "lidar") -> "ObstacleMeasurement":
        return cls(
            frame=ObstacleFrame(origin=Pose2D(0.0, 0.0, 0.0), confidence=0.0, valid=False),
            source=source,
        )


@dataclass(frozen=True)
class BoxMeasurement:
    valid: bool
    pose: Pose2D | None
    source: str
    confidence: float = 0.0

    @classmethod
    def invalid(cls, source: str) -> "BoxMeasurement":
        return cls(valid=False, pose=None, source=source, confidence=0.0)


@dataclass(frozen=True)
class BoxState:
    pose_obstacle: Pose2D
    source: str
    confidence: float
    reward_seen: bool = False


@dataclass(frozen=True)
class ControlTarget:
    pose_obstacle: Pose2D
    phase: TaskDPhase
    description: str = ""
