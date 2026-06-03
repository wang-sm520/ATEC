"""Lightweight robot odometry for Task D."""

from __future__ import annotations

import math
from typing import Sequence

from .types import ObstacleFrame, Pose2D, RobotState, wrap_to_pi


class TaskDStateEstimator:
    """Dead-reckon Task D robot pose from proprioception.

    The estimator intentionally depends only on Python numerics and the shared
    Task D types so it can run in probes and tests without IsaacLab imports.
    """

    def __init__(
        self,
        dt: float = 0.02,
        initial_pose: Pose2D = Pose2D(-3.0, 0.0, 0.0),
        fallback_obstacle_frame: ObstacleFrame | None = None,
    ) -> None:
        self.dt = dt
        self.initial_pose = initial_pose
        self.fallback_obstacle_frame = fallback_obstacle_frame or ObstacleFrame(
            origin=Pose2D(0.0, 0.0, 0.0),
            confidence=1.0,
            valid=True,
        )
        self.reset()

    def reset(self) -> RobotState:
        self.pose_world = self.initial_pose
        self.step_count = 0
        self.body_vx = 0.0
        self.body_vy = 0.0
        self.yaw_rate = 0.0
        return self._state_from_pose(self.fallback_obstacle_frame)

    def update(self, proprio: Sequence[float], obstacle_frame: ObstacleFrame | None = None) -> RobotState:
        if len(proprio) < 12:
            raise ValueError("TaskDStateEstimator requires at least 12 proprio values")

        base_lin_vel = [_as_float(proprio[index]) for index in range(0, 3)]
        base_ang_vel = [_as_float(proprio[index]) for index in range(3, 6)]
        projected_gravity = [_as_float(proprio[index]) for index in range(9, 12)]

        up_body = _normalized([-projected_gravity[0], -projected_gravity[1], -projected_gravity[2]])
        yaw_rate_world = _dot(base_ang_vel, up_body)

        yaw = self.pose_world.yaw
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        vx_world = cos_yaw * base_lin_vel[0] - sin_yaw * base_lin_vel[1]
        vy_world = sin_yaw * base_lin_vel[0] + cos_yaw * base_lin_vel[1]

        self.pose_world = Pose2D(
            x=self.pose_world.x + vx_world * self.dt,
            y=self.pose_world.y + vy_world * self.dt,
            yaw=wrap_to_pi(yaw + yaw_rate_world * self.dt),
        )
        self.step_count += 1
        self.body_vx = base_lin_vel[0]
        self.body_vy = base_lin_vel[1]
        self.yaw_rate = yaw_rate_world

        frame = obstacle_frame if obstacle_frame is not None and obstacle_frame.valid else self.fallback_obstacle_frame
        return self._state_from_pose(frame)

    def _state_from_pose(self, frame: ObstacleFrame) -> RobotState:
        return RobotState(
            pose_world=self.pose_world,
            pose_obstacle=frame.world_to_local(self.pose_world),
            body_vx=self.body_vx,
            body_vy=self.body_vy,
            yaw_rate=self.yaw_rate,
            step_count=self.step_count,
        )


def _as_float(value: object) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _dot(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    return sum(left * right for left, right in zip(lhs, rhs))


def _normalized(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(_dot(vector, vector))
    if norm <= 1.0e-8:
        return [0.0, 0.0, 1.0]
    return [component / norm for component in vector]
