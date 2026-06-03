"""Body-frame velocity controller for Task D policy input."""

from __future__ import annotations

import math

from .types import ControlTarget, RobotState, TaskDPhase, VelocityCommand, clamp, wrap_to_pi


class TaskDVelocityController:
    def __init__(self, k_xy: float = 1.4, k_yaw: float = 2.0) -> None:
        self.k_xy = k_xy
        self.k_yaw = k_yaw

    def command(self, robot_state: RobotState, target: ControlTarget) -> VelocityCommand:
        robot_pose = robot_state.pose_obstacle
        target_pose = target.pose_obstacle

        error_x = target_pose.x - robot_pose.x
        error_y = target_pose.y - robot_pose.y
        cos_yaw = math.cos(robot_pose.yaw)
        sin_yaw = math.sin(robot_pose.yaw)

        body_error_x = cos_yaw * error_x + sin_yaw * error_y
        body_error_y = -sin_yaw * error_x + cos_yaw * error_y
        yaw_error = wrap_to_pi(target_pose.yaw - robot_pose.yaw)

        command = VelocityCommand(
            vx=self.k_xy * body_error_x,
            vy=self.k_xy * body_error_y,
            wz=self.k_yaw * yaw_error,
        ).clamped(vx_range=(-0.5, 1.2), vy_abs=0.6, wz_abs=1.57)

        if target.phase == TaskDPhase.PUSH_BOX_TO_BRIDGE:
            command = VelocityCommand(
                vx=clamp(command.vx, 0.35, 0.8),
                vy=clamp(command.vy, -0.25, 0.25),
                wz=command.wz,
            )

        return command
