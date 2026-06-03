"""MAPush-style deterministic waypoint planner for Task D."""

from __future__ import annotations

from .types import BoxState, ControlTarget, Pose2D, RobotState, TaskDPhase


class TaskDWaypointPlanner:
    def target(
        self,
        phase: TaskDPhase,
        robot_state: RobotState,
        box_state: BoxState,
    ) -> ControlTarget:
        box_pose = box_state.pose_obstacle
        robot_pose = robot_state.pose_obstacle

        if phase == TaskDPhase.MOVE_TO_BOX_LANE:
            pose = Pose2D(-3.0, 1.35, 0.0)
            description = "move_to_box_lane"
        elif phase == TaskDPhase.ALIGN_BEHIND_BOX:
            pose = Pose2D(box_pose.x - 0.75, box_pose.y, 0.0)
            description = "align_behind_box"
        elif phase == TaskDPhase.PUSH_BOX_TO_BRIDGE:
            pose = Pose2D(box_pose.x - 0.65, box_pose.y, 0.0)
            description = "push_contact"
        elif phase == TaskDPhase.BACK_OFF_AND_CENTER:
            pose = Pose2D(robot_pose.x - 0.4, box_pose.y, 0.0)
            description = "back_off_and_center"
        elif phase == TaskDPhase.CROSS_ON_BOX:
            pose = Pose2D(2.2, box_pose.y, 0.0)
            description = "cross_on_box"
        elif phase == TaskDPhase.FINISH:
            pose = Pose2D(3.8, 1.2, 0.0)
            description = "finish"
        else:
            pose = robot_pose
            description = "warmup_hold"

        return ControlTarget(pose_obstacle=pose, phase=phase, description=description)
