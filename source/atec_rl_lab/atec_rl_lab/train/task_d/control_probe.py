"""Small deterministic probe for Task D planning and control."""

from __future__ import annotations

from .phase_machine import TaskDPhaseMachine
from .types import BoxState, Pose2D, RobotState
from .velocity_controller import TaskDVelocityController
from .waypoint_planner import TaskDWaypointPlanner


def _robot(x: float, y: float, yaw: float, step: int) -> RobotState:
    return RobotState(
        pose_world=Pose2D(x, y, yaw),
        pose_obstacle=Pose2D(x, y, yaw),
        body_vx=0.0,
        body_vy=0.0,
        yaw_rate=0.0,
        step_count=step,
    )


def _box(x: float, y: float) -> BoxState:
    return BoxState(
        pose_obstacle=Pose2D(x, y, 0.0),
        source="probe",
        confidence=1.0,
    )


def main() -> None:
    machine = TaskDPhaseMachine(warmup_steps=0)
    planner = TaskDWaypointPlanner()
    controller = TaskDVelocityController()
    samples = (
        (_robot(-4.0, 1.0, 0.0, 0), _box(-3.0, 1.6), 0.0),
        (_robot(-3.0, 1.35, 0.0, 1), _box(-3.0, 1.6), 0.0),
        (_robot(-3.75, 1.6, 0.0, 2), _box(-3.0, 1.6), 0.0),
        (_robot(-1.55, 1.6, 0.0, 3), _box(-0.9, 1.6), 0.0),
        (_robot(2.1, 1.6, 0.0, 4), _box(-0.9, 1.6), 1.0),
    )

    for robot_state, box_state, score in samples:
        phase = machine.update(robot_state, box_state, score)
        target = planner.target(phase, robot_state, box_state)
        command = controller.command(robot_state, target)
        print(
            f"phase={phase.value} "
            f"target=({target.pose_obstacle.x:.2f}, {target.pose_obstacle.y:.2f}, {target.pose_obstacle.yaw:.2f}) "
            f"command=({command.vx:.2f}, {command.vy:.2f}, {command.wz:.2f})"
        )


if __name__ == "__main__":
    main()
