"""Task D closed-loop velocity-command controller."""

from __future__ import annotations

from typing import Any

from .box_tracker import TaskDBoxTracker
from .lidar_perception import TaskDLidarPerception
from .phase_machine import TaskDPhaseMachine
from .state_estimator import TaskDStateEstimator
from .types import BoxMeasurement, VelocityCommand
from .velocity_controller import TaskDVelocityController
from .waypoint_planner import TaskDWaypointPlanner


class TaskDController:
    """Coordinate perception, tracking, planning, and velocity control."""

    def __init__(
        self,
        warmup_steps: int = 20,
        lidar_perception: TaskDLidarPerception | None = None,
        state_estimator: TaskDStateEstimator | None = None,
        box_tracker: TaskDBoxTracker | None = None,
        phase_machine: TaskDPhaseMachine | None = None,
        waypoint_planner: TaskDWaypointPlanner | None = None,
        velocity_controller: TaskDVelocityController | None = None,
    ) -> None:
        self.lidar_perception = lidar_perception or TaskDLidarPerception()
        self.state_estimator = state_estimator or TaskDStateEstimator()
        self.box_tracker = box_tracker or TaskDBoxTracker()
        self.phase_machine = phase_machine or TaskDPhaseMachine(warmup_steps=warmup_steps)
        self.waypoint_planner = waypoint_planner or TaskDWaypointPlanner()
        self.velocity_controller = velocity_controller or TaskDVelocityController()
        self.last_debug: dict[str, Any] = {}

    def reset(self) -> None:
        self.state_estimator.reset()
        self.box_tracker.reset()
        self.phase_machine.reset()
        self.last_debug = {}

    def update(self, obs: dict[str, Any], current_score: float) -> VelocityCommand:
        proprio = _first_row(obs.get("proprio"))
        extero = obs.get("extero")

        obstacle_measurement, box_measurement = self.lidar_perception.update(extero)
        obstacle_frame = obstacle_measurement.frame

        robot_state = self.state_estimator.update(proprio, obstacle_frame=obstacle_frame)
        previous_phase = self.phase_machine.phase
        box_state = self.box_tracker.update(
            robot_state=robot_state,
            box_measurement=box_measurement if isinstance(box_measurement, BoxMeasurement) else None,
            current_score=float(current_score),
            phase=previous_phase,
        )
        phase = self.phase_machine.update(robot_state, box_state, float(current_score))
        target = self.waypoint_planner.target(phase, robot_state, box_state)
        command = self.velocity_controller.command(robot_state, target)

        self.last_debug = {
            "phase": phase.value,
            "robot_in_obstacle": (
                robot_state.pose_obstacle.x,
                robot_state.pose_obstacle.y,
                robot_state.pose_obstacle.yaw,
            ),
            "box_in_obstacle": (
                box_state.pose_obstacle.x,
                box_state.pose_obstacle.y,
                box_state.pose_obstacle.yaw,
            ),
            "obstacle_valid": obstacle_frame.valid,
            "obstacle_confidence": obstacle_frame.confidence,
            "obstacle_source": obstacle_measurement.source,
            "box_source": box_state.source,
            "box_confidence": box_state.confidence,
            "box_lidar_valid": box_measurement.valid,
            "target": (
                target.pose_obstacle.x,
                target.pose_obstacle.y,
                target.pose_obstacle.yaw,
            ),
            "target_description": target.description,
            "cmd": (command.vx, command.vy, command.wz),
            "lidar_box_debug": self.lidar_perception.box_debug(),
        }
        return command


def _first_row(value: Any) -> Any:
    if value is None:
        raise ValueError("TaskDController requires obs['proprio']")
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 2:
        return value[0]
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return value[0]
    return value
