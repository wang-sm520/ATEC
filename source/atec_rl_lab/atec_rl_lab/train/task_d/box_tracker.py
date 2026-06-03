"""Task D box position tracker in obstacle-frame coordinates."""

from __future__ import annotations

from .types import BoxMeasurement, BoxState, Pose2D, RobotState, TaskDPhase, clamp, wrap_to_pi


class TaskDBoxTracker:
    """Track the push box from sparse measurements, contact, and score cues."""

    def __init__(
        self,
        initial_pose_obstacle: Pose2D = Pose2D(-3.0, 1.6, 0.0),
        measurement_alpha: float = 0.35,
        contact_offset: float = 0.65,
    ) -> None:
        self.initial_pose_obstacle = initial_pose_obstacle
        self.measurement_alpha = measurement_alpha
        self.contact_offset = contact_offset
        self.reset()

    def reset(self) -> BoxState:
        self.pose_obstacle = self.initial_pose_obstacle
        self.source = "initial"
        self.confidence = 0.25
        self.reward_seen = False
        return self._state()

    def update(
        self,
        robot_state: RobotState,
        box_measurement: BoxMeasurement | None,
        current_score: float,
        phase: TaskDPhase | str,
    ) -> BoxState:
        if box_measurement is not None and box_measurement.valid and box_measurement.pose is not None:
            self._blend_measurement(box_measurement)

        if self._is_pushing_phase(phase) and self._contact_likely(robot_state):
            robot_pose = robot_state.pose_obstacle
            self.pose_obstacle = Pose2D(
                x=robot_pose.x + self.contact_offset,
                y=_blend(self.pose_obstacle.y, robot_pose.y, 0.25),
                yaw=self.pose_obstacle.yaw,
            )
            self.source = "contact"
            self.confidence = max(self.confidence, 0.55)

        if current_score >= 16.0:
            self.reward_seen = True
            self.pose_obstacle = Pose2D(
                x=clamp(self.pose_obstacle.x, -1.4, -0.9),
                y=self.pose_obstacle.y,
                yaw=self.pose_obstacle.yaw,
            )
            self.source = "score"
            self.confidence = max(self.confidence, 0.8)

        return self._state()

    def _blend_measurement(self, measurement: BoxMeasurement) -> None:
        pose = measurement.pose
        if pose is None:
            return
        alpha = self.measurement_alpha
        self.pose_obstacle = Pose2D(
            x=_blend(self.pose_obstacle.x, pose.x, alpha),
            y=_blend(self.pose_obstacle.y, pose.y, alpha),
            yaw=wrap_to_pi(_blend(self.pose_obstacle.yaw, pose.yaw, alpha)),
        )
        self.source = measurement.source
        self.confidence = max(0.35, min(1.0, measurement.confidence))

    def _contact_likely(self, robot_state: RobotState) -> bool:
        robot_pose = robot_state.pose_obstacle
        predicted_box_x = robot_pose.x + self.contact_offset
        return (
            abs(predicted_box_x - self.pose_obstacle.x) <= 0.8
            and abs(robot_pose.y - self.pose_obstacle.y) <= 0.7
        )

    def _state(self) -> BoxState:
        return BoxState(
            pose_obstacle=self.pose_obstacle,
            source=self.source,
            confidence=self.confidence,
            reward_seen=self.reward_seen,
        )

    @staticmethod
    def _is_pushing_phase(phase: TaskDPhase | str) -> bool:
        value = phase.value if isinstance(phase, TaskDPhase) else str(phase)
        return "PUSH" in value


def _blend(current: float, measurement: float, alpha: float) -> float:
    return current * (1.0 - alpha) + measurement * alpha
