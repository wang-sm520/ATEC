"""Deterministic phase machine for Task D obstacle-frame pushing."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .types import BoxState, RobotState, TaskDPhase


@dataclass
class TaskDPhaseMachine:
    warmup_steps: int = 20
    phase_timeout_steps: int = 250
    lane_tolerance: float = 0.18
    align_tolerance: float = 0.20
    bridge_box_x: float = -0.95
    crossed_robot_x: float = 2.0
    finish_score: float = 35.0

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._phase = TaskDPhase.WARMUP
        self._phase_enter_step: int | None = None

    @property
    def phase(self) -> TaskDPhase:
        return self._phase

    def update(
        self,
        robot_state: RobotState,
        box_state: BoxState,
        current_score: float,
    ) -> TaskDPhase:
        step = robot_state.step_count
        if self._phase_enter_step is None:
            self._phase_enter_step = step

        if self._is_finished(robot_state, current_score):
            return self._set_phase(TaskDPhase.FINISH, step)

        if self._phase == TaskDPhase.FINISH:
            return self._phase

        if self._phase == TaskDPhase.WARMUP:
            if step - self._phase_enter_step >= self.warmup_steps:
                return self._set_phase(TaskDPhase.MOVE_TO_BOX_LANE, step)
            return self._phase

        if self._timed_out(step):
            return self._advance(step)

        robot_pose = robot_state.pose_obstacle
        box_pose = box_state.pose_obstacle

        if self._phase == TaskDPhase.MOVE_TO_BOX_LANE:
            if math.hypot(robot_pose.x + 3.0, robot_pose.y - 1.35) <= self.lane_tolerance:
                return self._set_phase(TaskDPhase.ALIGN_BEHIND_BOX, step)
        elif self._phase == TaskDPhase.ALIGN_BEHIND_BOX:
            if math.hypot(robot_pose.x - (box_pose.x - 0.75), robot_pose.y - box_pose.y) <= self.align_tolerance:
                return self._set_phase(TaskDPhase.PUSH_BOX_TO_BRIDGE, step)
        elif self._phase == TaskDPhase.PUSH_BOX_TO_BRIDGE:
            if box_pose.x >= self.bridge_box_x:
                return self._set_phase(TaskDPhase.BACK_OFF_AND_CENTER, step)
        elif self._phase == TaskDPhase.BACK_OFF_AND_CENTER:
            if box_pose.x - robot_pose.x >= 0.35 and abs(robot_pose.y - box_pose.y) <= 0.25:
                return self._set_phase(TaskDPhase.CROSS_ON_BOX, step)
        elif self._phase == TaskDPhase.CROSS_ON_BOX:
            if robot_pose.x >= self.crossed_robot_x:
                return self._set_phase(TaskDPhase.FINISH, step)

        return self._phase

    def _is_finished(self, robot_state: RobotState, current_score: float) -> bool:
        return current_score >= self.finish_score or robot_state.pose_obstacle.x >= self.crossed_robot_x

    def _timed_out(self, step: int) -> bool:
        enter_step = self._phase_enter_step if self._phase_enter_step is not None else step
        return step - enter_step >= self.phase_timeout_steps

    def _advance(self, step: int) -> TaskDPhase:
        order = (
            TaskDPhase.WARMUP,
            TaskDPhase.MOVE_TO_BOX_LANE,
            TaskDPhase.ALIGN_BEHIND_BOX,
            TaskDPhase.PUSH_BOX_TO_BRIDGE,
            TaskDPhase.BACK_OFF_AND_CENTER,
            TaskDPhase.CROSS_ON_BOX,
            TaskDPhase.FINISH,
        )
        next_index = min(order.index(self._phase) + 1, len(order) - 1)
        return self._set_phase(order[next_index], step)

    def _set_phase(self, phase: TaskDPhase, step: int) -> TaskDPhase:
        if phase != self._phase:
            self._phase = phase
            self._phase_enter_step = step
        return self._phase
