"""Small Task D tracker probe that runs without IsaacLab."""

from __future__ import annotations

import argparse

from .box_tracker import TaskDBoxTracker
from .state_estimator import TaskDStateEstimator
from .types import BoxMeasurement, Pose2D, TaskDPhase


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mock Task D odometry and box tracking steps.")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--score", type=float, default=0.0)
    parser.add_argument("--phase", default=TaskDPhase.PUSH_BOX_TO_BRIDGE.value)
    args = parser.parse_args()

    estimator = TaskDStateEstimator()
    tracker = TaskDBoxTracker()

    for step in range(args.steps):
        proprio = [0.0] * 12
        proprio[0] = args.vx
        proprio[5] = args.wz
        proprio[11] = -1.0
        robot_state = estimator.update(proprio)

        measurement = BoxMeasurement.invalid("probe")
        if step == 0:
            measurement = BoxMeasurement(True, Pose2D(-3.0, 1.6, 0.0), "probe", 0.4)

        box_state = tracker.update(robot_state, measurement, args.score, args.phase)
        print(
            f"step={step:03d} "
            f"robot=({robot_state.pose_obstacle.x:.3f}, {robot_state.pose_obstacle.y:.3f}, "
            f"{robot_state.pose_obstacle.yaw:.3f}) "
            f"box=({box_state.pose_obstacle.x:.3f}, {box_state.pose_obstacle.y:.3f}) "
            f"source={box_state.source} confidence={box_state.confidence:.2f} "
            f"reward_seen={box_state.reward_seen}"
        )


if __name__ == "__main__":
    main()
