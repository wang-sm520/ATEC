import math
import unittest

from atec_rl_lab.train.task_d.box_tracker import TaskDBoxTracker
from atec_rl_lab.train.task_d.state_estimator import TaskDStateEstimator
from atec_rl_lab.train.task_d.types import BoxMeasurement, ObstacleFrame, Pose2D, RobotState, TaskDPhase


class TaskDStateEstimatorTest(unittest.TestCase):
    def test_integrates_yaw_with_upright_projected_gravity(self):
        estimator = TaskDStateEstimator()
        proprio = [0.0] * 12
        proprio[5] = 1.0
        proprio[11] = -1.0

        state = estimator.update(proprio)

        self.assertAlmostEqual(state.pose_world.yaw, 0.02, places=7)
        self.assertAlmostEqual(state.yaw_rate, 1.0, places=7)

    def test_integrates_body_forward_velocity_into_world_x(self):
        estimator = TaskDStateEstimator()
        proprio = [0.0] * 12
        proprio[0] = 1.0
        proprio[11] = -1.0

        for _ in range(10):
            state = estimator.update(proprio)

        self.assertAlmostEqual(state.pose_world.x, -2.8, places=7)
        self.assertAlmostEqual(state.pose_world.y, 0.0, places=7)
        self.assertAlmostEqual(state.pose_obstacle.x, -2.8, places=7)

    def test_uses_valid_obstacle_frame_transform(self):
        estimator = TaskDStateEstimator()
        frame = ObstacleFrame(origin=Pose2D(-3.0, 0.0, math.pi / 2.0), confidence=0.9, valid=True)
        proprio = [0.0] * 12
        proprio[11] = -1.0

        state = estimator.update(proprio, obstacle_frame=frame)

        self.assertAlmostEqual(state.pose_obstacle.x, 0.0, places=7)
        self.assertAlmostEqual(state.pose_obstacle.y, 0.0, places=7)
        self.assertAlmostEqual(state.pose_obstacle.yaw, -math.pi / 2.0, places=7)


class TaskDBoxTrackerTest(unittest.TestCase):
    def test_blends_valid_measurement(self):
        tracker = TaskDBoxTracker()
        measurement = BoxMeasurement(valid=True, pose=Pose2D(-2.0, 1.0, 0.0), source="lidar", confidence=0.8)
        robot_state = TaskDStateEstimator().update([0.0] * 12)

        state = tracker.update(robot_state, measurement, current_score=0.0, phase="ALIGN_BEHIND_BOX")

        self.assertAlmostEqual(state.pose_obstacle.x, -2.65, places=7)
        self.assertAlmostEqual(state.pose_obstacle.y, 1.39, places=7)
        self.assertEqual(state.source, "lidar")
        self.assertGreater(state.confidence, 0.0)

    def test_score_seen_clamps_reward_box_x(self):
        tracker = TaskDBoxTracker()
        measurement = BoxMeasurement.invalid("none")
        robot_state = TaskDStateEstimator().update([0.0] * 12)

        state = tracker.update(robot_state, measurement, current_score=16.0, phase="PUSH_BOX_TO_BRIDGE")

        self.assertTrue(state.reward_seen)
        self.assertGreaterEqual(state.pose_obstacle.x, -1.4)
        self.assertLessEqual(state.pose_obstacle.x, -0.9)
        self.assertEqual(state.source, "score")

    def test_default_contact_offset_matches_push_geometry(self):
        tracker = TaskDBoxTracker()
        robot_state = RobotState(
            pose_world=Pose2D(-3.65, 1.6, 0.0),
            pose_obstacle=Pose2D(-3.65, 1.6, 0.0),
            body_vx=0.0,
            body_vy=0.0,
            yaw_rate=0.0,
            step_count=1,
        )

        state = tracker.update(
            robot_state,
            BoxMeasurement.invalid("none"),
            current_score=0.0,
            phase=TaskDPhase.PUSH_BOX_TO_BRIDGE,
        )

        self.assertAlmostEqual(state.pose_obstacle.x, -3.0)
        self.assertEqual(state.source, "contact")


if __name__ == "__main__":
    unittest.main()
