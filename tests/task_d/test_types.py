import math
import unittest

from atec_rl_lab.train.task_d.types import (
    BoxMeasurement,
    ObstacleFrame,
    Pose2D,
    TaskDPhase,
    VelocityCommand,
    wrap_to_pi,
)


class TaskDTypesTest(unittest.TestCase):
    def test_obstacle_frame_round_trip_transform(self):
        frame = ObstacleFrame(origin=Pose2D(x=1.0, y=2.0, yaw=math.pi / 2), confidence=0.9)
        world_pose = Pose2D(x=1.0, y=3.0, yaw=math.pi)

        local_pose = frame.world_to_local(world_pose)
        restored = frame.local_to_world(local_pose)

        self.assertAlmostEqual(local_pose.x, 1.0)
        self.assertAlmostEqual(local_pose.y, 0.0)
        self.assertAlmostEqual(local_pose.yaw, math.pi / 2)
        self.assertAlmostEqual(restored.x, world_pose.x)
        self.assertAlmostEqual(restored.y, world_pose.y)
        self.assertAlmostEqual(restored.yaw, world_pose.yaw)

    def test_velocity_command_clamps_per_axis(self):
        cmd = VelocityCommand(vx=2.0, vy=-2.0, wz=3.0)
        clipped = cmd.clamped(vx_range=(-0.5, 1.2), vy_abs=0.6, wz_abs=1.57)

        self.assertAlmostEqual(clipped.vx, 1.2)
        self.assertAlmostEqual(clipped.vy, -0.6)
        self.assertAlmostEqual(clipped.wz, 1.57)

    def test_box_measurement_invalid_has_no_pose(self):
        measurement = BoxMeasurement.invalid(source="lidar")

        self.assertFalse(measurement.valid)
        self.assertIsNone(measurement.pose)
        self.assertEqual(measurement.source, "lidar")

    def test_task_d_phase_values_are_stable(self):
        self.assertEqual(TaskDPhase.WARMUP.value, "WARMUP")
        self.assertEqual(TaskDPhase.PUSH_BOX_TO_BRIDGE.value, "PUSH_BOX_TO_BRIDGE")
        self.assertEqual(TaskDPhase.FINISH.value, "FINISH")

    def test_wrap_to_pi_bounds_angle(self):
        self.assertAlmostEqual(wrap_to_pi(3 * math.pi), math.pi)
        self.assertAlmostEqual(wrap_to_pi(-3 * math.pi), math.pi)


if __name__ == "__main__":
    unittest.main()
