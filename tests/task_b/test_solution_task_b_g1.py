import math
import os
import sys
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment-specific
    torch = None

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from demo import solution_task_b_g1 as sol  # noqa: E402


class GeometryHelperTest(unittest.TestCase):
    def test_wrap_to_pi_keeps_angles_in_closed_range(self):
        self.assertAlmostEqual(sol._wrap_to_pi(0.0), 0.0)
        self.assertAlmostEqual(sol._wrap_to_pi(3.0 * math.pi), math.pi)
        self.assertAlmostEqual(sol._wrap_to_pi(-3.0 * math.pi), math.pi)
        self.assertLessEqual(sol._wrap_to_pi(123.4), math.pi)
        self.assertGreaterEqual(sol._wrap_to_pi(123.4), -math.pi)

    def test_clamp_limits_value(self):
        self.assertEqual(sol._clamp(2.0, -1.0, 1.0), 1.0)
        self.assertEqual(sol._clamp(-2.0, -1.0, 1.0), -1.0)
        self.assertEqual(sol._clamp(0.25, -1.0, 1.0), 0.25)

    def test_pose_distance_and_bearing(self):
        pose = sol.Pose2D(x=-10.0, y=-10.0, yaw=0.0)
        self.assertAlmostEqual(pose.distance_to((-9.0, -10.0)), 1.0)
        self.assertAlmostEqual(pose.bearing_to((-10.0, -9.0)), math.pi / 2.0)


if __name__ == "__main__":
    unittest.main()
