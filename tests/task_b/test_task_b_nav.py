import math
import os
import sys
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from demo import task_b_nav as nav  # noqa: E402


class GeometryHelperTest(unittest.TestCase):
    def test_wrap_to_pi_keeps_angles_in_closed_range(self):
        self.assertAlmostEqual(nav._wrap_to_pi(0.0), 0.0)
        self.assertAlmostEqual(nav._wrap_to_pi(3.0 * math.pi), math.pi)
        self.assertAlmostEqual(nav._wrap_to_pi(-3.0 * math.pi), math.pi)
        self.assertLessEqual(nav._wrap_to_pi(123.4), math.pi)
        self.assertGreaterEqual(nav._wrap_to_pi(123.4), -math.pi)

    def test_clamp_limits_value(self):
        self.assertEqual(nav._clamp(2.0, -1.0, 1.0), 1.0)
        self.assertEqual(nav._clamp(-2.0, -1.0, 1.0), -1.0)
        self.assertEqual(nav._clamp(0.25, -1.0, 1.0), 0.25)

    def test_pose_distance_and_bearing(self):
        pose = nav.Pose2D(x=-10.0, y=-10.0, yaw=0.0)
        self.assertAlmostEqual(pose.distance_to((-9.0, -10.0)), 1.0)
        self.assertAlmostEqual(pose.bearing_to((-10.0, -9.0)), math.pi / 2.0)


class DeadReckoningOdometryTest(unittest.TestCase):
    def make_row(self, vx=0.0, vy=0.0, yaw_rate=0.0):
        row = [0.0] * (12 + 3 * 33)
        row[0] = vx
        row[1] = vy
        row[3:6] = [0.0, 0.0, yaw_rate]
        row[9:12] = [0.0, 0.0, -1.0]
        return row

    def test_integrates_body_velocity_in_world_frame(self):
        odom = nav.DeadReckoningOdometry(dt=0.02, x0=-10.0, y0=-10.0)
        pose = odom.update(self.make_row(vx=1.0, vy=0.0))
        self.assertAlmostEqual(pose.x, -9.98, places=5)
        self.assertAlmostEqual(pose.y, -10.0, places=5)
        self.assertAlmostEqual(pose.yaw, 0.0, places=5)

    def test_integrates_yaw_rate_projected_on_up_axis(self):
        odom = nav.DeadReckoningOdometry(dt=0.02, x0=-10.0, y0=-10.0)
        pose = odom.update(self.make_row(yaw_rate=1.0))
        self.assertAlmostEqual(pose.yaw, 0.02, places=5)

    def test_reset_restores_initial_pose(self):
        odom = nav.DeadReckoningOdometry(dt=0.02, x0=-10.0, y0=-10.0)
        odom.update(self.make_row(vx=1.0, yaw_rate=1.0))
        pose = odom.reset()
        self.assertEqual(pose, nav.Pose2D(-10.0, -10.0, 0.0))


class PostureGuardTest(unittest.TestCase):
    def row(self, gx, gy, gz):
        r = [0.0] * (12 + 3 * 33)
        r[9:12] = [gx, gy, gz]
        return r

    def test_upright_is_ok(self):
        guard = nav.PostureGuard()
        self.assertEqual(guard.check(self.row(0.0, 0.0, -1.0)), "ok")

    def test_large_tilt_is_recover(self):
        guard = nav.PostureGuard()
        self.assertEqual(guard.check(self.row(0.5, 0.0, -0.86)), "recover")

    def test_reset(self):
        guard = nav.PostureGuard()
        guard.check(self.row(0.5, 0.0, -0.86))
        guard.reset()
        self.assertEqual(guard.state, "ok")

    def test_accepts_2d_batch_list(self):
        guard = nav.PostureGuard()
        batch = [[0.0] * 9 + [0.5, 0.0, -0.86] + [0.0] * (3 * 33)]
        self.assertEqual(guard.check(batch), "recover")

    @unittest.skipIf(np is None, "numpy not installed")
    def test_accepts_2d_numpy(self):
        guard = nav.PostureGuard()
        arr = np.zeros((1, 12 + 3 * 33), dtype=np.float32)
        arr[0, 9] = 0.5
        self.assertEqual(guard.check(arr), "recover")
        upright = np.zeros((1, 12 + 3 * 33), dtype=np.float32)
        upright[0, 11] = -1.0
        self.assertEqual(guard.check(upright), "ok")

    def test_ndim1_tensor_like_not_unwrapped(self):
        class TensorLike:
            def __init__(self, data):
                self._d = list(data); self.ndim = 1
            def __len__(self):  # mimic torch: method exists even for scalars
                return len(self._d)
            def __getitem__(self, i):
                v = self._d[i]
                class Scalar:
                    def __init__(self, x): self._x = x
                    def __len__(self): raise TypeError("len() of unsized scalar")
                    def item(self): return self._x
                return Scalar(v)
        guard = nav.PostureGuard()
        row = [0.0]*9 + [0.5, 0.0, -0.86] + [0.0]*(3*33)
        self.assertEqual(guard.check(TensorLike(row)), "recover")


if __name__ == "__main__":
    unittest.main()
