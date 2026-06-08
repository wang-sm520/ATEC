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


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class G1VelocityPolicyBridgeTest(unittest.TestCase):
    class FakePolicy:
        def __init__(self):
            self.calls = []

        def eval(self):
            return self

        def __call__(self, policy_input):
            self.calls.append(policy_input.detach().clone())
            return torch.zeros((policy_input.shape[0], sol.G1VelocityPolicyBridge.ACTION_DIM_BODY))

    def make_proprio(self, action_dim=33):
        proprio = torch.zeros((1, 12 + 3 * action_dim), dtype=torch.float32)
        proprio[0, 9:12] = torch.tensor([0.0, 0.0, -1.0])
        return proprio

    def test_policy_input_dim_is_960(self):
        bridge = sol.G1VelocityPolicyBridge(policy_path="missing.pt", device="cpu")
        self.assertEqual(bridge.policy_input_dim, 960)

    def test_act_builds_term_major_history_and_returns_full_action_dim(self):
        bridge = sol.G1VelocityPolicyBridge(policy_path="missing.pt", device="cpu")
        fake = self.FakePolicy()
        bridge.policy = fake
        action = bridge.act(self.make_proprio(action_dim=33), (0.25, 0.0, 0.1))
        self.assertEqual(len(action), 33)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(tuple(fake.calls[0].shape), (1, 960))
        cmd_slice = fake.calls[0][0, 30 - 3:30]
        self.assertTrue(torch.allclose(cmd_slice, torch.tensor([0.25, 0.0, 0.1])))

    def test_reset_clears_history_buffers(self):
        bridge = sol.G1VelocityPolicyBridge(policy_path="missing.pt", device="cpu")
        bridge.policy = self.FakePolicy()
        bridge.act(self.make_proprio(action_dim=33), (0.25, 0.0, 0.1))
        self.assertGreater(float(bridge._buf_cmd.abs().sum()), 0.0)
        bridge.reset()
        self.assertEqual(float(bridge._buf_cmd.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
