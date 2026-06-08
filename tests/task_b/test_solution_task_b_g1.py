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
        cmd_slice = fake.calls[0][0, 30 + 30 - 3:30 + 30]
        self.assertTrue(torch.allclose(cmd_slice, torch.tensor([0.25, 0.0, 0.1])))

    def test_reset_clears_history_buffers(self):
        bridge = sol.G1VelocityPolicyBridge(policy_path="missing.pt", device="cpu")
        bridge.policy = self.FakePolicy()
        bridge.act(self.make_proprio(action_dim=33), (0.25, 0.0, 0.1))
        self.assertGreater(float(bridge._buf_cmd.abs().sum()), 0.0)
        bridge.reset()
        self.assertEqual(float(bridge._buf_cmd.abs().sum()), 0.0)


class DeadReckoningOdometryTest(unittest.TestCase):
    def make_row(self, vx=0.0, vy=0.0, yaw_rate=0.0):
        row = [0.0] * (12 + 3 * 33)
        row[0] = vx
        row[1] = vy
        row[3:6] = [0.0, 0.0, yaw_rate]
        row[9:12] = [0.0, 0.0, -1.0]
        return row

    def test_integrates_body_velocity_in_world_frame(self):
        odom = sol.DeadReckoningOdometry(dt=0.02, x0=-10.0, y0=-10.0)
        pose = odom.update(self.make_row(vx=1.0, vy=0.0))
        self.assertAlmostEqual(pose.x, -9.98, places=5)
        self.assertAlmostEqual(pose.y, -10.0, places=5)
        self.assertAlmostEqual(pose.yaw, 0.0, places=5)

    def test_integrates_yaw_rate_projected_on_up_axis(self):
        odom = sol.DeadReckoningOdometry(dt=0.02, x0=-10.0, y0=-10.0)
        pose = odom.update(self.make_row(yaw_rate=1.0))
        self.assertAlmostEqual(pose.yaw, 0.02, places=5)

    def test_reset_restores_initial_pose(self):
        odom = sol.DeadReckoningOdometry(dt=0.02, x0=-10.0, y0=-10.0)
        odom.update(self.make_row(vx=1.0, yaw_rate=1.0))
        pose = odom.reset()
        self.assertEqual(pose, sol.Pose2D(-10.0, -10.0, 0.0))


class TaskBPlannerTest(unittest.TestCase):
    def detection(self, track_id=1, world=(-9.0, -10.0), rel=(1.0, 0.0), distance=1.0):
        return sol.Detection(
            track_id=track_id,
            label="object",
            rel_x=rel[0],
            rel_y=rel[1],
            distance=distance,
            confidence=0.9,
            world_x=world[0],
            world_y=world[1],
            bbox=(10, 10, 20, 20),
        )

    def test_search_drives_toward_first_waypoint_without_detections(self):
        planner = sol.TaskBPlanner()
        out = planner.step(sol.Pose2D(-10.0, -10.0, 0.0), [], current_score=0.0)
        self.assertEqual(out.phase, "search")
        self.assertEqual(out.arm_mode, "stow")
        self.assertEqual(len(out.command), 3)

    def test_detection_interrupts_search_and_enters_approach(self):
        planner = sol.TaskBPlanner()
        out = planner.step(sol.Pose2D(-10.0, -10.0, 0.0), [self.detection()], current_score=0.0)
        self.assertEqual(out.phase, "approach_object")
        self.assertEqual(out.target_world, (-9.0, -10.0))

    def test_approach_expires_stale_detection_after_missing_frames(self):
        planner = sol.TaskBPlanner()
        planner.step(sol.Pose2D(-10.0, -10.0, 0.0), [self.detection()], current_score=0.0)

        out = None
        for _ in range(31):
            out = planner.step(sol.Pose2D(-10.0, -10.0, 0.0), [], current_score=0.0)

        self.assertEqual(out.phase, "search")
        self.assertIsNone(planner.active_detection)

    def test_close_detection_enters_touch_phase(self):
        planner = sol.TaskBPlanner()
        planner.step(sol.Pose2D(-10.0, -10.0, 0.0), [self.detection()], current_score=0.0)
        out = planner.step(
            sol.Pose2D(-9.45, -10.0, 0.0),
            [self.detection(world=(-9.0, -10.0), rel=(0.45, 0.0), distance=0.45)],
            current_score=0.0,
        )
        self.assertEqual(out.phase, "touch_object")
        self.assertEqual(out.arm_mode, "left_touch")

    def test_score_delta_marks_contact_and_verifies_next(self):
        planner = sol.TaskBPlanner()
        planner.step(sol.Pose2D(-10.0, -10.0, 0.0), [self.detection()], current_score=0.0)
        planner.step(sol.Pose2D(-9.45, -10.0, 0.0), [self.detection(distance=0.45)], current_score=0.0)
        out = planner.step(sol.Pose2D(-9.35, -10.0, 0.0), [self.detection(distance=0.35)], current_score=1.0)
        self.assertEqual(out.phase, "verify_or_next")
        self.assertIn(1, planner.touched_track_ids)

    def test_push_score_marks_placed_without_new_touch(self):
        planner = sol.TaskBPlanner()
        pose = sol.Pose2D(-6.95, -10.0, 0.0)
        det = self.detection(track_id=7, world=(-6.5, -10.0), rel=(0.45, 0.0), distance=0.45)
        planner.step(pose, [det], current_score=0.0)
        planner.step(pose, [det], current_score=0.0)
        out = None
        for _ in range(41):
            out = planner.step(pose, [det], current_score=0.0)
        self.assertEqual(out.phase, "push_to_goal")
        self.assertNotIn(7, planner.touched_track_ids)

        out = planner.step(pose, [det], current_score=1.0)

        self.assertEqual(out.phase, "verify_or_next")
        self.assertIn(7, planner.placed_track_ids)
        self.assertNotIn(7, planner.touched_track_ids)


if __name__ == "__main__":
    unittest.main()
