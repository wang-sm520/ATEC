import math
import os
import sys
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment-specific
    torch = None

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

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

    def test_touch_expires_stale_detection_after_missing_frames(self):
        planner = sol.TaskBPlanner()
        planner.step(sol.Pose2D(-10.0, -10.0, 0.0), [self.detection()], current_score=0.0)
        planner.step(sol.Pose2D(-9.45, -10.0, 0.0), [self.detection(distance=0.45)], current_score=0.0)

        out = None
        for _ in range(31):
            out = planner.step(sol.Pose2D(-9.45, -10.0, 0.0), [], current_score=0.0)

        self.assertEqual(out.phase, "verify_or_next")
        self.assertEqual(out.arm_mode, "stow")

    def test_push_expires_stale_detection_after_missing_frames(self):
        planner = sol.TaskBPlanner()
        pose = sol.Pose2D(-6.95, -10.0, 0.0)
        det = self.detection(track_id=7, world=(-6.5, -10.0), rel=(0.45, 0.0), distance=0.45)
        planner.step(pose, [det], current_score=0.0)
        planner.step(pose, [det], current_score=0.0)
        out = None
        for _ in range(41):
            out = planner.step(pose, [det], current_score=0.0)
        self.assertEqual(out.phase, "push_to_goal")

        for _ in range(31):
            out = planner.step(pose, [], current_score=0.0)

        self.assertEqual(out.phase, "verify_or_next")
        self.assertEqual(out.arm_mode, "stow")

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

    def test_search_skips_already_placed_detection(self):
        planner = sol.TaskBPlanner()
        pose = sol.Pose2D(-6.95, -10.0, 0.0)
        det = self.detection(track_id=7, world=(-6.5, -10.0), rel=(0.45, 0.0), distance=0.45)
        planner.step(pose, [det], current_score=0.0)
        planner.step(pose, [det], current_score=0.0)
        for _ in range(41):
            planner.step(pose, [det], current_score=0.0)
        planner.step(pose, [det], current_score=1.0)
        self.assertIn(7, planner.placed_track_ids)

        out = None
        for _ in range(20):
            out = planner.step(pose, [], current_score=1.0)
        self.assertEqual(out.phase, "search")

        out = planner.step(pose, [det], current_score=1.0)

        self.assertEqual(out.phase, "search")
        self.assertNotEqual(out.target_world, (det.world_x, det.world_y))


class LocalObjectInteractionTest(unittest.TestCase):
    def test_stow_leaves_action_unchanged(self):
        interaction = sol.LocalObjectInteraction()
        action = [0.1] * 33
        out = interaction.apply_arm_override(action, "stow")
        self.assertEqual(out, action)

    def test_left_touch_overrides_left_arm_and_hands_only(self):
        interaction = sol.LocalObjectInteraction()
        action = [0.1] * 33
        out = interaction.apply_arm_override(action, "left_touch")
        changed = {i for i, (a, b) in enumerate(zip(action, out)) if a != b}
        self.assertTrue({15, 16, 17, 18, 19, 20, 21}.issubset(changed))
        self.assertTrue({29, 30}.issubset(changed))
        self.assertNotIn(0, changed)
        self.assertNotIn(6, changed)

    def test_left_push_uses_more_forward_pose_than_left_touch(self):
        interaction = sol.LocalObjectInteraction()
        touch = interaction.apply_arm_override([0.0] * 33, "left_touch")
        push = interaction.apply_arm_override([0.0] * 33, "left_push")
        self.assertGreaterEqual(push[15], touch[15])
        self.assertGreaterEqual(push[18], touch[18])


class TaskBRgbdPerceptionHelperTest(unittest.TestCase):
    def test_default_hfov_matches_head_camera_config_lateral_scale(self):
        perception = sol.TaskBRgbdPerception()
        expected_hfov = 2.0 * math.atan(20.955 / (2.0 * 24.0))
        self.assertAlmostEqual(perception.hfov, expected_hfov, places=6)

        right_rel_x, right_rel_y = perception._pixel_to_robot_xy(cx=95, width=96, depth=1.0)
        right_edge_x_norm = (95.5 / 96.0) - 0.5
        self.assertAlmostEqual(right_rel_x, 1.0)
        self.assertAlmostEqual(right_rel_y, -math.tan(right_edge_x_norm * expected_hfov), places=6)

        left_rel_x, left_rel_y = perception._pixel_to_robot_xy(cx=0, width=96, depth=1.0)
        left_edge_x_norm = (0.5 / 96.0) - 0.5
        self.assertAlmostEqual(left_rel_x, 1.0)
        self.assertAlmostEqual(left_rel_y, -math.tan(left_edge_x_norm * expected_hfov), places=6)
        self.assertGreater(left_rel_y, 0.0)
        self.assertLess(right_rel_y, 0.0)

    def test_assign_track_reserves_used_ids_within_frame(self):
        perception = sol.TaskBRgbdPerception(track_match_dist=1.0)
        perception.tracks[7] = (0.0, 0.0)
        perception.next_track_id = 8
        used_track_ids = set()

        first = perception._assign_track(0.1, 0.0, used_track_ids)
        second = perception._assign_track(0.2, 0.0, used_track_ids)

        self.assertEqual(first, 7)
        self.assertEqual(second, 8)
        self.assertEqual(used_track_ids, {7, 8})


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class TaskBRgbdPerceptionTest(unittest.TestCase):
    def make_image_obs(self):
        rgb = torch.zeros((1, 64, 96, 3), dtype=torch.uint8)
        depth = torch.full((1, 64, 96, 1), 4.0, dtype=torch.float32)
        rgb[0, 28:38, 44:54, 0] = 230
        rgb[0, 28:38, 44:54, 1] = 190
        rgb[0, 28:38, 44:54, 2] = 30
        depth[0, 28:38, 44:54, 0] = 2.0
        return {"head_rgb": rgb, "head_depth": depth}

    def test_detects_synthetic_colored_blob(self):
        perception = sol.TaskBRgbdPerception(min_pixels=20)
        detections = perception.update(self.make_image_obs(), sol.Pose2D(-10.0, -10.0, 0.0))
        self.assertEqual(len(detections), 1)
        det = detections[0]
        self.assertEqual(det.label, "colored_object")
        self.assertGreater(det.confidence, 0.2)
        self.assertAlmostEqual(det.distance, 2.0, delta=0.1)
        self.assertGreater(det.world_x, -10.0)

    def test_suppresses_large_target_colored_blob_but_keeps_small_blob(self):
        pose = sol.Pose2D(-5.0, -10.0, 0.0)
        large_rgb = torch.zeros((1, 64, 96, 3), dtype=torch.uint8)
        large_depth = torch.full((1, 64, 96, 1), 2.0, dtype=torch.float32)
        large_rgb[0, 5:59, 8:88, 0] = 230
        large_rgb[0, 5:59, 8:88, 1] = 90
        large_rgb[0, 5:59, 8:88, 2] = 20

        large_detections = sol.TaskBRgbdPerception(min_pixels=20).update(
            {"head_rgb": large_rgb, "head_depth": large_depth}, pose
        )

        self.assertEqual(large_detections, [])

        small_rgb = torch.zeros((1, 64, 96, 3), dtype=torch.uint8)
        small_depth = torch.full((1, 64, 96, 1), 2.0, dtype=torch.float32)
        small_rgb[0, 28:38, 44:54, 0] = 230
        small_rgb[0, 28:38, 44:54, 1] = 90
        small_rgb[0, 28:38, 44:54, 2] = 20
        small_detections = sol.TaskBRgbdPerception(min_pixels=20).update(
            {"head_rgb": small_rgb, "head_depth": small_depth}, pose
        )

        self.assertEqual(len(small_detections), 1)

    def test_returns_empty_when_no_image_keys_exist(self):
        perception = sol.TaskBRgbdPerception(min_pixels=20)
        detections = perception.update({}, sol.Pose2D(-10.0, -10.0, 0.0))
        self.assertEqual(detections, [])

    def test_returns_empty_when_rgb_depth_spatial_sizes_mismatch(self):
        rgb = torch.zeros((1, 64, 96, 3), dtype=torch.uint8)
        depth = torch.full((1, 32, 48, 1), 2.0, dtype=torch.float32)
        perception = sol.TaskBRgbdPerception(min_pixels=20)

        detections = perception.update({"head_rgb": rgb, "head_depth": depth}, sol.Pose2D(-10.0, -10.0, 0.0))

        self.assertEqual(detections, [])

    def test_tracks_same_blob_with_stable_id(self):
        perception = sol.TaskBRgbdPerception(min_pixels=20)
        first = perception.update(self.make_image_obs(), sol.Pose2D(-10.0, -10.0, 0.0))[0]
        second = perception.update(self.make_image_obs(), sol.Pose2D(-10.0, -10.0, 0.0))[0]
        self.assertEqual(first.track_id, second.track_id)

    def test_distance_uses_only_colored_component_pixels(self):
        rgb = torch.zeros((1, 16, 16, 3), dtype=torch.uint8)
        depth = torch.full((1, 16, 16, 1), 7.0, dtype=torch.float32)
        depth[0, 3:12, 3:12, 0] = 0.4
        border = []
        for y in range(3, 12):
            border.append((y, 3))
            border.append((y, 11))
        for x in range(4, 11):
            border.append((3, x))
            border.append((11, x))
        for y, x in border:
            rgb[0, y, x, 0] = 230
            rgb[0, y, x, 1] = 190
            rgb[0, y, x, 2] = 30
            depth[0, y, x, 0] = 2.0

        perception = sol.TaskBRgbdPerception(min_pixels=20)
        detections = perception.update({"head_rgb": rgb, "head_depth": depth}, sol.Pose2D(-10.0, -10.0, 0.0))

        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].distance, 2.0, delta=0.1)

    def test_two_same_frame_components_get_distinct_track_ids(self):
        rgb = torch.zeros((1, 16, 16, 3), dtype=torch.uint8)
        depth = torch.full((1, 16, 16, 1), 2.0, dtype=torch.float32)
        rgb[0, 4:7, 4:7, 0] = 230
        rgb[0, 4:7, 4:7, 1] = 190
        rgb[0, 4:7, 4:7, 2] = 30
        rgb[0, 4:7, 9:12, 0] = 230
        rgb[0, 4:7, 9:12, 1] = 190
        rgb[0, 4:7, 9:12, 2] = 30
        perception = sol.TaskBRgbdPerception(min_pixels=4, track_match_dist=10.0)
        perception.tracks[5] = (-8.0, -10.0)
        perception.next_track_id = 6

        detections = perception.update({"head_rgb": rgb, "head_depth": depth}, sol.Pose2D(-10.0, -10.0, 0.0))
        track_ids = [det.track_id for det in detections]

        self.assertEqual(len(track_ids), 2)
        self.assertEqual(len(set(track_ids)), 2)
        self.assertIn(5, track_ids)

    def test_large_contiguous_mask_does_not_fragment_into_many_tracks(self):
        rgb = torch.zeros((1, 180, 180, 3), dtype=torch.uint8)
        depth = torch.full((1, 180, 180, 1), 2.0, dtype=torch.float32)
        rgb[0, :, :, 0] = 230
        rgb[0, :, :, 1] = 190
        rgb[0, :, :, 2] = 30
        perception = sol.TaskBRgbdPerception(min_pixels=20)

        detections = perception.update({"head_rgb": rgb, "head_depth": depth}, sol.Pose2D(-20.0, -20.0, 0.0))

        self.assertLessEqual(len(detections), 1)


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class AlgSolutionGlueTest(unittest.TestCase):
    class FakeBridge:
        def __init__(self):
            self.reset_calls = 0
            self.commands = []

        def reset(self):
            self.reset_calls += 1

        def act(self, proprio, command):
            self.commands.append(tuple(command))
            return [0.0] * 33

    class FakePerception:
        def __init__(self, detections):
            self.detections = detections
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

        def update(self, image_obs, pose):
            return self.detections

    class SequencePerception:
        def __init__(self, detection_frames):
            self.detection_frames = list(detection_frames)
            self.reset_calls = 0
            self.update_calls = 0

        def reset(self):
            self.reset_calls += 1

        def update(self, image_obs, pose):
            frame_idx = min(self.update_calls, len(self.detection_frames) - 1)
            self.update_calls += 1
            return self.detection_frames[frame_idx]

    class RecordingPlanner:
        def __init__(self):
            self.phase = "search"
            self.reset_calls = 0
            self.detection_batches = []

        def reset(self):
            self.phase = "search"
            self.reset_calls += 1
            self.detection_batches.clear()

        def step(self, pose, detections, current_score):
            self.detection_batches.append(list(detections))
            return sol.PlannerOutput("record", (0.1, 0.0, 0.0), "stow", None)

    def make_solution_with_fakes(self, detections):
        instance = sol.AlgSolution.__new__(sol.AlgSolution)
        instance.bridge = self.FakeBridge()
        instance.odom = sol.DeadReckoningOdometry()
        instance.perception = self.FakePerception(detections)
        instance.planner = sol.TaskBPlanner()
        instance.interaction = sol.LocalObjectInteraction()
        return instance

    def proprio(self):
        row = torch.zeros((1, 12 + 3 * 33), dtype=torch.float32)
        row[0, 9:12] = torch.tensor([0.0, 0.0, -1.0])
        return row

    def assertCommandAlmostEqual(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=6)

    def test_predicts_returns_33_dim_action_no_giveup_and_uses_planner_command(self):
        det = sol.Detection(1, "object", 1.0, 0.0, 1.0, 0.9, -9.0, -10.0, (0, 0, 3, 3))
        proprio = self.proprio()
        expected_pose = sol.DeadReckoningOdometry().update(proprio[0])
        expected_plan = sol.TaskBPlanner().step(expected_pose, [det], current_score=0.0)
        solution = self.make_solution_with_fakes([det])

        out = solution.predicts({"proprio": proprio, "image": {}}, current_score=0.0)

        self.assertFalse(out["giveup"])
        self.assertEqual(len(out["action"]), 33)
        self.assertEqual(len(solution.bridge.commands), 1)
        self.assertCommandAlmostEqual(solution.bridge.commands[0], expected_plan.command)

    def test_predicts_throttles_perception_and_reuses_cached_detections(self):
        det = sol.Detection(2, "object", 1.0, 0.0, 1.0, 0.9, -8.5, -10.0, (0, 0, 3, 3))
        solution = self.make_solution_with_fakes([])
        solution.perception = self.SequencePerception([[det], []])
        solution.planner = self.RecordingPlanner()

        solution.predicts({"proprio": self.proprio(), "image": {"frame": 1}}, current_score=0.0)
        solution.predicts({"proprio": self.proprio(), "image": {"frame": 2}}, current_score=0.0)

        self.assertEqual(solution.perception.update_calls, 1)
        self.assertEqual(solution.planner.detection_batches, [[det], [det]])
        self.assertEqual(len(solution.bridge.commands), 2)

    def test_reset_resets_all_components_and_perception_cache(self):
        det = sol.Detection(3, "object", 1.0, 0.0, 1.0, 0.9, -8.0, -10.0, (0, 0, 3, 3))
        solution = self.make_solution_with_fakes([])
        solution._perception_step = 3
        solution._cached_detections = [det]

        solution.reset()

        self.assertEqual(solution.bridge.reset_calls, 1)
        self.assertEqual(solution.perception.reset_calls, 1)
        self.assertEqual(solution.planner.phase, "search")
        self.assertEqual(solution.odom.pose, sol.Pose2D(-10.0, -10.0, 0.0))
        self.assertEqual(solution._perception_step, 0)
        self.assertEqual(solution._cached_detections, [])


@unittest.skipIf(np is None, "numpy not installed")
class OpenWBTSquatBridgeObsTest(unittest.TestCase):
    class FakeRunner:
        def __init__(self):
            self.last_obs = None
        def run(self, obs, hidden):
            self.last_obs = obs.copy()
            return np.zeros((1, 12), dtype=np.float32), hidden

    def make_proprio(self, n_joints=33):
        row = [0.0] * (12 + 3 * n_joints)
        row[3:6] = [0.0, 0.0, 0.0]
        row[9:12] = [0.0, 0.0, -1.0]
        return [row]

    def test_obs_is_78_dims_and_command_first(self):
        runner = self.FakeRunner()
        bridge = sol.OpenWBTSquatBridge(policy_runner=runner)
        bridge.act(self.make_proprio(), sol.SquatCommand(height=0.5, pitch=0.1))
        self.assertEqual(runner.last_obs.shape, (1, 78))
        self.assertAlmostEqual(float(runner.last_obs[0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(runner.last_obs[0, 1]), 0.1, places=5)

    def test_act_returns_12_leg_actions(self):
        bridge = sol.OpenWBTSquatBridge(policy_runner=self.FakeRunner())
        leg = bridge.act(self.make_proprio(), sol.SquatCommand())
        self.assertEqual(len(leg), 12)


@unittest.skipIf(np is None, "numpy not installed")
class OpenWBTSquatGainCompTest(unittest.TestCase):
    class ConstRunner:
        def __init__(self, value):
            self.value = value
        def run(self, obs, hidden):
            return np.full((1, 12), self.value, dtype=np.float32), hidden

    def make_proprio(self):
        # joint_pos_rel = 0 -> q_abs = taskb_default
        return [[0.0] * (12 + 3 * 33)]

    def test_ankle_roll_indices_zeroed(self):
        bridge = sol.OpenWBTSquatBridge(policy_runner=self.ConstRunner(0.4))
        leg = bridge.act(self.make_proprio(), sol.SquatCommand(height=0.4))
        self.assertEqual(leg[5], 0.0)
        self.assertEqual(leg[11], 0.0)

    def test_hip_pitch_uses_kp_ratio(self):
        # raw=0.4: target_WBT0 = 0.4*0.25 + (-0.1) = 0.0 ; q_abs0 = -0.2
        # comp = -0.2 + 0.5*(0.0 - (-0.2)) = -0.2 + 0.1 = -0.1
        # action = (-0.1 - (-0.2))/0.5 = 0.2
        bridge = sol.OpenWBTSquatBridge(policy_runner=self.ConstRunner(0.4))
        leg = bridge.act(self.make_proprio(), sol.SquatCommand(height=0.4))
        self.assertAlmostEqual(leg[0], 0.2, places=4)

    def test_ankle_pitch_amplified_vs_no_comp(self):
        # ankle ratio 1.4 -> compensated magnitude larger than static-offset magnitude
        bridge = sol.OpenWBTSquatBridge(policy_runner=self.ConstRunner(0.4))
        leg = bridge.act(self.make_proprio(), sol.SquatCommand(height=0.4))
        # compensated[4] = (-0.23 + 1.4*((-0.1)-(-0.23)) - (-0.23))/0.5 = (1.4*0.13)/0.5 = 0.364
        self.assertAlmostEqual(leg[4], 0.364, places=3)


class GroundSweepArmControllerTest(unittest.TestCase):
    LEFT_SH_PITCH = 15
    LEFT_ELBOW = 18
    LEFT_SH_ROLL = 16
    RIGHT_SH_ROLL = 23

    def test_progress_zero_arms_near_stow(self):
        sweep = sol.GroundSweepArmController()
        out = sweep.step(squat_progress=0.0)
        self.assertAlmostEqual(out.get(self.LEFT_SH_PITCH, 0.0), 0.0, places=6)
        self.assertAlmostEqual(out.get(self.LEFT_ELBOW, 0.0), 0.0, places=6)

    def test_progress_one_reaches_down(self):
        sweep = sol.GroundSweepArmController()
        out = sweep.step(squat_progress=1.0)
        self.assertGreater(out[self.LEFT_SH_PITCH], 0.3)
        self.assertGreater(out[self.LEFT_ELBOW], 0.3)

    def test_sweep_oscillates_left_right(self):
        sweep = sol.GroundSweepArmController()
        sweep.step(squat_progress=1.0)  # phase advances internally
        a = sweep.step(squat_progress=1.0)[self.LEFT_SH_ROLL]
        for _ in range(20):
            b_out = sweep.step(squat_progress=1.0)
        b = b_out[self.LEFT_SH_ROLL]
        self.assertNotAlmostEqual(a, b, places=3)

    def test_left_right_roll_antisymmetric(self):
        sweep = sol.GroundSweepArmController()
        out = sweep.step(squat_progress=1.0)
        self.assertAlmostEqual(out[self.RIGHT_SH_ROLL], -out[self.LEFT_SH_ROLL], places=10)

    def test_fingers_open_at_full_squat(self):
        sweep = sol.GroundSweepArmController()
        out = sweep.step(squat_progress=1.0)
        for idx in (29, 30, 31, 32):
            self.assertAlmostEqual(out[idx], 0.4, places=6)

    def test_only_upper_body_indices(self):
        sweep = sol.GroundSweepArmController()
        out = sweep.step(squat_progress=1.0)
        self.assertTrue(all(15 <= i <= 32 for i in out.keys()))

    def test_reset_clears_phase(self):
        sweep = sol.GroundSweepArmController()
        for _ in range(5):
            sweep.step(squat_progress=1.0)
        sweep.reset()
        self.assertEqual(sweep.phase, 0)


class PostureGuardTest(unittest.TestCase):
    def row(self, gx, gy, gz):
        r = [0.0] * (12 + 3 * 33)
        r[9:12] = [gx, gy, gz]
        return r

    def test_upright_is_ok(self):
        guard = sol.PostureGuard()
        self.assertEqual(guard.check(self.row(0.0, 0.0, -1.0)), "ok")

    def test_large_tilt_is_recover(self):
        guard = sol.PostureGuard()
        self.assertEqual(guard.check(self.row(0.5, 0.0, -0.86)), "recover")

    def test_reset(self):
        guard = sol.PostureGuard()
        guard.check(self.row(0.5, 0.0, -0.86))
        guard.reset()
        self.assertEqual(guard.state, "ok")


if __name__ == "__main__":
    unittest.main()
