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

from demo import task_b_perception as perc  # noqa: E402
from demo.task_b_nav import Pose2D  # noqa: E402


class HeadCameraGroundProjectionTest(unittest.TestCase):
    """Pure pinhole + camera-pitch back-projection (no torch needed)."""

    def test_center_pixel_forward_uses_pitch(self):
        # center pixel, slant depth 2.0 -> forward = offset + depth*cos(pitch)
        fwd, left = perc.TaskBRgbdPerception._project_pixel_to_base(320, 240, 640, 480, 2.0)
        self.assertAlmostEqual(fwd, 0.03 + 2.0 * math.cos(math.radians(47.6)), delta=0.01)
        self.assertAlmostEqual(left, 0.0, delta=1e-6)

    def test_lower_pixel_is_closer_forward(self):
        # a pixel lower in the image (larger cy) maps to a closer ground point
        fwd_center, _ = perc.TaskBRgbdPerception._project_pixel_to_base(320, 240, 640, 480, 1.5)
        fwd_low, _ = perc.TaskBRgbdPerception._project_pixel_to_base(320, 480, 640, 480, 1.5)
        self.assertLess(fwd_low, fwd_center)
        self.assertAlmostEqual(fwd_low, 0.679, delta=0.02)

    def test_right_pixel_gives_negative_left(self):
        fwd, left = perc.TaskBRgbdPerception._project_pixel_to_base(420, 240, 640, 480, 2.0)
        self.assertLess(left, 0.0)
        self.assertAlmostEqual(left, -(100.0 / (24.0 / 20.955 * 640.0)) * 2.0, delta=1e-4)


class TaskBRgbdPerceptionHelperTest(unittest.TestCase):
    def test_projection_lateral_sign_matches_image_side(self):
        # right-of-center pixel -> object to the robot's right -> negative "left"
        _, right_left = perc.TaskBRgbdPerception._project_pixel_to_base(95, 32, 96, 64, 1.0)
        _, left_left = perc.TaskBRgbdPerception._project_pixel_to_base(0, 32, 96, 64, 1.0)
        self.assertLess(right_left, 0.0)
        self.assertGreater(left_left, 0.0)

    def test_assign_track_reserves_used_ids_within_frame(self):
        perception = perc.TaskBRgbdPerception(track_match_dist=1.0)
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
        perception = perc.TaskBRgbdPerception(min_pixels=20)
        detections = perception.update(self.make_image_obs(), Pose2D(-10.0, -10.0, 0.0))
        self.assertEqual(len(detections), 1)
        det = detections[0]
        self.assertEqual(det.label, "colored_object")
        self.assertGreater(det.confidence, 0.2)
        # distance is now ground distance (projected), strictly less than the 2.0 slant depth
        self.assertGreater(det.distance, 0.5)
        self.assertLess(det.distance, 2.0)
        self.assertGreater(det.world_x, -10.0)

    def test_suppresses_large_target_colored_blob_but_keeps_small_blob(self):
        pose = Pose2D(-5.0, -10.0, 0.0)
        large_rgb = torch.zeros((1, 64, 96, 3), dtype=torch.uint8)
        large_depth = torch.full((1, 64, 96, 1), 2.0, dtype=torch.float32)
        large_rgb[0, 5:59, 8:88, 0] = 230
        large_rgb[0, 5:59, 8:88, 1] = 90
        large_rgb[0, 5:59, 8:88, 2] = 20

        large_detections = perc.TaskBRgbdPerception(min_pixels=20).update(
            {"head_rgb": large_rgb, "head_depth": large_depth}, pose
        )

        self.assertEqual(large_detections, [])

        small_rgb = torch.zeros((1, 64, 96, 3), dtype=torch.uint8)
        small_depth = torch.full((1, 64, 96, 1), 2.0, dtype=torch.float32)
        small_rgb[0, 28:38, 44:54, 0] = 230
        small_rgb[0, 28:38, 44:54, 1] = 90
        small_rgb[0, 28:38, 44:54, 2] = 20
        small_detections = perc.TaskBRgbdPerception(min_pixels=20).update(
            {"head_rgb": small_rgb, "head_depth": small_depth}, pose
        )

        self.assertEqual(len(small_detections), 1)

    def test_returns_empty_when_no_image_keys_exist(self):
        perception = perc.TaskBRgbdPerception(min_pixels=20)
        detections = perception.update({}, Pose2D(-10.0, -10.0, 0.0))
        self.assertEqual(detections, [])

    def test_returns_empty_when_rgb_depth_spatial_sizes_mismatch(self):
        rgb = torch.zeros((1, 64, 96, 3), dtype=torch.uint8)
        depth = torch.full((1, 32, 48, 1), 2.0, dtype=torch.float32)
        perception = perc.TaskBRgbdPerception(min_pixels=20)

        detections = perception.update({"head_rgb": rgb, "head_depth": depth}, Pose2D(-10.0, -10.0, 0.0))

        self.assertEqual(detections, [])

    def test_tracks_same_blob_with_stable_id(self):
        perception = perc.TaskBRgbdPerception(min_pixels=20)
        first = perception.update(self.make_image_obs(), Pose2D(-10.0, -10.0, 0.0))[0]
        second = perception.update(self.make_image_obs(), Pose2D(-10.0, -10.0, 0.0))[0]
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

        perception = perc.TaskBRgbdPerception(min_pixels=20)
        detections = perception.update({"head_rgb": rgb, "head_depth": depth}, Pose2D(-10.0, -10.0, 0.0))

        self.assertEqual(len(detections), 1)
        # ground-projected distance reflects the colored border depth (2.0), not the
        # interior depth (0.4): depth-2.0 projects to ~1.4m ground, depth-0.4 to ~0.3m.
        self.assertGreater(detections[0].distance, 1.0)

    def test_two_same_frame_components_get_distinct_track_ids(self):
        rgb = torch.zeros((1, 16, 16, 3), dtype=torch.uint8)
        depth = torch.full((1, 16, 16, 1), 2.0, dtype=torch.float32)
        rgb[0, 4:7, 4:7, 0] = 230
        rgb[0, 4:7, 4:7, 1] = 190
        rgb[0, 4:7, 4:7, 2] = 30
        rgb[0, 4:7, 9:12, 0] = 230
        rgb[0, 4:7, 9:12, 1] = 190
        rgb[0, 4:7, 9:12, 2] = 30
        perception = perc.TaskBRgbdPerception(min_pixels=4, track_match_dist=10.0)
        perception.tracks[5] = (-8.0, -10.0)
        perception.next_track_id = 6

        detections = perception.update({"head_rgb": rgb, "head_depth": depth}, Pose2D(-10.0, -10.0, 0.0))
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
        perception = perc.TaskBRgbdPerception(min_pixels=20)

        detections = perception.update({"head_rgb": rgb, "head_depth": depth}, Pose2D(-20.0, -20.0, 0.0))

        self.assertLessEqual(len(detections), 1)


if __name__ == "__main__":
    unittest.main()
