"""Task B G1 RGB-D perception: colored-object detection with ground projection.

Ported verbatim from the legacy task-b module (retired 2026-06-12). The camera intrinsics/extrinsics and
the pinhole back-projection through the measured 47.6° head-camera pitch are
calibrated in simulation — do not alter any constant or math expression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    from task_b_nav import Pose2D, _clamp, _near_target_circle
except ImportError:  # pragma: no cover - local dev path
    from demo.task_b_nav import Pose2D, _clamp, _near_target_circle

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - the eval image provides torch.
    torch = None


@dataclass(frozen=True)
class Detection:
    track_id: int
    label: str
    rel_x: float
    rel_y: float
    distance: float
    confidence: float
    world_x: float
    world_y: float
    bbox: tuple[int, int, int, int]


# Head-camera intrinsics/extrinsics. The camera is pitched steeply DOWN, so a
# floor object's slant depth is much larger than its ground distance — projecting
# the depth pixel through these is required to place objects accurately.
#
# CALIBRATION IS ISAACLAB-VERSION-DEPENDENT. These values were re-measured
# 2026-06-12 on isaaclab 0.54.4 / isaacsim 5.1 by a least-squares fit over 216
# truth-matched detections (scripts/audit_task_b_perception.py data; localization
# median 0.171m -> 0.078m). They supersede the 2026-06-10 probe values
# (pitch 47.6 deg, focal ratio 24.0/20.955 ~= 1.145) which were measured on an
# older IsaacLab and are WRONG on this stack. Re-fit after any env upgrade.
_HEAD_CAM_FOCAL_OVER_APERTURE = 700.0 / 640.0   # fx = fy = this * image_width (square pixels)
_HEAD_CAM_PITCH_RAD = math.radians(56.9)        # downward tilt from horizontal
_HEAD_CAM_HEIGHT_ABOVE_BASE = 0.56              # camera height above robot base (m); not used by projection
_HEAD_CAM_FORWARD_OFFSET = 0.03                 # camera forward offset from base (m)

_HEAD_CAMERA_HFOV_DEG = math.degrees(2.0 * math.atan(0.5 / _HEAD_CAM_FOCAL_OVER_APERTURE))


class TaskBRgbdPerception:
    def __init__(
        self,
        hfov_deg: float = _HEAD_CAMERA_HFOV_DEG,
        min_pixels: int = 35,
        max_depth: float = 8.0,
        track_match_dist: float = 0.75,
    ):
        self.hfov = math.radians(float(hfov_deg))
        self.min_pixels = int(min_pixels)
        self.max_depth = float(max_depth)
        self.track_match_dist = float(track_match_dist)
        self.next_track_id = 1
        self.tracks: dict[int, tuple[float, float]] = {}

    def reset(self) -> None:
        self.next_track_id = 1
        self.tracks.clear()

    def update(self, image_obs: dict, pose: Pose2D) -> list[Detection]:
        if torch is None:
            return []
        rgb, depth = self._extract_head_rgbd(image_obs)
        if rgb is None or depth is None:
            return []
        rgb = rgb.detach().float().cpu()
        depth = depth.detach().float().cpu()
        if rgb.ndim == 4:
            rgb = rgb[0]
        if depth.ndim == 4:
            depth = depth[0]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if rgb.ndim != 3 or rgb.shape[-1] < 3 or depth.ndim != 2:
            return []
        if tuple(rgb.shape[:2]) != tuple(depth.shape[:2]):
            return []

        mask = self._colored_object_mask(rgb[..., :3], depth)
        components = self._components(mask)
        detections: list[Detection] = []
        used_track_ids: set[int] = set()
        for pixels in components:
            if len(pixels) < self.min_pixels:
                continue
            ys = [p[0] for p in pixels]
            xs = [p[1] for p in pixels]
            y0, y1 = min(ys), max(ys)
            x0, x1 = min(xs), max(xs)
            cy = int(round(sum(ys) / len(ys)))
            cx = int(round(sum(xs) / len(xs)))
            y_idx = torch.as_tensor(ys, dtype=torch.long)
            x_idx = torch.as_tensor(xs, dtype=torch.long)
            pixel_depth = depth[y_idx, x_idx]
            finite = torch.isfinite(pixel_depth) & (pixel_depth > 0.05) & (pixel_depth < self.max_depth)
            if not bool(finite.any()):
                continue
            dist = float(pixel_depth[finite].median().item())
            rel_x, rel_y = self._project_pixel_to_base(cx, cy, rgb.shape[1], rgb.shape[0], dist)
            ground_dist = math.hypot(rel_x, rel_y)
            world_x = pose.x + math.cos(pose.yaw) * rel_x - math.sin(pose.yaw) * rel_y
            world_y = pose.y + math.sin(pose.yaw) * rel_x + math.cos(pose.yaw) * rel_y
            if self._is_large_target_background(rgb[..., :3], y_idx, x_idx, x0, y0, x1, y1, world_x, world_y):
                continue
            track_id = self._assign_track(world_x, world_y, used_track_ids)
            confidence = _clamp(len(pixels) / 250.0, 0.05, 1.0)
            detections.append(
                Detection(
                    track_id=track_id,
                    label="colored_object",
                    rel_x=rel_x,
                    rel_y=rel_y,
                    distance=ground_dist,
                    confidence=confidence,
                    world_x=world_x,
                    world_y=world_y,
                    bbox=(x0, y0, x1, y1),
                )
            )
        detections.sort(key=lambda d: (d.distance, -d.confidence))
        return detections[:5]

    @staticmethod
    def _extract_head_rgbd(image_obs: dict):
        if not isinstance(image_obs, dict):
            return None, None
        rgb = image_obs.get("head_rgb")
        depth = image_obs.get("head_depth")
        return rgb, depth

    def _colored_object_mask(self, rgb, depth):
        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]
        maxc = torch.maximum(torch.maximum(r, g), b)
        minc = torch.minimum(torch.minimum(r, g), b)
        saturation = maxc - minc
        yellow = (r > 120.0) & (g > 90.0) & (b < 130.0)
        red_or_orange = (r > 130.0) & (g > 45.0) & (b < 150.0) & (r > b + 35.0)
        bright_colored = (maxc > 110.0) & (saturation > 45.0)
        depth_ok = torch.isfinite(depth) & (depth > 0.15) & (depth < self.max_depth)
        return (yellow | red_or_orange | bright_colored) & depth_ok

    @staticmethod
    def _is_large_target_background(rgb, ys, xs, x0: int, y0: int, x1: int, y1: int, world_x: float, world_y: float) -> bool:
        height = int(rgb.shape[0])
        width = int(rgb.shape[1])
        bbox_area = max(1, (int(x1) - int(x0) + 1) * (int(y1) - int(y0) + 1))
        frame_area = max(1, width * height)
        if bbox_area < 0.45 * frame_area:
            return False
        r = rgb[ys, xs, 0]
        g = rgb[ys, xs, 1]
        b = rgb[ys, xs, 2]
        orange_red = (r > 150.0) & (g > 40.0) & (g < 160.0) & (b < 120.0) & (r > g + 40.0)
        if float(orange_red.float().mean().item()) < 0.7:
            return False
        return _near_target_circle((world_x, world_y), max_distance=1.0)

    @staticmethod
    def _components(mask) -> list[list[tuple[int, int]]]:
        coords_tensor = mask.nonzero(as_tuple=False)
        if coords_tensor.numel() == 0:
            return []
        true_pixels = {(int(y), int(x)) for y, x in coords_tensor.tolist()}
        components: list[list[tuple[int, int]]] = []
        while true_pixels:
            start = true_pixels.pop()
            stack = [start]
            pixels = [start]
            while stack:
                cy, cx = stack.pop()
                for neighbor in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if neighbor in true_pixels:
                        true_pixels.remove(neighbor)
                        stack.append(neighbor)
                        pixels.append(neighbor)
            components.append(pixels)
        return components

    @staticmethod
    def _project_pixel_to_base(cx: float, cy: float, width: int, height: int, depth: float) -> tuple[float, float]:
        """Back-project a depth pixel to robot-base ground coords (forward, left) in metres.

        Pinhole back-projection (fx=fy=focal/aperture*width, principal point at image
        centre) into the camera optical frame, then rotate by the measured head-camera
        downward pitch to recover the horizontal forward distance and lateral offset.
        ``depth`` is the camera "depth" channel (distance to image plane along the optical
        axis). Using only the forward component (old behaviour) badly over-estimated the
        ground distance for the steeply down-pitched head camera.
        """
        f = _HEAD_CAM_FOCAL_OVER_APERTURE * float(width)
        if f <= 0.0:
            return float(depth), 0.0
        xc = (float(cx) - float(width) / 2.0) / f * float(depth)    # camera-right
        yc = (float(cy) - float(height) / 2.0) / f * float(depth)   # camera-down
        zc = float(depth)                                           # camera-forward (image plane)
        cth = math.cos(_HEAD_CAM_PITCH_RAD)
        sth = math.sin(_HEAD_CAM_PITCH_RAD)
        forward = _HEAD_CAM_FORWARD_OFFSET + zc * cth - yc * sth
        left = -xc
        return forward, left

    def _assign_track(self, world_x: float, world_y: float, used_track_ids: set[int] | None = None) -> int:
        if used_track_ids is None:
            used_track_ids = set()
        best_id = None
        best_dist = self.track_match_dist
        for track_id, (tx, ty) in self.tracks.items():
            if track_id in used_track_ids:
                continue
            d = math.hypot(world_x - tx, world_y - ty)
            if d < best_dist:
                best_dist = d
                best_id = track_id
        if best_id is None:
            best_id = self.next_track_id
            while best_id in used_track_ids or best_id in self.tracks:
                best_id += 1
            self.next_track_id = best_id + 1
        self.tracks[best_id] = (world_x, world_y)
        used_track_ids.add(best_id)
        return best_id
