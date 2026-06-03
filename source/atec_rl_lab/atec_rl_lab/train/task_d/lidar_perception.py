"""Lightweight LiDAR/height-scan perception for Task D.

This module intentionally avoids IsaacLab imports.  The current Task D
RayCaster is configured against ``/World/ground``, so dynamic box visibility is
not assumed.  When terrain extraction is ambiguous, the perception layer emits
the known deterministic Task D obstacle frame and marks that source explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from .types import BoxMeasurement, ObstacleFrame, ObstacleMeasurement, Pose2D


DEFAULT_CHANNELS = 16
DEFAULT_BINS = 360
DEFAULT_FLAT_LENGTH = DEFAULT_CHANNELS * DEFAULT_BINS
TASK_D_KNOWN_OBSTACLE_FRAME = ObstacleFrame(
    origin=Pose2D(0.0, 0.0, 0.0),
    confidence=0.25,
    valid=True,
)


@dataclass(frozen=True)
class ScanGrid:
    """Small list-backed grid with a tensor-like ``shape`` attribute."""

    rows: tuple[tuple[float, ...], ...]

    @property
    def shape(self) -> tuple[int, int]:
        if not self.rows:
            return (0, 0)
        return (len(self.rows), len(self.rows[0]))

    def __getitem__(self, index: int) -> tuple[float, ...]:
        return self.rows[index]


@dataclass(frozen=True)
class LidarScan:
    """Normalized scan representation independent of tensor libraries."""

    flat: tuple[float, ...]
    channels: int
    bins: int
    grid: ScanGrid
    original_shape: tuple[int, ...]

    @property
    def flat_length(self) -> int:
        return len(self.flat)

    @property
    def empty(self) -> bool:
        return not self.flat


class TaskDLidarPerception:
    """Parse Task D extero observations into obstacle and optional box measurements."""

    def __init__(
        self,
        fallback_obstacle_frame: ObstacleFrame = TASK_D_KNOWN_OBSTACLE_FRAME,
        elevated_threshold: float = 0.20,
        min_cluster_points: int = 8,
    ) -> None:
        self.fallback_obstacle_frame = fallback_obstacle_frame
        self.elevated_threshold = elevated_threshold
        self.min_cluster_points = min_cluster_points
        self.last_debug: dict[str, object] = {}

    def parse_scan(self, extero: object) -> LidarScan:
        """Return a list-backed scan from tensors, lists, tuples, or obs dicts.

        A 5760-value flat scan is treated as the expected 16 x 360 layout.
        Other one-dimensional lengths degrade to a single-channel scan.
        """

        if isinstance(extero, dict):
            extero = extero.get("extero")

        original_shape = _shape_of(extero)
        flat = tuple(_flatten_numbers(extero))
        if len(flat) == DEFAULT_FLAT_LENGTH:
            channels = DEFAULT_CHANNELS
            bins = DEFAULT_BINS
        elif original_shape and len(original_shape) >= 2 and original_shape[-2] > 0 and original_shape[-1] > 0:
            channels = int(original_shape[-2])
            bins = int(original_shape[-1])
            if channels * bins != len(flat):
                channels = 1
                bins = len(flat)
        else:
            channels = 1 if flat else 0
            bins = len(flat)

        rows = _rows_from_flat(flat, channels, bins)
        return LidarScan(flat=flat, channels=channels, bins=bins, grid=ScanGrid(rows), original_shape=original_shape)

    def measure(self, extero: object) -> tuple[ObstacleMeasurement, BoxMeasurement]:
        """Measure the obstacle frame and, only with evidence, the dynamic box."""

        scan = self.parse_scan(extero)
        obstacle = self._measure_obstacle(scan)
        box = self._measure_box(scan)
        self.last_debug.update(
            {
                "extero_shape": scan.original_shape,
                "scan_shape": scan.grid.shape,
                "obstacle_source": obstacle.source,
                "obstacle_confidence": obstacle.frame.confidence,
                "box_valid": box.valid,
                "box_source": box.source,
            }
        )
        return obstacle, box

    def update(self, extero: object) -> tuple[ObstacleMeasurement, BoxMeasurement]:
        """Alias used by controller/probe code."""

        return self.measure(extero)

    def box_debug(self) -> str:
        return str(self.last_debug.get("box_reason", "box perception has not run"))

    def _measure_obstacle(self, scan: LidarScan) -> ObstacleMeasurement:
        if scan.empty:
            self.last_debug["obstacle_reason"] = "no extero scan; using deterministic Task D frame"
            return ObstacleMeasurement(frame=self.fallback_obstacle_frame, source="lidar_fallback:no_extero")

        terrain_signal = _robust_range(scan.flat)
        confidence = min(0.55, 0.30 + terrain_signal)
        frame = ObstacleFrame(
            origin=self.fallback_obstacle_frame.origin,
            confidence=confidence,
            valid=True,
        )
        self.last_debug["obstacle_reason"] = (
            "terrain extraction ambiguous without calibrated ray geometry; "
            "using deterministic Task D frame"
        )
        return ObstacleMeasurement(frame=frame, source="lidar_fallback:task_d_known_frame")

    def _measure_box(self, scan: LidarScan) -> BoxMeasurement:
        if scan.empty:
            self.last_debug["box_reason"] = "no extero scan; dynamic box cannot be inferred"
            return BoxMeasurement.invalid("lidar")
        if scan.channels != DEFAULT_CHANNELS or scan.bins != DEFAULT_BINS:
            self.last_debug["box_reason"] = "scan geometry is not 16x360; no calibrated cluster coordinates"
            return BoxMeasurement.invalid("lidar")

        cluster = self._find_non_ground_cluster(scan)
        if cluster is None:
            self.last_debug["box_reason"] = "no plausible non-ground cluster in LiDAR scan"
            return BoxMeasurement.invalid("lidar")

        center_bin, center_channel, count, spread_bins = cluster
        angle = (center_bin / DEFAULT_BINS) * 2.0 * math.pi - math.pi
        # Coarse placeholder projection: sufficient to flag a measured source,
        # not suitable for competition control until validated in Isaac Sim.
        radial_distance = 1.6
        pose = Pose2D(
            x=radial_distance * math.cos(angle),
            y=radial_distance * math.sin(angle),
            yaw=0.0,
        )
        confidence = min(0.60, 0.25 + count / 100.0)
        self.last_debug["box_reason"] = (
            f"localized non-ground cluster: count={count}, channel={center_channel:.1f}, "
            f"bin={center_bin:.1f}, spread_bins={spread_bins}"
        )
        return BoxMeasurement(valid=True, pose=pose, source="lidar", confidence=confidence)

    def _find_non_ground_cluster(self, scan: LidarScan) -> tuple[float, float, int, int] | None:
        baseline = _median(scan.flat)
        candidates: list[tuple[int, int, float]] = []
        for channel, row in enumerate(scan.grid.rows):
            for bin_index, value in enumerate(row):
                if value - baseline >= self.elevated_threshold:
                    candidates.append((channel, bin_index, value))

        if len(candidates) < self.min_cluster_points:
            return None

        bins = [bin_index for _, bin_index, _ in candidates]
        channels = [channel for channel, _, _ in candidates]
        spread_bins = max(bins) - min(bins)
        spread_channels = max(channels) - min(channels)
        if spread_bins > 24 or spread_channels > 8:
            return None

        center_bin = sum(bins) / len(bins)
        center_channel = sum(channels) / len(channels)
        return center_bin, center_channel, len(candidates), spread_bins


def _rows_from_flat(flat: Sequence[float], channels: int, bins: int) -> tuple[tuple[float, ...], ...]:
    if channels <= 0 or bins <= 0:
        return ()
    return tuple(tuple(flat[channel * bins : (channel + 1) * bins]) for channel in range(channels))


def _shape_of(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    if isinstance(value, dict):
        return _shape_of(value.get("extero"))
    if isinstance(value, (str, bytes)):
        return ()
    if isinstance(value, Sequence):
        if not value:
            return (0,)
        inner = _shape_of(value[0])
        return (len(value), *inner)
    return ()


def _flatten_numbers(value: object) -> Iterable[float]:
    if value is None:
        return ()
    if isinstance(value, dict):
        return _flatten_numbers(value.get("extero"))
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        try:
            return _flatten_numbers(value.reshape(-1).tolist())
        except TypeError:
            return _flatten_numbers(value.tolist())
    if hasattr(value, "tolist"):
        return _flatten_numbers(value.tolist())
    if isinstance(value, (str, bytes)):
        return ()
    if isinstance(value, Sequence):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numbers(item))
        return flattened
    try:
        return (float(value),)
    except (TypeError, ValueError):
        return ()


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[middle]
    return 0.5 * (sorted_values[middle - 1] + sorted_values[middle])


def _robust_range(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    sorted_values = sorted(values)
    low = sorted_values[int(0.10 * (len(sorted_values) - 1))]
    high = sorted_values[int(0.90 * (len(sorted_values) - 1))]
    return min(1.0, max(0.0, abs(high - low)))
