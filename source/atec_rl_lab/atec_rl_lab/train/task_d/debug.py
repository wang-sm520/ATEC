"""Lightweight debug summaries for Task D validation."""

from __future__ import annotations

from math import isfinite
from typing import Any


FAILURE_LABELS = {
    "lidar_no_obstacle",
    "lidar_no_box",
    "bad_box_tracking",
    "missed_contact_pose",
    "push_too_weak",
    "box_misaligned",
    "fall_before_cross",
    "crossed_without_box_reward",
    "running",
}


def summarize_controller_debug(debug: dict, current_score: float) -> str:
    """Return a compact one-line summary of a TaskDController debug dict."""

    phase = _text(debug.get("phase"), default="unknown")
    robot = _pose_text(debug.get("robot_in_obstacle"))
    box = _pose_text(debug.get("box_in_obstacle"))
    target = _pose_text(debug.get("target"))
    cmd = _pose_text(debug.get("cmd"))
    label = classify_failure(debug, current_score)

    parts = [
        f"phase={phase}",
        f"robot={robot}",
        f"box={box}",
        f"target={target}",
        f"cmd={cmd}",
        f"score={float(current_score):.2f}",
        f"classification={label}",
        f"obstacle_valid={bool(debug.get('obstacle_valid', False))}",
        f"box_lidar_valid={bool(debug.get('box_lidar_valid', False))}",
    ]

    if debug.get("target_description") is not None:
        parts.append(f"target_desc={_text(debug.get('target_description'))}")
    if debug.get("box_source") is not None:
        parts.append(f"box_source={_text(debug.get('box_source'))}")
    if debug.get("lidar_box_debug") is not None:
        parts.append(f"lidar_box_debug={_text(debug.get('lidar_box_debug'))}")

    return " ".join(parts)


def classify_failure(debug: dict, current_score: float) -> str:
    """Classify the most likely Task D failure mode from controller debug data."""

    explicit = debug.get("failure") or debug.get("classification")
    if explicit in FAILURE_LABELS:
        return str(explicit)

    phase = _text(debug.get("phase")).upper()
    robot = _tuple3(debug.get("robot_in_obstacle"))
    box = _tuple3(debug.get("box_in_obstacle"))
    cmd = _tuple3(debug.get("cmd"))
    score = float(current_score)

    if not bool(debug.get("obstacle_valid", False)):
        return "lidar_no_obstacle"

    if bool(debug.get("obstacle_valid", False)) and not bool(debug.get("box_lidar_valid", False)):
        return "lidar_no_box"

    if _fell_before_cross(debug, robot):
        return "fall_before_cross"

    if robot is not None and robot[0] >= 2.0 and score < 35.0:
        return "crossed_without_box_reward"

    if _bad_pose(box) or float(debug.get("box_confidence", 1.0)) < 0.05:
        return "bad_box_tracking"

    if box is not None and (abs(box[1] - 1.6) > 0.45 or abs(box[2]) > 0.55):
        return "box_misaligned"

    if "PUSH" in phase:
        if robot is not None and box is not None:
            contact_x_error = abs(robot[0] - (box[0] - 0.65))
            contact_y_error = abs(robot[1] - box[1])
            if contact_x_error > 0.45 or contact_y_error > 0.35:
                return "missed_contact_pose"
        if cmd is not None and cmd[0] < 0.15:
            return "push_too_weak"

    return "running"


def _pose_text(value: Any) -> str:
    pose = _tuple3(value)
    if pose is None:
        return "(unknown)"
    return f"({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f})"


def _tuple3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        pose = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    return pose


def _bad_pose(value: tuple[float, float, float] | None) -> bool:
    if value is None:
        return True
    return not all(isfinite(component) for component in value)


def _fell_before_cross(debug: dict, robot: tuple[float, float, float] | None) -> bool:
    if bool(debug.get("fell") or debug.get("robot_fallen") or debug.get("fall_detected")):
        return robot is None or robot[0] < 2.0
    height = debug.get("robot_height")
    if height is None:
        return False
    try:
        return float(height) < 0.25 and (robot is None or robot[0] < 2.0)
    except (TypeError, ValueError):
        return False


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)
