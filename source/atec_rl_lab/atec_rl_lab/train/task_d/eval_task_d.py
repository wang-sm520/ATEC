"""Lightweight Task D validation entrypoint.

This script intentionally avoids launching Isaac Sim. It prints the command a
human can run in the real challenge environment and can show a mock debug line.
"""

from __future__ import annotations

import argparse

from .debug import summarize_controller_debug


REAL_ENV_COMMAND = (
    "PYTHONPATH=source/atec_rl_lab python demo/solution_d.py "
    "# run inside the official ATEC/Isaac task runner"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Task D lightweight validation helper")
    parser.add_argument(
        "--mock-summary",
        action="store_true",
        help="print a representative controller debug summary without running simulation",
    )
    args = parser.parse_args()

    print("Recommended real-env command:")
    print(REAL_ENV_COMMAND)

    if args.mock_summary:
        print("Mock debug summary:")
        print(summarize_controller_debug(_mock_debug(), current_score=12.5))

    return 0


def _mock_debug() -> dict:
    return {
        "phase": "PUSH_BOX_TO_BRIDGE",
        "robot_in_obstacle": (-3.6, 1.58, 0.01),
        "box_in_obstacle": (-2.95, 1.6, 0.0),
        "obstacle_valid": True,
        "obstacle_confidence": 0.95,
        "box_source": "lidar",
        "box_confidence": 0.8,
        "box_lidar_valid": True,
        "target": (-3.6, 1.6, 0.0),
        "target_description": "push_contact",
        "cmd": (0.8, 0.0, 0.0),
        "lidar_box_debug": "box cluster accepted",
    }


if __name__ == "__main__":
    raise SystemExit(main())
