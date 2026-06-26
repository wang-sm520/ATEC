import argparse
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[2]
_ACT_DIR = _REPO / "scripts" / "act"
if str(_ACT_DIR) not in sys.path:
    sys.path.insert(0, str(_ACT_DIR))

from cli_args import add_collect_demo_args  # noqa: E402


def _parse(args):
    parser = argparse.ArgumentParser()
    add_collect_demo_args(parser)
    return parser.parse_args(args)


def test_collect_demo_args_include_full_order_and_grasp_controls():
    args = _parse([
        "--full_order_123",
        "--optimized_grasp_flow",
        "--grasp_z_offsets", "1:0.080", "2:0.085", "3:0.075",
        "--grasp_offset_json", "datasets/atec_task_e/grasp_offset_sweep.json",
        "--max_attempts", "10",
        "--tool_center_offset_local", "0.0", "0.01", "0.0",
    ])

    assert args.full_order_123 is True
    assert args.optimized_grasp_flow is True
    assert args.grasp_z_offsets == ["1:0.080", "2:0.085", "3:0.075"]
    assert args.grasp_offset_json == "datasets/atec_task_e/grasp_offset_sweep.json"
    assert args.max_attempts == 10
    assert args.tool_center_offset_local == [0.0, 0.01, 0.0]


def test_collect_demo_args_default_new_controls_are_disabled():
    args = _parse([])

    assert args.full_order_123 is False
    assert args.optimized_grasp_flow is False
    assert args.grasp_z_offsets is None
    assert args.grasp_offset_json is None
    assert args.max_attempts is None
    assert args.tool_center_offset_local is None
