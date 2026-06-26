from pathlib import Path
import json
import sys

_REPO = Path(__file__).resolve().parents[2]
_ACT_DIR = _REPO / "scripts" / "act"
if str(_ACT_DIR) not in sys.path:
    sys.path.insert(0, str(_ACT_DIR))

from task_e.options import (  # noqa: E402
    choose_best_offset,
    load_grasp_offset_json,
    parse_grasp_z_offsets,
    resolve_grasp_z_offsets,
    resolve_pick_objects,
)


def test_parse_grasp_z_offsets_accepts_object_colon_value_pairs():
    offsets = parse_grasp_z_offsets(["1:0.080", "2:0.085", "3:0.075"])

    assert offsets == {1: 0.080, 2: 0.085, 3: 0.075}


def test_parse_grasp_z_offsets_rejects_unknown_object_id():
    try:
        parse_grasp_z_offsets(["4:0.080"])
    except ValueError as exc:
        assert "object id must be one of [1, 2, 3]" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_parse_grasp_z_offsets_rejects_bad_pair_format():
    try:
        parse_grasp_z_offsets(["1=0.080"])
    except ValueError as exc:
        assert "expected OBJECT_ID:OFFSET" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_resolve_pick_objects_forces_full_order():
    assert resolve_pick_objects([3], full_order_123=True) == [1, 2, 3]


def test_resolve_pick_objects_sorts_and_deduplicates_normal_mode():
    assert resolve_pick_objects([3, 1, 3, 2], full_order_123=False) == [1, 2, 3]


def test_load_grasp_offset_json_reads_best_offsets(tmp_path):
    path = tmp_path / "sweep.json"
    path.write_text(
        json.dumps({"best_offsets": {"1": 0.080, "2": 0.085, "3": 0.075}}),
        encoding="utf-8",
    )

    assert load_grasp_offset_json(str(path)) == {1: 0.080, 2: 0.085, 3: 0.075}


def test_resolve_grasp_z_offsets_prefers_cli_over_json(tmp_path):
    path = tmp_path / "sweep.json"
    path.write_text(
        json.dumps({"best_offsets": {"1": 0.070, "2": 0.070, "3": 0.070}}),
        encoding="utf-8",
    )

    offsets = resolve_grasp_z_offsets(
        cli_offsets=["1:0.080", "2:0.085"],
        json_path=str(path),
        defaults={1: 0.090, 2: 0.090, 3: 0.090},
    )

    assert offsets == {1: 0.080, 2: 0.085, 3: 0.070}


def test_choose_best_offset_prefers_fewer_early_terminations_before_middle_tie():
    rows = [
        {"offset": 0.070, "attempts": 5, "successes": 5, "success_rate": 1.0, "early_terminations": 2},
        {"offset": 0.075, "attempts": 5, "successes": 5, "success_rate": 1.0, "early_terminations": 0},
        {"offset": 0.080, "attempts": 5, "successes": 5, "success_rate": 1.0, "early_terminations": 1},
    ]

    assert choose_best_offset(rows) == 0.075


def test_choose_best_offset_uses_success_rate_then_middle_tie_breaker():
    rows = [
        {"offset": 0.070, "attempts": 5, "successes": 4, "success_rate": 0.8},
        {"offset": 0.075, "attempts": 5, "successes": 5, "success_rate": 1.0},
        {"offset": 0.080, "attempts": 5, "successes": 5, "success_rate": 1.0},
        {"offset": 0.085, "attempts": 5, "successes": 5, "success_rate": 1.0},
        {"offset": 0.090, "attempts": 5, "successes": 3, "success_rate": 0.6},
    ]

    assert choose_best_offset(rows) == 0.080
