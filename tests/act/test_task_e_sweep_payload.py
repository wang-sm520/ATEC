from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[2]
_ACT_DIR = _REPO / "scripts" / "act"
if str(_ACT_DIR) not in sys.path:
    sys.path.insert(0, str(_ACT_DIR))

from sweep_task_e_grasp_offsets import build_sweep_payload  # noqa: E402


def test_build_sweep_payload_records_best_offsets_and_results():
    sweep_results = {
        1: ([{"object_id": 1, "offset": 0.08, "attempts": 5, "successes": 5, "success_rate": 1.0}], 0.08),
        2: ([{"object_id": 2, "offset": 0.085, "attempts": 5, "successes": 4, "success_rate": 0.8}], 0.085),
        3: ([{"object_id": 3, "offset": 0.075, "attempts": 5, "successes": 5, "success_rate": 1.0}], 0.075),
    }

    payload = build_sweep_payload(
        objects=[1, 2, 3],
        offsets=[0.075, 0.08, 0.085],
        attempts_per_offset=5,
        optimized_grasp_flow=True,
        tool_center_offset_local=[0.0, 0.01, 0.0],
        sweep_results=sweep_results,
    )

    assert payload["objects"] == [1, 2, 3]
    assert payload["offsets"] == [0.075, 0.08, 0.085]
    assert payload["attempts_per_offset"] == 5
    assert payload["optimized_grasp_flow"] is True
    assert payload["tool_center_offset_local"] == [0.0, 0.01, 0.0]
    assert payload["best_offsets"] == {"1": 0.08, "2": 0.085, "3": 0.075}
    assert payload["results"]["1"][0]["successes"] == 5
