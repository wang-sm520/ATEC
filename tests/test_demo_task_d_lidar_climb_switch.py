"""Tests for Task D demo LiDAR-based climb switching.

The demo solution must stay self-contained for submission, so these tests import
its private helper classes directly. They avoid constructing AlgSolution because
that would load TorchScript policy files.
"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from demo.solution import (  # noqa: E402
    _LidarClimbObservation,
    _TaskDLidarClimbDetector,
    _WallPushController,
    _should_giveup_for_score,
)

_CHANNELS = 16
_BINS = 360
_FRONT_BIN = 180


def test_solution_giveup_after_task_d_score_is_complete():
    assert not _should_giveup_for_score(35.0)
    assert _should_giveup_for_score(35.01)
    assert _should_giveup_for_score(36.0)
    assert not _should_giveup_for_score("not-a-number")


def _synthetic_scan(
    start_bin: int,
    end_bin: int,
    *,
    pit_height_value: float = 0.65,
    box_top_value: float = -0.45,
) -> list[float]:
    """Build a 16x360 height scan with a pit and box-top signal.

    Most values are flat ground around 0.0. Positive values represent deeper
    hits such as a pit floor. Negative values represent elevated hits such as
    the top/front face of a box according to IsaacLab height_scan semantics.
    """

    grid = [[0.0 for _ in range(_BINS)] for _ in range(_CHANNELS)]
    for channel in range(0, 4):
        for bin_index in range(start_bin, end_bin + 1):
            grid[channel][bin_index % _BINS] = pit_height_value
    for channel in range(8, 13):
        for bin_index in range(start_bin, end_bin + 1):
            grid[channel][bin_index % _BINS] = box_top_value
    return [value for row in grid for value in row]


def test_detector_confirms_centered_box_in_pit_from_height_scan():
    detector = _TaskDLidarClimbDetector()
    scan = _synthetic_scan(_FRONT_BIN - 4, _FRONT_BIN + 4)

    observation = detector.measure(scan)

    assert observation.valid, observation.reason
    assert observation.box_in_pit, observation.reason
    assert observation.confidence >= 0.5
    assert abs(observation.alignment_angle) < 0.03
    assert abs(observation.alignment_vy) < 0.05
    assert "pit_bins=9" in observation.reason
    assert "top_bins=9" in observation.reason


def test_detector_outputs_positive_lateral_alignment_for_left_shifted_box():
    detector = _TaskDLidarClimbDetector()
    scan = _synthetic_scan(_FRONT_BIN + 18, _FRONT_BIN + 26)

    observation = detector.measure(scan)

    assert observation.valid, observation.reason
    assert observation.box_in_pit, observation.reason
    assert observation.alignment_angle > 0.25
    assert observation.alignment_vy > 0.20


def test_detector_reports_front_elevated_count():
    detector = _TaskDLidarClimbDetector()
    scan = _synthetic_scan(_FRONT_BIN - 4, _FRONT_BIN + 4)  # 9 elevated bins in front

    observation = detector.measure(scan)

    assert observation.front_elevated_count == 9
    flat = [0.0] * (_CHANNELS * _BINS)
    assert detector.measure(flat).front_elevated_count == 0


def test_detector_reports_front_top_height():
    detector = _TaskDLidarClimbDetector()
    # box_top_value=-0.45 over a ~0.0 median -> tallest object reads 0.45 m.
    scan = _synthetic_scan(_FRONT_BIN - 4, _FRONT_BIN + 4, box_top_value=-0.45)

    observation = detector.measure(scan)

    assert abs(observation.front_top_height - 0.45) < 1e-6
    flat = [0.0] * (_CHANNELS * _BINS)
    assert detector.measure(flat).front_top_height == 0.0


def test_detector_rejects_flat_ground_as_not_box_in_pit():
    detector = _TaskDLidarClimbDetector()
    scan = [0.0] * (_CHANNELS * _BINS)

    observation = detector.measure(scan)

    assert observation.valid, observation.reason
    assert not observation.box_in_pit
    assert observation.confidence == 0.0
    assert observation.alignment_angle == 0.0
    assert "pit_bins=0" in observation.reason
    assert "top_bins=0" in observation.reason


def test_detector_marks_missing_or_wrong_size_extero_invalid():
    detector = _TaskDLidarClimbDetector()

    missing = detector.measure(None)
    short = detector.measure([0.1, 0.2, 0.3])

    assert not missing.valid
    assert not missing.box_in_pit
    assert missing.reason == "missing extero"
    assert not short.valid
    assert not short.box_in_pit
    assert short.reason == "expected 5760 lidar values, got 3"


def test_detector_reports_loose_bridge_candidate_without_strict_confirmation():
    detector = _TaskDLidarClimbDetector()
    scan = _synthetic_scan(
        _FRONT_BIN - 2,
        _FRONT_BIN + 2,
        pit_height_value=0.55,
        box_top_value=-0.05,
    )

    observation = detector.measure(scan)

    assert observation.valid, observation.reason
    assert observation.may_have_bridge, observation.reason
    assert not observation.box_in_pit
    assert "may_have_bridge=True" in observation.reason


def test_detector_rejects_flat_ground_as_no_bridge_candidate():
    detector = _TaskDLidarClimbDetector()
    scan = [0.0] * (_CHANNELS * _BINS)

    observation = detector.measure(scan)

    assert observation.valid, observation.reason
    assert not observation.may_have_bridge
    assert not observation.box_in_pit
    assert "may_have_bridge=False" in observation.reason


def test_controller_stays_in_push_x_pit_when_lidar_has_no_bridge_candidate():
    controller = _WallPushController()
    controller.phase = "push_x_pit"
    controller.step = 100
    controller.retry_confirm_after_step = 0
    controller._last_lidar = _LidarClimbObservation(
        valid=True,
        may_have_bridge=False,
        box_in_pit=False,
        alignment_angle=0.0,
        alignment_vy=0.0,
        confidence=0.0,
        reason="test no bridge candidate",
    )

    controller._update_phase(controller.PIT_EDGE_X + 0.01, 0.0, 0.0)

    assert controller.phase == "push_x_pit"


def _front_lidar(front_top_height: float, reason: str) -> _LidarClimbObservation:
    """LiDAR observation carrying only the front object-top height used by the
    box-drop climb-switch gate (other fields are irrelevant to that path)."""
    return _LidarClimbObservation(
        valid=True,
        may_have_bridge=False,
        box_in_pit=False,
        alignment_angle=0.0,
        alignment_vy=0.0,
        confidence=0.0,
        reason=reason,
        front_top_height=front_top_height,
    )


def test_controller_marks_box_seen_and_stays_while_box_on_ground():
    controller = _WallPushController()
    controller.phase = "push_x_pit"
    controller._last_lidar = _front_lidar(0.6, "box ~0.6m tall on the ground ahead")

    controller._update_phase(controller.PIT_EDGE_X + 0.05, 0.0, 0.0)

    # Tall object ahead -> remember the box, keep pushing, do not switch yet.
    assert controller.phase == "push_x_pit"
    assert controller._front_box_seen
    assert controller._drop_below_count == 0


def test_controller_settles_when_box_top_drops_below_threshold():
    controller = _WallPushController()
    controller.phase = "push_x_pit"
    controller.step = 100
    controller._front_box_seen = True  # box was seen standing a moment ago
    controller._drop_below_count = controller.DROP_DEBOUNCE_FRAMES - 1
    # Object top now below 0.40 m -> box has fallen into the pit.
    controller._last_lidar = _front_lidar(0.05, "box top sank below 0.4m")

    controller._update_phase(controller.PIT_EDGE_X + 0.05, 0.0, 0.0)

    # Box dropped -> settle/centre first, do NOT rush straight into forward/climb.
    assert controller.phase == "settle_climb"
    assert controller.settle_start_step == 100
    assert controller._settle_count == 0


def test_controller_requires_debounce_before_declaring_box_dropped():
    controller = _WallPushController()
    controller.phase = "push_x_pit"
    controller._front_box_seen = True
    controller._drop_below_count = 0
    # A single below-threshold frame (e.g. a transient dip) must not trigger.
    controller._last_lidar = _front_lidar(0.10, "single low frame")

    controller._update_phase(controller.PIT_EDGE_X + 0.05, 0.0, 0.0)

    assert controller.phase == "push_x_pit"
    assert controller._drop_below_count == 1


def test_controller_settle_holds_until_centered_on_box():
    controller = _WallPushController()
    controller.phase = "settle_climb"
    controller.by = 0.0
    controller.settle_start_step = 0
    controller.step = 10  # well within SETTLE_TIMEOUT_STEPS
    controller._settle_count = 0

    # ry far from the box centre -> not aligned -> keep settling.
    controller._update_phase(controller.PIT_EDGE_X + 0.1, 0.5, 0.0)

    assert controller.phase == "settle_climb"
    assert controller._settle_count == 0


def test_controller_settle_advances_to_forward_when_centered():
    controller = _WallPushController()
    controller.phase = "settle_climb"
    controller.by = 0.0
    controller.settle_start_step = 0
    controller.step = 10
    controller._settle_count = controller.SETTLE_REQUIRED_FRAMES - 1

    # ry == box_y and yaw == 0 -> aligned -> final frame commits to forward.
    controller._update_phase(controller.PIT_EDGE_X + 0.1, 0.0, 0.0)

    assert controller.phase == "forward"


def test_controller_settle_commits_to_forward_on_timeout():
    controller = _WallPushController()
    controller.phase = "settle_climb"
    controller.by = 0.0
    controller.settle_start_step = 0
    controller.step = controller.SETTLE_TIMEOUT_STEPS  # timeout reached
    controller._settle_count = 0

    # Still not aligned, but timeout fires so we commit rather than stall forever.
    controller._update_phase(controller.PIT_EDGE_X + 0.1, 0.6, 0.4)

    assert controller.phase == "forward"


def test_controller_settle_command_strafes_to_box_center_without_advancing():
    controller = _WallPushController()
    controller.phase = "settle_climb"
    controller.by = 0.5  # box centre to the +y side of the robot

    cmd = controller._control(controller.PIT_EDGE_X + 0.1, 0.0, 0.0)

    assert abs(cmd[0]) <= 0.05   # not advancing forward while settling
    assert cmd[1] > 0.20         # strafing toward the box centre (+y)


def test_controller_does_not_switch_to_climb_before_box_seen():
    controller = _WallPushController()
    controller.phase = "push_x_pit"
    assert not controller._front_box_seen
    controller._drop_below_count = controller.DROP_DEBOUNCE_FRAMES  # debounce satisfied
    # Low/empty front but the box was never observed standing -> not a drop.
    controller._last_lidar = _front_lidar(0.0, "empty front, box never seen")

    controller._update_phase(controller.PIT_EDGE_X + 0.05, 0.0, 0.0)

    assert controller.phase == "push_x_pit"


def test_controller_waits_for_pit_edge_before_climb_switch():
    controller = _WallPushController()
    controller.phase = "push_x_pit"
    controller._front_box_seen = True
    controller._drop_below_count = controller.DROP_DEBOUNCE_FRAMES  # debounce satisfied
    controller._last_lidar = _front_lidar(0.05, "box dropped but robot not at pit edge")

    controller._update_phase(controller.PIT_EDGE_X - 0.2, 0.0, 0.0)

    assert controller.phase == "push_x_pit"


def test_controller_requires_consecutive_lidar_confirmation_before_forward():
    controller = _WallPushController()
    controller.phase = "confirm_pit_box"
    controller.step = 130
    controller.confirm_start_step = 120
    controller._lidar_confirm_count = controller.CONFIRM_REQUIRED_FRAMES - 1
    controller._last_lidar = _LidarClimbObservation(
        valid=True,
        may_have_bridge=True,
        box_in_pit=True,
        alignment_angle=0.0,
        alignment_vy=0.0,
        confidence=0.9,
        reason="test centered confirmation",
    )

    controller._update_phase(controller.PIT_EDGE_X + 0.02, 0.0, 0.0)

    assert controller.phase == "forward"


def test_controller_aligns_before_forward_when_lidar_target_is_off_center():
    controller = _WallPushController()
    controller.phase = "confirm_pit_box"
    controller.step = 130
    controller.confirm_start_step = 120
    controller._lidar_confirm_count = controller.CONFIRM_REQUIRED_FRAMES - 1
    controller._last_lidar = _LidarClimbObservation(
        valid=True,
        may_have_bridge=True,
        box_in_pit=True,
        alignment_angle=0.28,
        alignment_vy=0.31,
        confidence=0.9,
        reason="test off-center confirmation",
    )

    controller._update_phase(controller.PIT_EDGE_X + 0.02, 0.0, 0.0)
    cmd = controller._control(controller.PIT_EDGE_X + 0.02, 0.0, 0.0)

    assert controller.phase == "align_climb"
    assert cmd[0] > 0.0
    assert cmd[1] > 0.20


def test_controller_confirmation_timeout_returns_to_push_x_pit_with_cooldown():
    controller = _WallPushController()
    controller.phase = "confirm_pit_box"
    controller.step = 200
    controller.confirm_start_step = 200 - controller.CONFIRM_TIMEOUT_STEPS
    controller._lidar_confirm_count = 0
    controller._last_lidar = _LidarClimbObservation(
        valid=True,
        may_have_bridge=True,
        box_in_pit=False,
        alignment_angle=0.0,
        alignment_vy=0.0,
        confidence=0.0,
        reason="test confirmation timeout",
    )

    controller._update_phase(controller.PIT_EDGE_X + 0.02, 0.0, 0.0)

    assert controller.phase == "push_x_pit"
    assert controller.retry_confirm_after_step == 200 + controller.MIN_REPUSH_STEPS


def test_controller_confirmation_command_does_not_probe_forward():
    controller = _WallPushController()
    controller.phase = "confirm_pit_box"
    controller._last_lidar = _LidarClimbObservation(
        valid=True,
        may_have_bridge=True,
        box_in_pit=False,
        alignment_angle=0.0,
        alignment_vy=0.27,
        confidence=0.0,
        reason="test confirmation command",
    )

    cmd = controller._control(controller.PIT_EDGE_X + 0.02, 0.0, 0.0)

    assert cmd[0] <= 0.05
    assert cmd[1] > 0.20


def test_controller_cooldown_blocks_immediate_reentry_to_confirmation():
    controller = _WallPushController()
    controller.phase = "push_x_pit"
    controller.step = 120
    controller.retry_confirm_after_step = 150
    controller._last_lidar = _LidarClimbObservation(
        valid=True,
        may_have_bridge=True,
        box_in_pit=True,
        alignment_angle=0.0,
        alignment_vy=0.0,
        confidence=0.9,
        reason="test cooldown bridge candidate",
    )

    controller._update_phase(controller.PIT_EDGE_X + 0.01, 0.0, 0.0)

    assert controller.phase == "push_x_pit"


def test_controller_align_timeout_returns_to_push_x_pit_with_cooldown():
    controller = _WallPushController()
    controller.phase = "align_climb"
    controller.step = 300
    controller.align_start_step = 300 - controller.ALIGN_TIMEOUT_STEPS
    controller._align_stable_count = 0
    controller._last_lidar = _LidarClimbObservation(
        valid=True,
        may_have_bridge=True,
        box_in_pit=True,
        alignment_angle=0.30,
        alignment_vy=0.30,
        confidence=0.9,
        reason="test align timeout",
    )

    controller._update_phase(controller.PIT_EDGE_X + 0.02, 0.0, 0.0)

    assert controller.phase == "push_x_pit"
    assert controller.retry_confirm_after_step == 300 + controller.MIN_REPUSH_STEPS


def test_controller_align_loses_lidar_confirmation_returns_to_push_x_pit():
    controller = _WallPushController()
    controller.phase = "align_climb"
    controller.step = 310
    controller.align_start_step = 300
    controller._last_lidar = _LidarClimbObservation(
        valid=True,
        may_have_bridge=True,
        box_in_pit=False,
        alignment_angle=0.02,
        alignment_vy=0.02,
        confidence=0.0,
        reason="test lost strict confirmation",
    )

    controller._update_phase(controller.PIT_EDGE_X + 0.02, 0.0, 0.0)

    assert controller.phase == "push_x_pit"
    assert controller.retry_confirm_after_step == 310 + controller.MIN_REPUSH_STEPS


def test_alg_solution_passes_extero_to_controller_and_command_to_bridge(monkeypatch):
    import demo.solution as solution

    def fail_if_policy_loads(*args, **kwargs):
        raise AssertionError("test must not load TorchScript policy files")

    monkeypatch.setattr(solution.torch.jit, "load", fail_if_policy_loads)

    proprio = solution.torch.tensor([[1.0, 2.0, 3.0]])
    expected_row = proprio[0]
    sentinel_extero = object()
    returned_command = (0.4, -0.1, 0.2)
    returned_action = [0.7, 0.8]

    class FakeController:
        phase = "push_x_pit"

        def __init__(self):
            self.update_calls = []

        def update(self, proprio_row, extero):
            self.update_calls.append((proprio_row, extero))
            return returned_command

    class FakeBridge:
        def __init__(self):
            self.act_calls = []
            self.select_calls = []

        def select(self, policy_name):
            self.select_calls.append(policy_name)

        def act(self, proprio, cmd):
            self.act_calls.append((proprio, cmd))
            return returned_action

    alg = solution.AlgSolution.__new__(solution.AlgSolution)
    alg.controller = FakeController()
    alg.bridge = FakeBridge()
    obs = {"proprio": proprio, "extero": sentinel_extero}

    result = alg.predicts(obs, current_score=0.0)

    [(controller_row, controller_extero)] = alg.controller.update_calls
    [(bridge_proprio, bridge_command)] = alg.bridge.act_calls
    assert solution.torch.equal(controller_row, expected_row)
    assert controller_extero is sentinel_extero
    assert bridge_proprio is proprio
    assert bridge_command is returned_command
    assert alg.bridge.select_calls == []
    assert result == {"action": returned_action, "giveup": False}
