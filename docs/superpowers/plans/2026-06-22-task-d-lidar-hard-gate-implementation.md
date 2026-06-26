# Task D LiDAR Hard Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Task D late-stage climb handoff safer by requiring LiDAR bridge evidence before `confirm_pit_box`, removing timeout-to-`forward` paths, and retrying `push_x_pit` for a short debounce interval when confirmation fails.

**Architecture:** Keep the demo submission self-contained in `demo/solution.py`. Extend the existing LiDAR observation with a loose `may_have_bridge` signal, use that signal to gate `push_x_pit -> confirm_pit_box`, and make `confirm_pit_box`/`align_climb` return to `push_x_pit` on failure instead of entering `forward`. Preserve the current walking/climbing policy bridge and only change high-level state transitions and commands.

**Tech Stack:** Python 3.11, PyTorch tensors as runtime observations, pure-Python LiDAR height-scan parsing, pytest, Isaac Sim Task D probe.

---

## File Structure

- Modify: `demo/solution.py`
  - Extend `_LidarClimbObservation` with `may_have_bridge`.
  - Extend `_TaskDLidarClimbDetector.measure()` to emit both loose bridge candidate and strict box-in-pit confirmation.
  - Add retry debounce fields/constants to `_WallPushController`.
  - Change `push_x_pit`, `confirm_pit_box`, and `align_climb` transitions to hard-gate `forward`.
  - Change `confirm_pit_box` command to near-zero forward velocity.

- Modify: `tests/test_demo_task_d_lidar_climb_switch.py`
  - Add tests for loose `may_have_bridge` detector behavior.
  - Replace old transition expectations that allow odometry-only confirmation.
  - Add tests for confirmation timeout returning to `push_x_pit`, cooldown behavior, and `align_climb` failure fallback.

- Verify with existing runtime harness:
  - `scripts/probe_task_d_groundtruth.py`

**Commit note:** The user explicitly said to ignore commits for now, so this plan omits commit steps. Leave changes uncommitted unless the user later asks to commit.

---

## Task 1: Add Failing Tests for Loose LiDAR Bridge Candidate

**Files:**
- Modify: `tests/test_demo_task_d_lidar_climb_switch.py`

- [ ] **Step 1: Add tests for `may_have_bridge`**

Append these tests after `test_detector_marks_missing_or_wrong_size_extero_invalid`:

```python
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
```

- [ ] **Step 2: Run tests to verify the new tests fail for the expected reason**

Run:

```bash
conda activate atec && PYTHONPATH=/home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch:/home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch/source/atec_rl_lab pytest tests/test_demo_task_d_lidar_climb_switch.py -q
```

Expected: FAIL with `AttributeError: '_LidarClimbObservation' object has no attribute 'may_have_bridge'`.

---

## Task 2: Implement Loose Bridge Candidate in the Detector

**Files:**
- Modify: `demo/solution.py`
- Verify: `tests/test_demo_task_d_lidar_climb_switch.py`

- [ ] **Step 1: Extend `_LidarClimbObservation` constructor and invalid factory**

In `demo/solution.py`, replace `_LidarClimbObservation.__init__` with:

```python
    def __init__(
        self,
        valid: bool,
        may_have_bridge: bool,
        box_in_pit: bool,
        alignment_angle: float,
        alignment_vy: float,
        confidence: float,
        reason: str,
    ):
        self.valid = valid
        self.may_have_bridge = may_have_bridge
        self.box_in_pit = box_in_pit
        self.alignment_angle = alignment_angle
        self.alignment_vy = alignment_vy
        self.confidence = confidence
        self.reason = reason
```

Replace `_LidarClimbObservation.invalid` with:

```python
    @classmethod
    def invalid(cls, reason: str) -> "_LidarClimbObservation":
        return cls(False, False, False, 0.0, 0.0, 0.0, reason)
```

- [ ] **Step 2: Add loose detector thresholds**

In `_TaskDLidarClimbDetector.__init__`, replace the signature with:

```python
    def __init__(
        self,
        front_half_width_bins: int = 35,
        pit_delta: float = 0.35,
        elevated_delta: float = 0.25,
        min_pit_bins: int = 5,
        min_top_bins: int = 4,
        loose_min_pit_bins: int = 4,
        loose_min_top_bins: int = 8,
    ):
        self.front_half_width_bins = front_half_width_bins
        self.pit_delta = pit_delta
        self.elevated_delta = elevated_delta
        self.min_pit_bins = min_pit_bins
        self.min_top_bins = min_top_bins
        self.loose_min_pit_bins = loose_min_pit_bins
        self.loose_min_top_bins = loose_min_top_bins
```

- [ ] **Step 3: Emit `may_have_bridge` from `measure()`**

In `_TaskDLidarClimbDetector.measure()`, replace:

```python
        box_in_pit = len(pit_bins) >= self.min_pit_bins and len(top_bins) >= self.min_top_bins
```

with:

```python
        may_have_bridge = (
            len(pit_bins) >= self.loose_min_pit_bins
            or len(top_bins) >= self.loose_min_top_bins
        )
        box_in_pit = len(pit_bins) >= self.min_pit_bins and len(top_bins) >= self.min_top_bins
```

Replace the `reason` block with:

```python
        reason = (
            f"median={median:.3f} pit_bins={len(pit_bins)} top_bins={len(top_bins)} "
            f"may_have_bridge={may_have_bridge} angle={alignment_angle:.3f} vy={alignment_vy:.3f}"
        )
        return _LidarClimbObservation(
            True, may_have_bridge, box_in_pit, alignment_angle, alignment_vy, confidence, reason
        )
```

- [ ] **Step 4: Update existing test construction of `_LidarClimbObservation`**

In `tests/test_demo_task_d_lidar_climb_switch.py`, every direct `_LidarClimbObservation(...)` construction must include `may_have_bridge=True` between `valid=True` and `box_in_pit=True`. For example:

```python
    controller._last_lidar = _LidarClimbObservation(
        valid=True,
        may_have_bridge=True,
        box_in_pit=True,
        alignment_angle=0.0,
        alignment_vy=0.0,
        confidence=0.9,
        reason="test centered confirmation",
    )
```

- [ ] **Step 5: Run detector tests**

Run:

```bash
conda activate atec && PYTHONPATH=/home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch:/home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch/source/atec_rl_lab pytest tests/test_demo_task_d_lidar_climb_switch.py -q
```

Expected: detector tests pass; controller hard-gate tests are not added yet.

---

## Task 3: Add Failing Tests for Hard-Gated `push_x_pit` and `confirm_pit_box`

**Files:**
- Modify: `tests/test_demo_task_d_lidar_climb_switch.py`

- [ ] **Step 1: Replace old odometry-only confirmation test**

Replace `test_controller_enters_lidar_confirmation_before_forward` with:

```python
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


def test_controller_enters_confirmation_when_lidar_has_bridge_candidate():
    controller = _WallPushController()
    controller.phase = "push_x_pit"
    controller.step = 100
    controller.retry_confirm_after_step = 0
    controller._last_lidar = _LidarClimbObservation(
        valid=True,
        may_have_bridge=True,
        box_in_pit=False,
        alignment_angle=0.0,
        alignment_vy=0.0,
        confidence=0.0,
        reason="test loose bridge candidate",
    )

    controller._update_phase(controller.PIT_EDGE_X + 0.01, 0.0, 0.0)

    assert controller.phase == "confirm_pit_box"
    assert controller.confirm_start_step == 100
    assert controller._lidar_confirm_count == 0
```

- [ ] **Step 2: Add tests for confirmation timeout fallback and near-zero forward command**

Append after the confirmation success/alignment tests:

```python
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
```

- [ ] **Step 3: Add cooldown test**

Append:

```python
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
```

- [ ] **Step 4: Run tests and verify expected failures**

Run:

```bash
conda activate atec && PYTHONPATH=/home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch:/home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch/source/atec_rl_lab pytest tests/test_demo_task_d_lidar_climb_switch.py -q
```

Expected: FAIL because `_WallPushController` does not yet define `MIN_REPUSH_STEPS` / `retry_confirm_after_step`, and because current `confirm_pit_box` timeout still enters `forward`.

---

## Task 4: Implement Hard Gate for `push_x_pit` and `confirm_pit_box`

**Files:**
- Modify: `demo/solution.py`
- Verify: `tests/test_demo_task_d_lidar_climb_switch.py`

- [ ] **Step 1: Add retry debounce constant and reset field**

In `_WallPushController`, add this constant after `ALIGN_ANGLE_TOL`:

```python
    MIN_REPUSH_STEPS = 30
```

In `reset()`, add:

```python
        self.retry_confirm_after_step = 0
```

- [ ] **Step 2: Add helper for returning to repush**

In `_WallPushController`, add this method before `_update_phase`:

```python
    def _return_to_push_x_pit(self) -> None:
        self.phase = "push_x_pit"
        self.retry_confirm_after_step = self.step + self.MIN_REPUSH_STEPS
        self._lidar_confirm_count = 0
        self._align_stable_count = 0
```

- [ ] **Step 3: Gate `push_x_pit` using LiDAR candidate and cooldown**

In `_update_phase`, replace the `push_x_pit` block with:

```python
        elif self.phase == "push_x_pit":
            can_retry_confirm = self.step >= self.retry_confirm_after_step
            if rx >= self.PIT_EDGE_X and can_retry_confirm and self._last_lidar.may_have_bridge:
                self.phase = "confirm_pit_box"
                self.confirm_start_step = self.step
                self._lidar_confirm_count = 0
```

- [ ] **Step 4: Make `confirm_pit_box` timeout return to repush**

In `_update_phase`, replace:

```python
            elif self.step - self.confirm_start_step >= self.CONFIRM_TIMEOUT_STEPS:
                self.phase = "forward"
```

with:

```python
            elif self.step - self.confirm_start_step >= self.CONFIRM_TIMEOUT_STEPS:
                self._return_to_push_x_pit()
```

- [ ] **Step 5: Stop probing forward during confirmation**

In `_control`, replace the `confirm_pit_box` block with:

```python
        if self.phase == "confirm_pit_box":
            vy = self._last_lidar.alignment_vy if self._last_lidar.valid else 0.0
            return 0.0, _clamp(vy, -0.35, 0.35), _clamp(-2.2 * ryaw, -0.7, 0.7)
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
conda activate atec && PYTHONPATH=/home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch:/home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch/source/atec_rl_lab pytest tests/test_demo_task_d_lidar_climb_switch.py -q
```

Expected: hard-gated `push_x_pit` and `confirm_pit_box` tests pass. `align_climb` hard-gate tests are not added yet.

---

## Task 5: Add and Implement Hard Gate for `align_climb`

**Files:**
- Modify: `tests/test_demo_task_d_lidar_climb_switch.py`
- Modify: `demo/solution.py`

- [ ] **Step 1: Add failing tests for `align_climb` fallback**

Append to `tests/test_demo_task_d_lidar_climb_switch.py`:

```python
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
```

- [ ] **Step 2: Run tests and verify expected failure**

Run:

```bash
conda activate atec && PYTHONPATH=/home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch:/home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch/source/atec_rl_lab pytest tests/test_demo_task_d_lidar_climb_switch.py -q
```

Expected: FAIL because current `align_climb` timeout still enters `forward` and lost strict confirmation only resets `_align_stable_count`.

- [ ] **Step 3: Implement align hard gate**

In `_update_phase`, replace the `align_climb` block with:

```python
        elif self.phase == "align_climb":
            if not (self._last_lidar.valid and self._last_lidar.box_in_pit):
                self._return_to_push_x_pit()
            elif abs(self._last_lidar.alignment_angle) <= self.ALIGN_ANGLE_TOL:
                self._align_stable_count += 1
                if self._align_stable_count >= self.ALIGN_REQUIRED_FRAMES:
                    self.phase = "forward"
            elif self.step - self.align_start_step >= self.ALIGN_TIMEOUT_STEPS:
                self._return_to_push_x_pit()
            else:
                self._align_stable_count = 0
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
conda activate atec && PYTHONPATH=/home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch:/home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch/source/atec_rl_lab pytest tests/test_demo_task_d_lidar_climb_switch.py -q
```

Expected: all hard-gate unit tests pass.

---

## Task 6: Run Regression Tests and Runtime Probe

**Files:**
- Verify: `demo/solution.py`
- Verify: `tests/test_demo_task_d_lidar_climb_switch.py`
- Verify: `tests/task_d/test_lidar_perception.py`
- Runtime: `scripts/probe_task_d_groundtruth.py`

- [ ] **Step 1: Run focused regression tests**

Run:

```bash
conda activate atec && PYTHONPATH=/home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch:/home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch/source/atec_rl_lab pytest \
  tests/test_demo_task_d_lidar_climb_switch.py \
  tests/task_d/test_lidar_perception.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run syntax and whitespace checks**

Run:

```bash
conda activate atec && python -m py_compile demo/solution.py tests/test_demo_task_d_lidar_climb_switch.py
git diff --check -- demo/solution.py tests/test_demo_task_d_lidar_climb_switch.py
```

Expected: both commands exit successfully with no whitespace errors.

- [ ] **Step 3: Run Task D runtime probe**

Run:

```bash
conda activate atec && PYTHONPATH=/home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch:/home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch/source/atec_rl_lab python scripts/probe_task_d_groundtruth.py \
  --task=ATEC-TaskD-G1 \
  --num_envs=1 \
  --headless \
  --enable_cameras \
  --num_steps 2600 \
  --log_every 50 \
  --snapshot_every 500 \
  --out outputs/task_d_lidar_hard_gate
```

Expected:

- Probe starts Isaac Sim and reaches Task D runtime.
- Score remains at least 16.0, unless the hard gate intentionally prevents unsafe late crossing after collecting the safe points.
- Phase records must not show an unconfirmed timeout path into `forward`.
- If `forward` appears, it must occur after strict confirmation or stable alignment.

- [ ] **Step 4: Inspect runtime phase records**

Run:

```bash
python - <<'PY'
import json
from collections import Counter
from pathlib import Path
trace = json.loads(Path('outputs/task_d_lidar_hard_gate/trace.json').read_text())
records = trace['records']
print('summary=', json.dumps(trace['summary'], sort_keys=True))
print('phase_counts=', dict(Counter(r['phase'] for r in records)))
for r in records[-25:]:
    print(r['step'], r['phase'], r['score'], r['robot_w'], r['box_w'], r['cmd'])
PY
```

Expected: phase summary and tail records are available for diagnosis. If the score is still 16, classify it as one of:

```text
safe hard gate prevented unsafe forward
confirmed forward occurred but climb policy failed
```

Use the phase records and final robot pose to choose the classification.

---

## Self-Review

- Spec coverage: The plan covers the hard `confirm_pit_box` gate, LiDAR-gated `push_x_pit` transition, retry cooldown, `align_climb` fallback, invalid LiDAR behavior, focused tests, and runtime probe verification.
- Placeholder scan: The plan uses concrete paths, exact method/field names, complete test code, exact implementation snippets, and exact commands. It contains no placeholder implementation steps.
- Type consistency: `_LidarClimbObservation.may_have_bridge`, `_TaskDLidarClimbDetector.measure()`, `_WallPushController.retry_confirm_after_step`, `_WallPushController.MIN_REPUSH_STEPS`, and `_return_to_push_x_pit()` are defined before later tasks use them.
