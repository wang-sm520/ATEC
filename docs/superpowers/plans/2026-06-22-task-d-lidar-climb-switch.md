# Task D LiDAR Climb Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use Task D's 16×360 LiDAR height scan to confirm the box is in the pit, align the robot with the detected climb path, and only then switch from the walking policy to `policy_climb.pt`.

**Architecture:** Keep `demo/solution.py` self-contained for submission: add a small LiDAR height-scan detector in the same file, feed `obs["extero"]` into the existing wall-push controller, and add two controller phases between `push_x_pit` and `forward`: `confirm_pit_box` and `align_climb`. The detector is advisory and gated by odometry, with a timeout fallback to the current behavior so a noisy scan cannot deadlock the run.

**Tech Stack:** Python 3.11, PyTorch tensors as optional input objects, pure-Python LiDAR parsing, pytest unit tests, existing ATEC Task D FastAPI/demo submission flow.

---

## File Structure

- Modify: `demo/solution.py`
  - Add `_LidarClimbObservation` and `_TaskDLidarClimbDetector` after `_G1VelocityPolicyBridge`.
  - Change `_WallPushController.update()` to accept optional `extero` and store per-step LiDAR diagnostics.
  - Add `confirm_pit_box` and `align_climb` phases between `push_x_pit` and `forward`.
  - Change `AlgSolution.predicts()` to pass `obs.get("extero")` into the controller.

- Create: `tests/test_demo_task_d_lidar_climb_switch.py`
  - Pure-Python unit tests for 16×360 LiDAR parsing, box-in-pit confirmation, lateral alignment sign, invalid scan behavior, and controller phase transitions.
  - Imports `demo.solution` but does not instantiate `AlgSolution`, so policy files are not loaded during tests.

- Existing files to run as regression checks:
  - `tests/task_d/test_lidar_perception.py`
  - `scripts/probe_task_d_groundtruth.py`

---

## Task 1: Add LiDAR Climb Detector Tests

**Files:**
- Create: `tests/test_demo_task_d_lidar_climb_switch.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_demo_task_d_lidar_climb_switch.py` with this full content:

```python
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
)

_CHANNELS = 16
_BINS = 360
_FRONT_BIN = 180


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


def test_controller_enters_lidar_confirmation_before_forward():
    controller = _WallPushController()
    controller.phase = "push_x_pit"
    controller.step = 100
    controller._last_lidar = _LidarClimbObservation.invalid("test has not measured scan")

    controller._update_phase(controller.PIT_EDGE_X + 0.01, 0.0, 0.0)

    assert controller.phase == "confirm_pit_box"
    assert controller.confirm_start_step == 100
    assert controller._lidar_confirm_count == 0


def test_controller_requires_consecutive_lidar_confirmation_before_forward():
    controller = _WallPushController()
    controller.phase = "confirm_pit_box"
    controller.step = 130
    controller.confirm_start_step = 120
    controller._lidar_confirm_count = controller.CONFIRM_REQUIRED_FRAMES - 1
    controller._last_lidar = _LidarClimbObservation(
        valid=True,
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


def test_alg_solution_passes_extero_to_controller_source_contract():
    source = (_REPO / "demo" / "solution.py").read_text(encoding="utf-8")

    assert "extero = obs.get(\"extero\")" in source
    assert "cmd = self.controller.update(row, extero)" in source
```

- [ ] **Step 2: Run the new tests and verify they fail for the expected reason**

Run:

```bash
PYTHONPATH=/home/air/wang-sm/ATEC:/home/air/wang-sm/ATEC/source/atec_rl_lab pytest tests/test_demo_task_d_lidar_climb_switch.py -q
```

Expected: FAIL during import or collection because `_LidarClimbObservation` and `_TaskDLidarClimbDetector` do not exist in `demo.solution` yet.

- [ ] **Step 3: Commit the failing tests**

Run:

```bash
git add tests/test_demo_task_d_lidar_climb_switch.py
git commit -m "test: cover task d lidar climb switching"
```

Expected: commit succeeds with only the new test file staged.

---

## Task 2: Add Self-Contained LiDAR Height-Scan Detector

**Files:**
- Modify: `demo/solution.py`
- Test: `tests/test_demo_task_d_lidar_climb_switch.py`

- [ ] **Step 1: Insert detector classes after `_G1VelocityPolicyBridge`**

In `demo/solution.py`, insert this block after `_G1VelocityPolicyBridge.act()` and before the `# Closed-loop pushing state machine` section:

```python
# --------------------------------------------------------------------------- #
# Task D LiDAR height-scan detector for pit confirmation and climb alignment
# --------------------------------------------------------------------------- #
class _LidarClimbObservation:
    def __init__(
        self,
        valid: bool,
        box_in_pit: bool,
        alignment_angle: float,
        alignment_vy: float,
        confidence: float,
        reason: str,
    ):
        self.valid = valid
        self.box_in_pit = box_in_pit
        self.alignment_angle = alignment_angle
        self.alignment_vy = alignment_vy
        self.confidence = confidence
        self.reason = reason

    @classmethod
    def invalid(cls, reason: str) -> "_LidarClimbObservation":
        return cls(False, False, 0.0, 0.0, 0.0, reason)


class _TaskDLidarClimbDetector:
    """Detect whether the box is usable as a pit bridge from Task D extero.

    Task D exposes a flattened 16 x 360 height_scan. Values are not raw ranges:
    large positive values indicate lower ray hits such as the pit floor, while
    negative values indicate elevated hits such as box faces or box top.
    """

    CHANNELS = 16
    BINS = 360
    FLAT_LENGTH = CHANNELS * BINS
    FRONT_BIN = BINS // 2

    def __init__(
        self,
        front_half_width_bins: int = 35,
        pit_delta: float = 0.35,
        elevated_delta: float = 0.25,
        min_pit_bins: int = 5,
        min_top_bins: int = 4,
    ):
        self.front_half_width_bins = front_half_width_bins
        self.pit_delta = pit_delta
        self.elevated_delta = elevated_delta
        self.min_pit_bins = min_pit_bins
        self.min_top_bins = min_top_bins

    def measure(self, extero: Any) -> _LidarClimbObservation:
        flat = tuple(self._flatten_numbers(extero))
        if not flat:
            return _LidarClimbObservation.invalid("missing extero")
        if len(flat) != self.FLAT_LENGTH:
            return _LidarClimbObservation.invalid(f"expected 5760 lidar values, got {len(flat)}")

        finite = [value for value in flat if math.isfinite(value)]
        if not finite:
            return _LidarClimbObservation.invalid("lidar scan has no finite values")

        rows = [flat[c * self.BINS : (c + 1) * self.BINS] for c in range(self.CHANNELS)]
        median = self._median(finite)
        pit_bins: list[int] = []
        top_bins: list[int] = []
        for bin_index in self._front_bins():
            column = [row[bin_index] for row in rows if math.isfinite(row[bin_index])]
            if not column:
                continue
            col_min = min(column)
            col_max = max(column)
            if col_max - median >= self.pit_delta:
                pit_bins.append(bin_index)
            if median - col_min >= self.elevated_delta:
                top_bins.append(bin_index)

        box_in_pit = len(pit_bins) >= self.min_pit_bins and len(top_bins) >= self.min_top_bins
        if box_in_pit:
            center_bins = top_bins
            center_offset = sum(self._signed_bin_offset(b) for b in center_bins) / len(center_bins)
            alignment_angle = center_offset * (2.0 * math.pi / self.BINS)
            alignment_vy = _clamp(1.1 * math.sin(alignment_angle), -0.35, 0.35)
            confidence = min(1.0, 0.40 + 0.03 * min(len(pit_bins), 10) + 0.03 * min(len(top_bins), 10))
        else:
            alignment_angle = 0.0
            alignment_vy = 0.0
            confidence = 0.0

        reason = (
            f"median={median:.3f} pit_bins={len(pit_bins)} top_bins={len(top_bins)} "
            f"angle={alignment_angle:.3f} vy={alignment_vy:.3f}"
        )
        return _LidarClimbObservation(True, box_in_pit, alignment_angle, alignment_vy, confidence, reason)

    def _front_bins(self) -> list[int]:
        return [
            (self.FRONT_BIN + offset) % self.BINS
            for offset in range(-self.front_half_width_bins, self.front_half_width_bins + 1)
        ]

    def _signed_bin_offset(self, bin_index: int) -> int:
        return ((bin_index - self.FRONT_BIN + self.BINS // 2) % self.BINS) - self.BINS // 2

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return 0.5 * (ordered[middle - 1] + ordered[middle])

    @classmethod
    def _flatten_numbers(cls, value: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, dict):
            return cls._flatten_numbers(value.get("extero"))
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "reshape") and hasattr(value, "tolist"):
            try:
                return cls._flatten_numbers(value.reshape(-1).tolist())
            except TypeError:
                return cls._flatten_numbers(value.tolist())
        if hasattr(value, "tolist"):
            return cls._flatten_numbers(value.tolist())
        if isinstance(value, (str, bytes)):
            return []
        if isinstance(value, Sequence):
            flattened: list[float] = []
            for item in value:
                flattened.extend(cls._flatten_numbers(item))
            return flattened
        try:
            number = float(value)
        except (TypeError, ValueError):
            return []
        return [number]
```

- [ ] **Step 2: Run detector-focused tests and verify partial progress**

Run:

```bash
PYTHONPATH=/home/air/wang-sm/ATEC:/home/air/wang-sm/ATEC/source/atec_rl_lab pytest tests/test_demo_task_d_lidar_climb_switch.py -q
```

Expected: detector tests pass; controller/source-contract tests still fail because the controller has not been wired to the detector or `extero` yet.

- [ ] **Step 3: Commit the detector implementation**

Run:

```bash
git add demo/solution.py tests/test_demo_task_d_lidar_climb_switch.py
git commit -m "feat: detect task d box bridge from lidar scan"
```

Expected: commit succeeds with the detector implementation and the already-added tests.

---

## Task 3: Wire LiDAR Confirmation into the Wall-Push Controller

**Files:**
- Modify: `demo/solution.py`
- Test: `tests/test_demo_task_d_lidar_climb_switch.py`

- [ ] **Step 1: Add controller constants and detector initialization**

In `_WallPushController`, add these class constants under `PIT_EDGE_X` and before `DOWN`:

```python
    CONFIRM_REQUIRED_FRAMES = 3
    CONFIRM_TIMEOUT_STEPS = 80
    ALIGN_REQUIRED_FRAMES = 10
    ALIGN_TIMEOUT_STEPS = 80
    ALIGN_ANGLE_TOL = 0.08
```

Then replace `_WallPushController.__init__` with:

```python
    def __init__(self, warmup_steps: int = 20, dt: float = 0.02):
        self.warmup_steps = warmup_steps
        self.odom = _Odometry(dt=dt)
        self.lidar = _TaskDLidarClimbDetector()
        self.reset()
```

Replace `_WallPushController.reset` with:

```python
    def reset(self) -> None:
        self.odom.reset()
        self.phase = "warmup"
        self.bx, self.by = self.BOX_START
        self.step = 0
        self.wpi = 0
        self.jam_x, self.jam_step = -1e9, 0
        self.x_line = None
        self.around_top = None
        self.around_back2 = None
        self.confirm_start_step = 0
        self.align_start_step = 0
        self._lidar_confirm_count = 0
        self._align_stable_count = 0
        self._last_lidar = _LidarClimbObservation.invalid("lidar has not been measured")
        self.last_debug: dict[str, Any] = {}
```

- [ ] **Step 2: Change `update` to accept `extero` and record debug data**

Replace `_WallPushController.update` with:

```python
    def update(self, proprio_row: Sequence[float], extero: Any = None) -> tuple[float, float, float]:
        rx, ry, ryaw = self.odom.update(proprio_row)
        self._last_lidar = self.lidar.measure(extero)
        self._update_box(rx, ry)
        self._update_phase(rx, ry, ryaw)
        cmd = self._control(rx, ry, ryaw)
        self.last_debug = {
            "phase": self.phase,
            "robot": (round(rx, 3), round(ry, 3), round(ryaw, 3)),
            "box_est": (round(self.bx, 3), round(self.by, 3)),
            "lidar_valid": self._last_lidar.valid,
            "lidar_box_in_pit": self._last_lidar.box_in_pit,
            "lidar_confidence": round(self._last_lidar.confidence, 3),
            "lidar_alignment_angle": round(self._last_lidar.alignment_angle, 3),
            "lidar_alignment_vy": round(self._last_lidar.alignment_vy, 3),
            "lidar_reason": self._last_lidar.reason,
            "cmd": tuple(round(v, 3) for v in cmd),
        }
        self.step += 1
        return cmd
```

- [ ] **Step 3: Replace `push_x_pit` transition with confirmation and alignment phases**

In `_WallPushController._update_phase`, replace the existing `push_x_pit` block:

```python
        elif self.phase == "push_x_pit":
            if rx >= self.PIT_EDGE_X:                       # robot at pit edge → box pushed in
                self.phase = "forward"
```

with this block:

```python
        elif self.phase == "push_x_pit":
            if rx >= self.PIT_EDGE_X:                       # robot at pit edge → confirm before climbing
                self.phase = "confirm_pit_box"
                self.confirm_start_step = self.step
                self._lidar_confirm_count = 0
        elif self.phase == "confirm_pit_box":
            if self._last_lidar.box_in_pit and self._last_lidar.confidence >= 0.5:
                self._lidar_confirm_count += 1
            else:
                self._lidar_confirm_count = 0

            if self._lidar_confirm_count >= self.CONFIRM_REQUIRED_FRAMES:
                if abs(self._last_lidar.alignment_angle) > self.ALIGN_ANGLE_TOL:
                    self.phase = "align_climb"
                    self.align_start_step = self.step
                    self._align_stable_count = 0
                else:
                    self.phase = "forward"
            elif self.step - self.confirm_start_step >= self.CONFIRM_TIMEOUT_STEPS:
                self.phase = "forward"
        elif self.phase == "align_climb":
            if self._last_lidar.valid and abs(self._last_lidar.alignment_angle) <= self.ALIGN_ANGLE_TOL:
                self._align_stable_count += 1
            else:
                self._align_stable_count = 0

            if self._align_stable_count >= self.ALIGN_REQUIRED_FRAMES:
                self.phase = "forward"
            elif self.step - self.align_start_step >= self.ALIGN_TIMEOUT_STEPS:
                self.phase = "forward"
```

- [ ] **Step 4: Add control commands for confirmation and alignment phases**

In `_WallPushController._control`, replace this block:

```python
        if self.phase == "push_x_pit":
            return 0.9, _clamp(1.2 * (self.by - ry), -0.4, 0.4), _clamp(-2.2 * ryaw, -0.8, 0.8)
        # forward — keep walking +x toward the finish (onto the box / across; never stop)
```

with this block:

```python
        if self.phase == "push_x_pit":
            return 0.9, _clamp(1.2 * (self.by - ry), -0.4, 0.4), _clamp(-2.2 * ryaw, -0.8, 0.8)
        if self.phase == "confirm_pit_box":
            vy = self._last_lidar.alignment_vy if self._last_lidar.valid else 1.2 * (self.by - ry)
            return 0.45, _clamp(vy, -0.35, 0.35), _clamp(-2.2 * ryaw, -0.7, 0.7)
        if self.phase == "align_climb":
            vy = self._last_lidar.alignment_vy if self._last_lidar.valid else 0.0
            return 0.15, _clamp(vy, -0.35, 0.35), _clamp(-2.2 * ryaw, -0.6, 0.6)
        # forward — keep walking +x toward the finish (onto the box / across; never stop)
```

- [ ] **Step 5: Run controller tests**

Run:

```bash
PYTHONPATH=/home/air/wang-sm/ATEC:/home/air/wang-sm/ATEC/source/atec_rl_lab pytest tests/test_demo_task_d_lidar_climb_switch.py -q
```

Expected: detector and controller tests pass except the source-contract test for `AlgSolution.predicts`, because `extero` is not passed into the controller yet.

- [ ] **Step 6: Commit controller wiring**

Run:

```bash
git add demo/solution.py tests/test_demo_task_d_lidar_climb_switch.py
git commit -m "feat: gate task d climb switch with lidar confirmation"
```

Expected: commit succeeds with controller-only changes.

---

## Task 4: Pass `obs["extero"]` from `AlgSolution` into the Controller

**Files:**
- Modify: `demo/solution.py:309-316`
- Test: `tests/test_demo_task_d_lidar_climb_switch.py`

- [ ] **Step 1: Replace `AlgSolution.predicts`**

In `demo/solution.py`, replace `AlgSolution.predicts` with:

```python
    def predicts(self, obs: dict, current_score: float):
        proprio = obs["proprio"]
        extero = obs.get("extero")
        row = proprio[0] if hasattr(proprio, "shape") and len(proprio.shape) >= 2 else proprio
        cmd = self.controller.update(row, extero)
        if self.controller.phase == "forward":
            self.bridge.select("climb")     # box confirmed/aligned or timeout fallback → climb to finish
        action = self.bridge.act(proprio, cmd)
        return {"action": action, "giveup": False}
```

- [ ] **Step 2: Run all new tests**

Run:

```bash
PYTHONPATH=/home/air/wang-sm/ATEC:/home/air/wang-sm/ATEC/source/atec_rl_lab pytest tests/test_demo_task_d_lidar_climb_switch.py -q
```

Expected: all tests in `tests/test_demo_task_d_lidar_climb_switch.py` pass.

- [ ] **Step 3: Run existing Task D LiDAR perception regression tests**

Run:

```bash
PYTHONPATH=/home/air/wang-sm/ATEC/source/atec_rl_lab pytest tests/task_d/test_lidar_perception.py -q
```

Expected: existing Task D perception tests pass. These tests cover the train-side LiDAR parser and guard against accidental import/path breakage.

- [ ] **Step 4: Commit `AlgSolution` wiring**

Run:

```bash
git add demo/solution.py tests/test_demo_task_d_lidar_climb_switch.py
git commit -m "feat: pass task d lidar scan into demo controller"
```

Expected: commit succeeds with the final source-contract test passing.

---

## Task 5: Run Lightweight Regression Suite

**Files:**
- Verify: `demo/solution.py`
- Verify: `tests/test_demo_task_d_lidar_climb_switch.py`
- Verify: `tests/task_d/test_lidar_perception.py`
- Verify: `tests/test_rsl_rl_script_compat.py`

- [ ] **Step 1: Run the focused local tests together**

Run:

```bash
PYTHONPATH=/home/air/wang-sm/ATEC:/home/air/wang-sm/ATEC/source/atec_rl_lab pytest \
  tests/test_demo_task_d_lidar_climb_switch.py \
  tests/task_d/test_lidar_perception.py \
  tests/test_rsl_rl_script_compat.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Inspect the resulting diff for accidental policy/model changes**

Run:

```bash
git diff --stat HEAD~3..HEAD
```

Expected: the implementation commits only include `demo/solution.py` and `tests/test_demo_task_d_lidar_climb_switch.py`. If other files appear, inspect them with `git diff -- <path>` and either commit intentional changes separately or restore accidental changes.

- [ ] **Step 3: Commit any small fixes discovered during regression**

If Step 1 or Step 2 required a correction, run:

```bash
git add demo/solution.py tests/test_demo_task_d_lidar_climb_switch.py
git commit -m "fix: stabilize task d lidar climb gate"
```

Expected: commit succeeds only when a correction was necessary. If no correction was necessary, do not create an empty commit.

---

## Task 6: Validate in Isaac Sim Probe

**Files:**
- Verify: `scripts/probe_task_d_groundtruth.py`
- Verify: `demo/solution.py`

- [ ] **Step 1: Run the Task D privileged probe**

Run:

```bash
PYTHONPATH=/home/air/wang-sm/ATEC:/home/air/wang-sm/ATEC/source/atec_rl_lab python scripts/probe_task_d_groundtruth.py \
  --task=ATEC-TaskD-G1 \
  --num_envs=1 \
  --headless \
  --enable_cameras \
  --num_steps 700 \
  --log_every 10 \
  --snapshot_every 100 \
  --out outputs/task_d_lidar_climb
```

Expected:
- The script starts Isaac Sim and does not crash during `AlgSolution.predicts`.
- Probe logs include phase transitions through `push_x_pit`, `confirm_pit_box`, optionally `align_climb`, and `forward`.
- The summary reports `box_reached_target_x: true` or a final box x inside `[-1.4, 0.7]`.
- `fell_in_pit` is `false` for a successful crossing attempt. If it is `true`, keep the local tests passing and adjust only detector thresholds or alignment timeout in a new small commit.

- [ ] **Step 2: Inspect LiDAR debug output in the probe trace**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
trace = json.loads(Path('outputs/task_d_lidar_climb/trace.json').read_text())
for rec in trace[-20:]:
    print(rec)
PY
```

Expected: late trace entries show `phase` moving to `forward`. If the trace never leaves `confirm_pit_box`, lower `min_top_bins` from `4` to `3` or increase `CONFIRM_TIMEOUT_STEPS` from `80` to `100`, then rerun the unit tests and probe.

- [ ] **Step 3: Commit probe-calibrated threshold change only if needed**

If Step 1 or Step 2 shows a repeatable false negative or early false positive, make the smallest threshold change and run:

```bash
PYTHONPATH=/home/air/wang-sm/ATEC:/home/air/wang-sm/ATEC/source/atec_rl_lab pytest tests/test_demo_task_d_lidar_climb_switch.py tests/task_d/test_lidar_perception.py -q
git add demo/solution.py tests/test_demo_task_d_lidar_climb_switch.py
git commit -m "fix: calibrate task d lidar climb thresholds"
```

Expected: threshold commit exists only if the Isaac Sim probe demonstrated the need.

---

## Self-Review

- Spec coverage: The plan implements the requested LiDAR-based confirmation of whether the box has entered the pit, LiDAR-based alignment before stepping onto the box, and keeps the final `policy_climb.pt` switch tied to `phase == "forward"` after confirmation/alignment or timeout fallback.
- Placeholder scan: The plan contains concrete file paths, commands, tests, class names, method names, constants, and code blocks for each source change.
- Type consistency: `_LidarClimbObservation`, `_TaskDLidarClimbDetector`, `_WallPushController.update(row, extero)`, `confirm_pit_box`, `align_climb`, and `AlgSolution.predicts` use the same names across tests and implementation steps.
