# Task E Bottle Side-Grasp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated object_2 Mustard bottle side-grasp flow that approaches from the robot side along +X→object, descends to bottle-body height, advances in X, closes, lifts, and then reuses the existing basket placement flow.

**Architecture:** Keep the existing `PickPlaceStateMachine` as the single source of demo trajectory state. Add object_2-specific constants in `scripts/act/task_e/config.py`, then branch `PickPlaceStateMachine._current_state_order()`, `_get_target_pos_gripper()`, and `_get_target_quat()` for bottle-specific states while leaving object_1 and object_3 flows unchanged.

**Tech Stack:** Python, PyTorch tensors, pytest, existing ACT fake IsaacLab stubs in `tests/act/test_task_e_state_machine.py`.

---

## Known Baseline Note

At the time this plan was written, `conda run -n atec pytest tests/act -q` has one existing failure unrelated to bottle side-grasp:

```text
tests/act/test_task_e_state_machine.py::test_optimized_steps_keep_grasp_phases_conservative
assert config.OPTIMIZED_STEPS["PRE_GRASP"] == 180
# actual current value is 40
```

This plan does not change optimized step-speed policy. Verification commands below use targeted tests or exclude that known timing assertion.

## File Structure

- Modify: `scripts/act/task_e/config.py`
  - Responsibility: ACT Task E constants.
  - Add named bottle side-grasp constants and object_2 bottle-state step durations.
- Modify: `scripts/act/task_e/state_machine.py`
  - Responsibility: Produce target end-effector pose/quaternion/gripper command per state.
  - Add object_2 side-grasp state order, target positions, cached object position handling, and side-grasp orientation selection.
- Modify: `tests/act/test_task_e_state_machine.py`
  - Responsibility: Unit-test state-machine behavior without Isaac Sim.
  - Replace the object_2 default pick expectation with bottle side-grasp expectations and add path/quaternion target tests.
- No changes to collector success judgment, basket placement, object randomization, CLI parsing, or sweep payload format.

## Task 1: Add failing tests for object_2 bottle side-grasp behavior

**Files:**
- Modify: `tests/act/test_task_e_state_machine.py`
- Test: `tests/act/test_task_e_state_machine.py`

- [ ] **Step 1: Replace the old object_2 state-order test**

In `tests/act/test_task_e_state_machine.py`, replace the existing function:

```python
def test_object_two_skips_align_gripper_state():
    steps = {state: 1 for state in [*config.STATE_ORDER, "ALIGN_GRIPPER"]}
    sm = PickPlaceStateMachine([2], "cpu", steps=steps, grasp_z_offsets={2: 0.10})
    obj_pos = torch.tensor([1.0, 2.0, 0.30])

    visited = []
    while not sm.done:
        visited.append(sm.state)
        sm.tick(obj_pos)

    assert "ALIGN_GRIPPER" not in visited
    assert visited[:4] == ["INIT", "PRE_GRASP", "REACH", "CLOSE"]
```

with this new desired behavior test:

```python
def test_object_two_uses_bottle_side_grasp_state_sequence():
    bottle_states = [
        "BOTTLE_ALIGN_SIDE",
        "BOTTLE_DESCEND_MID",
        "BOTTLE_APPROACH_X",
        "BOTTLE_CLOSE",
        "BOTTLE_LIFT",
    ]
    steps = {state: 1 for state in [*config.STATE_ORDER, *bottle_states]}
    object_state_steps = {2: {state: 1 for state in ["PRE_GRASP", *bottle_states]}}
    sm = PickPlaceStateMachine(
        [2],
        "cpu",
        steps=steps,
        grasp_z_offsets={2: 0.10},
        object_state_steps=object_state_steps,
    )
    obj_pos = torch.tensor([1.0, 2.0, 0.30])

    visited = []
    while not sm.done:
        visited.append(sm.state)
        sm.tick(obj_pos)

    assert visited[:8] == [
        "INIT",
        "PRE_GRASP",
        "BOTTLE_ALIGN_SIDE",
        "BOTTLE_DESCEND_MID",
        "BOTTLE_APPROACH_X",
        "BOTTLE_CLOSE",
        "BOTTLE_LIFT",
        "TRANSPORT",
    ]
    assert "ALIGN_GRIPPER" not in visited
    assert "REACH" not in visited
    assert "CLOSE" not in visited
    assert "LIFT" not in visited
```

- [ ] **Step 2: Add a bottle side-grasp target pose and quaternion test**

Append this test immediately after `test_object_two_uses_bottle_side_grasp_state_sequence()`:

```python
def test_object_two_side_grasp_targets_follow_robot_side_x_path():
    bottle_states = [
        "BOTTLE_ALIGN_SIDE",
        "BOTTLE_DESCEND_MID",
        "BOTTLE_APPROACH_X",
        "BOTTLE_CLOSE",
        "BOTTLE_LIFT",
    ]
    steps = {state: 1 for state in [*config.STATE_ORDER, *bottle_states]}
    object_state_steps = {2: {state: 1 for state in ["PRE_GRASP", *bottle_states]}}
    sm = PickPlaceStateMachine(
        [2],
        "cpu",
        steps=steps,
        object_state_steps=object_state_steps,
    )
    obj_pos = torch.tensor([1.0, 2.0, 0.30])
    moved_pos = torch.tensor([9.0, 9.0, 0.90])
    side_quat = torch.tensor(config.BOTTLE_SIDE_GRASP_QUAT_W, dtype=torch.float32)

    sm.tick(obj_pos)  # INIT
    pre_grasp_pos, _, pre_grasp_gripper = sm.tick(obj_pos)
    align_pos, align_quat, align_gripper = sm.tick(moved_pos)
    descend_pos, descend_quat, descend_gripper = sm.tick(moved_pos)
    approach_pos, approach_quat, approach_gripper = sm.tick(moved_pos)
    close_pos, close_quat, close_gripper = sm.tick(moved_pos)
    lift_pos, lift_quat, lift_gripper = sm.tick(moved_pos)

    assert torch.allclose(pre_grasp_pos, torch.tensor([1.0, 2.0, config.CARRY_Z]))
    assert pre_grasp_gripper == "open"

    assert torch.allclose(
        align_pos,
        torch.tensor([1.0 + config.BOTTLE_SIDE_APPROACH_X, 2.0, config.CARRY_Z]),
    )
    assert torch.allclose(align_quat, side_quat)
    assert align_gripper == "open"

    assert torch.allclose(
        descend_pos,
        torch.tensor([1.0 + config.BOTTLE_SIDE_APPROACH_X, 2.0, config.BOTTLE_SIDE_GRASP_Z]),
    )
    assert torch.allclose(descend_quat, side_quat)
    assert descend_gripper == "open"

    assert torch.allclose(
        approach_pos,
        torch.tensor([1.0 + config.BOTTLE_SIDE_GRASP_X_OFFSET, 2.0, config.BOTTLE_SIDE_GRASP_Z]),
    )
    assert torch.allclose(approach_quat, side_quat)
    assert approach_gripper == "open"

    assert torch.allclose(close_pos, approach_pos)
    assert torch.allclose(close_quat, side_quat)
    assert close_gripper == "close"

    assert torch.allclose(
        lift_pos,
        torch.tensor([1.0 + config.BOTTLE_SIDE_GRASP_X_OFFSET, 2.0, config.CARRY_Z]),
    )
    assert torch.allclose(lift_quat, side_quat)
    assert lift_gripper == "close"
```

- [ ] **Step 3: Run the new tests and verify they fail before production code changes**

Run:

```bash
conda run -n atec pytest \
  tests/act/test_task_e_state_machine.py::test_object_two_uses_bottle_side_grasp_state_sequence \
  tests/act/test_task_e_state_machine.py::test_object_two_side_grasp_targets_follow_robot_side_x_path \
  -v
```

Expected result before implementation:

```text
FAILED tests/act/test_task_e_state_machine.py::test_object_two_uses_bottle_side_grasp_state_sequence
FAILED tests/act/test_task_e_state_machine.py::test_object_two_side_grasp_targets_follow_robot_side_x_path
```

The failures should be due to missing bottle-specific constants/states or the state machine still using the previous object_2 route.

- [ ] **Step 4: Do not commit yet**

Leave the failing tests in the working tree. Task 2 will add the minimal implementation to make them pass.

## Task 2: Add bottle side-grasp constants

**Files:**
- Modify: `scripts/act/task_e/config.py:10-139`
- Test: `tests/act/test_task_e_state_machine.py`

- [ ] **Step 1: Export the new bottle constants**

In `scripts/act/task_e/config.py`, update the `__all__` state-machine/geometry section from:

```python
    # state machine
    "STEPS", "OPTIMIZED_STEPS", "STATE_ORDER", "OBJECT_STATE_STEPS", "OBJECT_GRASP_Z_OFFSETS",
    # geometry
    "PRE_GRASP_CLEARANCE", "GRASP_Z_OFFSET", "TOOL_CENTER_OFFSET_LOCAL",
    "CARRY_Z", "PLACE_HEIGHT", "BASKET_DROP_Z",
```

to:

```python
    # state machine
    "STEPS", "OPTIMIZED_STEPS", "STATE_ORDER", "OBJECT_STATE_STEPS", "OBJECT_GRASP_Z_OFFSETS",
    # geometry
    "PRE_GRASP_CLEARANCE", "GRASP_Z_OFFSET", "TOOL_CENTER_OFFSET_LOCAL",
    "BOTTLE_SIDE_APPROACH_X", "BOTTLE_SIDE_GRASP_X_OFFSET",
    "BOTTLE_SIDE_GRASP_Z", "BOTTLE_SIDE_GRASP_QUAT_W",
    "CARRY_Z", "PLACE_HEIGHT", "BASKET_DROP_Z",
```

- [ ] **Step 2: Add object_2 bottle state durations**

In `scripts/act/task_e/config.py`, add the object_2 entry at the start of the existing `OBJECT_STATE_STEPS` dictionary, preserving any existing commented experiments below it:

```python
OBJECT_STATE_STEPS: dict[int, dict[str, int]] = {
    2: {
        "PRE_GRASP": 40,
        "BOTTLE_ALIGN_SIDE": 40,
        "BOTTLE_DESCEND_MID": 50,
        "BOTTLE_APPROACH_X": 50,
        "BOTTLE_CLOSE": 60,
        "BOTTLE_LIFT": 70,
    },
    # 1: {
    #       #"INIT":          10,
            # "PRE_GRASP":   40,
            # "REACH":       50,
            # "CLOSE":        20,
            # "LIFT":        35,
            # "TRANSPORT":    25,
            # "PLACE":        25,
            # "OPEN":         10,
            # "LIFT_RETRACT": 10,
            # "RETRACT":      10,
    # },
    # 3:{
    #     "INIT":          10,
    #     "PRE_GRASP":   10,
    #     "REACH":       50,
    #     "CLOSE":        30,
    #     "LIFT":        50,
    #     "TRANSPORT":    20,
    #     "PLACE":        15,
    #     "OPEN":         5,
    #     "LIFT_RETRACT": 10,
    #     "RETRACT":      15,
    # }
}
```

If the surrounding commented block has drifted, preserve it and only ensure the live `2: {...}` entry appears inside the dictionary.

- [ ] **Step 3: Add side-grasp geometry constants**

In `scripts/act/task_e/config.py`, immediately after `TOOL_CENTER_OFFSET_LOCAL = [0.0, 0.0, 0.0]`, add:

```python
BOTTLE_SIDE_APPROACH_X = 0.12
BOTTLE_SIDE_GRASP_X_OFFSET = 0.035
BOTTLE_SIDE_GRASP_Z = TABLE_TOP_Z + 0.10
BOTTLE_SIDE_GRASP_QUAT_W = [0.70710678, 0.0, -0.70710678, 0.0]
```

This quaternion is the initial candidate for a side-grasp frame with local Z pointing toward world `-X` and local Y aligned with world `+Y`. Validate the physical orientation with video after unit tests pass.

- [ ] **Step 4: Do not run tests yet**

The tests still need state-machine behavior from Task 3. Continue without committing.

## Task 3: Implement object_2 bottle side-grasp state machine branch

**Files:**
- Modify: `scripts/act/task_e/state_machine.py:8-266`
- Test: `tests/act/test_task_e_state_machine.py`

- [ ] **Step 1: Import the new bottle constants**

In `scripts/act/task_e/state_machine.py`, replace the current config import block:

```python
from .config import (
    STEPS, STATE_ORDER, OBJECT_STATE_STEPS,
    CARRY_Z, PLACE_HEIGHT, BASKET_DROP_Z,
    RETRACT_POS_X, RETRACT_POS_Y,
    GRASP_Z_OFFSET, TOOL_CENTER_OFFSET_LOCAL,
    BASKET_CENTER_X, BASKET_CENTER_Y,
    DEFAULT_PLACE_QUAT_W,
)
```

with:

```python
from .config import (
    STEPS, STATE_ORDER, OBJECT_STATE_STEPS,
    CARRY_Z, PLACE_HEIGHT, BASKET_DROP_Z,
    RETRACT_POS_X, RETRACT_POS_Y,
    GRASP_Z_OFFSET, TOOL_CENTER_OFFSET_LOCAL,
    BASKET_CENTER_X, BASKET_CENTER_Y,
    DEFAULT_PLACE_QUAT_W,
    BOTTLE_SIDE_APPROACH_X, BOTTLE_SIDE_GRASP_X_OFFSET,
    BOTTLE_SIDE_GRASP_Z, BOTTLE_SIDE_GRASP_QUAT_W,
)
```

- [ ] **Step 2: Add private bottle state constants**

In `scripts/act/task_e/state_machine.py`, after the import block and before `_build_grasp_matrix()`, add:

```python
_BOTTLE_OBJECT_IDX = 2
_BOTTLE_PICK_STATES = (
    "BOTTLE_ALIGN_SIDE",
    "BOTTLE_DESCEND_MID",
    "BOTTLE_APPROACH_X",
    "BOTTLE_CLOSE",
    "BOTTLE_LIFT",
)
_CACHED_OBJECT_POS_STATES = (
    "ALIGN_GRIPPER",
    "REACH",
    "CLOSE",
    "LIFT",
    *_BOTTLE_PICK_STATES,
)
```

- [ ] **Step 3: Use the shared cached-position state tuple in `tick()`**

In `PickPlaceStateMachine.tick()`, replace:

```python
        if s in ("ALIGN_GRIPPER", "REACH", "CLOSE", "LIFT") and self._cached_obj_pos is not None:
            obj_pos = self._cached_obj_pos
```

with:

```python
        if s in _CACHED_OBJECT_POS_STATES and self._cached_obj_pos is not None:
            obj_pos = self._cached_obj_pos
```

- [ ] **Step 4: Branch object_2 state order**

In `_current_state_order()`, replace:

```python
    def _current_state_order(self) -> list[str]:
        order = list(STATE_ORDER)
        if self._current_object_idx() in self._object_state_steps and "ALIGN_GRIPPER" not in order:
            order.insert(order.index("REACH"), "ALIGN_GRIPPER")
        return order
```

with:

```python
    def _current_state_order(self) -> list[str]:
        order = list(STATE_ORDER)
        if self._current_object_idx() == _BOTTLE_OBJECT_IDX:
            reach_idx = order.index("REACH")
            transport_idx = order.index("TRANSPORT")
            return order[:reach_idx] + list(_BOTTLE_PICK_STATES) + order[transport_idx:]
        if self._current_object_idx() in self._object_state_steps and "ALIGN_GRIPPER" not in order:
            order.insert(order.index("REACH"), "ALIGN_GRIPPER")
        return order
```

- [ ] **Step 5: Add target positions for bottle side-grasp states**

In `_get_target_pos_gripper()`, insert these cases after the existing `ALIGN_GRIPPER` case and before the existing `REACH` case:

```python
        elif s == "BOTTLE_ALIGN_SIDE":
            p = obj_pos.clone(); p[0] += BOTTLE_SIDE_APPROACH_X; p[2] = CARRY_Z
            return self._jaw_center_to_gripper_base(p, ee_quat), "open"
        elif s == "BOTTLE_DESCEND_MID":
            p = obj_pos.clone(); p[0] += BOTTLE_SIDE_APPROACH_X; p[2] = BOTTLE_SIDE_GRASP_Z
            return self._jaw_center_to_gripper_base(p, ee_quat), "open"
        elif s == "BOTTLE_APPROACH_X":
            p = obj_pos.clone(); p[0] += BOTTLE_SIDE_GRASP_X_OFFSET; p[2] = BOTTLE_SIDE_GRASP_Z
            return self._jaw_center_to_gripper_base(p, ee_quat), "open"
        elif s == "BOTTLE_CLOSE":
            p = obj_pos.clone(); p[0] += BOTTLE_SIDE_GRASP_X_OFFSET; p[2] = BOTTLE_SIDE_GRASP_Z
            return self._jaw_center_to_gripper_base(p, ee_quat), "close"
        elif s == "BOTTLE_LIFT":
            p = obj_pos.clone(); p[0] += BOTTLE_SIDE_GRASP_X_OFFSET; p[2] = CARRY_Z
            return self._jaw_center_to_gripper_base(p, ee_quat), "close"
```

- [ ] **Step 6: Return the side-grasp quaternion for bottle states**

In `_get_target_quat()`, replace:

```python
    def _get_target_quat(self, s: str, d: str) -> torch.Tensor:
        default_quat = torch.tensor(DEFAULT_PLACE_QUAT_W, dtype=torch.float32, device=d)
        if s in ("ALIGN_GRIPPER", "REACH", "CLOSE", "LIFT"):
            cur_idx = self._current_object_idx()
            return self._grasp_quat_cache.get(cur_idx, default_quat)
        if s == "PRE_GRASP" and self._current_object_idx() not in self._object_state_steps:
            cur_idx = self._current_object_idx()
            return self._grasp_quat_cache.get(cur_idx, default_quat)
        return default_quat
```

with:

```python
    def _get_target_quat(self, s: str, d: str) -> torch.Tensor:
        default_quat = torch.tensor(DEFAULT_PLACE_QUAT_W, dtype=torch.float32, device=d)
        if s in _BOTTLE_PICK_STATES:
            return torch.tensor(BOTTLE_SIDE_GRASP_QUAT_W, dtype=torch.float32, device=d)
        if s in ("ALIGN_GRIPPER", "REACH", "CLOSE", "LIFT"):
            cur_idx = self._current_object_idx()
            return self._grasp_quat_cache.get(cur_idx, default_quat)
        if s == "PRE_GRASP" and self._current_object_idx() not in self._object_state_steps:
            cur_idx = self._current_object_idx()
            return self._grasp_quat_cache.get(cur_idx, default_quat)
        return default_quat
```

- [ ] **Step 7: Run the new targeted tests and verify they pass**

Run:

```bash
conda run -n atec pytest \
  tests/act/test_task_e_state_machine.py::test_object_two_uses_bottle_side_grasp_state_sequence \
  tests/act/test_task_e_state_machine.py::test_object_two_side_grasp_targets_follow_robot_side_x_path \
  -v
```

Expected result after implementation:

```text
PASSED tests/act/test_task_e_state_machine.py::test_object_two_uses_bottle_side_grasp_state_sequence
PASSED tests/act/test_task_e_state_machine.py::test_object_two_side_grasp_targets_follow_robot_side_x_path
```

- [ ] **Step 8: Commit if commits are authorized for this execution**

Only run this step if the user has explicitly authorized commits in the execution session.

```bash
git add scripts/act/task_e/config.py scripts/act/task_e/state_machine.py tests/act/test_task_e_state_machine.py
git commit -m "feat: add Task E bottle side grasp flow"
```

If commits are not authorized, skip this step and report the modified files.

## Task 4: Verify state-machine compatibility and provide operational validation command

**Files:**
- Test: `tests/act/test_task_e_state_machine.py`
- No production modifications expected in this task.

- [ ] **Step 1: Run state-machine tests except the known optimized timing baseline failure**

Run:

```bash
conda run -n atec pytest tests/act/test_task_e_state_machine.py -q \
  -k "not optimized_steps_keep_grasp_phases_conservative"
```

Expected result:

```text
passed
```

All selected tests should pass. The excluded test is the known pre-existing timing expectation mismatch documented at the top of this plan.

- [ ] **Step 2: Inspect the implementation diff**

Run:

```bash
git diff -- scripts/act/task_e/config.py scripts/act/task_e/state_machine.py tests/act/test_task_e_state_machine.py
```

Expected content:

- `config.py` exports and defines `BOTTLE_SIDE_APPROACH_X`, `BOTTLE_SIDE_GRASP_X_OFFSET`, `BOTTLE_SIDE_GRASP_Z`, and `BOTTLE_SIDE_GRASP_QUAT_W`.
- `config.py` adds live `OBJECT_STATE_STEPS[2]` bottle-state durations.
- `state_machine.py` adds object_2 bottle pick states and uses them instead of default `REACH/CLOSE/LIFT` for object_2.
- `state_machine.py` returns `BOTTLE_SIDE_GRASP_QUAT_W` for all bottle pick states.
- `test_task_e_state_machine.py` verifies object_2 state order and target poses.

- [ ] **Step 3: Report the focused video validation command**

Give the user this command for manual video validation in a fresh directory:

```bash
python scripts/act/sweep_task_e_grasp_offsets.py \
  --objects 2 \
  --offsets 0.09 \
  --attempts_per_offset 10 \
  --optimized_grasp_flow \
  --save_video \
  --video_dir datasets/atec_task_e/sweep_videos_bottle_side_grasp_obj2 \
  --tool_center_offset_local 0.00 -0.050 -0.0 \
  --output datasets/atec_task_e/grasp_offset_sweep_object2_side_grasp.json
```

Tell the user to check the first few videos for these concrete signs:

```text
1. Gripper starts on the robot/+X side of the bottle.
2. Gripper descends beside the bottle, not onto the top.
3. Gripper advances in X at bottle-body height.
4. Bottle stays upright until close.
5. Lift happens before transport to basket.
6. Final success/fail uses the official basket judgment already aligned earlier.
```

- [ ] **Step 4: If manual video shows wrong gripper orientation, tune only quaternion first**

If the user reports that the gripper is rotated incorrectly in video, change only this constant before touching approach distances or state durations:

```python
BOTTLE_SIDE_GRASP_QUAT_W = [0.70710678, 0.0, -0.70710678, 0.0]
```

Use the video observation to decide the next quaternion candidate. Keep `BOTTLE_SIDE_APPROACH_X`, `BOTTLE_SIDE_GRASP_X_OFFSET`, and `BOTTLE_SIDE_GRASP_Z` unchanged during the first orientation tuning pass.

- [ ] **Step 5: Commit verification notes if commits are authorized and Task 3 was not committed**

No separate commit is required if Task 3 already committed implementation and tests. If commits are not authorized, do not commit; report modified files and verification results.

## Self-Review Notes

- Spec coverage: The plan implements object_2-only side-grasp state order, X-axis approach from robot side, mid-height descent, X insertion, close, lift, side-grasp quaternion, tests, and video validation guidance.
- Placeholder scan: No plan step contains TBD/TODO placeholders. The initial quaternion is explicit: `[0.70710678, 0.0, -0.70710678, 0.0]`.
- Type consistency: State names are consistent across config, state-machine, and tests: `BOTTLE_ALIGN_SIDE`, `BOTTLE_DESCEND_MID`, `BOTTLE_APPROACH_X`, `BOTTLE_CLOSE`, `BOTTLE_LIFT`.
