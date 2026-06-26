# Task E Bottle Top-Down Grasp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the object_2 bottle side-grasp experiment with a simple top-down grasp: above bottle, no special gripper rotation, descend, close, lift, and reuse basket placement.

**Architecture:** Remove object_2-specific bottle side states from the active state-machine branch. Keep `PickPlaceStateMachine` using normal `PRE_GRASP → REACH → CLOSE → LIFT` for object_2, with object_2-specific grasp z offset and timing in `config.py`.

**Tech Stack:** Python, PyTorch tensors, pytest, existing ACT state-machine tests with IsaacLab math stubs.

## Global Constraints

- Do not change object_1 Sugar box behavior.
- Do not change object_3 Banana behavior.
- Do not change official basket success judgment.
- Do not add a new CLI for this first pass.
- Known baseline: full `tests/act` includes an optimized timing assertion that may be stale; use targeted tests and the documented exclusion when needed.

---

## File Structure

- Modify: `scripts/act/task_e/config.py`
  - Remove active object_2 bottle side-state durations.
  - Remove unused side-grasp constants from active config exports/definitions.
  - Set object_2 `OBJECT_GRASP_Z_OFFSETS[2] = 0.06` as initial top-down grasp offset.
  - Add object_2 normal-state durations: `PRE_GRASP`, `REACH`, `CLOSE`, `LIFT`.
- Modify: `scripts/act/task_e/state_machine.py`
  - Remove active object_2 side-grasp state branch.
  - Remove bottle side-state quaternion handling.
  - Keep object_2 on normal state order.
  - Force object_2 pick states to default top-down quaternion.
- Modify: `tests/act/test_task_e_state_machine.py`
  - Replace bottle side-grasp tests with top-down object_2 tests.
  - Verify object_2 normal state order, default quaternion, and object_2 grasp offset.

## Task 1: Write failing tests for object_2 top-down behavior

**Files:**
- Modify: `tests/act/test_task_e_state_machine.py`
- Test: `tests/act/test_task_e_state_machine.py`

**Interfaces:**
- Consumes: `PickPlaceStateMachine`, `config.STATE_ORDER`, `config.DEFAULT_PLACE_QUAT_W`, `config.OBJECT_GRASP_Z_OFFSETS`, `config.CARRY_Z`.
- Produces: Tests that require object_2 to use normal `REACH/CLOSE/LIFT` top-down states and no bottle side states.

- [ ] **Step 1: Replace object_2 side-grasp state sequence test**

In `tests/act/test_task_e_state_machine.py`, replace `test_object_two_uses_bottle_side_grasp_state_sequence()` with:

```python
def test_object_two_uses_normal_top_down_state_sequence():
    side_states = [
        "BOTTLE_FACE_MINUS_X",
        "BOTTLE_DESCEND_HALF",
        "BOTTLE_PUSH_MINUS_X",
        "BOTTLE_CLOSE",
        "BOTTLE_LIFT",
    ]
    steps = {state: 1 for state in [*config.STATE_ORDER, *side_states]}
    object_state_steps = {2: {"PRE_GRASP": 1, "REACH": 1, "CLOSE": 1, "LIFT": 1}}
    sm = PickPlaceStateMachine(
        [2],
        "cpu",
        steps=steps,
        object_state_steps=object_state_steps,
    )
    obj_pos = torch.tensor([1.0, 2.0, 0.30])

    visited = []
    while not sm.done:
        visited.append(sm.state)
        sm.tick(obj_pos)

    assert visited[:6] == ["INIT", "PRE_GRASP", "REACH", "CLOSE", "LIFT", "TRANSPORT"]
    for state in side_states:
        assert state not in visited
```

- [ ] **Step 2: Replace object_2 side-grasp target pose test**

In `tests/act/test_task_e_state_machine.py`, replace `test_object_two_side_grasp_targets_follow_robot_side_x_path()` with:

```python
def test_object_two_top_down_reach_close_and_lift_targets():
    steps = {state: 1 for state in config.STATE_ORDER}
    object_state_steps = {2: {"PRE_GRASP": 1, "REACH": 1, "CLOSE": 1, "LIFT": 1}}
    sm = PickPlaceStateMachine(
        [2],
        "cpu",
        steps=steps,
        object_state_steps=object_state_steps,
    )
    obj_pos = torch.tensor([1.0, 2.0, 0.30])
    moved_pos = torch.tensor([9.0, 9.0, 0.90])
    default_quat = torch.tensor(config.DEFAULT_PLACE_QUAT_W, dtype=torch.float32)

    sm.tick(obj_pos)  # INIT
    pre_grasp_pos, pre_grasp_quat, pre_grasp_gripper = sm.tick(obj_pos)
    reach_pos, reach_quat, reach_gripper = sm.tick(moved_pos)
    close_pos, close_quat, close_gripper = sm.tick(moved_pos)
    lift_pos, lift_quat, lift_gripper = sm.tick(moved_pos)

    expected_grasp_z = 0.30 + config.OBJECT_GRASP_Z_OFFSETS[2]

    assert torch.allclose(pre_grasp_pos, torch.tensor([1.0, 2.0, config.CARRY_Z]))
    assert torch.allclose(pre_grasp_quat, default_quat)
    assert pre_grasp_gripper == "open"

    assert torch.allclose(reach_pos, torch.tensor([1.0, 2.0, expected_grasp_z]))
    assert torch.allclose(reach_quat, default_quat)
    assert reach_gripper == "open"

    assert torch.allclose(close_pos, torch.tensor([1.0, 2.0, expected_grasp_z]))
    assert torch.allclose(close_quat, default_quat)
    assert close_gripper == "close"

    assert torch.allclose(lift_pos, torch.tensor([1.0, 2.0, config.CARRY_Z]))
    assert torch.allclose(lift_quat, default_quat)
    assert lift_gripper == "close"
```

- [ ] **Step 3: Run the new tests and verify they fail before implementation**

Run:

```bash
conda run -n atec pytest \
  tests/act/test_task_e_state_machine.py::test_object_two_uses_normal_top_down_state_sequence \
  tests/act/test_task_e_state_machine.py::test_object_two_top_down_reach_close_and_lift_targets \
  -v
```

Expected before implementation:

```text
FAILED tests/act/test_task_e_state_machine.py::test_object_two_uses_normal_top_down_state_sequence
FAILED tests/act/test_task_e_state_machine.py::test_object_two_top_down_reach_close_and_lift_targets
```

Failures should show object_2 still using side states or missing normal top-down expectations.

## Task 2: Revert object_2 production behavior to normal top-down grasp

**Files:**
- Modify: `scripts/act/task_e/config.py`
- Modify: `scripts/act/task_e/state_machine.py`
- Test: `tests/act/test_task_e_state_machine.py`

**Interfaces:**
- Consumes: failing tests from Task 1.
- Produces: object_2 uses default state order with object-specific `REACH/CLOSE/LIFT` timings and `OBJECT_GRASP_Z_OFFSETS[2]`.

- [ ] **Step 1: Update object_2 config**

In `scripts/act/task_e/config.py`, remove active bottle side-state constants from `__all__`:

```python
    "BOTTLE_SIDE_APPROACH_X", "BOTTLE_SIDE_GRASP_X_OFFSET",
    "BOTTLE_SIDE_GRASP_Z", "BOTTLE_SIDE_GRASP_QUAT_W",
```

Then set the active `OBJECT_STATE_STEPS` object_2 entry to normal states:

```python
OBJECT_STATE_STEPS: dict[int, dict[str, int]] = {
    2: {
        "PRE_GRASP": 40,
        "REACH": 60,
        "CLOSE": 60,
        "LIFT": 70,
    },
    # keep existing commented experiments below this entry
}
```

Then set object_2 grasp offset:

```python
OBJECT_GRASP_Z_OFFSETS: dict[int, float] = {
    1: GRASP_Z_OFFSET,
    2: 0.06,
    3: GRASP_Z_OFFSET,
}
```

Remove these active constants if present:

```python
BOTTLE_SIDE_APPROACH_X = 0.12
BOTTLE_SIDE_GRASP_X_OFFSET = 0.035
BOTTLE_SIDE_GRASP_Z = TABLE_TOP_Z + 0.10
BOTTLE_SIDE_GRASP_QUAT_W = [0.9238795, 0.0, -0.3826834, 0.0]
```

- [ ] **Step 2: Remove object_2 side branch from state machine**

In `scripts/act/task_e/state_machine.py`:

Remove these imports if present:

```python
    BOTTLE_SIDE_APPROACH_X, BOTTLE_SIDE_GRASP_X_OFFSET,
    BOTTLE_SIDE_GRASP_Z, BOTTLE_SIDE_GRASP_QUAT_W,
```

Remove these module-level constants if present:

```python
_BOTTLE_OBJECT_IDX = 2
_BOTTLE_PICK_STATES = (...)
```

Set cached states back to normal pick states plus any object_1 align state:

```python
_CACHED_OBJECT_POS_STATES = (
    "ALIGN_GRIPPER",
    "REACH",
    "CLOSE",
    "LIFT",
)
```

Replace `_current_state_order()` with:

```python
    def _current_state_order(self) -> list[str]:
        order = list(STATE_ORDER)
        if self._current_object_idx() in self._object_state_steps and "ALIGN_GRIPPER" not in order:
            order.insert(order.index("REACH"), "ALIGN_GRIPPER")
        return order
```

Remove all `elif s == "BOTTLE_..."` cases from `_get_target_pos_gripper()`.

Remove this bottle quaternion branch from `_get_target_quat()`:

```python
        if s in _BOTTLE_PICK_STATES:
            return torch.tensor(BOTTLE_SIDE_GRASP_QUAT_W, dtype=torch.float32, device=d)
```

- [ ] **Step 3: Force object_2 to default top-down quaternion**

In `_get_target_quat()`, before the generic `ALIGN_GRIPPER/REACH/CLOSE/LIFT` cached-quat branch, add:

```python
        if self._current_object_idx() == 2 and s in ("PRE_GRASP", "REACH", "CLOSE", "LIFT"):
            return default_quat
```

This ensures object_2 does not use `set_grasp_quat()` computed orientation.

- [ ] **Step 4: Run targeted tests and verify pass**

Run:

```bash
conda run -n atec pytest \
  tests/act/test_task_e_state_machine.py::test_object_two_uses_normal_top_down_state_sequence \
  tests/act/test_task_e_state_machine.py::test_object_two_top_down_reach_close_and_lift_targets \
  -v
```

Expected:

```text
PASSED tests/act/test_task_e_state_machine.py::test_object_two_uses_normal_top_down_state_sequence
PASSED tests/act/test_task_e_state_machine.py::test_object_two_top_down_reach_close_and_lift_targets
```

## Task 3: Verify compatibility and provide video command

**Files:**
- Test: `tests/act/test_task_e_state_machine.py`
- No production modifications expected.

**Interfaces:**
- Consumes: top-down object_2 state-machine behavior from Task 2.
- Produces: verified local test state and a command for manual video inspection.

- [ ] **Step 1: Run ACT tests excluding known optimized timing assertion**

Run:

```bash
conda run -n atec pytest tests/act -q -k "not optimized_steps_keep_grasp_phases_conservative"
```

Expected:

```text
passed
```

- [ ] **Step 2: Run full ACT tests and record known failure if still present**

Run:

```bash
conda run -n atec pytest tests/act -q
```

Expected current branch behavior may still include:

```text
FAILED tests/act/test_task_e_state_machine.py::test_optimized_steps_keep_grasp_phases_conservative
```

If that is the only failure, report it as the known optimized timing assertion mismatch. Do not change timing policy unless the user asks.

- [ ] **Step 3: Provide short video validation command**

Report this command:

```bash
python scripts/act/sweep_task_e_grasp_offsets.py \
  --objects 2 \
  --offsets 0.06 \
  --attempts_per_offset 2 \
  --optimized_grasp_flow \
  --save_video \
  --video_dir datasets/atec_task_e/sweep_videos_bottle_topdown_check \
  --tool_center_offset_local 0.00 -0.050 -0.0 \
  --output datasets/atec_task_e/grasp_offset_sweep_object2_topdown_check.json
```

Tell the user to check:

```text
1. Gripper stays top-down and does not rotate sideways.
2. Gripper moves above bottle before descending.
3. Gripper descends enough to close on bottle.
4. Bottle lifts after close.
5. Bottle reaches basket success region.
```

## Self-Review Notes

- Spec coverage: Plan removes object_2 side states, restores normal top-down sequence, uses default quaternion, tunes object_2 grasp z offset, and provides video validation.
- Placeholder scan: No placeholders remain; all code snippets and commands are explicit.
- Type consistency: Tests and implementation use existing `PickPlaceStateMachine`, `OBJECT_STATE_STEPS`, `OBJECT_GRASP_Z_OFFSETS`, and `DEFAULT_PLACE_QUAT_W` names.
