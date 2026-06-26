# Task E Basket Placement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Release Task E objects at separate in-basket targets so `object_1`, `object_2`, and `object_3` no longer stack at the basket center during ACT data collection.

**Architecture:** Add a per-object basket XY offset map to the Task E ACT collection config, then have `PickPlaceStateMachine` resolve the current object's basket target for `TRANSPORT`, `PLACE`, `OPEN`, and `LIFT_RETRACT`. The change is local to the scripted collection state machine and preserves existing Z heights, pick order, object spawn randomization, grasp/tool offsets, HDF5 format, and success bounds.

**Tech Stack:** Python, PyTorch tensors, pytest, IsaacLab Task E config constants.

## Global Constraints

- Do not change object pick order.
- Do not change object spawn randomization.
- Do not change grasp offsets, tool-center offsets, or state dwell times.
- Do not change the HDF5 output format.
- Do not change success bounds.
- Use these exact per-object basket target offsets relative to basket center: `{1: (0.0, 0.07), 2: (0.0, 0.0), 3: (0.0, -0.07)}`.
- Use per-object basket targets for `TRANSPORT`, `PLACE`, `OPEN`, and `LIFT_RETRACT` only.
- Keep `CARRY_Z`, `BASKET_DROP_Z`, and `PLACE_HEIGHT` behavior unchanged.

---

## File Structure

- Modify `scripts/act/task_e/config.py`
  - Responsibility: Task E ACT collection constants.
  - Add exported `OBJECT_BASKET_TARGET_OFFSETS: dict[int, tuple[float, float]]`.

- Modify `scripts/act/task_e/state_machine.py`
  - Responsibility: Generate per-state end-effector targets for scripted pick-and-place.
  - Import `OBJECT_BASKET_TARGET_OFFSETS`.
  - Store the offset map in `PickPlaceStateMachine`.
  - Add helper methods for resolving the current object's basket target.
  - Replace hard-coded basket-center targets for basket states.

- Modify `tests/act/test_task_e_state_machine.py`
  - Responsibility: Pure unit tests for state-machine target generation using stubs; no simulator launch.
  - Add tests for per-object basket targets, bounds safety, and center fallback.

---

### Task 1: Add per-object basket target support to the state machine

**Files:**
- Modify: `tests/act/test_task_e_state_machine.py`
- Modify: `scripts/act/task_e/config.py`
- Modify: `scripts/act/task_e/state_machine.py`

**Interfaces:**
- Consumes:
  - Existing `PickPlaceStateMachine(object_indices: list[int], device: str, steps: Mapping[str, int] | None = None, grasp_z_offsets: Mapping[int, float] | None = None, tool_center_offset_local: list[float] | tuple[float, float, float] | None = None, object_state_steps: Mapping[int, Mapping[str, int]] | None = None, use_basket_drop_height: bool = False)`.
  - Existing constants `BASKET_CENTER_X`, `BASKET_CENTER_Y`, `BASKET_IN_X`, `BASKET_IN_Y`, `CARRY_Z`, `PLACE_HEIGHT`, and `BASKET_DROP_Z`.
- Produces:
  - `config.OBJECT_BASKET_TARGET_OFFSETS: dict[int, tuple[float, float]]`.
  - `PickPlaceStateMachine._basket_target_xy() -> tuple[float, float]`.
  - `PickPlaceStateMachine._basket_target(z: float, device: str) -> torch.Tensor` returning a `(3,)` tensor `[target_x, target_y, z]`.

- [ ] **Step 1: Add failing state-machine tests**

Append the following helper and tests to `tests/act/test_task_e_state_machine.py` after `test_baseline_place_and_open_use_original_place_height()` and before `test_object_two_uses_normal_top_down_state_sequence()`:

```python
def _collect_basket_state_targets(sm: PickPlaceStateMachine, obj_pos: torch.Tensor):
    targets = {}
    while not sm.done:
        state = sm.state
        object_key = sm.current_object_key
        ee_pos, _, _ = sm.tick(obj_pos)
        if state in ("TRANSPORT", "PLACE", "OPEN", "LIFT_RETRACT"):
            targets[(object_key, state)] = ee_pos
    return targets


def _expected_basket_target(obj_idx: int, z: float) -> torch.Tensor:
    dx, dy = config.OBJECT_BASKET_TARGET_OFFSETS[obj_idx]
    return torch.tensor([
        config.BASKET_CENTER_X + dx,
        config.BASKET_CENTER_Y + dy,
        z,
    ])


def test_state_machine_uses_per_object_basket_targets_for_drop_states():
    steps = {state: 1 for state in config.STATE_ORDER}
    sm = PickPlaceStateMachine(
        [1, 2, 3],
        "cpu",
        steps=steps,
        object_state_steps={},
        use_basket_drop_height=True,
    )

    targets = _collect_basket_state_targets(sm, torch.tensor([1.0, 2.0, 0.30]))

    for obj_idx in (1, 2, 3):
        object_key = f"object_{obj_idx}"
        assert torch.allclose(
            targets[(object_key, "TRANSPORT")],
            _expected_basket_target(obj_idx, config.CARRY_Z),
        )
        assert torch.allclose(
            targets[(object_key, "PLACE")],
            _expected_basket_target(obj_idx, config.BASKET_DROP_Z),
        )
        assert torch.allclose(
            targets[(object_key, "OPEN")],
            _expected_basket_target(obj_idx, config.BASKET_DROP_Z),
        )
        assert torch.allclose(
            targets[(object_key, "LIFT_RETRACT")],
            _expected_basket_target(obj_idx, config.CARRY_Z),
        )


def test_basket_target_offsets_stay_inside_success_bounds():
    assert config.OBJECT_BASKET_TARGET_OFFSETS == {
        1: (0.0, 0.07),
        2: (0.0, 0.0),
        3: (0.0, -0.07),
    }
    for dx, dy in config.OBJECT_BASKET_TARGET_OFFSETS.values():
        assert abs(dx) <= config.BASKET_IN_X
        assert abs(dy) <= config.BASKET_IN_Y


def test_state_machine_falls_back_to_basket_center_when_object_has_no_basket_offset():
    steps = {state: 1 for state in config.STATE_ORDER}
    sm = PickPlaceStateMachine(
        [1],
        "cpu",
        steps=steps,
        object_state_steps={},
        use_basket_drop_height=False,
    )
    sm._basket_target_offsets = {}

    while sm.state != "PLACE":
        sm.tick(torch.tensor([1.0, 2.0, 0.30]))
    place_pos, _, place_gripper = sm.tick(torch.tensor([1.0, 2.0, 0.30]))

    assert torch.allclose(
        place_pos,
        torch.tensor([config.BASKET_CENTER_X, config.BASKET_CENTER_Y, config.PLACE_HEIGHT]),
    )
    assert place_gripper == "close"
```

- [ ] **Step 2: Run the new tests to verify RED**

Run:

```bash
conda run -n atec pytest \
  tests/act/test_task_e_state_machine.py::test_state_machine_uses_per_object_basket_targets_for_drop_states \
  tests/act/test_task_e_state_machine.py::test_basket_target_offsets_stay_inside_success_bounds \
  tests/act/test_task_e_state_machine.py::test_state_machine_falls_back_to_basket_center_when_object_has_no_basket_offset \
  -v
```

Expected before implementation: FAIL because `task_e.config` does not yet define `OBJECT_BASKET_TARGET_OFFSETS` and `PickPlaceStateMachine` does not yet define `_basket_target_offsets`.

- [ ] **Step 3: Export and define basket target offsets in config**

In `scripts/act/task_e/config.py`, update the `__all__` geometry section from:

```python
    "PRE_GRASP_CLEARANCE", "GRASP_Z_OFFSET", "TOOL_CENTER_OFFSET_LOCAL",
    "OBJECT_TOOL_CENTER_OFFSETS_LOCAL",
    "CARRY_Z", "PLACE_HEIGHT", "BASKET_DROP_Z",
```

to:

```python
    "PRE_GRASP_CLEARANCE", "GRASP_Z_OFFSET", "TOOL_CENTER_OFFSET_LOCAL",
    "OBJECT_TOOL_CENTER_OFFSETS_LOCAL", "OBJECT_BASKET_TARGET_OFFSETS",
    "CARRY_Z", "PLACE_HEIGHT", "BASKET_DROP_Z",
```

Then add the new constant immediately after `OBJECT_TOOL_CENTER_OFFSETS_LOCAL`:

```python
OBJECT_BASKET_TARGET_OFFSETS: dict[int, tuple[float, float]] = {
    1: (0.0, 0.07),
    2: (0.0, 0.0),
    3: (0.0, -0.07),
}
```

- [ ] **Step 4: Import basket target offsets in the state machine**

In `scripts/act/task_e/state_machine.py`, update the config import from:

```python
    GRASP_Z_OFFSET, OBJECT_GRASP_Z_OFFSETS,
    TOOL_CENTER_OFFSET_LOCAL, OBJECT_TOOL_CENTER_OFFSETS_LOCAL,
    BASKET_CENTER_X, BASKET_CENTER_Y,
```

to:

```python
    GRASP_Z_OFFSET, OBJECT_GRASP_Z_OFFSETS,
    TOOL_CENTER_OFFSET_LOCAL, OBJECT_TOOL_CENTER_OFFSETS_LOCAL,
    OBJECT_BASKET_TARGET_OFFSETS,
    BASKET_CENTER_X, BASKET_CENTER_Y,
```

- [ ] **Step 5: Store the per-object basket target map**

In `PickPlaceStateMachine.__init__`, insert this block after `self._grasp_z_offsets.update(...)` and before the existing `if tool_center_offset_local is None:` block:

```python
        self._basket_target_offsets = {
            int(k): (float(v[0]), float(v[1]))
            for k, v in OBJECT_BASKET_TARGET_OFFSETS.items()
        }
```

The resulting section should read:

```python
        self._grasp_z_offsets = {
            int(k): float(v) for k, v in OBJECT_GRASP_Z_OFFSETS.items()
        }
        self._grasp_z_offsets.update({
            int(k): float(v) for k, v in (grasp_z_offsets or {}).items()
        })
        self._basket_target_offsets = {
            int(k): (float(v[0]), float(v[1]))
            for k, v in OBJECT_BASKET_TARGET_OFFSETS.items()
        }
        if tool_center_offset_local is None:
```

- [ ] **Step 6: Add basket target helper methods**

In `scripts/act/task_e/state_machine.py`, insert these methods after `_get_grasp_z_offset()` and before `_basket_release_z()`:

```python
    def _basket_target_xy(self) -> tuple[float, float]:
        dx, dy = self._basket_target_offsets.get(self._current_object_idx(), (0.0, 0.0))
        return BASKET_CENTER_X + dx, BASKET_CENTER_Y + dy

    def _basket_target(self, z: float, d: str) -> torch.Tensor:
        x, y = self._basket_target_xy()
        return torch.tensor([x, y, z], dtype=torch.float32, device=d)
```

- [ ] **Step 7: Use the current object's basket target in basket states**

In `_get_target_pos_gripper()`, replace:

```python
        elif s == "TRANSPORT":
            return torch.tensor([BASKET_CENTER_X, BASKET_CENTER_Y, CARRY_Z], device=d), "close"
        elif s == "PLACE":
            return torch.tensor([BASKET_CENTER_X, BASKET_CENTER_Y, self._basket_release_z()], device=d), "close"
        elif s == "OPEN":
            return torch.tensor([BASKET_CENTER_X, BASKET_CENTER_Y, self._basket_release_z()], device=d), "open"
        elif s == "LIFT_RETRACT":
            return torch.tensor([BASKET_CENTER_X, BASKET_CENTER_Y, CARRY_Z], device=d), "open"
```

with:

```python
        elif s == "TRANSPORT":
            return self._basket_target(CARRY_Z, d), "close"
        elif s == "PLACE":
            return self._basket_target(self._basket_release_z(), d), "close"
        elif s == "OPEN":
            return self._basket_target(self._basket_release_z(), d), "open"
        elif s == "LIFT_RETRACT":
            return self._basket_target(CARRY_Z, d), "open"
```

Do not change `INIT`, `PRE_GRASP`, `REACH`, `CLOSE`, `LIFT`, or `RETRACT`.

- [ ] **Step 8: Run the new tests to verify GREEN**

Run:

```bash
conda run -n atec pytest \
  tests/act/test_task_e_state_machine.py::test_state_machine_uses_per_object_basket_targets_for_drop_states \
  tests/act/test_task_e_state_machine.py::test_basket_target_offsets_stay_inside_success_bounds \
  tests/act/test_task_e_state_machine.py::test_state_machine_falls_back_to_basket_center_when_object_has_no_basket_offset \
  -v
```

Expected after implementation: all 3 tests PASS.

- [ ] **Step 9: Run existing basket-height tests**

Run:

```bash
conda run -n atec pytest \
  tests/act/test_task_e_state_machine.py::test_optimized_place_and_open_use_medium_basket_drop_height \
  tests/act/test_task_e_state_machine.py::test_baseline_place_and_open_use_original_place_height \
  -v
```

Expected after implementation: both tests PASS after updating their expected XY targets to include the configured object 2 basket offset. Because object 2's configured offset is `(0.0, 0.0)`, the existing expected positions remain `[config.BASKET_CENTER_X, config.BASKET_CENTER_Y, z]` and should not need source changes.

- [ ] **Step 10: Run ACT state-machine regression tests for the changed file**

Run:

```bash
conda run -n atec pytest tests/act/test_task_e_state_machine.py -q -k "not optimized_steps_keep_grasp_phases_conservative"
```

Expected after implementation: all selected tests PASS. If a selected test fails because it asserts old state dwell-time values that differ from current user-tuned values, do not change dwell times in this task; report that failure as pre-existing configuration/test drift and keep the basket-placement targeted tests as the evidence for this change.

- [ ] **Step 11: Run the ACT test selection from the spec**

Run:

```bash
conda run -n atec pytest tests/act -q -k "not optimized_steps_keep_grasp_phases_conservative"
```

Expected after implementation: all selected tests PASS. If an unrelated existing test fails because it asserts old state dwell-time values that differ from current user-tuned values, do not change dwell times in this task; report the exact failing test and output.

- [ ] **Step 12: Inspect the diff**

Run:

```bash
git diff -- scripts/act/task_e/config.py scripts/act/task_e/state_machine.py tests/act/test_task_e_state_machine.py
```

Expected diff contents:

- `OBJECT_BASKET_TARGET_OFFSETS` is added and exported in `config.py`.
- `state_machine.py` imports `OBJECT_BASKET_TARGET_OFFSETS`.
- `PickPlaceStateMachine` stores `_basket_target_offsets`.
- `TRANSPORT`, `PLACE`, `OPEN`, and `LIFT_RETRACT` call `_basket_target(...)`.
- Tests cover object-specific basket targets, success-bound safety, and missing-offset fallback.
- No changes appear to object pick order, object spawn randomization, grasp offsets, tool-center offsets, state dwell times, HDF5 output format, or success bounds.

- [ ] **Step 13: Commit if commit authorization is active**

Only run this step if the user has explicitly authorized commits for this implementation session. If commits are not authorized, leave the changes uncommitted and report the diff and verification output.

```bash
git add scripts/act/task_e/config.py scripts/act/task_e/state_machine.py tests/act/test_task_e_state_machine.py docs/superpowers/specs/2026-06-24-task-e-basket-placement-design.md docs/superpowers/plans/2026-06-24-task-e-basket-placement.md
git commit -m "feat: separate task e basket drop targets"
```

Expected if run: git creates one commit containing the basket-placement config, state-machine implementation, tests, spec, and plan.
