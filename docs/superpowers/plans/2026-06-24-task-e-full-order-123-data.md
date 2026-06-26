# Task E Full-Order 1-2-3 Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure Task E ACT demo collection for one continuous `object_1 → object_2 → object_3` trajectory with per-object tuned state steps, grasp z offsets, and tool center offsets.

**Architecture:** Add per-object tool-center-offset support to `PickPlaceStateMachine` while preserving the existing global `--tool_center_offset_local` override. Store the tuned full-order values in `scripts/act/task_e/config.py`, then verify with unit tests and use existing `--full_order_123 --only_success` for filtering successful demos.

**Tech Stack:** Python, PyTorch tensors, pytest.

## Global Constraints

- Keep existing `--tool_center_offset_local` behavior as a global override for old workflows.
- Use official basket success filtering via existing `--only_success`.
- Force collection order using existing `--full_order_123`.
- Do not change object randomization or output file format.

---

## Task 1: Add tests for per-object tool offsets and full-order config

**Files:**
- Modify: `tests/act/test_task_e_state_machine.py`

**Interfaces:**
- Consumes: `PickPlaceStateMachine`, `config.OBJECT_TOOL_CENTER_OFFSETS_LOCAL`, `config.OBJECT_STATE_STEPS`, `config.OBJECT_GRASP_Z_OFFSETS`.
- Produces: Failing tests that require per-object tool offsets and tuned full-order configuration.

- [ ] **Step 1: Add config assertion test**

Append to `tests/act/test_task_e_state_machine.py` after `test_optimized_steps_keep_grasp_phases_conservative()`:

```python
def test_full_order_123_tuned_object_config():
    assert config.OBJECT_STATE_STEPS[1] == {
        "PRE_GRASP": 40,
        "REACH": 50,
        "CLOSE": 20,
        "LIFT": 35,
        "TRANSPORT": 25,
        "PLACE": 25,
        "OPEN": 10,
        "LIFT_RETRACT": 10,
        "RETRACT": 10,
    }
    assert config.OBJECT_STATE_STEPS[2] == {
        "PRE_GRASP": 10,
        "REACH": 40,
        "CLOSE": 20,
        "LIFT": 80,
    }
    assert config.OBJECT_STATE_STEPS[3] == {
        "INIT": 10,
        "PRE_GRASP": 10,
        "REACH": 50,
        "CLOSE": 30,
        "LIFT": 50,
        "TRANSPORT": 20,
        "PLACE": 15,
        "OPEN": 5,
        "LIFT_RETRACT": 10,
        "RETRACT": 15,
    }
    assert config.OBJECT_GRASP_Z_OFFSETS == {1: 0.08, 2: 0.08, 3: 0.08}
    assert config.OBJECT_TOOL_CENTER_OFFSETS_LOCAL == {
        1: [0.0, -0.05, 0.0],
        2: [-0.04, -0.01, 0.0],
        3: [0.0, 0.0, 0.0],
    }
```

- [ ] **Step 2: Add per-object tool offset behavior test**

Append after the config assertion test:

```python
def test_state_machine_uses_per_object_tool_center_offsets():
    steps = {state: 1 for state in config.STATE_ORDER}
    sm = PickPlaceStateMachine([1, 2, 3], "cpu", steps=steps)

    obj_pos = torch.tensor([1.0, 2.0, 0.30])

    sm.tick(obj_pos)  # object_1 INIT
    object_1_pre_grasp, _, _ = sm.tick(obj_pos)
    while sm.current_object_key == "object_1" and not sm.done:
        sm.tick(obj_pos)

    object_2_pre_grasp, _, _ = sm.tick(obj_pos)
    while sm.current_object_key == "object_2" and not sm.done:
        sm.tick(obj_pos)

    object_3_pre_grasp, _, _ = sm.tick(obj_pos)

    assert torch.allclose(object_1_pre_grasp, torch.tensor([1.0, 2.05, config.CARRY_Z]))
    assert torch.allclose(object_2_pre_grasp, torch.tensor([1.04, 2.01, config.CARRY_Z]))
    assert torch.allclose(object_3_pre_grasp, torch.tensor([1.0, 2.0, config.CARRY_Z]))
```

- [ ] **Step 3: Run tests and verify failure before implementation**

Run:

```bash
conda run -n atec pytest \
  tests/act/test_task_e_state_machine.py::test_full_order_123_tuned_object_config \
  tests/act/test_task_e_state_machine.py::test_state_machine_uses_per_object_tool_center_offsets \
  -v
```

Expected: both tests fail because the object_1/object_3 config is still commented out and per-object tool offsets do not exist.

## Task 2: Implement full-order tuned config and per-object tool offsets

**Files:**
- Modify: `scripts/act/task_e/config.py`
- Modify: `scripts/act/task_e/state_machine.py`
- Test: `tests/act/test_task_e_state_machine.py`

**Interfaces:**
- Consumes: failing tests from Task 1.
- Produces: `OBJECT_TOOL_CENTER_OFFSETS_LOCAL: dict[int, list[float]]`; `PickPlaceStateMachine` uses per-object offsets unless global override supplied.

- [ ] **Step 1: Add config export and tuned values**

In `scripts/act/task_e/config.py`, add `"OBJECT_TOOL_CENTER_OFFSETS_LOCAL"` to `__all__` geometry exports after `"TOOL_CENTER_OFFSET_LOCAL"`.

Set:

```python
OBJECT_STATE_STEPS: dict[int, dict[str, int]] = {
    1: {
        "PRE_GRASP": 40,
        "REACH": 50,
        "CLOSE": 20,
        "LIFT": 35,
        "TRANSPORT": 25,
        "PLACE": 25,
        "OPEN": 10,
        "LIFT_RETRACT": 10,
        "RETRACT": 10,
    },
    2: {
        "PRE_GRASP": 10,
        "REACH": 40,
        "CLOSE": 20,
        "LIFT": 80,
    },
    3: {
        "INIT": 10,
        "PRE_GRASP": 10,
        "REACH": 50,
        "CLOSE": 30,
        "LIFT": 50,
        "TRANSPORT": 20,
        "PLACE": 15,
        "OPEN": 5,
        "LIFT_RETRACT": 10,
        "RETRACT": 15,
    },
}

OBJECT_GRASP_Z_OFFSETS: dict[int, float] = {
    1: 0.08,
    2: 0.08,
    3: 0.08,
}
TOOL_CENTER_OFFSET_LOCAL = [0.0, 0.0, 0.0]
OBJECT_TOOL_CENTER_OFFSETS_LOCAL: dict[int, list[float]] = {
    1: [0.0, -0.05, 0.0],
    2: [-0.04, -0.01, 0.0],
    3: [0.0, 0.0, 0.0],
}
```

- [ ] **Step 2: Import object tool offsets in state machine**

In `scripts/act/task_e/state_machine.py`, import `OBJECT_TOOL_CENTER_OFFSETS_LOCAL` next to `TOOL_CENTER_OFFSET_LOCAL`.

- [ ] **Step 3: Store per-object tool offsets**

In `PickPlaceStateMachine.__init__`, replace the single tensor assignment with:

```python
        if tool_center_offset_local is None:
            self._tool_center_offsets_local = {
                int(k): torch.tensor(v, dtype=torch.float32, device=device)
                for k, v in OBJECT_TOOL_CENTER_OFFSETS_LOCAL.items()
            }
            self._tool_center_offset_local = torch.tensor(
                TOOL_CENTER_OFFSET_LOCAL,
                dtype=torch.float32,
                device=device,
            )
        else:
            global_offset = torch.tensor(tool_center_offset_local, dtype=torch.float32, device=device)
            self._tool_center_offsets_local = {}
            self._tool_center_offset_local = global_offset
```

- [ ] **Step 4: Use current object offset in `_jaw_center_to_gripper_base()`**

Replace:

```python
        world_offset = quat_apply(
            ee_quat.unsqueeze(0),
            self._tool_center_offset_local.unsqueeze(0),
        ).squeeze(0)
```

with:

```python
        local_offset = self._tool_center_offsets_local.get(
            self._current_object_idx(),
            self._tool_center_offset_local,
        )
        world_offset = quat_apply(
            ee_quat.unsqueeze(0),
            local_offset.unsqueeze(0),
        ).squeeze(0)
```

- [ ] **Step 5: Run targeted tests and verify pass**

Run:

```bash
conda run -n atec pytest \
  tests/act/test_task_e_state_machine.py::test_full_order_123_tuned_object_config \
  tests/act/test_task_e_state_machine.py::test_state_machine_uses_per_object_tool_center_offsets \
  -v
```

Expected: both pass.

## Task 3: Verify and provide collection command

**Files:**
- Test: `tests/act/test_task_e_state_machine.py`

**Interfaces:**
- Consumes: Task 2 implementation.
- Produces: verified command for collection.

- [ ] **Step 1: Run ACT tests excluding known timing assertion**

Run:

```bash
conda run -n atec pytest tests/act -q -k "not optimized_steps_keep_grasp_phases_conservative"
```

Expected: all selected tests pass.

- [ ] **Step 2: Provide collection command**

Report:

```bash
python scripts/act/collect_demos_task_e.py \
  --num_demos 50 \
  --output_dir datasets/atec_task_e/full_order_123 \
  --full_order_123 \
  --optimized_grasp_flow \
  --only_success \
  --max_attempts 300
```

For debugging with videos:

```bash
python scripts/act/collect_demos_task_e.py \
  --num_demos 5 \
  --output_dir datasets/atec_task_e/full_order_123_debug \
  --full_order_123 \
  --optimized_grasp_flow \
  --only_success \
  --save_video \
  --max_attempts 100
```
