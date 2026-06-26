# Task E Official Judgment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ACT Task E sweep/demo success labels match the official Task E basket judgment in `source/atec_rl_lab/atec_rl_lab/tasks/task_e`.

**Architecture:** Keep the existing ACT collector API, `get_objects_in_basket(env, pick_objects)`, and align its z-bound check with the official Task E reward/termination logic. The change is intentionally local to ACT collection/sweep evaluation and does not modify grasp trajectories, state-machine timing, object placement, or the official environment.

**Tech Stack:** Python, PyTorch tensors in lightweight fake env tests, pytest, IsaacLab-compatible module stubs used by the existing `tests/act` suite.

---

## File Structure

- Modify: `tests/act/test_task_e_collector_success.py`
  - Responsibility: Unit-test ACT collector basket success behavior without launching Isaac Sim, using existing fake env objects and module stubs.
  - Add official z-boundary coverage for `TABLE_TOP_Z <= obj_z <= TABLE_TOP_Z + 0.15`.
- Modify: `scripts/act/task_e/collector.py`
  - Responsibility: Demo collection and ACT/sweep success checking.
  - Change only the basket z predicate in `get_objects_in_basket()` to match official Task E logic.
- No new production files.
- No changes to `scripts/act/task_e/state_machine.py`, grasp offsets, `tool_center_offset_local`, sweep payload format, or video saving.

## Task 1: Add failing tests for official z bounds

**Files:**
- Modify: `tests/act/test_task_e_collector_success.py`
- Test: `tests/act/test_task_e_collector_success.py`

- [ ] **Step 1: Add the boundary test**

Append this test after `test_get_objects_in_basket_reports_each_requested_object()` and before `test_check_objects_in_basket_remains_all_success_wrapper()`:

```python
def test_get_objects_in_basket_matches_official_z_bounds():
    table_top = collector.TABLE_TOP_Z
    official_max_z = table_top + 0.15

    env = _fake_env({
        1: [0.40, 0.00, table_top],
        2: [0.40, 0.00, official_max_z],
        3: [0.40, 0.00, official_max_z + 0.001],
    })

    assert collector.get_objects_in_basket(env, [1, 2, 3]) == {
        1: True,
        2: True,
        3: False,
    }

    below_table_env = _fake_env({1: [0.40, 0.00, table_top - 0.001]})

    assert collector.get_objects_in_basket(below_table_env, [1]) == {1: False}
```

- [ ] **Step 2: Run the new test and verify it fails before implementation**

Run:

```bash
pytest tests/act/test_task_e_collector_success.py::test_get_objects_in_basket_matches_official_z_bounds -v
```

Expected result before implementation:

```text
FAILED tests/act/test_task_e_collector_success.py::test_get_objects_in_basket_matches_official_z_bounds
```

The failure should show at least one mismatch because current collector logic accepts `z <= TABLE_TOP_Z + 0.10` and does not enforce the official lower z bound.

- [ ] **Step 3: Do not commit yet**

Leave the failing test in the working tree. The implementation task will make it pass before any commit step.

## Task 2: Align collector z predicate with official Task E judgment

**Files:**
- Modify: `scripts/act/task_e/collector.py:64-76`
- Test: `tests/act/test_task_e_collector_success.py`

- [ ] **Step 1: Replace the collector z constants**

In `scripts/act/task_e/collector.py`, replace the existing constant:

```python
_BASKET_MAX_Z = TABLE_TOP_Z + 0.1   # object must be below this to count as inside
```

with:

```python
_BASKET_MIN_Z = TABLE_TOP_Z
_BASKET_MAX_Z = TABLE_TOP_Z + 0.15
```

- [ ] **Step 2: Replace the z predicate in `get_objects_in_basket()`**

In `scripts/act/task_e/collector.py`, replace the current `in_basket` block:

```python
        in_basket = (
            abs(pos[0].item() - BASKET_CENTER_X) <= BASKET_IN_X and
            abs(pos[1].item() - BASKET_CENTER_Y) <= BASKET_IN_Y and
            pos[2].item() <= _BASKET_MAX_Z
        )
```

with this implementation:

```python
        z = pos[2].item()
        in_basket = (
            abs(pos[0].item() - BASKET_CENTER_X) <= BASKET_IN_X and
            abs(pos[1].item() - BASKET_CENTER_Y) <= BASKET_IN_Y and
            _BASKET_MIN_Z <= z <= _BASKET_MAX_Z
        )
```

- [ ] **Step 3: Run the new targeted test and verify it passes**

Run:

```bash
pytest tests/act/test_task_e_collector_success.py::test_get_objects_in_basket_matches_official_z_bounds -v
```

Expected result after implementation:

```text
PASSED tests/act/test_task_e_collector_success.py::test_get_objects_in_basket_matches_official_z_bounds
```

- [ ] **Step 4: Run the existing collector success tests**

Run:

```bash
pytest tests/act/test_task_e_collector_success.py -v
```

Expected result:

```text
3 passed
```

If pytest reports more than 3 tests because future tests were added, all tests in this file should pass.

- [ ] **Step 5: Commit if commits are authorized for this execution**

Only run this step if the user has explicitly authorized commits in the execution session.

```bash
git add scripts/act/task_e/collector.py tests/act/test_task_e_collector_success.py
git commit -m "fix: align Task E ACT success check with official judgment"
```

Expected result:

```text
[worktree-task-e-act-grasp-data ...] fix: align Task E ACT success check with official judgment
```

If commits are not authorized, skip this step and report the modified files instead.

## Task 3: Verify broader ACT tests and explain rerun behavior

**Files:**
- Test: `tests/act/test_task_e_collector_success.py`
- Test: `tests/act/test_task_e_sweep_payload.py`
- Test: `tests/act/test_task_e_options.py`
- Test: `tests/act/test_task_e_cli_args.py`
- Test: `tests/act/test_task_e_state_machine.py`

- [ ] **Step 1: Run the ACT unit test suite**

Run:

```bash
pytest tests/act -q
```

Expected result:

```text
passed
```

The exact number of passing tests may vary as this branch evolves, but there should be no failures.

- [ ] **Step 2: Inspect git diff for only intended files**

Run:

```bash
git diff -- scripts/act/task_e/collector.py tests/act/test_task_e_collector_success.py
```

Expected content:

- `collector.py` introduces `_BASKET_MIN_Z = TABLE_TOP_Z` and `_BASKET_MAX_Z = TABLE_TOP_Z + 0.15`.
- `collector.py` changes the z predicate to `_BASKET_MIN_Z <= z <= _BASKET_MAX_Z`.
- `test_task_e_collector_success.py` adds `test_get_objects_in_basket_matches_official_z_bounds()`.

- [ ] **Step 3: Explain how to rerun sweep with clean video output**

Report this command to the user for a clean rerun, using a fresh `--video_dir` so old `success`/`fail` filenames do not cause confusion:

```bash
python scripts/act/sweep_task_e_grasp_offsets.py \
  --objects 1 \
  --offsets 0.08 \
  --attempts_per_offset 10 \
  --optimized_grasp_flow \
  --save_video \
  --video_dir datasets/atec_task_e/sweep_videos_official_judgment_obj1_offset008 \
  --tool_center_offset_local 0.00 -0.050 -0.0 \
  --output datasets/atec_task_e/grasp_offset_sweep_object1.json
```

Explain that after the code change, `success`/`fail` labels are still based on final object root position after physics stepping, not visual inspection of any intermediate video frame.

- [ ] **Step 4: Commit verification notes if commits are authorized and Task 2 was committed**

No code commit is needed for this task if Task 2 already committed the implementation and tests. If the execution session requires a final commit and Task 2's commit was skipped, ask the user for commit authorization before running any `git commit` command.

## Self-Review Notes

- Spec coverage: The plan implements the chosen minimal collector-side alignment, preserves the existing API, keeps x/y bounds unchanged, adds official lower and upper z-bound tests, and leaves trajectory/state-machine behavior out of scope.
- Placeholder scan: No implementation steps contain TBD/TODO placeholders. Code changes and commands are explicit.
- Type consistency: Tests use the existing `_fake_env()` helper and `collector.TABLE_TOP_Z`; implementation uses existing module-level scalar constants and the existing `pos[2].item()` value.
