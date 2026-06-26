# Task E Terminated Success Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Treat official Task E basket-success termination as a successful collected demo instead of discarding it as early termination.

**Architecture:** Add a small classification helper in `scripts/act/task_e/collector.py` that distinguishes `continue`, `success`, and `failure` after each `env.step()`. Keep the environment's official `basket_success` termination unchanged, and make only the collection loop success-aware.

**Tech Stack:** Python, PyTorch boolean tensors, pytest, lightweight fake environment objects.

## Global Constraints

- Do not disable or modify the Task E environment's `basket_success` termination.
- Do not change basket success bounds.
- Do not change state-machine timing, placement, grasp offsets, tool-center offsets, or object spawn randomization.
- Do not change the HDF5 output format.
- Do not change the outer `--only_success` filtering; it should still validate the final returned trajectory.
- `truncated.any()` must return `"failure"`.
- `terminated.any()` with all requested objects in basket must return `"success"`.
- `terminated.any()` without all requested objects in basket must return `"failure"`.
- No termination and no truncation must return `"continue"`.

---

## File Structure

- Modify `scripts/act/task_e/collector.py`
  - Responsibility: one-episode Task E ACT demo collection and success checking.
  - Add `_classify_step_end()` near the existing basket helper functions.
  - Replace the current unconditional terminated/truncated early-failure block with helper-based control flow.

- Modify `tests/act/test_task_e_collector_success.py`
  - Responsibility: pure unit tests for basket success checks and collector step-end classification using fake env objects.
  - Add four focused tests for `_classify_step_end()`.

---

### Task 1: Classify basket-success termination as successful collection

**Files:**
- Modify: `tests/act/test_task_e_collector_success.py`
- Modify: `scripts/act/task_e/collector.py`

**Interfaces:**
- Consumes:
  - `collector.check_objects_in_basket(env: ManagerBasedRLEnv, pick_objects: list[int]) -> bool`
  - `collector.get_objects_in_basket(env: ManagerBasedRLEnv, pick_objects: list[int]) -> dict[int, bool]`
  - `terminated` and `truncated` tensors or tensor-like values that support `.any()`.
- Produces:
  - `collector._classify_step_end(env: ManagerBasedRLEnv, pick_objects: list[int], terminated, truncated) -> str`
  - Return values are exactly `"continue"`, `"success"`, or `"failure"`.

- [ ] **Step 1: Add failing tests for step-end classification**

Append these tests to `tests/act/test_task_e_collector_success.py` after `test_check_objects_in_basket_remains_all_success_wrapper()`:

```python
def _flag(value: bool) -> torch.Tensor:
    return torch.tensor([value])


def test_classify_step_end_returns_failure_for_truncation_before_success_check():
    env = _fake_env({1: [0.40, 0.00, 0.22], 2: [0.40, 0.00, 0.22], 3: [0.40, 0.00, 0.22]})

    result = collector._classify_step_end(env, [1, 2, 3], _flag(False), _flag(True))

    assert result == "failure"


def test_classify_step_end_returns_success_for_basket_success_termination():
    env = _fake_env({1: [0.40, 0.00, 0.22], 2: [0.40, 0.00, 0.22], 3: [0.40, 0.00, 0.22]})

    result = collector._classify_step_end(env, [1, 2, 3], _flag(True), _flag(False))

    assert result == "success"


def test_classify_step_end_returns_failure_for_non_success_termination():
    env = _fake_env({1: [0.40, 0.00, 0.22], 2: [0.90, 0.00, 0.22], 3: [0.40, 0.00, 0.22]})

    result = collector._classify_step_end(env, [1, 2, 3], _flag(True), _flag(False))

    assert result == "failure"


def test_classify_step_end_returns_continue_without_done_flags():
    env = _fake_env({1: [0.40, 0.00, 0.22], 2: [0.40, 0.00, 0.22], 3: [0.40, 0.00, 0.22]})

    result = collector._classify_step_end(env, [1, 2, 3], _flag(False), _flag(False))

    assert result == "continue"
```

- [ ] **Step 2: Run classification tests to verify RED**

Run:

```bash
conda run -n atec pytest \
  tests/act/test_task_e_collector_success.py::test_classify_step_end_returns_failure_for_truncation_before_success_check \
  tests/act/test_task_e_collector_success.py::test_classify_step_end_returns_success_for_basket_success_termination \
  tests/act/test_task_e_collector_success.py::test_classify_step_end_returns_failure_for_non_success_termination \
  tests/act/test_task_e_collector_success.py::test_classify_step_end_returns_continue_without_done_flags \
  -v
```

Expected before implementation: all four tests FAIL with `AttributeError: module 'task_e.collector' has no attribute '_classify_step_end'`.

- [ ] **Step 3: Add `_classify_step_end()` to collector**

In `scripts/act/task_e/collector.py`, insert this helper immediately after `check_objects_in_basket()` and before `collect_one_demo()`:

```python
def _classify_step_end(env: ManagerBasedRLEnv, pick_objects: list[int], terminated, truncated) -> str:
    """Classify environment done flags for Task E demo collection.

    Returns:
      "continue" — keep recording
      "success"  — official basket-success termination; keep recorded data
      "failure"  — timeout/truncation or non-success termination; discard demo
    """
    if truncated.any():
        return "failure"
    if terminated.any():
        if check_objects_in_basket(env, pick_objects):
            return "success"
        return "failure"
    return "continue"
```

- [ ] **Step 4: Use `_classify_step_end()` inside `collect_one_demo()`**

In `scripts/act/task_e/collector.py`, replace this block:

```python
        _, _, terminated, truncated, _ = env.step(env_action)

        if terminated.any() or truncated.any():
            print("[WARN] Episode ended early — skipping demo.")
            return None
```

with:

```python
        _, _, terminated, truncated, _ = env.step(env_action)

        step_end = _classify_step_end(env, pick_objects, terminated, truncated)
        if step_end == "success":
            print("[INFO] Episode terminated by basket success.")
            break
        if step_end == "failure":
            print("[WARN] Episode ended before basket success — skipping demo.")
            return None
```

Do not change the recording buffers or result construction after the loop.

- [ ] **Step 5: Run classification tests to verify GREEN**

Run:

```bash
conda run -n atec pytest \
  tests/act/test_task_e_collector_success.py::test_classify_step_end_returns_failure_for_truncation_before_success_check \
  tests/act/test_task_e_collector_success.py::test_classify_step_end_returns_success_for_basket_success_termination \
  tests/act/test_task_e_collector_success.py::test_classify_step_end_returns_failure_for_non_success_termination \
  tests/act/test_task_e_collector_success.py::test_classify_step_end_returns_continue_without_done_flags \
  -v
```

Expected after implementation: all four tests PASS.

- [ ] **Step 6: Run collector success test file**

Run:

```bash
conda run -n atec pytest tests/act/test_task_e_collector_success.py -v
```

Expected after implementation: all tests in the file PASS.

- [ ] **Step 7: Re-run focused basket-placement tests**

Run:

```bash
conda run -n atec pytest \
  tests/act/test_task_e_state_machine.py::test_state_machine_uses_per_object_basket_targets_for_drop_states \
  tests/act/test_task_e_state_machine.py::test_basket_target_offsets_stay_inside_success_bounds \
  tests/act/test_task_e_state_machine.py::test_state_machine_falls_back_to_basket_center_when_object_has_no_basket_offset \
  -v
```

Expected after implementation: all three tests PASS. This verifies the prior basket-placement behavior still holds.

- [ ] **Step 8: Inspect the diff**

Run:

```bash
git diff -- scripts/act/task_e/collector.py tests/act/test_task_e_collector_success.py docs/superpowers/specs/2026-06-24-task-e-terminated-success-design.md docs/superpowers/plans/2026-06-24-task-e-terminated-success.md
```

Expected diff contents:

- `collector.py` adds `_classify_step_end()`.
- `collector.py` treats `"success"` as a loop break that preserves recorded data.
- `collector.py` treats `"failure"` as `return None`.
- Tests cover truncation failure, basket-success termination success, non-success termination failure, and continue.
- No changes appear to environment termination definitions, success bounds, timing, placement, grasp offsets, tool-center offsets, spawn randomization, or HDF5 schema.

- [ ] **Step 9: Commit if commit authorization is active**

Only run this step if the user has explicitly authorized commits for this implementation session. If commits are not authorized, leave the changes uncommitted and report the diff and verification output.

```bash
git add scripts/act/task_e/collector.py tests/act/test_task_e_collector_success.py docs/superpowers/specs/2026-06-24-task-e-terminated-success-design.md docs/superpowers/plans/2026-06-24-task-e-terminated-success.md
git commit -m "fix: keep task e basket success demos"
```

Expected if run: git creates one commit containing the collector terminated-success handling, tests, spec, and plan.
