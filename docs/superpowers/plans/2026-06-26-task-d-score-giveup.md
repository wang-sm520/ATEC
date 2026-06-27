# Task D Score Giveup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** End Task D submissions early once the reported score is above 35 without changing the Task D environment termination threshold.

**Architecture:** Add a small pure helper in the Task D submission solution that converts `current_score` to `float` and returns whether the score is above `35.0`. `AlgSolution.predicts()` will still compute the normal action, then return that action with `giveup` set from the helper. Mirror the same helper and wiring in `demo/submit_d_6.25/solution.py` so the archived submission bundle stays consistent.

**Tech Stack:** Python, existing Task D `demo/solution.py`, pytest.

## Global Constraints

- Do not modify Task D environment termination (`x_threshold` remains `3.5`).
- Score-based giveup threshold is `current_score > 35.0`.
- Keep returning a valid action even when `giveup=True`.
- Handle non-numeric `current_score` by not giving up.
- Preserve existing Task D controller and policy-selection behavior.

---

### Task 1: Add score-threshold giveup helper and tests

**Files:**
- Modify: `tests/test_demo_task_d_lidar_climb_switch.py`
- Modify: `demo/solution.py`
- Modify: `demo/submit_d_6.25/solution.py`

**Interfaces:**
- Consumes: `current_score` passed to `AlgSolution.predicts(obs: dict, current_score: float)`.
- Produces: `_should_giveup_for_score(current_score: object) -> bool` in both solution files.

- [ ] **Step 1: Write the failing test**

Add `_should_giveup_for_score` to the import list in `tests/test_demo_task_d_lidar_climb_switch.py` and add this test:

```python
def test_solution_giveup_after_task_d_score_is_complete():
    assert not _should_giveup_for_score(35.0)
    assert _should_giveup_for_score(35.01)
    assert _should_giveup_for_score(36.0)
    assert not _should_giveup_for_score("not-a-number")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch
conda run -n atec pytest tests/test_demo_task_d_lidar_climb_switch.py::test_solution_giveup_after_task_d_score_is_complete -q
```

Expected: FAIL during import or test execution because `_should_giveup_for_score` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

In both `demo/solution.py` and `demo/submit_d_6.25/solution.py`, add this helper near the ATEC entry point:

```python
_SCORE_GIVEUP_THRESHOLD = 35.0


def _should_giveup_for_score(current_score: object) -> bool:
    try:
        return float(current_score) > _SCORE_GIVEUP_THRESHOLD
    except (TypeError, ValueError):
        return False
```

Then update `AlgSolution.predicts()` in both files to compute the normal action first and return:

```python
        return {"action": action, "giveup": _should_giveup_for_score(current_score)}
```

- [ ] **Step 4: Run targeted test to verify it passes**

Run:

```bash
cd /home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch
conda run -n atec pytest tests/test_demo_task_d_lidar_climb_switch.py::test_solution_giveup_after_task_d_score_is_complete -q
```

Expected: PASS.

- [ ] **Step 5: Run full Task D unit test file**

Run:

```bash
cd /home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch
conda run -n atec pytest tests/test_demo_task_d_lidar_climb_switch.py -q
```

Expected: all Task D tests pass.

- [ ] **Step 6: Review diff**

Run:

```bash
git -C /home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch diff -- demo/solution.py demo/submit_d_6.25/solution.py tests/test_demo_task_d_lidar_climb_switch.py docs/superpowers/plans/2026-06-26-task-d-score-giveup.md
```

Expected: only the helper, `giveup` wiring, test, and this plan changed.

- [ ] **Step 7: Commit if requested**

Only if the user asks to commit/push this follow-up change:

```bash
git -C /home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch add demo/solution.py demo/submit_d_6.25/solution.py tests/test_demo_task_d_lidar_climb_switch.py docs/superpowers/plans/2026-06-26-task-d-score-giveup.md
git -C /home/air/wang-sm/ATEC/.claude/worktrees/task-d-lidar-climb-switch commit -m "Add Task D score-based giveup"
```

## Self-Review

- Spec coverage: The plan changes only solution-level giveup, keeps environment termination unchanged, and mirrors the active and archived Task D submission files.
- Placeholder scan: No placeholders or open-ended implementation steps remain.
- Type consistency: `_should_giveup_for_score(current_score: object) -> bool` is defined before use and tested without loading policy files.
