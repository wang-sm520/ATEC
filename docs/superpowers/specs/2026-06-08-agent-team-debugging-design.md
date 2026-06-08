# Long-Running ATEC Agent Team Debugging Design

日期：2026-06-08

## Goal

Create a repeatable agent-team workflow for long-running ATEC competition debugging, starting with the approved Task B G1 visual contact baseline and later extending to Task D integration and score tuning.

## Team Roles

- **Coordinator**: Maintains the worktree, task queue, commits, quality gates, and final reporting. The coordinator does not make large implementation edits unless unblocking small mechanical issues.
- **Implementer**: Executes exactly one plan task at a time using TDD. It writes tests first, implements the minimal code, runs verification, commits, and reports `DONE`, `DONE_WITH_CONCERNS`, `NEEDS_CONTEXT`, or `BLOCKED`.
- **Spec Reviewer**: Checks the task implementation against the approved plan and design. It focuses on under-build, over-build, missing files, wrong interfaces, and skipped verification.
- **Code Quality Reviewer**: Runs after spec review passes. It checks correctness risks, maintainability, simplification, edge cases, and test quality.
- **Experiment/Probe Agent**: Runs Isaac probes/evaluations when the code is statically stable. It records runtime observations, sensor keys, scores, failure modes, and tuning suggestions.
- **Score Analyst**: Converts run logs into prioritized next actions: detection tuning, navigation tuning, interaction tuning, or packaging/submission work.

## Execution Protocol

1. Work only inside the isolated branch `worktree-task-b-g1-agent-team`.
2. Execute the approved Task B plan task-by-task from `docs/superpowers/plans/2026-06-08-task-b-g1-implementation.md`.
3. For each task:
   - Dispatch a fresh implementer subagent.
   - Require tests or syntax checks appropriate to the task.
   - Require a commit for completed implementation tasks.
   - Dispatch a spec reviewer.
   - If spec review finds issues, return to an implementer/fix agent and re-review.
   - Dispatch a code quality reviewer only after spec review passes.
   - If quality review finds issues, fix and re-review.
4. Do not dispatch multiple implementers that edit overlapping files in parallel.
5. Isaac runtime probes start only after non-Isaac tests and syntax checks pass.
6. If local environment lacks `pytest`, non-Isaac `unittest` tests may be run with `python -m unittest` as an equivalent verification path, and the limitation must be reported.

## Long-Term Cadence

- **Build phase**: Complete plan Tasks 1-10 to create the Task B baseline code and tools.
- **Verification phase**: Run Task 11 static checks, camera probe, object probe, and short evaluator smoke tests.
- **Experiment phase**: Run Task 12 longer evaluation and record notes in `docs/task_b_g1_baseline_notes.md`.
- **Tuning loop**: Each loop records score, phase traces, detection counts, failure mode, and next tuning action. Tune one subsystem per loop.
- **Submission phase**: Once baseline score is useful, create a separate packaging plan to copy/switch `solution_task_b_g1.py` into the platform-required `demo/solution.py` without losing Task D work.

## Quality Gates

- A task is not complete until implementation, spec review, and code quality review all pass.
- A run is not called successful without command output or saved artifacts proving it.
- Runtime failures are recorded as findings, not hidden.
- All score/tuning notes must distinguish contact score, placement score, and failure mode when observable.
