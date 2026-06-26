# Task E Terminated Success Design

## Goal

When Task E data collection triggers the official basket-success termination because all requested objects are in the basket, count that episode as one successful collected demo instead of treating it as early failure.

## Root cause

The Task E environment defines `basket_success` as a termination condition. When all three Task E objects enter the basket success region, `env.step()` can return `terminated=True`.

`collect_one_demo()` currently treats any `terminated` or `truncated` value as an early-failure signal:

```python
if terminated.any() or truncated.any():
    print("[WARN] Episode ended early — skipping demo.")
    return None
```

That means an officially successful Task E episode can be discarded before the outer collection loop runs `get_objects_in_basket()` and saves the trajectory.

## Desired behavior

After each `env.step()` inside `collect_one_demo()`:

1. If `truncated.any()` is true, treat the episode as failed and return `None`.
2. If `terminated.any()` is true and all requested objects are in the basket, treat termination as successful, stop the recording loop, and return the recorded data.
3. If `terminated.any()` is true but all requested objects are not in the basket, treat the episode as failed and return `None`.
4. If neither flag is true, continue recording.

This preserves the official environment termination semantics while making the collector success-aware.

## Implementation approach

Add a small pure control-flow helper in `scripts/act/task_e/collector.py`:

```python
def _classify_step_end(env: ManagerBasedRLEnv, pick_objects: list[int], terminated, truncated) -> str:
    if truncated.any():
        return "failure"
    if terminated.any():
        if check_objects_in_basket(env, pick_objects):
            return "success"
        return "failure"
    return "continue"
```

It returns one of:

- `"continue"`
- `"success"`
- `"failure"`

`collect_one_demo()` uses the helper immediately after `env.step(env_action)`:

- `"continue"`: keep looping
- `"success"`: print an info message and `break`
- `"failure"`: print a warning and `return None`

## Non-goals

- Do not disable or modify the Task E environment's `basket_success` termination.
- Do not change basket success bounds.
- Do not change state-machine timing, placement, grasp offsets, tool-center offsets, or object spawn randomization.
- Do not change the HDF5 output format.
- Do not change the outer `--only_success` filtering; it should still validate the final returned trajectory.

## Testing

Add unit tests for `_classify_step_end()` using lightweight fake env objects:

1. `truncated=True` returns `"failure"` even if basket success would be true.
2. `terminated=True` and all requested objects in basket returns `"success"`.
3. `terminated=True` and at least one requested object outside basket returns `"failure"`.
4. `terminated=False` and `truncated=False` returns `"continue"`.

Run targeted tests:

```bash
conda run -n atec pytest tests/act/test_task_e_collector_success.py -v
```

Run the focused basket placement tests after implementation to make sure the previous change still holds:

```bash
conda run -n atec pytest \
  tests/act/test_task_e_state_machine.py::test_state_machine_uses_per_object_basket_targets_for_drop_states \
  tests/act/test_task_e_state_machine.py::test_basket_target_offsets_stay_inside_success_bounds \
  tests/act/test_task_e_state_machine.py::test_state_machine_falls_back_to_basket_center_when_object_has_no_basket_offset \
  -v
```
