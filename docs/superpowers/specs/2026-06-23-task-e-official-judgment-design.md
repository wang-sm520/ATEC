# Task E Official Basket Judgment Alignment Design

## Goal

Make the ACT Task E sweep/demo success label match the official Task E basket judgment used by `/home/air/wang-sm/ATEC/source/atec_rl_lab/atec_rl_lab/tasks/task_e`.

The immediate symptom is that sweep videos can look successful while `sweep_task_e_grasp_offsets.py` labels the attempt as `fail`. Investigation showed that the ACT collector uses a stricter basket z upper bound than the official environment logic.

## Official judgment to match

The official Task E environment checks object root positions in the per-environment local frame using these bounds:

- `abs(obj_x - BASKET_CENTER_X) <= 0.20`
- `abs(obj_y - BASKET_CENTER_Y) <= 0.11`
- `TABLE_TOP_Z <= obj_z <= TABLE_TOP_Z + 0.15`

This logic appears in both official reward and termination checks:

- `source/atec_rl_lab/atec_rl_lab/tasks/task_e/env_cfg.py`
- `source/atec_rl_lab/atec_rl_lab/tasks/task_e/mdp/rewards.py`
- `source/atec_rl_lab/atec_rl_lab/tasks/task_e/mdp/terminations.py`

## Current ACT mismatch

`get_objects_in_basket()` in `scripts/act/task_e/collector.py` currently checks:

- `abs(obj_x - BASKET_CENTER_X) <= BASKET_IN_X`
- `abs(obj_y - BASKET_CENTER_Y) <= BASKET_IN_Y`
- `obj_z <= TABLE_TOP_Z + 0.10`

The x/y bounds match the official values, but the z bound is stricter and lacks the official lower bound. This can label an object as failed when its root z is between `TABLE_TOP_Z + 0.10` and `TABLE_TOP_Z + 0.15`, even though the official Task E logic would count it as inside the basket.

## Chosen approach

Use a minimal collector-side alignment:

1. Keep `get_objects_in_basket(env, pick_objects)` as the ACT/sweep API.
2. Change its z check to match the official logic exactly for each requested object:
   - introduce `_BASKET_MIN_Z = TABLE_TOP_Z`
   - set `_BASKET_MAX_Z = TABLE_TOP_Z + 0.15`
   - require `_BASKET_MIN_Z <= obj_z <= _BASKET_MAX_Z`
3. Keep existing x/y constants `BASKET_IN_X = 0.20` and `BASKET_IN_Y = 0.11`, because they already match official `BASKET_SUCCESS_HALF_X/Y`.
4. Do not change grasp trajectories, offsets, state timings, or video generation.

This is preferred over calling `ObjectsInBasketDone` directly because sweep evaluates one requested object at a time, while the official termination class is written for all Task E objects. A direct dependency on manager termination internals would add complexity without improving correctness for this targeted alignment.

## Data flow after change

`sweep_task_e_grasp_offsets.py` will continue to run one attempt, call `collect_one_demo()`, then call:

```python
success = get_objects_in_basket(env, [obj_id])[obj_id]
```

The returned `success` will use the same x/y/z root-position bounds as the official Task E environment for that object. The saved video status suffix (`success` or `fail`) and JSON success counts will therefore match the official basket-region logic.

## Edge cases

- Object exactly on x/y/z boundaries counts as success, matching official `<=` and `>=` comparisons.
- Object below `TABLE_TOP_Z` counts as fail, matching official lower z bound.
- Object above `TABLE_TOP_Z + 0.15` counts as fail.
- This change does not guarantee visual agreement in every frame; success is still based on the final physics object root position after collection, not on the rendered video appearance.
- Existing old videos in `sweep_videos/` may still show stale `success`/`fail` filenames from previous runs. Use a fresh `--video_dir` or clean old videos when comparing results.

## Testing plan

Add or update unit tests around `get_objects_in_basket()` to assert official boundary behavior:

- z at `TABLE_TOP_Z` succeeds.
- z at `TABLE_TOP_Z + 0.15` succeeds.
- z below `TABLE_TOP_Z` fails.
- z above `TABLE_TOP_Z + 0.15` fails.
- x/y outside `BASKET_IN_X/BASKET_IN_Y` fail.

If a fake environment helper already exists in tests, reuse it. Otherwise create a minimal fake object/env structure sufficient for `get_objects_in_basket()`.

## Out of scope

- Changing the official Task E environment implementation.
- Changing basket geometry or placement.
- Changing object grasp offsets, `tool_center_offset_local`, or state-machine step durations.
- Automatically deleting old sweep videos.
