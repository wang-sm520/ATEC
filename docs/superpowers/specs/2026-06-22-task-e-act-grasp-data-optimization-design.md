# Task E ACT Grasp Data Collection Optimization Design

Date: 2026-06-22

## Goal

Optimize Task E ACT demonstration collection so the dataset reliably contains full successful pick-and-place trajectories.

Each saved demonstration must execute the fixed object order:

1. `object_1` — box
2. `object_2` — bottle
3. `object_3` — banana

A trajectory is valid only when all three objects are placed into the basket. The final target is at least 100 saved successful demonstrations.

The collection flow must not begin the final 100-demo run until the adjusted state machine has been validated on the full three-object sequence.

## Current Baseline

The existing ACT data collection pipeline is implemented in:

- `scripts/act/collect_demos_task_e.py`
- `scripts/act/task_e/collector.py`
- `scripts/act/task_e/state_machine.py`
- `scripts/act/task_e/config.py`

The current `PickPlaceStateMachine` drives a scripted pick-and-place sequence using fixed dwell times and a global `GRASP_Z_OFFSET`. The collector converts target end-effector poses into joint-position actions via IK and writes trajectories to HDF5.

Current limitations for this goal:

- `--pick_objects 3` means “only pick object 3,” not “pick three objects.” This can accidentally collect banana-only data.
- Grasp height uses one global offset for all objects, even though the box, bottle, and banana may need different approach heights.
- The state durations are conservative and include redundant waiting.
- There is no dedicated sweep tool to choose grasp offsets from measured success rates.
- There is no explicit full-sequence validation gate before collecting the final dataset.

## Proposed Flow

Use a three-stage workflow:

1. Per-object grasp offset sweep.
2. Full-order state-machine validation.
3. Final 100-demo data collection.

Final data collection is allowed only after stage 2 confirms that the adjusted state machine can complete the full `1 → 2 → 3` placement sequence.

## Stage 1: Per-Object Grasp Offset Sweep

Add a generic sweep script:

```text
scripts/act/sweep_task_e_grasp_offsets.py
```

The script tests each object independently:

```text
for object_id in [1, 2, 3]:
    for grasp_z_offset in candidate_offsets:
        run N single-object attempts
        count attempts where that object ends in the basket
    choose the best offset for this object
```

The sweep should support:

```bash
python scripts/act/sweep_task_e_grasp_offsets.py \
    --objects 1 2 3 \
    --offsets 0.070 0.075 0.080 0.085 0.090 0.095 \
    --attempts_per_offset 5 \
    --optimized_grasp_flow \
    --headless \
    --enable_cameras \
    --output datasets/atec_task_e/grasp_offset_sweep.json
```

If time allows, use `--attempts_per_offset 10` for a more stable estimate.

### Sweep Success Rule

For a single-object sweep attempt, success means the selected object is inside the basket at the end of the episode.

The sweep should record for each object and offset:

- attempts
- successes
- success rate
- optional failure counts
- optional final object position summaries

### Best Offset Selection

For each object:

1. Select the offset with the highest success rate.
2. If tied, prefer the offset with fewer unstable failures or early terminations.
3. If still tied, choose the middle value among tied candidates to avoid extremes that may collide with the table or grasp too high.

The output JSON should contain:

```json
{
  "best_offsets": {
    "1": 0.080,
    "2": 0.080,
    "3": 0.075
  },
  "results": {
    "1": [],
    "2": [],
    "3": []
  }
}
```

The exact numeric values above are examples. Final values come from the sweep.

## Stage 2: Full-Order State-Machine Validation Gate

Before collecting the final 100 demonstrations, run a validation pass using the best offsets from stage 1.

This validation must execute the full fixed order:

```text
object_1 box → object_2 bottle → object_3 banana
```

The purpose is to confirm that the adjusted state machine can complete all three placements in one continuous episode.

### Validation Command

Add support for a validation-only or bounded-attempt mode. Example command:

```bash
python scripts/act/collect_demos_task_e.py \
    --full_order_123 \
    --num_demos 5 \
    --max_attempts 10 \
    --only_success \
    --headless \
    --enable_cameras \
    --optimized_grasp_flow \
    --grasp_offset_json datasets/atec_task_e/grasp_offset_sweep.json \
    --output_dir datasets/atec_task_e/validation_full_order
```

The validation output directory must be separate from the final dataset directory so validation data does not overwrite or mix with final training data.

### Validation Pass Criteria

The validation gate passes only if:

- At least one full episode places all three objects into the basket.
- Preferably, at least 5 successful full-order trajectories are collected within 10 attempts.
- The log shows per-object success for `object_1`, `object_2`, and `object_3`.
- No systematic failure is observed for a specific object.

If validation fails, do not start the final 100-demo collection. Instead, rerun offset sweep with a narrower range for the failing object, or relax the optimized dwell times for the failing phase.

## Stage 3: Final 100-Demo Collection

After validation passes, collect the final dataset with the same state-machine settings and per-object offsets.

Example:

```bash
python scripts/act/collect_demos_task_e.py \
    --full_order_123 \
    --num_demos 100 \
    --only_success \
    --headless \
    --enable_cameras \
    --save_images \
    --optimized_grasp_flow \
    --grasp_offset_json datasets/atec_task_e/grasp_offset_sweep.json \
    --output_dir datasets/atec_task_e/final_100
```

Each saved `traj_N` must contain the full three-object sequence in order. Failed attempts are skipped and not written to HDF5.

## CLI Changes

### `--full_order_123`

New explicit mode for this dataset goal.

Behavior:

- Forces `pick_objects = [1, 2, 3]`.
- Prevents the common mistake where `--pick_objects 3` collects only object 3.
- Requires success checking for all three objects.

### `--optimized_grasp_flow`

Uses shorter, less redundant state dwell times.

Keep the old baseline timings available when this flag is not provided.

### `--grasp_z_offsets`

Optional direct CLI override:

```bash
--grasp_z_offsets 1:0.080 2:0.080 3:0.075
```

### `--grasp_offset_json`

Loads per-object offsets from sweep output:

```bash
--grasp_offset_json datasets/atec_task_e/grasp_offset_sweep.json
```

Precedence:

1. `--grasp_z_offsets`
2. `--grasp_offset_json`
3. default object offsets
4. fallback global `GRASP_Z_OFFSET`

### `--max_attempts`

Optional safety bound for validation runs.

If `--num_demos` successful trajectories are not collected within `--max_attempts`, the command exits with a non-zero status and prints a failure summary.

This prevents validation or collection from looping indefinitely when the state machine is broken.

## State Machine Changes

### Per-Object Grasp Offset

Replace direct use of global `GRASP_Z_OFFSET` during `REACH` and `CLOSE` with object-specific lookup.

Current behavior:

```python
p[2] += GRASP_Z_OFFSET
```

Desired behavior:

```python
p[2] += self._get_grasp_z_offset()
```

The state machine should accept a mapping such as:

```python
{
    1: 0.080,
    2: 0.080,
    3: 0.075,
}
```

If the current object is missing from the mapping, use the global fallback.

### Optimized Dwell Times

Keep baseline timings as the default. Add an optimized timing table used by `--optimized_grasp_flow`.

Initial optimized timings:

```text
INIT:           60
PRE_GRASP:     120
REACH:         100
CLOSE:          60
LIFT:          120
TRANSPORT:     150
PLACE:          70
OPEN:           50
LIFT_RETRACT:   60
RETRACT:        70
```

Rationale:

- Reduce redundant movement and waiting time.
- Keep `REACH` conservative so the gripper reaches the target before closing.
- Increase `CLOSE` from the old 40 to 60 to improve grasp stability.
- Use the same optimized flow during sweep, validation, and final collection so the tuned offsets match the actual collection behavior.

If full-order validation fails, the first recovery should be to increase dwell times for the failing phase before changing unrelated behavior.

## Success Checking and Logging

Improve attempt logging for both sweep and collection.

Each attempt should print or record:

- attempt index
- requested object order
- active per-object offsets
- whether the episode ended early
- per-object basket success
- trajectory length when saved

Example success log:

```text
[INFO] Demo 12/100 attempt 17
[INFO] order=[1,2,3] offsets={1:0.080,2:0.080,3:0.075}
[INFO] success: object_1=True object_2=True object_3=True
[INFO] traj_11: 2530 steps, images saved
```

Example failure log:

```text
[WARN] Failed attempt 18
[WARN] success: object_1=False object_2=True object_3=True
[WARN] skipping (--only_success)
```

## Data Safety

The collection script currently truncates `trajectory.hdf5` in the selected output directory. To avoid losing data:

- Use a separate output directory for sweep artifacts.
- Use a separate output directory for full-order validation.
- Use a separate output directory for the final 100-demo dataset.
- Do not reuse the final output directory for validation.

Recommended paths:

```text
datasets/atec_task_e/grasp_offset_sweep.json
datasets/atec_task_e/validation_full_order/trajectory.hdf5
datasets/atec_task_e/final_100/trajectory.hdf5
```

## Post-Collection Training Flow

After final collection succeeds:

```bash
python scripts/act/filter_demos.py \
    --input datasets/atec_task_e/final_100/trajectory.hdf5 \
    --output datasets/atec_task_e/final_100/trajectory_filtered.hdf5 \
    --threshold 0.001
```

Then train ACT with the filtered dataset. The existing `baseline.sh` may need its `demo_path` updated to point to `datasets/atec_task_e/final_100/trajectory_filtered.hdf5`.

## Out of Scope

This design does not change:

- ACT model architecture.
- HDF5 training schema.
- Reward definitions.
- Task E environment assets.
- Final policy evaluation logic.

The focus is only on making scripted expert data collection faster, more reliable, and explicitly validated before collecting the 100-demo training dataset.

## Implementation Notes

The implementation adds a validation gate in the operator workflow rather than automatically starting the final collection after sweep. Operators must run the full-order validation command and confirm it collects successful `object_1 → object_2 → object_3` trajectories before running the final 100-demo command.

The final dataset should be collected into `datasets/atec_task_e/final_100/` so validation runs do not overwrite training data.
