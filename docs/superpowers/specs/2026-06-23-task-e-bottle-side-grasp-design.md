# Task E Object 2 Bottle Side-Grasp Design

## Goal

Add a dedicated grasp flow for Task E `object_2` (Mustard bottle) that approaches from the robot/mechanical-arm side along world X, descends to bottle-body height, advances into the bottle, closes the gripper, lifts, and then reuses the existing basket placement flow.

This should not change the Sugar box (`object_1`) or Banana (`object_3`) grasp flows.

## Context

Task E object mapping in `source/atec_rl_lab/atec_rl_lab/tasks/task_e/env_cfg.py`:

- `object_1`: Sugar box
- `object_2`: Mustard bottle
- `object_3`: Banana

The current ACT state machine uses a mostly top-down pick sequence:

```text
INIT → PRE_GRASP → REACH → CLOSE → LIFT → TRANSPORT → PLACE → OPEN → LIFT_RETRACT → RETRACT
```

For bottle picking, the user wants a side-grasp motion from the mechanical-arm direction. In this workspace the Piper arm starts on the table `+X` side, so “from mechanical-arm direction toward the bottle” means approach from larger X toward the object, not along Y.

## Proposed object_2 sequence

For `object_2` only, replace the default pick section with a side-grasp pick section:

```text
INIT
PRE_GRASP
BOTTLE_ALIGN_SIDE
BOTTLE_DESCEND_MID
BOTTLE_APPROACH_X
BOTTLE_CLOSE
BOTTLE_LIFT
TRANSPORT
PLACE
OPEN
LIFT_RETRACT
RETRACT
```

The existing place/release states remain unchanged.

## State behavior

### `PRE_GRASP`

Keep the existing role of freezing/caching the current object position at the start of the pick. This gives the bottle sequence a stable target even if the bottle drifts during approach.

For object_2, `PRE_GRASP` can remain a safe high pose above or near the bottle. It should keep the gripper open.

### `BOTTLE_ALIGN_SIDE`

Move to a safe side-approach pose on the robot side of the bottle:

```text
x = object_x + BOTTLE_SIDE_APPROACH_X
y = object_y
z = CARRY_Z or another safe high z
```

The gripper remains open. This state should also switch to the side-grasp orientation.

### `BOTTLE_DESCEND_MID`

Hold the same x/y side-approach position and lower the gripper to bottle-body height:

```text
x = object_x + BOTTLE_SIDE_APPROACH_X
y = object_y
z = BOTTLE_SIDE_GRASP_Z
```

This separates vertical descent from horizontal insertion, reducing the chance of knocking the bottle over while descending.

### `BOTTLE_APPROACH_X`

Advance toward the bottle along world X:

```text
x = object_x + BOTTLE_SIDE_GRASP_X_OFFSET
y = object_y
z = BOTTLE_SIDE_GRASP_Z
```

The gripper remains open. `BOTTLE_SIDE_GRASP_X_OFFSET` is smaller than `BOTTLE_SIDE_APPROACH_X` and represents the final side-grasp position relative to the bottle center.

### `BOTTLE_CLOSE`

Hold the final side-grasp position and close the gripper:

```text
x = object_x + BOTTLE_SIDE_GRASP_X_OFFSET
y = object_y
z = BOTTLE_SIDE_GRASP_Z
gripper = close
```

This state should be long enough to avoid kicking the bottle away with a fast close.

### `BOTTLE_LIFT`

Lift vertically while keeping the same x/y and side-grasp orientation:

```text
x = object_x + BOTTLE_SIDE_GRASP_X_OFFSET
y = object_y
z = CARRY_Z
gripper = close
```

After this, reuse the existing `TRANSPORT → PLACE → OPEN` behavior.

## Initial tunable constants

Add object_2 side-grasp constants in `scripts/act/task_e/config.py`:

```python
BOTTLE_SIDE_APPROACH_X = 0.12
BOTTLE_SIDE_GRASP_X_OFFSET = 0.035
BOTTLE_SIDE_GRASP_Z = TABLE_TOP_Z + 0.10
BOTTLE_SIDE_GRASP_QUAT_W = [...]
```

Recommended initial state durations:

```python
OBJECT_STATE_STEPS[2] = {
    "PRE_GRASP": 40,
    "BOTTLE_ALIGN_SIDE": 40,
    "BOTTLE_DESCEND_MID": 50,
    "BOTTLE_APPROACH_X": 50,
    "BOTTLE_CLOSE": 60,
    "BOTTLE_LIFT": 70,
}
```

The numeric values are starting points for video-based tuning. `BOTTLE_SIDE_APPROACH_X` must be greater than `BOTTLE_SIDE_GRASP_X_OFFSET` so the approach starts outside the bottle and then advances inward.

## Orientation strategy

The side-grasp orientation should be a dedicated constant for object_2, not reused from the top-down `DEFAULT_PLACE_QUAT_W` or the object long-axis computed grasp quaternion.

Requirements for the side-grasp quaternion:

- The gripper approaches from the robot side (`+X`) toward the bottle (`-X`).
- The gripper jaw opening direction should span the bottle body diameter rather than push the bottle sideways.
- The gripper should remain in this orientation through `BOTTLE_ALIGN_SIDE`, `BOTTLE_DESCEND_MID`, `BOTTLE_APPROACH_X`, `BOTTLE_CLOSE`, and `BOTTLE_LIFT`.

Because the exact Piper frame convention is easy to misread, implement the quaternion as a named constant and test that object_2 states return this constant. Validate the physical orientation by saving a short sweep video. If the gripper is rotated incorrectly, tune only `BOTTLE_SIDE_GRASP_QUAT_W` first before changing the path geometry.

## State-machine integration

Update `PickPlaceStateMachine._current_state_order()` so that when the current object is `2`, the order inserts bottle-specific pick states between `PRE_GRASP` and `TRANSPORT` instead of using `REACH`, `CLOSE`, and `LIFT`.

Recommended object_2 order:

```text
INIT → PRE_GRASP → BOTTLE_ALIGN_SIDE → BOTTLE_DESCEND_MID → BOTTLE_APPROACH_X → BOTTLE_CLOSE → BOTTLE_LIFT → TRANSPORT → PLACE → OPEN → LIFT_RETRACT → RETRACT
```

The object position should remain cached for all bottle-specific pick states, just like current `ALIGN_GRIPPER`, `REACH`, `CLOSE`, and `LIFT` do.

## Testing strategy

Add unit tests in `tests/act/test_task_e_state_machine.py` to verify state-machine behavior without Isaac Sim:

1. Object 2 uses the bottle-specific state order and does not visit default `REACH`, `CLOSE`, or `LIFT` during its pick section.
2. `BOTTLE_ALIGN_SIDE` targets `[obj_x + BOTTLE_SIDE_APPROACH_X, obj_y, CARRY_Z]` with gripper open.
3. `BOTTLE_DESCEND_MID` targets `[obj_x + BOTTLE_SIDE_APPROACH_X, obj_y, BOTTLE_SIDE_GRASP_Z]` with gripper open.
4. `BOTTLE_APPROACH_X` targets `[obj_x + BOTTLE_SIDE_GRASP_X_OFFSET, obj_y, BOTTLE_SIDE_GRASP_Z]` with gripper open.
5. `BOTTLE_CLOSE` holds the final side-grasp pose and returns gripper close.
6. `BOTTLE_LIFT` keeps the final x/y and lifts to `CARRY_Z` with gripper close.
7. Bottle-specific states use `BOTTLE_SIDE_GRASP_QUAT_W`.
8. Object 1 and object 3 keep their existing state ordering, except for existing branch-local changes already present in this worktree.

## Operational validation

After implementation, run a focused object_2 sweep with video in a fresh directory, for example:

```bash
python scripts/act/sweep_task_e_grasp_offsets.py \
  --objects 2 \
  --offsets 0.09 \
  --attempts_per_offset 10 \
  --optimized_grasp_flow \
  --save_video \
  --video_dir datasets/atec_task_e/sweep_videos_bottle_side_grasp_obj2 \
  --tool_center_offset_local 0.00 -0.050 -0.0 \
  --output datasets/atec_task_e/grasp_offset_sweep_object2_side_grasp.json
```

Evaluate the first videos for:

- Did the gripper approach from the robot side along X?
- Did it descend beside the bottle rather than onto the top?
- Did it advance into the bottle at body height?
- Did the bottle stay upright until close?
- Did the object end inside the official basket success region?

## Risks and tuning guidance

### Quaternion risk

The largest risk is the side-grasp quaternion being rotated incorrectly. If the gripper looks wrong in video, tune `BOTTLE_SIDE_GRASP_QUAT_W` before changing offsets or durations.

### Pushing the bottle over

If the bottle falls during `BOTTLE_APPROACH_X`, reduce `BOTTLE_SIDE_GRASP_X_OFFSET`, increase `BOTTLE_APPROACH_X` steps, or lower `BOTTLE_SIDE_GRASP_Z` slightly so the gripper contacts the body below the unstable upper section.

### Missing the bottle laterally

If the gripper passes beside the bottle, adjust y alignment or consider a small object_2-specific local tool-center offset. Do not mix this with the first quaternion tuning pass; change one variable at a time.

### Closing too aggressively

If the bottle pops out during close, increase `BOTTLE_CLOSE` steps and verify the gripper is open before approach.

## Out of scope

- Changing object_1 Sugar box grasp flow.
- Changing object_3 Banana grasp flow.
- Changing official basket success judgment.
- Automatically optimizing bottle side-grasp constants.
- Adding a new CLI for sweeping side-grasp x/z/orientation parameters in the first implementation pass.
