# Task E Object 2 Bottle Top-Down Grasp Design

## Goal

Replace the object_2 bottle side-grasp experiment with a simpler top-down grasp flow: move the gripper above the bottle, keep the normal top-down gripper orientation, descend to a tuned grasp height, close, lift, and reuse the existing basket placement states.

This supersedes `2026-06-23-task-e-bottle-side-grasp-design.md` for object_2.

## User intent

The side-grasp state machine was too complex and physically awkward. The desired bottle behavior is now:

1. Move the gripper to the bottle top/above-bottle position.
2. Do not add a special side-facing gripper rotation.
3. Descend a fixed tuned distance toward the bottle.
4. Close the gripper on the bottle.
5. Lift and place into the basket using the existing transport/place/open flow.

## State sequence

For object_2, use the normal pick-place state sequence:

```text
INIT → PRE_GRASP → REACH → CLOSE → LIFT → TRANSPORT → PLACE → OPEN → LIFT_RETRACT → RETRACT
```

Do not insert object_2-specific side-grasp states such as:

```text
BOTTLE_FACE_MINUS_X
BOTTLE_DESCEND_HALF
BOTTLE_PUSH_MINUS_X
BOTTLE_CLOSE
BOTTLE_LIFT
```

## Pose behavior

### `PRE_GRASP`

Move above the bottle at safe height:

```text
x = object_x
y = object_y
z = CARRY_Z
gripper = open
quat = default top-down orientation
```

### `REACH`

Descend toward the bottle using object_2's tuned grasp offset:

```text
x = object_x
y = object_y
z = object_z + OBJECT_GRASP_Z_OFFSETS[2]
gripper = open
quat = default top-down orientation
```

### `CLOSE`

Hold the same target pose and close:

```text
x = object_x
y = object_y
z = object_z + OBJECT_GRASP_Z_OFFSETS[2]
gripper = close
quat = default top-down orientation
```

### `LIFT`

Lift vertically after close:

```text
x = object_x
y = object_y
z = CARRY_Z
gripper = close
quat = default top-down orientation
```

Then use the existing `TRANSPORT → PLACE → OPEN` behavior.

## Initial tuning values

Set object_2-specific timings and grasp height in `scripts/act/task_e/config.py`:

```python
OBJECT_GRASP_Z_OFFSETS[2] = 0.06
OBJECT_STATE_STEPS[2] = {
    "PRE_GRASP": 40,
    "REACH": 60,
    "CLOSE": 60,
    "LIFT": 70,
}
```

`0.06` is a starting point. If the gripper closes too high, reduce it. If it collides too deep or pushes the bottle, increase it.

## Orientation rule

Object_2 should not use the computed object grasp quaternion or a bottle side-grasp quaternion. It should keep the default top-down orientation used by `DEFAULT_PLACE_QUAT_W` for `PRE_GRASP`, `REACH`, `CLOSE`, and `LIFT`.

This differs from object_1 if object_1 keeps any special cached grasp orientation logic. The priority for object_2 is simplicity and reachable IK.

## Testing strategy

Update `tests/act/test_task_e_state_machine.py` so object_2 tests assert:

1. Object_2 does not visit any bottle side-grasp states.
2. Object_2 visits the normal `PRE_GRASP → REACH → CLOSE → LIFT` sequence.
3. `REACH` and `CLOSE` use `OBJECT_GRASP_Z_OFFSETS[2]`.
4. Object_2 pick states use `DEFAULT_PLACE_QUAT_W`.
5. Existing multi-object retract behavior still works.

## Operational validation

Run a short object_2 sweep with video:

```bash
python scripts/act/sweep_task_e_grasp_offsets.py \
  --objects 2 \
  --offsets 0.06 \
  --attempts_per_offset 2 \
  --optimized_grasp_flow \
  --save_video \
  --video_dir datasets/atec_task_e/sweep_videos_bottle_topdown_check \
  --tool_center_offset_local 0.00 -0.050 -0.0 \
  --output datasets/atec_task_e/grasp_offset_sweep_object2_topdown_check.json
```

Evaluate:

- Does the gripper stay top-down without awkward side rotation?
- Does it descend to a useful bottle height?
- Does close happen around the bottle rather than above it?
- Does lift hold the bottle?
- Does the bottle reach the official basket success region?

## Out of scope

- Reintroducing side-grasp states for object_2.
- Changing object_1 or object_3 behavior.
- Changing official basket success judgment.
- Adding CLI support for sweeping object_2 top-down offset beyond existing `--offsets` / `--grasp_z_offsets` mechanisms.
