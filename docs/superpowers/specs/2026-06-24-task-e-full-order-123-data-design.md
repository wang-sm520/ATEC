# Task E Full-Order 1-2-3 Data Collection Design

## Goal

Collect Task E demonstrations as one continuous trajectory that picks objects in order `object_1 → object_2 → object_3`, and keep only demos where all three objects end in the official basket success region.

## Success filter

Use the existing collection flags:

```bash
--full_order_123 --only_success
```

`--full_order_123` forces order `[1, 2, 3]`. `--only_success` keeps only demos where all requested objects are in the basket success region at the end.

## Tuned object parameters

Use the user's tuned values:

```python
OBJECT_STATE_STEPS = {
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

OBJECT_GRASP_Z_OFFSETS = {
    1: 0.08,
    2: 0.08,
    3: 0.08,
}

OBJECT_TOOL_CENTER_OFFSETS_LOCAL = {
    1: [0.00, -0.05, 0.0],
    2: [-0.04, -0.01, 0.0],
    3: [0.00, 0.00, 0.0],
}
```

## Required implementation change

The state machine already supports per-object step counts and per-object grasp z offsets. It currently treats `tool_center_offset_local` as one global offset for the whole trajectory. To collect one 1→2→3 trajectory with per-object tuned offsets, add per-object tool center offset support:

- Add `OBJECT_TOOL_CENTER_OFFSETS_LOCAL` to `config.py`.
- Have `PickPlaceStateMachine` default to that map.
- Keep existing `--tool_center_offset_local` as an optional global override for old one-object workflows.
- During `_jaw_center_to_gripper_base()`, select the offset for the current object.

## Collection command

After implementation, collect data with:

```bash
python scripts/act/collect_demos_task_e.py \
  --num_demos 50 \
  --output_dir datasets/atec_task_e/full_order_123 \
  --full_order_123 \
  --optimized_grasp_flow \
  --only_success \
  --max_attempts 300
```

Add `--save_video` for debugging runs.
