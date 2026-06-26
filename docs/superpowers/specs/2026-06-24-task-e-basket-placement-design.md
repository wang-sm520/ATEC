# Task E Basket Placement Design

## Goal

Prevent the three Task E objects from stacking at the same basket center during scripted ACT data collection. The collection order remains `object_1 → object_2 → object_3`, but each object should be released at a separate in-basket target arranged from nearest to the robot arm to farthest from the robot arm.

## Root cause

`PickPlaceStateMachine` currently sends every object to the same basket target:

```python
[BASKET_CENTER_X, BASKET_CENTER_Y, ...]
```

for `TRANSPORT`, `PLACE`, `OPEN`, and `LIFT_RETRACT`. The success checker then evaluates each object's root pose against basket XY and Z bounds. When objects stack, root poses can be pushed upward or sideways, so an object that visually landed in the basket may fail the success predicate.

## Placement convention

Use per-object XY offsets relative to the basket center. The Y axis is used because the basket lies at negative Y from the table center; larger basket Y is closer to the robot/object side, and smaller basket Y is farther away.

```python
OBJECT_BASKET_TARGET_OFFSETS = {
    1: (0.0,  0.07),  # nearest robot arm
    2: (0.0,  0.00),  # middle
    3: (0.0, -0.07),  # farthest from robot arm
}
```

These offsets stay within the current success bounds:

```python
abs(x - BASKET_CENTER_X) <= 0.20
abs(y - BASKET_CENTER_Y) <= 0.11
```

so the scripted release points should still be accepted by the official basket-region check while reducing object overlap.

## State-machine behavior

Add per-object basket target support to `PickPlaceStateMachine`:

- Store a default map from `OBJECT_BASKET_TARGET_OFFSETS`.
- Resolve the current object's basket target as `(BASKET_CENTER_X + dx, BASKET_CENTER_Y + dy)`.
- Use that target for `TRANSPORT`, `PLACE`, `OPEN`, and `LIFT_RETRACT`.
- Keep the existing Z behavior unchanged:
  - carry/retract over basket uses `CARRY_Z`
  - release uses `BASKET_DROP_Z` when `--optimized_grasp_flow` is enabled
  - release uses `PLACE_HEIGHT` otherwise

## Non-goals

- Do not change object pick order.
- Do not change object spawn randomization.
- Do not change grasp offsets, tool-center offsets, or state dwell times.
- Do not change the HDF5 output format.
- Do not change success bounds.

## Testing

Add focused state-machine tests that verify:

1. `TRANSPORT`, `PLACE`, `OPEN`, and `LIFT_RETRACT` use object-specific basket XY targets.
2. The configured offsets keep every release point inside the existing basket success bounds.
3. Existing global behavior still falls back to basket center for any object without a configured offset.

Run the ACT test selection after implementation:

```bash
conda run -n atec pytest tests/act -q -k "not optimized_steps_keep_grasp_phases_conservative"
```
