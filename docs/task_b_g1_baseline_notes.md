# Task B G1 Baseline Notes

## Run 1

Command:

```bash
PYTHONPATH=. python scripts/eval_task_b_g1.py \
  --task ATEC-TaskB-G1 \
  --num_envs 1 \
  --enable_cameras \
  --headless \
  --max_steps 6000 \
  --debug
```

Observed result:

- Score: 0.00
- Elapsed simulation time: 119.98 seconds
- Evaluator steps: 6000
- Last phase: approach_object
- Tracks detected: seen=[1, 2, 3, 4, 5, 6], active=4
- Touched tracks: []
- Placed tracks: []
- First nonzero score event: none
- Failure mode: RGB-D perception found candidate tracks and the planner repeatedly entered approach/touch/verify/search phases, but no object contact or placement reward was earned during the 6000-step baseline run.

Next tuning action:

- Inspect `outputs/task_b_g1_camera/summary.json` and `outputs/task_b_g1_objects/detections_vs_truth.json` to compare visible objects, detection positions, and nearest-object errors.
- If detections are offset from object truth, calibrate `TaskBRgbdPerception._pixel_to_robot_xy`, camera yaw assumptions, or depth filtering.
- If detections are good but no touch reward appears, tune `TaskBPlanner` standoff thresholds and `LocalObjectInteraction` left-arm touch/push poses.
- If the robot is not physically reaching objects, inspect the 6000-step evaluator log around `touch_object` and `approach_object` transitions before changing RGB thresholds.
