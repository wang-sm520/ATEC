# G1 AMP Motion Data

73-dim per-frame motion clips consumed by `MotionLoaderG1` for AMP discriminator training.

Frame layout: `[joint_pos(29), joint_vel(29), ee_pos_in_pelvis(15)]` where the 5 EE bodies are
`left_ankle_roll_link`, `right_ankle_roll_link`, `waist_yaw_link`, `left_wrist_yaw_link`,
`right_wrist_yaw_link` (matches `G1_AMP_EE_BODIES` in `rough_env_cfg.py`). Joint order matches
`G1_BODY_29_JOINT_NAMES` (== GMR's mujoco G1 joint order).

## Pipeline

```
LAFAN1 BVH ─▶ GMR retargeting (gmr env) ─▶ pickle ─▶ gmr_to_amp_json.py ─▶ JSON
```

Step 1 – retarget human motion to G1 (run from `~/wsm/GMR`):
```bash
conda activate gmr
python scripts/bvh_to_robot.py \
    --bvh_file data/lafan1/walk1_subject1.bvh \
    --robot unitree_g1 \
    --save_path /tmp/walk1_subject1_g1.pkl
```
The script also opens a viewer for visual sanity-checking; it still saves the pkl when done.

Step 2 – convert pickle to our 73-dim AMP JSON (also `gmr` env – uses `mujoco` for FK):
```bash
python motion_data/gmr_to_amp_json.py \
    --pkl /tmp/walk1_subject1_g1.pkl \
    --output motion_data/g1_29dof/walk1_subject1.json
```

## Current contents

`g1_29dof/`:
- `walk1_subject1.json`, `walk1_subject2.json` — straight-line walking (~261 s @ 30 fps each)
- `walk2_subject1.json`, `walk3_subject1.json` — varied walking (~238 s, ~247 s)

Stairs / climb motions are **TODO** (placeholder only for now). To add later, retarget any
LAFAN1 stair-climb BVH (or AMASS clip) through the same pipeline and drop the JSON in
`g1_29dof/`. Then point `--amp_motion_files` at it during training.

## Verification

The motion loader has a self-test mode:
```bash
python -m atec_rl_lab.algorithms.amp.motion_loader_g1
```

Visual inspection of a single retarget can be done via GMR's viewer (Step 1 above) before
saving — keep an eye on foot sliding and arm swing.
