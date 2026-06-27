# Task A G1 Corridor Training Design

Date: 2026-06-16

## Goal

Improve Unitree G1 performance on Task A evaluation, especially:

1. walking straight instead of drifting laterally;
2. climbing and descending stairs reliably;
3. preserving compatibility with the existing G1 AMP 79-dimensional motion format.

The current observed behavior is that the robot can walk, but does not walk straight and cannot yet handle stairs robustly. The user has an existing exported policy at `/demo/policy_a.pt` and stair motion source data at:

- `/home/air/wsm/GMR/data/downstairs01_stageii.npz`
- `/home/air/wsm/GMR/data/upstairs01_stageii.npz`

The GMR root is:

- `/home/air/wsm/GMR`

## Current project context

Task A evaluation terrain is a single 300 m long corridor:

- `source/atec_rl_lab/atec_rl_lab/tasks/task_a/terrain.py`
- tile size: `20 m x 20 m`
- `num_rows = 15`
- `num_cols = 1`
- sequence:
  - `flat`, `flat`
  - `random_rough` x4
  - `hf_pyramid_slope`, `hf_pyramid_slope_inv`, `hf_pyramid_slope`, `hf_pyramid_slope_inv`
  - `pyramid_stairs`, `pyramid_stairs_inv`, `pyramid_stairs`, `pyramid_stairs_inv`
  - `flat`

Task A evaluation G1 reset starts near:

- `x = -141`
- `y = 0`
- `z = 0.8`

The existing G1 AMP Task A training environment uses a broad random terrain grid rather than multiple evaluation-like corridors:

- `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/task_a_env_cfg.py`
- current training grid defaults to `10 x 20`
- active terrain mix currently uses flat, random rough, and inverted stairs only
- it does not fully reproduce the ordered Task A evaluation corridor

The existing G1 AMP observation/data format is 79-dimensional:

```text
[0:6]    root_lin_vel_b(3), root_ang_vel_b(3)
[6:35]   joint_pos, 29 DoF, raw joint position
[35:64]  joint_vel, 29 DoF
[64:79]  ee_pos_b, 5 end-effectors x 3
```

This layout is defined consistently in:

- `source/atec_rl_lab/atec_rl_lab/algorithms/amp/motion_loader_g1.py`
- `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/observations.py`
- `motion_data/gmr_to_amp_json.py`

## Design decision

Add a new Task A-specific G1 AMP training environment instead of modifying the existing broad Task A training environment in place.

Proposed new environment id:

```text
ATEC-Isaac-AMP-Unitree-G1-TaskA-Corridor-v0
```

This environment is specialized for Task A evaluation performance. It should train from scratch with straight-walking and centerline-following incentives, while preserving limited lateral/yaw recovery ability.

The existing environment should remain available:

```text
ATEC-Isaac-AMP-Unitree-G1-TaskA-v0
```

This keeps broad locomotion experiments and Task A corridor specialization separate.

## Corridor terrain design

The new environment should create multiple parallel Task A corridors:

```text
num_rows = 15
num_cols = N
size = (20.0, 20.0)
```

Each column should be one full 300 m Task A-style corridor. Every corridor should contain the same 15-tile ordered sequence used by the evaluation environment:

```text
flat
flat
random_rough
random_rough
random_rough
random_rough
hf_pyramid_slope
hf_pyramid_slope_inv
hf_pyramid_slope
hf_pyramid_slope_inv
pyramid_stairs
pyramid_stairs_inv
pyramid_stairs
pyramid_stairs_inv
flat
```

Initial recommended `num_cols` is 16 or 20, depending on memory and desired parallelism.

### Terrain sequence implementation requirement

Do not blindly reuse the 15-element evaluation `terrain_sequence` for a `15 x N` grid. First verify the cell traversal order in `BetterTerrainGenerator._get_terrain_mesh`.

If traversal is row-major:

```text
for row in rows:
    for col in cols:
```

then the expanded sequence should repeat the same row terrain across all columns:

```text
seq[0] repeated N times,
seq[1] repeated N times,
...
seq[14] repeated N times
```

If traversal is column-major:

```text
for col in cols:
    for row in rows:
```

then the expanded sequence should be:

```text
seq repeated N times
```

The implementation must ensure each column is a complete Task A corridor, not a row-wise scrambled set of terrain types.

## Reset design

Reset should place each robot near the start of its assigned corridor, not at the geometric center of the full 300 m corridor.

Recommended reset behavior:

```text
x: start of assigned corridor, with small perturbation
y: assigned corridor centerline, with small perturbation
yaw: near +x, with small perturbation
```

Suggested initial yaw range:

```text
yaw = (-0.10, 0.10)
```

A later curriculum may widen reset yaw:

```text
initial: (-0.05, 0.05)
middle:  (-0.10, 0.10)
late:    (-0.20, 0.20)
```

This preserves recovery ability without turning Task A training into a generic all-heading locomotion task.

If the terrain importer only provides per-tile origins rather than per-corridor start origins, implement a custom reset assignment or origin mapping so environments spawn at the start region of each corridor.

## Command design

The corridor environment should be straight-walking focused, but should not completely remove lateral and yaw control. The policy needs limited recovery behavior when it starts slightly off-axis or drifts during locomotion.

Recommended command ranges for the first version:

```text
lin_vel_x:  (0.6, 1.6)
lin_vel_y:  (-0.10, 0.10)
ang_vel_z:  (-0.25, 0.25)
heading:    (-0.10, 0.10)
```

This replaces the broader existing ranges such as:

```text
lin_vel_x:  (-0.6, 2.0)
lin_vel_y:  (-0.5, 0.5)
ang_vel_z:  (-1.57, 1.57)
heading:    (-0.3, 0.3)
```

Do not set `lin_vel_y`, `ang_vel_z`, and `heading` to exactly zero in the first version. That would train a narrow straight-line policy but reduce its ability to correct drift or recover before stairs.

## Centerline reward design

Add a Task A corridor-specific reward term that penalizes lateral displacement from the current corridor centerline.

Recommended function name:

```text
corridor_centerline_l1
```

Recommended location:

```text
source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/rewards.py
```

Core logic:

```text
robot_y = robot root world y
center_y = assigned corridor centerline y
y_error = robot_y - center_y
penalty = clamp(abs(y_error), max=y_clip)
```

The reward term should return a positive penalty value and use a negative reward weight, or return a negative reward directly following the surrounding code style.

Important: in a multi-corridor terrain, the centerline is not global `y = 0`. The reward must use per-environment centerline information, such as `env.scene.env_origins[:, 1]` if that corresponds to the corridor center. If env origins do not correspond to corridor centers, add an explicit per-env corridor center mapping.

Initial parameter recommendation:

```text
y_clip = 2.0 to 4.0 m
weight = -0.5 to -2.0
```

Use clipped absolute error before trying squared error. Squared error can dominate forward progress when the robot is already far from the centerline.

Keep existing lateral velocity penalty, but avoid making it so strong that it prevents corrective side motion:

```text
lin_vel_y_l2.weight = -0.5 initially, possibly -1.0 after comparison
```

## Stair motion data design

The provided stair data is source data, not yet verified as AMP-ready JSON:

```text
/home/air/wsm/GMR/data/downstairs01_stageii.npz
/home/air/wsm/GMR/data/upstairs01_stageii.npz
```

The target outputs should be:

```text
motion_data/g1_29dof/downstairs01_stageii.json
motion_data/g1_29dof/upstairs01_stageii.json
```

The intended pipeline is:

```text
stageii npz / GMR source data
-> GMR retargeted Unitree G1 29DoF pkl
-> motion_data/gmr_to_amp_json.py
-> 79-dimensional AMP JSON
-> train_amp.py --amp_motion_files ...
```

Use:

```text
--gmr_root /home/air/wsm/GMR
```

when running `motion_data/gmr_to_amp_json.py`.

Intermediate retargeted pkl outputs may be stored under a temporary or ignored directory such as:

```text
motion_data/g1_29dof/retargeted/
```

Large intermediate pkl files should not be committed unless the project explicitly wants to version them.

## AMP dimension constraints

The G1 AMP data dimension is strictly 79. This is a hard requirement.

The new stair JSON files must have:

```text
Frames.shape == (N, 79)
```

The exact layout must be:

```text
[0:6]    root velocity in pelvis/body frame
[6:35]   29 raw joint positions
[35:64]  29 joint velocities
[64:79]  15 end-effector positions in pelvis/body frame
```

Do not use or mix the older 73-dimensional format.

Training currently defaults to `--amp_obs_dim 79` in `scripts/rsl_rl/train_amp.py`, and `AMPPPO` checks that `MotionLoaderG1.observation_dim` matches `amp_obs_dim`.

However, `scripts/rsl_rl/play_amp.py` contains a fallback default of 73. This should be changed to 79, or replaced with an environment-derived AMP observation dimension, to avoid confusion when constructing AMPPPO during playback/export.

The motion loader currently accepts files with width greater than 79 and slices to 79. For this project, validation should be stricter: generated G1 AMP files should be exactly 79 wide, not merely at least 79 wide.

## Checkpoint handling

`/demo/policy_a.pt` is treated as an exported TorchScript actor policy for inference/submission. It should not be assumed to be a resumable PPO/AMP training checkpoint.

For resume or fine-tuning, prefer original training checkpoints under logs, e.g.:

```text
logs/.../model_*.pt
```

If no original checkpoint exists, using `/demo/policy_a.pt` would require a separate distillation or behavior-cloning initialization design. That is out of scope for the first implementation.

## Files to add or modify

Likely additions/modifications:

1. Add corridor env config:

   ```text
   source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/task_a_corridor_env_cfg.py
   ```

2. Register new task id:

   ```text
   source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/__init__.py
   ```

3. Add centerline reward function:

   ```text
   source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/rewards.py
   ```

4. Add or update AMP motion validation script:

   ```text
   motion_data/validate_amp_motion_g1.py
   ```

   or:

   ```text
   scripts/rsl_rl/validate_amp_motion_g1.py
   ```

5. Fix playback/export AMP dim fallback:

   ```text
   scripts/rsl_rl/play_amp.py
   ```

6. Generate stair motion JSON files after confirming the GMR retargeting path:

   ```text
   motion_data/g1_29dof/upstairs01_stageii.json
   motion_data/g1_29dof/downstairs01_stageii.json
   ```

## Validation plan

### 1. Motion data validation

Before training, validate every motion JSON used for AMP training:

```text
Frames.ndim == 2
Frames.shape[1] == 79
FrameDuration > 0
MotionWeight > 0
no NaN
no Inf
```

Then instantiate `MotionLoaderG1` and sample expert batches:

```text
loader.observation_dim == 79
s.shape == (batch, 79)
s_next.shape == (batch, 79)
```

### 2. Terrain validation

Verify the new corridor terrain:

```text
num_rows == 15
num_cols == N
each column follows the full Task A sequence
robot reset is near the start of a corridor
robot yaw is near +x
centerline reward uses the assigned corridor center, not global y=0
```

### 3. Training smoke test

Run a short training job:

```bash
python scripts/rsl_rl/train_amp.py \
  --task ATEC-Isaac-AMP-Unitree-G1-TaskA-Corridor-v0 \
  --num_envs 64 \
  --max_iterations 5 \
  --amp_motion_files motion_data/g1_29dof/*.json \
  --amp_obs_dim 79
```

Expected results:

```text
env creates successfully
terrain generates successfully
MotionLoaderG1 loads all motion files
AMP is enabled, not silently skipped
no 73/79 dimension mismatch
rollout and update complete for several iterations
```

### 4. Policy evaluation

Compare at least these policies:

1. existing `/demo/policy_a.pt` baseline;
2. new from-scratch corridor-trained policy;
3. optional fine-tuned policy if a resumable checkpoint exists.

Evaluate on the real Task A G1 environment:

```text
ATEC-TaskA-G1
```

Track straightness metrics:

```text
mean_abs_y_error
max_abs_y_error
final_abs_y_error
yaw_error_to_x
lin_vel_y_rms
distance_along_x
reach_goal_rate
```

Track stair metrics:

```text
stairs_entry_success_rate
stairs_exit_success_rate
fall_rate_on_stairs
mean_speed_on_stairs
stair_segment_progress
```

The key combined metric is forward progress divided by or compared against lateral drift. A policy that stays near the centerline by not moving forward should not be considered successful.

## Success criteria

Stage 1: straight walking improves.

```text
mean_abs_y_error decreases
final_abs_y_error decreases
forward progress does not regress
robot reaches the stair section more consistently
```

Stage 2: stairs improve.

```text
robot enters stairs aligned
fall rate on stairs decreases
upstairs/downstairs segments become passable
robot recovers straight walking after stairs
```

Stage 3: Task A evaluation improves.

```text
ATEC-TaskA-G1 score improves over the current /demo/policy_a.pt baseline
```

## Out of scope for first implementation

The first implementation should not:

1. overwrite `ATEC-Isaac-AMP-Unitree-G1-TaskA-v0`;
2. force `lin_vel_y`, `ang_vel_z`, and `heading` to exactly zero;
3. assume `/demo/policy_a.pt` is a resumable training checkpoint;
4. change the AMP observation layout;
5. change the discriminator architecture;
6. add a complex lookahead centerline controller;
7. commit large generated pkl files unless explicitly requested.

## Open implementation checks

Before coding, verify these details in the codebase:

1. the exact traversal order used by `BetterTerrainGenerator` for rows and columns;
2. whether `env.scene.env_origins[:, 1]` is the right centerline reference for every reset;
3. the GMR command needed to transform `upstairs01_stageii.npz` and `downstairs01_stageii.npz` into G1 29DoF pkl files;
4. whether the local Python/conda environment for GMR has `numpy` and `mujoco` available;
5. whether existing motion files in `motion_data/g1_29dof/*.json` are all 79-dimensional and compatible with the stricter validation script.
