# Task B B2Piper Pico Arm Teleoperation Design

Date: 2026-06-24

## Goal

Build a simulation-only data collection path for `ATEC-TaskB-B2Piper` where one Pico controller teleoperates only the Piper arm mounted on B2.

The B2 leg joints are not controlled by the operator and are not part of the recorded training dataset. They are only filled with zero action internally so the existing 20-dimensional B2Piper environment action can be stepped.

## Non-Goals

- Do not teleoperate B2 legs, body pose, base velocity, or locomotion.
- Do not record B2 leg state, base velocity, base pose, or locomotion commands as primary dataset fields.
- Do not train a policy in this change.
- Do not integrate real B2Piper hardware.
- Do not replace the current Task B solution or active `demo/solution.py`.

## Existing Context

`ATEC-TaskB-B2Piper` uses the B2Piper robot config. Its action joint order is:

1. 12 B2 leg joints.
2. 8 Piper joints: `arm_joint1` through `arm_joint8`.

The base environment action uses joint position targets with `scale=0.5` and default joint offsets. To step the simulator while controlling only the arm:

```text
full_action[0:12] = 0.0
full_action[12:20] = piper_action_8
```

ATEC already has `CartesianController`, which converts end-effector pose targets into arm joint targets through Differential IK. Task E already has an HDF5 demonstration collection pattern for arm data.

Sonic already demonstrates reading Pico/XR controller inputs through `xrobotoolkit_sdk` and, where available, IsaacTeleop-style reader adapters.

## Recommended Architecture

Create a standalone collection script under `ATEC/scripts/teleop/`:

```text
collect_task_b_b2piper_arm_pico.py
```

The first implementation can stay in one file to reduce integration overhead. If it grows, split it into:

```text
pico_input.py
pico_to_ee_target.py
b2piper_arm_teleop.py
arm_hdf5_recorder.py
```

The runtime pipeline is:

```text
Pico controller pose/buttons
  -> controller enable and calibration gate
  -> controller relative pose
  -> Piper end-effector target in robot base frame
  -> CartesianController.compute_base()
  -> piper joint target/action, 8 dimensions
  -> internal full_action, 20 dimensions
  -> env.step(full_action)
  -> record arm-only trajectory
```

## Control Semantics

Use one Pico controller as a relative end-effector device.

- A grip or primary button enables teleoperation.
- The first enable event captures calibration:
  - controller reference pose
  - current Piper `gripper_base` pose
- While enabled, controller translation delta maps to end-effector target translation.
- When disabled, the current end-effector target is held.
- Trigger controls gripper close amount.
- A secondary/menu button resets calibration.

The first version should use position control plus a stable end-effector orientation:

- Position: relative controller translation scaled into the Piper base frame.
- Orientation: fixed downward/top-grasp orientation, or yaw-only if that proves necessary.
- Full 6D orientation following is intentionally deferred because it is more likely to create IK instability during initial data collection.

## Coordinate Mapping

The controller pose is first converted into the robot convention used by ATEC:

```text
robot X: forward
robot Y: left
robot Z: up
```

The initial mapping should be conservative:

```text
controller delta x -> EE base-frame y or x, depending on operator comfort
controller delta y -> EE base-frame z
controller delta z -> EE base-frame x
```

The exact axis mapping should be configurable through constants in the script, not hard-coded across multiple call sites. The script should print the mapping at startup so bad calibration is visible during collection.

Suggested initial gains:

```text
translation_scale = 0.8 to 1.2
max_target_step_m = 0.03 per sim step
max_joint_delta_rad = 0.05 to 0.12 per sim step
```

## Workspace Limits

Clamp the end-effector target in B2 base frame before IK:

```text
x: [0.15, 0.85]
y: [-0.45, 0.45]
z: [-0.30, 0.35]
```

These are first-pass safety bounds. They should be adjusted after visual inspection of the B2Piper arm reach in `ATEC-TaskB-B2Piper`.

If IK returns NaN or an implausible joint target, hold the previous safe arm target for that step.

## Action Construction

At every simulator step:

1. Read current robot joint state.
2. Compute desired `arm_joint1..arm_joint6` from the EE target.
3. Compute `arm_joint7..arm_joint8` from trigger:

```text
open  = [0.035, -0.035]
close = [-0.015, 0.015]
gripper = lerp(open, close, trigger)
```

4. Build arm target positions for the last 8 B2Piper joints.
5. Convert target positions to environment actions:

```text
piper_action_8 = (target_arm_q - default_arm_q) / 0.5
```

6. Build full action:

```text
full_action_20 = zeros(20)
full_action_20[12:20] = piper_action_8
```

7. Step the environment with `full_action_20`.

Only `piper_action_8` is recorded as the dataset action.

## Dataset Format

Store one HDF5 file with groups `traj_0000`, `traj_0001`, and so on.

Required datasets per trajectory:

```text
obs/arm_qpos           float32, shape (T, 8)
obs/arm_qvel           float32, shape (T, 8)
actions/arm_action     float32, shape (T, 8)
ee_pos                 float32, shape (T, 3)
ee_quat                float32, shape (T, 4), wxyz
ee_target_pos          float32, shape (T, 3)
ee_target_quat         float32, shape (T, 4), wxyz
controller_pos         float32, shape (T, 3)
controller_quat        float32, shape (T, 4), xyzw or wxyz with metadata
controller_buttons     float32, shape (T, N)
```

Optional datasets:

```text
images/head_rgb
images/head_depth
score
```

File-level metadata:

```text
env_id
robot_joint_order
arm_joint_names
action_scale
controller_quat_order
controller_to_robot_axis_map
translation_scale
workspace_bounds
orientation_mode
calibration_policy
```

The dataset should not include B2 leg observations or B2 leg actions as normal learning fields.

## Episode Controls

The collection script should support keyboard commands in addition to Pico buttons:

```text
r: start/stop recording current episode
n: finish and save current episode
d: discard current episode
c: reset Pico/EE calibration
q: quit safely
```

If keyboard polling is too intrusive in Isaac Sim, use command-line flags plus Pico button events for the first version:

```text
--max_steps_per_episode
--num_episodes
--output_dir
--save_images
--headless
```

## Error Handling

- If Pico data is unavailable at startup, keep the simulator open and print a waiting message.
- If Pico data becomes stale during recording, pause target updates and hold the last safe target.
- If the controller is not enabled, do not update the target from hand motion.
- If IK fails, keep the previous safe arm joint target.
- If the env terminates early, save the episode only if it has enough recorded steps and the user explicitly accepts it.

## Testing And Verification

Unit-level tests should cover:

- Arm-only action packing: 8 recorded arm actions become 20 env actions with the first 12 entries zero.
- Trigger-to-gripper interpolation.
- Workspace clamping.
- Relative pose calibration math with synthetic controller poses.

Manual verification should cover:

1. Run `ATEC-TaskB-B2Piper` with zero leg action and confirm B2 remains stable enough for arm-only collection.
2. Confirm Pico enable/calibration does not jump the arm.
3. Move the controller along each axis and verify the end-effector moves in the expected direction.
4. Confirm trigger opens/closes `arm_joint7..arm_joint8`.
5. Record one short trajectory and inspect HDF5 shapes and metadata.

## Success Criteria

- A user can launch one script and collect arm-only teleoperation trajectories in `ATEC-TaskB-B2Piper`.
- The simulator receives valid 20-dimensional actions, but the saved dataset exposes only Piper arm state/action plus Pico/EE signals.
- B2 leg action remains zero throughout collection.
- The arm target is bounded and robust to Pico disconnects, IK NaNs, and disabled teleop.
- One saved trajectory can be loaded and replayed as an arm-only action sequence.
