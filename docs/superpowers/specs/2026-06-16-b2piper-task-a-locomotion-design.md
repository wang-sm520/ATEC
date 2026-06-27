# B2Piper Task-A-style Locomotion Training Design

Date: 2026-06-16

## Summary

Train a B2Piper locomotion policy for ATEC Task A and later Task B use. The policy controls only the 12 B2 leg joints while the Piper arm is held in a safe fixed posture. The training environment should use the B2Piper asset so the policy learns with the arm mass and inertia present, but the reinforcement-learning problem remains a quadruped locomotion problem rather than a 20-DoF whole-body manipulation problem.

The B2 training side will be refactored with Task-A-style terrain and reward ideas from the existing humanoid/G1 Task A configuration, while preserving quadruped-specific B2 reward structure.

## Goals

1. Add a B2Piper Task-A-style locomotion training environment.
2. Keep policy action output at 12 dimensions for B2 leg joints.
3. Keep Piper arm joints fixed or passively held in a stable carry posture.
4. Rework B2 reward configuration using transferable ideas from the G1 Task A humanoid reward design.
5. Produce RSL-RL checkpoints and exported TorchScript/ONNX policies suitable for later Task A/B integration.
6. Keep existing B2 flat/rough environments working and avoid destabilizing current baselines.

## Non-goals

1. Do not train a 20-dimensional whole-body B2Piper policy in the first version.
2. Do not add B2Piper manipulation learning in this phase.
3. Do not port G1 AMP training to B2Piper.
4. Do not directly copy humanoid-only two-foot or torso/arm reward terms to B2.
5. Do not modify Task B submission behavior in this phase.

## Proposed Environment

New Gym environment ID:

```text
ATEC-Isaac-Velocity-TaskA-Unitree-B2Piper-v0
```

New experiment name:

```text
unitree_b2piper_task_a
```

The environment will live with the existing Unitree B2 locomotion configs under:

```text
source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/
```

The new environment config should subclass or closely mirror the current B2 rough config, but change the robot asset to `UNITREE_B2_PIPER_CFG` and replace the terrain/reward setup with the Task-A-style design described below.

## Action and Observation Design

### Actions

The policy action space is 12-dimensional and controls only the B2 leg joints:

```text
FL_hip_joint
FL_thigh_joint
FL_calf_joint
FR_hip_joint
FR_thigh_joint
FR_calf_joint
RL_hip_joint
RL_thigh_joint
RL_calf_joint
RR_hip_joint
RR_thigh_joint
RR_calf_joint
```

The Piper arm joints are not exposed as policy actions in the first version. This keeps the locomotion policy compatible with a later Task B controller that can combine:

```text
12 leg actions from locomotion policy
+ 8 arm actions from a rule-based manipulation/contact controller
```

### Observations

The first version should keep the same policy observation style as the current B2 PPO setup:

```text
base angular velocity
projected gravity
velocity command
12 leg joint positions
12 leg joint velocities
last 12 actions
```

The critic may keep privileged base linear velocity as in the existing B2 setup. Arm joint positions/velocities should not be required by the actor in the first version. If arm passive dynamics make training unstable, arm joint state can be added to critic observations first, then actor observations only if needed.

## Piper Arm Fixed-Posture Strategy

Initial strategy:

1. Use `UNITREE_B2_PIPER_CFG` as the robot asset.
2. Set the policy action term to the 12 B2 leg joints only.
3. Rely on the B2Piper default joint configuration and actuator stiffness/damping to hold Piper in a stable carry pose.

If smoke testing shows the arm drops, oscillates, or destabilizes the base, upgrade to an explicit fixed-arm mechanism:

```text
policy still outputs 12 leg actions
wrapper or action manager appends fixed 8 arm targets
robot receives stable leg+arm targets
```

The policy interface should remain 12-dimensional even if the environment internally applies fixed arm targets.

## Terrain Design

Use a Task-A-style terrain curriculum inspired by the existing G1 Task A terrain configuration.

Initial terrain mixture:

```text
flat: 45%
random rough: 20%
inverted pyramid stairs: 35%
```

Initial terrain generator settings:

```text
tile size: 20m x 20m
num_rows: 10
num_cols: 20
curriculum: enabled
max_init_terrain_level: 0
```

This gives the policy enough flat terrain to learn stable early locomotion while exposing it to rough and stair-like terrain needed for Task A transfer.

If the first smoke or short training run is unstable, reduce difficulty in this order:

1. flat only
2. flat + rough
3. flat + rough + inverted stairs

## Reward Refactor

### Principle

Start from the current B2 quadruped reward base, then selectively migrate robot-agnostic reward ideas from the G1 Task A humanoid configuration.

Do not directly port humanoid-only reward terms that assume two feet, humanoid torso/arm structure, or AMP motion data.

### Keep from current B2 reward base

Keep or tune these existing B2-compatible terms:

```text
lin_vel_z_l2
ang_vel_xy_l2
joint_torques_l2
joint_acc_l2
joint_pos_limits
joint_power or energy, but not both at high weight
stand_still
joint_pos_penalty
joint_mirror
action_rate_l2
undesired_contacts
contact_forces
track_lin_vel_xy_exp
track_ang_vel_z_exp
feet_contact_without_cmd
feet_height_body
upward
```

### Transfer from G1 Task A reward design

Add or strengthen these robot-agnostic terms:

#### Stronger velocity tracking

Current B2 weights are lighter than G1 Task A. Use:

```text
track_lin_vel_xy_exp: 4.0
track_ang_vel_z_exp: 3.0
```

Do not start with G1's stronger yaw tracking weight of 6.0; that can be tried later if B2Piper turns too slowly.

#### Lateral drift penalty

Add:

```text
lin_vel_y_l2: -0.25 to -0.5
```

Purpose: reduce side drift and improve dead-reckoning reliability for Task A/B high-level controllers.

#### Action smoothness

Add:

```text
action_smoothness: -0.003
```

Purpose: reduce leg jitter and base shake, especially with Piper mass mounted.

#### Anti-idle penalty

Add:

```text
idle_when_commanded: -1.0 to -2.0
```

Purpose: avoid a stable standing policy when commanded velocity is non-zero.

#### Foot slip and stumble penalties

Enable conservatively:

```text
feet_slide: -0.1 to -0.25
feet_stumble: -1.0 to -2.0
```

These should start with mild weights so early training is not over-constrained by contact penalties.

### Do not port directly

Do not port these G1-specific reward designs in the first version:

```text
AMP reward or AMP observation stack
biped gait-clock rewards
feet_too_near_humanoid
feet_y_distance
feet_orientation_euler
humanoid arm/torso/knee contact terms
G1-specific joint deviation groups
```

For quadruped gait shaping, prefer existing quadruped gait/stance rewards if needed after the first short training run.

## PPO Runner Design

Add a new runner config that reuses the current B2 PPO network architecture:

```text
actor_hidden_dims: [512, 256, 128]
critic_hidden_dims: [512, 256, 128]
activation: elu
```

Set:

```text
experiment_name: unitree_b2piper_task_a
max_iterations: 10000
```

The current `scripts/rsl_rl/train.py` and `scripts/rsl_rl/play.py` already need the rsl-rl 5.x deprecated config migration. That compatibility has been added separately and should remain in place.

## Implementation Files

Expected edits:

```text
source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/task_a_b2piper_env_cfg.py
source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/__init__.py
source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_cfg.py
```

Expected tests:

```text
tests/test_b2piper_task_a_training_cfg.py
```

Optional documentation or run notes may be added if needed after smoke testing.

## Verification Plan

### Static/unit verification

Run:

```bash
python -m pytest tests/test_b2piper_task_a_training_cfg.py -q
python -m pytest tests/test_rsl_rl_script_compat.py -q
python -m py_compile \
  scripts/rsl_rl/play.py \
  scripts/rsl_rl/train.py \
  source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/task_a_b2piper_env_cfg.py \
  source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/__init__.py \
  source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_cfg.py
```

### Environment listing

Run:

```bash
python scripts/list_envs.py | grep B2Piper
```

Expected new env ID:

```text
ATEC-Isaac-Velocity-TaskA-Unitree-B2Piper-v0
```

### IsaacLab smoke training

Run:

```bash
python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-TaskA-Unitree-B2Piper-v0 \
  --num_envs 16 \
  --max_iterations 2 \
  --headless
```

Smoke success criteria:

1. Environment registers and creates successfully.
2. Action manager reports shape 12.
3. Policy observations have expected dimensions.
4. No NaN during the two training iterations.
5. B2Piper does not immediately fail because of uncontrolled arm dynamics.

### Short training

After smoke passes:

```bash
python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-TaskA-Unitree-B2Piper-v0 \
  --num_envs 512 \
  --max_iterations 500 \
  --headless
```

Short training success criteria:

1. Reward is finite.
2. Checkpoint is saved.
3. Termination rate is not immediately saturated.
4. Policy begins to track simple forward commands.

### Full training

Initial full training command for RTX 3090:

```bash
python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-TaskA-Unitree-B2Piper-v0 \
  --num_envs 2048 \
  --max_iterations 10000 \
  --headless
```

If stable and memory allows, increase to `--num_envs 4096`.

### Play and export

Use RSL-RL checkpoint files only. Replace `RUN_DIR` with the concrete timestamped run directory and `MODEL_FILE` with a saved training checkpoint such as `model_500.pt`:

```bash
python scripts/rsl_rl/play.py \
  --task ATEC-Isaac-Velocity-TaskA-Unitree-B2Piper-v0 \
  --checkpoint logs/rsl_rl/unitree_b2piper_task_a/RUN_DIR/MODEL_FILE
```

Do not pass exported TorchScript `policy.pt` back into `play.py --checkpoint`; exported policies are loaded directly with `torch.jit.load()` in deployment code.

## Risks and Mitigations

### Arm is unstable when not action-controlled

Mitigation: add an explicit fixed-arm target mechanism while keeping the policy output at 12 dimensions.

### Terrain is too hard for first training

Mitigation: temporarily reduce terrain to flat or flat+rough, then re-enable stairs after stable locomotion emerges.

### Reward terms over-constrain learning

Mitigation: start with mild weights for new foot contact and smoothness terms. Keep B2 quadruped reward base as the stable fallback.

### rsl-rl version compatibility

Mitigation: keep `handle_deprecated_rsl_rl_cfg` in standard PPO train/play scripts. New checkpoints trained under the current environment should match current rsl-rl format.

## Acceptance Criteria

The implementation is accepted when:

1. The new environment ID is registered and discoverable.
2. The new config uses `UNITREE_B2_PIPER_CFG`.
3. The policy action space remains 12-dimensional.
4. Piper arm is not controlled by the policy in the first version.
5. The reward config includes Task-A-inspired velocity tracking, lateral drift penalty, action smoothness, and anti-idle behavior.
6. Static tests pass.
7. IsaacLab smoke training runs for two iterations without config/runtime errors.
8. The path to short/full training is documented with exact commands.
