# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment and setup

This project depends on IsaacLab/Isaac Sim and CUDA/PyTorch. Run commands from the repository root unless a command explicitly changes directories.

```bash
# IsaacLab-capable conda env: torch 2.7.1 + CUDA 12.8 + onnxruntime + IsaacLab.
conda activate atec
# To activate inside a single non-interactive command:
source ~/miniconda3/etc/profile.d/conda.sh && conda activate atec

# Editable install of the task package (after IsaacLab is available).
pip install -e source/atec_rl_lab
```

`pxr` (USD) only exists after `AppLauncher` starts the sim app, so `import atec_rl_lab.tasks` fails outside that context — scripts import it AFTER constructing `AppLauncher`. Isaac startup takes minutes; run long evals/probes in the background and poll the log.

Some shells lack `pytest`. For pure-Python tests use `python -m unittest ...` (NOT pytest). `torch`/`onnxruntime` may be missing in the base shell but ARE present in `atec`; torch-dependent unit tests are guarded with `@unittest.skipIf` and skip when absent.

## Common commands

### Inspect / visualize / list envs

```bash
python scripts/list_envs.py
python scripts/view_task_b.py --enable_cameras    # also view_task_{a,d,e}.py, view_robots.py
```

### Local ATEC evaluation (official entry, loads demo/solution.py)

```bash
PYTHONPATH=. python scripts/play_atec_task.py --task ATEC-TaskB-G1 --enable_cameras --debug
```
`--task` picks arena+robot (see `OVERVIEW.md` matrix): `ATEC-TaskB-G1`, `ATEC-TaskD-G1`, `ATEC-TaskE-Piper`, B2/B2W/Tron variants. `--debug` prints score.

### Task B G1 (current active work — whole-body controller)

`demo/solution.py` IS the Task B WBC submission. Dev tooling:

```bash
# Fast non-Isaac unit tests (numpy-based run; torch/onnx parts skip if absent).
python -m unittest discover -s tests/task_b -p 'test_*.py' -v

# Eval the WBC solution + record a 3rd-person(follow)+head-cam+right-hand-cam video.
PYTHONPATH=. python scripts/eval_task_b_g1_wbc.py \
  --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless \
  --seconds 90 --out logs/videos/task_b_g1_wbc/run.mp4        # add --no_video for score-only

# Calibration / diagnostic probes (privileged: read env.scene truth, NOT for the solution):
PYTHONPATH=. python scripts/confirm_detection.py     ... --headless   # detection vs truth + annotated images
PYTHONPATH=. python scripts/probe_task_b_g1_camcalib.py ... --headless # head-camera extrinsics (pitch/height)
PYTHONPATH=. python scripts/probe_task_b_g1_handcam.py  ... --headless # what the eye-in-hand cameras see
PYTHONPATH=. python scripts/eval_miniwbc.py             ... --headless # WBC balance smoke (stand/walk/squat/reach)
```
Extracting a frame from a recorded mp4 to inspect: `python -c "import imageio.v3 as iio,PIL.Image as I; I.fromarray(iio.imread('run.mp4')[N]).save('f.png')"` then Read the PNG.

### Other tasks

```bash
# Task D / controller unit tests
python -m unittest discover -s tests/task_d -p 'test_*.py' -v
# Task A AMP train / playback (playback exports policy.pt + policy.onnx)
python scripts/rsl_rl/train_amp.py --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 --num_envs=4096 --max_iterations=30000 --headless --amp_motion_files motion_data/g1_29dof/*.json
python scripts/rsl_rl/play_amp.py --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 --num_envs=16 --real-time
# Task E ACT: scripts/act/collect_demos_task_e.py, filter_demos.py, then `cd scripts/act && bash baseline.sh`
```

## Submission interface

`demo/solution.py` must define `AlgSolution.predicts(obs, current_score) -> {"action": list[float], "giveup": bool}`. The platform does `from solution import AlgSolution` with the submission dir on the path (top-level import, NOT `from demo.solution`). Do NOT upload `run.sh` / `server.py` (injected by the platform). Container has no internet; package all deps/weights. Base image: Python 3.12 + PyTorch 2.7.1 + CUDA 12.8.

Observation groups: `obs["proprio"]` = [base_lin_vel(3), base_ang_vel(3), velocity_cmd(3), projected_gravity(3), joint_pos(N), joint_vel(N), last_actions(N)] (G1: N=33). `obs["image"]` = `head_rgb/head_depth` (640×480), `ee_rgb/ee_depth` (left hand), `ee_dual_rgb/ee_dual_depth` (right hand, G1 only). `obs["extero"]["lidar_scan"]`. Actions: `JointPositionActionCfg` scale 0.5, use_default_offset, preserve_order; G1 = 33 dims (29 body + 4 dex1 hand).

LiDAR raycasts only `/World/ground` (a terrain height scan) — it CANNOT see objects. The head camera is pitched ~47.6° down (sees a ~0.7–2.5m ground band ahead, blind closer than ~0.7m). Only the eye-in-hand cameras see near-field objects.

## High-level architecture

- `source/atec_rl_lab/atec_rl_lab/tasks/` registers/configures envs. `task_base/envs_base_cfg.py` defines shared obs/action/sensors (head/ee `CameraCfg`, `lidar_sensor`); `task_{a,b,d,e}/` specialize. `tasks/task_b/`: `env_cfg.py` (G1 uses `UNITREE_G1_29DOF_DEX1_CFG`, target circle `(-3,-10)` r=1.0), `mdp/rewards.py` (`GraspedObjectsByEE` grasp_dist_thresh **0.20** on `left/right_hand_base_link`; `ObjectsInCircle`).
- `source/atec_rl_lab/atec_rl_lab/assets/robots/g1/g1_29dof_dex1.py` — G1 actuator kp/kd, init joint angles, and the **ATEC dex1 joint order** (legs 0-11, waist 12-14, L arm 15-21, R arm 22-28, hands 29-32). Check this before indexing action vectors.
- `scripts/` — Isaac app entrypoints (AppLauncher first, then gym/isaaclab imports; MARL envs wrapped via `multi_agent_to_single_agent`).
- `demo/` — submission/runtime surface; solutions must be self-contained (no `atec_rl_lab` import at runtime).
- `tests/` — non-Isaac unit tests. `motion_data/`, `atec_robot_model/` — AMP clips and baseline USD/weights.

## Task B WBC solution (demo/solution.py) — current architecture

A single whole-body controller replaces walk+squat+sweep. The implementation is split into focused, single-responsibility modules wired together by a thin `solution.py`. Pipeline per `predicts()`: dead-reckon pose → posture guard → RGB-D perception (throttled every 5 steps) → planner FSM → map phase to WBC command.

- `demo/solution.py` — thin wiring ONLY (no planner/motion logic, constants, or state machine). `AlgSolution.predicts()` constructs and sequences `MiniWBC` + `DeadReckoningOdometry` + `TaskBRgbdPerception` + `TaskBPlanner` + `PostureGuard`. Uses a try/except import so it works both as platform top-level (`from task_b_planner import`) and dev (`from demo.task_b_planner import`).
- `demo/task_b_planner.py` — `TaskBPlanner`, the unified FSM. Phases: `search` → `approach` → `creep` → `squat_sweep` → `stand_up` (then back to `search`/`approach`). Emits one `WBCCommand` per step. Holds all planner constants and the phase logic; `PostureGuard`-driven `recover` interrupts any phase into `stand_up`.
- `demo/task_b_nav.py` — stdlib-only kinematics: `Pose2D`, `DeadReckoningOdometry` (integrates base velocity), `PostureGuard` (flags fall/recover from projected gravity).
- `demo/task_b_perception.py` — `TaskBRgbdPerception` (color+depth blob detector). **Ground-projects depth via the 47.6° camera pitch** in `_project_pixel_to_base` — using raw slant depth as ground distance is wrong and was the original bug.
- `demo/mini_wbc.py` `MiniWBC` — adapter for `mini/`'s loco-manip ONNX policy (`policy18.onnx`). Commands: base velocity(3), base height, waist rpy(3), **left/right hand pose (xyz+wxyz quat, base frame)**; obs 115 + 5-frame history + ik_input 15 → action 29 + ik_output 17. The policy uses IsaacLab *interleaved* joint order; mini's "mujoco" order == ATEC dex1 order, so `ATEC_TO_POLICY`/`POLICY_TO_ATEC` permute between them. The 4 dex1 fingers are held at default.
- Reach strategy: detect → rotate-scan/approach → creep in → squat to base height **0.30** → BOTH hands brush the front ground (Lissajous sweep at floor level) → stand → next. The generous 0.20m grasp sphere means a sweep beats precise single-point targeting.

**WBC command valid ranges (from `mini/command_gui.py`) — commanding outside misbehaves:** base height [0.3, 0.9]; hand x [-0.2, 0.6]; hand z [-0.2, 0.65]; left-hand y [-0.1, 0.6]; right-hand y [-0.6, 0.1]. Key consequence: only at base height ≈0.30 does hand z=-0.20 reach the floor (world ~0.12); a shallower squat cannot reach floor objects within valid range.

`mini/` is the standalone source of the controller (MuJoCo viewer + `g1_loco_manip.py` + `command_gui.py`); it is reference only — the ATEC adapter is `demo/mini_wbc.py`.

Two optimization-phase tunables worth preserving:
- **Sighting band gating (0.60–3.0m):** `task_b_planner.py` only trusts a detection whose ground distance falls in this band. The head cam is pitched ~47.6° down and is physically blind for ground objects closer than ~0.62m (they drop below the frame); beyond ~3.0m the depth projection is unreliable. Detections outside the band are noise or the robot's own body and must never select/refresh/enter memory — relaxing this re-introduces phantom locks and squats on empty floor.
- **Search cycle = full 360° spin + 1.8m leg:** each scan stop does a complete 360° in-place spin (`SEARCH_ROTATE_STEPS=242`) then walks a 1.8m leg (`SEARCH_RELOCATE_STEPS=165`) so consecutive scan annuli abut rather than overlap, giving full coverage of the 10×10m arena. The old under-rotated cycle (276° + 0.83m) left azimuth/coverage gaps.

Submission set (7 files, uploaded together flat as the import root): the five modules above (`solution.py`, `task_b_planner.py`, `task_b_nav.py`, `task_b_perception.py`, `mini_wbc.py`) + `policy18.onnx` + `requirements.txt` (adds `onnxruntime`). Staged copy at `demo/task_b/`; `tests/task_b/test_submission_staging.py` is a byte-equality guard that fails if any staged file drifts from its `demo/` source (re-sync the copy after editing a module). The prior Task D solution is preserved at `demo/solution_task_d.py`.

Design/run notes: `docs/superpowers/specs/2026-06-10-task-b-g1-squat-sweep-design.md`, `docs/superpowers/plans/2026-06-10-task-b-g1-squat-sweep.md`, `docs/task_b_g1_squat_notes.md` (chronological Isaac-run findings + tunables).

## Task A / D notes

- Task A: G1 AMP training — `docs/task_a_amp_training.md`, `docs/task_a_motion_data_pipeline.md`. README is Task-A focused; `play_amp.py` exports the TorchScript actor (960-dim term-major history: `[ang_vel, cmd, gravity, joint_pos, joint_vel, last_action]` × 10).
- Task D: `docs/task_d_submission.md`, `docs/task_d_mapush_integration_plan.md`; tests under `tests/task_d/`.
