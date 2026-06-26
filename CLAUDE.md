# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment and setup

Use the `atec` conda environment for local development:

```bash
cd /home/air/wang-sm/ATEC
conda activate atec
```

Install the repository extension after setting up Isaac Sim / IsaacLab:

```bash
cd /home/air/wang-sm/ATEC/source/atec_rl_lab
python -m pip install -e .
```

Task E ACT training needs additional Python packages that are declared in `source/atec_rl_lab/setup.py` and also called out in the README:

```bash
python -m pip install diffusers tyro h5py tensorboard wandb imageio
```

## Common commands

List registered ATEC Gym environments without launching Isaac Sim:

```bash
python scripts/list_envs.py | grep ATEC
```

View task environments. Task E and any camera environment must pass `--enable_cameras`:

```bash
python scripts/view_task_a.py --enable_cameras
python scripts/view_task_b.py --enable_cameras
python scripts/view_task_d.py --enable_cameras
python scripts/view_task_e.py --num_envs 1 --enable_cameras
```

Run the local evaluation entrypoint. This always imports `demo.solution.AlgSolution`, so only one task's solution can be active in `demo/solution.py` at a time:

```bash
PYTHONPATH=. python scripts/play_atec_task.py \
  --task <ATEC-TaskX-Robot> \
  --num_envs 1 \
  --enable_cameras \
  --debug
```

Task E environment ID:

```bash
PYTHONPATH=. python scripts/play_atec_task.py \
  --task ATEC-TaskE-Piper \
  --num_envs 1 \
  --enable_cameras \
  --debug
```

Run all lightweight tests:

```bash
conda run -n atec pytest tests -q
```

Run a single test:

```bash
conda run -n atec pytest tests/path/to/test_file.py::test_name -v
```

Run ACT/Task E unit tests in the current worktree:

```bash
conda run -n atec pytest tests/act -q
conda run -n atec pytest tests/act/test_task_e_state_machine.py -q
conda run -n atec pytest tests/act/test_task_e_collector_success.py -v
```

## Task E ACT data collection and training

Collect scripted Task E ACT demos with the state machine under `scripts/act/task_e/`:

```bash
python scripts/act/collect_demos_task_e.py \
  --num_demos 50 \
  --output_dir datasets/atec_task_e/full_order_123 \
  --full_order_123 \
  --optimized_grasp_flow \
  --only_success \
  --save_images \
  --max_attempts 300
```

Collection writes:

- `<output_dir>/trajectory.hdf5`
- `<output_dir>/trajectory.json`
- `<output_dir>/videos/*.mp4` when `--save_video` is used, unless `--video_dir` is provided
- `traj_N/images/rgb` inside the HDF5 when `--save_images` is used

Filter stationary steps:

```bash
python scripts/act/filter_demos.py \
  --input datasets/atec_task_e/full_order_123/trajectory.hdf5 \
  --output datasets/atec_task_e/full_order_123/trajectory_filtered.hdf5 \
  --threshold 0.001
```

Train Task E ACT on RGB demos:

```bash
python scripts/act/train_task_e.py \
  --demo-path datasets/atec_task_e/full_order_123/trajectory_filtered.hdf5 \
  --include-rgb \
  --total-iters 100000 \
  --batch-size 256 \
  --log-freq 1000 \
  --save-freq 10000 \
  --seed 1 \
  --exp-name act-task-e-rgb-full-order-123
```

For state-only smoke runs, use `--no-include-rgb --no-cuda`. Avoid `--total-iters 1`; the scheduler computes a zero step size. Use at least `--total-iters 2`.

## Architecture overview

This repository is an IsaacLab extension for ATEC tasks. The extension package lives in `source/atec_rl_lab/atec_rl_lab` and is installed editable during setup.

Task registration is in `source/atec_rl_lab/atec_rl_lab/tasks/*/__init__.py`. Importing `atec_rl_lab.tasks` registers the Gym IDs used by scripts. Shared environment machinery is in `tasks/task_base/`; task-specific environment configs, rewards, terrain, and terminations live under `tasks/task_a`, `tasks/task_b`, `tasks/task_d`, and `tasks/task_e`.

Assets are under `source/atec_rl_lab/atec_rl_lab/assets/`, split into robots and task objects. Training/model code is under `source/atec_rl_lab/atec_rl_lab/train/`, including ACT components and locomotion utilities.

Top-level `scripts/` are the main operational entrypoints:

- `scripts/play_atec_task.py` runs local evaluation using the currently active `demo/solution.py`.
- `scripts/view_task_*.py` launch task environments for visualization/smoke checks.
- `scripts/rsl_rl/train*.py` and `scripts/rsl_rl/play*.py` train/play locomotion policies.
- `scripts/act/collect_demos_task_e.py`, `filter_demos.py`, and `train_task_e.py` implement the Task E ACT data pipeline.

The submission/runtime surface is `demo/`. `scripts/play_atec_task.py` imports only `demo.solution.AlgSolution`, so switching active task solutions means replacing or editing `demo/solution.py` and ensuring any required policy files sit next to it.

Task E scripted collection is organized as small modules under `scripts/act/task_e/`:

- `config.py` contains robot names, timing, grasp offsets, tool offsets, basket target offsets, success bounds, spawn regions, camera constants, actuator settings, and warm-up settings.
- `state_machine.py` converts object poses into desired end-effector poses and gripper commands.
- `collector.py` runs one episode, records `obs/actions/qvel/ee_pos/ee_quat`, handles success checks, and treats official basket-success termination as a successful demo.
- `env_setup.py` builds the single-env Task E Piper collection environment.
- `options.py` parses collection options such as full-order object selection and grasp offsets.

The HDF5 format expected by `scripts/act/train_task_e.py` is `traj_N/obs` and `traj_N/actions`, with optional `traj_N/images/rgb` for RGB training.
