# Task E ACT Checkpoint Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a direct runner that loads the trained Task E RGB ACT checkpoint and proves it can produce actions before any submission integration.

**Architecture:** Create a standalone script under `scripts/act/` that reuses the existing training `Agent` and `Args` definitions from `scripts/act/train_task_e.py`. The runner supports a fast fake-observation smoke mode and an optional live Task E environment mode that launches Isaac Sim and steps `ATEC-TaskE-Piper` with ACT actions.

**Tech Stack:** Python, PyTorch, Torchvision transforms from existing ACT code, IsaacLab AppLauncher for live mode, pytest for helper tests, `conda run -n atec` for execution.

## Global Constraints

- Do not continue self-contained `demo/solution.py` integration for this task.
- Do not depend on the previous in-progress submission scaffold.
- Use checkpoint `runs/act-task-e-rgb-full-order-123-20260624-1925/checkpoints/best_loss.pt` by default.
- Prefer `ema_agent` weights and fall back to `agent` only if `ema_agent` is absent.
- Fake smoke mode must not launch Isaac Sim.
- Live mode must run `ATEC-TaskE-Piper` with `--enable_cameras` and use `obs["image"]["video_rgb"]` plus the first 8 proprio values.
- Do not commit unless the user explicitly authorizes commits.

---

## File Structure

- Create: `scripts/act/run_task_e_act_checkpoint.py`
  - Responsibility: CLI runner, checkpoint loading, observation preprocessing, fake smoke forward, optional live environment stepping.
- Create: `tests/act/test_task_e_act_checkpoint_runner.py`
  - Responsibility: fast helper tests for checkpoint selection, state preprocessing, RGB preprocessing, and action denormalization without launching Isaac Sim.

---

### Task 1: Direct checkpoint runner smoke mode

**Files:**
- Create: `scripts/act/run_task_e_act_checkpoint.py`
- Create: `tests/act/test_task_e_act_checkpoint_runner.py`

**Interfaces:**
- Produces `resolve_checkpoint(path: str | None) -> pathlib.Path`.
- Produces `load_policy(checkpoint_path: pathlib.Path, device: torch.device) -> tuple[Agent, dict, Args]`.
- Produces `prepare_state(proprio: torch.Tensor, norm_stats: dict, device: torch.device) -> torch.Tensor` returning `(1, 8)`.
- Produces `prepare_rgb(rgb: torch.Tensor, device: torch.device) -> torch.Tensor` returning `(1, 1, 3, 224, 224)`.
- Produces `denormalize_action_chunk(chunk: torch.Tensor, norm_stats: dict) -> torch.Tensor`.
- CLI supports `--smoke-only`, `--checkpoint`, `--device`, `--max_steps`, `--task`, `--headless`.

- [ ] **Step 1: Write failing helper tests**

Create `tests/act/test_task_e_act_checkpoint_runner.py` with tests for `prepare_state`, `prepare_rgb`, and `denormalize_action_chunk` using simple tensors.

- [ ] **Step 2: Verify RED**

Run:

```bash
conda run -n atec pytest tests/act/test_task_e_act_checkpoint_runner.py -q
```

Expected: FAIL because `scripts/act/run_task_e_act_checkpoint.py` does not exist.

- [ ] **Step 3: Implement runner helpers and fake smoke mode**

Implement `scripts/act/run_task_e_act_checkpoint.py` so `--smoke-only` loads the checkpoint, creates zero proprio and zero RGB inputs, runs `Agent.get_action`, denormalizes the action chunk, and prints the checkpoint path plus chunk/action shapes.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
conda run -n atec pytest tests/act/test_task_e_act_checkpoint_runner.py -q
```

Expected: PASS.

- [ ] **Step 5: Run fake checkpoint smoke**

Run:

```bash
conda run -n atec python scripts/act/run_task_e_act_checkpoint.py --smoke-only --device cpu
```

Expected output includes `loaded checkpoint`, `action_chunk shape=(1, 30, 8)`, and `first_action shape=(8,)`.

---

### Task 2: Optional live Task E environment mode

**Files:**
- Modify: `scripts/act/run_task_e_act_checkpoint.py`

**Interfaces:**
- Consumes helpers from Task 1.
- Produces `run_live(args) -> None` that launches Isaac Sim, creates `ATEC-TaskE-Piper`, and steps for `--max_steps` using first denormalized ACT action from each chunk.

- [ ] **Step 1: Implement live mode**

Add AppLauncher setup, environment creation through `parse_env_cfg`, and a loop that runs at most `--max_steps` steps. Print per-step action shape for early steps and final score/elapsed/done status.

- [ ] **Step 2: Run syntax check**

Run:

```bash
python -m py_compile scripts/act/run_task_e_act_checkpoint.py
```

Expected: PASS.

- [ ] **Step 3: Run live environment smoke if Isaac can launch**

Run:

```bash
PYTHONPATH=. conda run -n atec python scripts/act/run_task_e_act_checkpoint.py \
  --max_steps 20 \
  --headless \
  --enable_cameras
```

Expected: script imports the ACT checkpoint, launches Task E, steps up to 20 times without Python exceptions, and prints final score/elapsed status. If Isaac launch fails due to environment/display constraints, report the exact launcher error and keep the fake smoke evidence.

---

## Self-Review Notes

- Spec coverage: Task 1 provides direct checkpoint loading and fake forward smoke. Task 2 provides optional live Task E execution.
- Placeholder scan: no placeholders remain; exact files, commands, and expected outputs are listed.
- Type consistency: helper signatures are defined once and used by both tests and runner.
