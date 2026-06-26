# Task E self-contained ACT submission design

Date: 2026-06-24

## Goal

Build a competition-ready Task E `demo/solution.py` that can be submitted as a self-contained Python solution using the trained RGB ACT checkpoint:

```text
runs/act-task-e-rgb-full-order-123-20260624-1925/checkpoints/best_loss.pt
```

The submission must not import repository-only modules such as `atec_rl_lab` or `scripts.act.*`. It should run from the flat `demo/` submission surface with the active entrypoint `AlgSolution`.

## Submission artifacts

The submission package contains:

- `solution.py` — self-contained inference implementation defining `AlgSolution`.
- `policy.pt` — copied from `runs/act-task-e-rgb-full-order-123-20260624-1925/checkpoints/best_loss.pt`.

`solution.py` may depend on common runtime packages already needed by the ACT baseline, specifically `torch` and `torchvision`, but it must not depend on local repository packages.

## Recommended approach

Use a self-contained ACT inference implementation. The solution reconstructs the same model family used by `scripts/act/train_task_e.py`, loads the checkpoint in-process, preprocesses Task E observations, and returns 8-dimensional Piper joint-position actions.

This is preferred over exporting TorchScript because it avoids TorchScript compatibility risk for the DETR/Transformer/ResNet stack, and preferred over a scripted state-machine solution because submission observations do not expose the privileged object poses and IK helpers used during data collection.

## Runtime data flow

Each call to `AlgSolution.predicts(obs, current_score)` performs this flow:

1. Extract the current Piper joint state from `obs["proprio"]`.
   - The training dataset stores `robot.data.joint_pos` as `traj_N/obs`.
   - In submission, the first 8 proprio values are treated as joint positions for the Piper arm and gripper.
2. Extract RGB from the Task E global camera.
   - The primary expected key is `obs["image"]["video_rgb"]`.
   - The implementation should tolerate tensor or numpy inputs, batched or unbatched layout, and RGB or RGBA channel count.
3. Resize RGB to `224x224`, convert to `(B, 1, 3, 224, 224)`, scale to `[0, 1]`, and apply ImageNet normalization with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
4. Normalize state using checkpoint `norm_stats["state_mean"]` and `norm_stats["state_std"]`.
5. Run the ACT model with zero latent at inference to obtain a `(1, num_queries, action_dim)` action chunk.
6. Denormalize the predicted actions using checkpoint `norm_stats["action_mean"]` and `norm_stats["action_std"]`.
7. Apply temporal aggregation over overlapping action chunks, then return the aggregated 8-dimensional action as:

```python
{"action": action_list, "giveup": False}
```

## Model reconstruction

`solution.py` embeds only the inference-relevant model code:

- ResNet18 visual backbone with frozen batch norm.
- Sine positional encoding.
- DETR-style transformer encoder/decoder.
- ACT/DETRVAE wrapper with state projection, action head, learned queries, and zero latent inference path.

The architecture constants must match the training checkpoint:

- `include_rgb = True`
- `backbone = "resnet18"`
- `state_dim = 8` inferred from `state_mean`
- `action_dim = 8` inferred from `action_mean`
- `num_queries = 30`
- `hidden_dim = 256`
- `enc_layers = 2`
- `dec_layers = 4`
- `dim_feedforward = 512`
- `nheads = 8`
- `dropout = 0.1`
- `pre_norm = False`

Checkpoint loading prefers `ema_agent` and falls back to `agent` if the EMA weights are absent.

## Temporal aggregation

The inference wrapper maintains a fixed-size buffer of predicted action chunks. At step `t`, the newly predicted chunk contributes actions for timesteps `t` through `t + num_queries - 1`. The returned action for the current timestep is an exponentially weighted average of all valid predictions for that timestep, with newer predictions weighted more strongly.

`reset()` clears the chunk buffer and step counter.

## Robustness and failure behavior

- Weight lookup order:
  1. `demo/policy.pt`
  2. `demo/best_loss.pt`
  3. the original run checkpoint path, for local development convenience only.
- Missing checkpoint is a startup error, because a silent zero-action submission would be misleading.
- Missing or malformed per-step image/state input should not crash the evaluator. In that case, the solution returns a conservative fallback action, using the current 8-dimensional joint-position estimate when available, otherwise zeros.
- All inference uses `torch.inference_mode()` and selects CUDA when available, CPU otherwise.
- The returned action is always a plain Python list of 8 floats.

## Validation plan

Before calling the solution complete, run:

1. Python syntax check:
   ```bash
   python -m py_compile demo/solution.py
   ```
2. Checkpoint load smoke test in the `atec` conda environment:
   ```bash
   conda run -n atec python - <<'PY'
   from demo.solution import AlgSolution
   agent = AlgSolution()
   print(type(agent).__name__)
   PY
   ```
3. One fake-observation inference smoke test that verifies `predicts()` returns `giveup=False` and an 8-float action.
4. If Isaac Sim can launch, run local Task E evaluation:
   ```bash
   PYTHONPATH=. conda run -n atec python scripts/play_atec_task.py \
     --task ATEC-TaskE-Piper \
     --num_envs 1 \
     --enable_cameras \
     --debug
   ```

## Out of scope

- Retraining the policy.
- Re-exporting or changing checkpoint format unless the direct checkpoint loading path proves impossible.
- Implementing visual object detection or IK-based scripted manipulation in the submission solution.
- Changing Task E environment configuration or data-collection code.
