# Task E Self-Contained ACT Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained Task E `demo/solution.py` that loads the trained RGB ACT checkpoint and can be submitted with `policy.pt`.

**Architecture:** `demo/solution.py` will own the full inference stack: observation extraction, RGB preprocessing, ACT/DETR model reconstruction, checkpoint loading, temporal action aggregation, and `AlgSolution`. The implementation will not import repository-only modules; it will copy the inference-relevant model code into the submission file and use dependency injection in tests so most behavior can be verified without loading the large checkpoint.

**Tech Stack:** Python 3.10+, PyTorch, Torchvision, pytest, IsaacLab local evaluation through `scripts/play_atec_task.py`.

## Global Constraints

- Submission must be self-contained: `solution.py` must not import `atec_rl_lab`, `scripts.act.*`, or other repository-only modules.
- Submission artifacts are `demo/solution.py` and `demo/policy.pt`.
- `demo/policy.pt` must be copied from `runs/act-task-e-rgb-full-order-123-20260624-1925/checkpoints/best_loss.pt`.
- Runtime dependencies are limited to common ACT baseline dependencies: `torch` and `torchvision`.
- Model constants must match training: RGB enabled, ResNet18 backbone, `state_dim=8`, `action_dim=8`, `num_queries=30`, `hidden_dim=256`, `enc_layers=2`, `dec_layers=4`, `dim_feedforward=512`, `nheads=8`, `dropout=0.1`, `pre_norm=False`.
- Checkpoint loading must prefer `ema_agent` and fall back to `agent`.
- `predicts()` must always return `{"action": <8-float list>, "giveup": False}`.
- Missing checkpoint is a startup error.
- Missing or malformed per-step observation data must not crash `predicts()`; return a conservative fallback action.
- Inference uses CUDA when available, CPU otherwise.
- Do not run `git commit` unless the user has explicitly authorized commits in this session.

---

## File Structure

- Modify: `demo/solution.py`
  - Responsibility: one-file Task E submission entrypoint containing all helper functions, model classes, checkpoint wrapper, fallback behavior, and `AlgSolution`.
  - Important public test seams: `_extract_joint_state`, `_extract_video_rgb`, `_prepare_rgb`, `_fallback_action`, `_TemporalAggregator`, `_ActConfig`, `_ActAgent`, `_ActInferencePolicy`, `AlgSolution`.
- Create: `tests/task_e/test_submission_solution.py`
  - Responsibility: fast unit tests for observation preprocessing, fallback behavior, temporal aggregation, self-contained import restrictions, model forward shape, and `AlgSolution` dependency injection.
- Create local artifact: `demo/policy.pt`
  - Responsibility: local/submission copy of the trained checkpoint. This file can remain uncommitted unless the user explicitly wants binary artifacts tracked.

Useful source references while implementing the self-contained model:

- `scripts/act/train_task_e.py:208-261` — training `Agent` wrapper and image normalization behavior.
- `source/atec_rl_lab/atec_rl_lab/train/act/act/detr/detr_vae.py:33-124` — DETRVAE model.
- `source/atec_rl_lab/atec_rl_lab/train/act/act/detr/detr_vae.py:127-140` — encoder builder.
- `source/atec_rl_lab/atec_rl_lab/train/act/act/detr/backbone.py:20-128` — ResNet backbone and frozen batch norm.
- `source/atec_rl_lab/atec_rl_lab/train/act/act/detr/position_encoding.py:14-92` — sine positional embedding.
- `source/atec_rl_lab/atec_rl_lab/train/act/act/detr/transformer.py:20-302` — DETR transformer.
- `demo/server.py:89-110` — submission server observation image keys, including `video_rgb`.

---

### Task 1: Add observation preprocessing, fallback, and temporal aggregation

**Files:**
- Create: `tests/task_e/test_submission_solution.py`
- Modify: `demo/solution.py`

**Interfaces:**
- Consumes: raw `obs` dictionaries with `obs["proprio"]` and `obs["image"]["video_rgb"]` as tensors or numpy arrays.
- Produces:
  - `_extract_joint_state(obs: dict, device: torch.device) -> torch.Tensor` returning shape `(1, 8)` float32.
  - `_extract_video_rgb(obs: dict, device: torch.device) -> torch.Tensor` returning batched RGB/RGBA image tensor.
  - `_prepare_rgb(rgb: Any, device: torch.device) -> torch.Tensor` returning shape `(1, 1, 3, 224, 224)` float32 normalized RGB.
  - `_fallback_action(obs: dict | None) -> list[float]` returning exactly 8 floats.
  - `_TemporalAggregator.add(chunk: torch.Tensor) -> None`, `.current() -> torch.Tensor`, `.advance() -> None`, `.reset() -> None`.

- [ ] **Step 1: Write the failing preprocessing and aggregation tests**

Create `tests/task_e/test_submission_solution.py` with this content:

```python
import importlib
import math
from pathlib import Path

import pytest
import torch

solution = importlib.import_module("demo.solution")


def test_extract_joint_state_uses_first_eight_proprio_values():
    obs = {"proprio": torch.arange(24, dtype=torch.float32).view(1, 24)}

    state = solution._extract_joint_state(obs, torch.device("cpu"))

    assert state.shape == (1, 8)
    assert state.dtype == torch.float32
    assert state.device.type == "cpu"
    assert state.tolist() == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]


def test_extract_video_rgb_prefers_video_camera_and_strips_alpha():
    rgba = torch.zeros((1, 480, 640, 4), dtype=torch.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = 123
    obs = {"image": {"video_rgb": rgba, "ee_rgb": torch.ones((1, 480, 640, 3), dtype=torch.uint8)}}

    rgb = solution._extract_video_rgb(obs, torch.device("cpu"))

    assert rgb.shape == (1, 480, 640, 3)
    assert rgb.dtype == torch.uint8
    assert torch.all(rgb[..., 0] == 255)


def test_prepare_rgb_resizes_and_imagenet_normalizes_black_frame():
    rgb = torch.zeros((1, 480, 640, 3), dtype=torch.uint8)

    prepared = solution._prepare_rgb(rgb, torch.device("cpu"))

    assert prepared.shape == (1, 1, 3, 224, 224)
    assert prepared.dtype == torch.float32
    expected = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])
    assert torch.allclose(prepared[0, 0, :, 0, 0], expected, atol=1e-4)


def test_fallback_action_returns_current_joint_state_when_available():
    obs = {"proprio": torch.arange(10, dtype=torch.float32).view(1, 10)}

    assert solution._fallback_action(obs) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_fallback_action_returns_zeros_without_proprio():
    assert solution._fallback_action({}) == [0.0] * 8


def test_temporal_aggregator_weights_newer_predictions_more():
    agg = solution._TemporalAggregator(
        num_queries=3,
        action_dim=1,
        decay=math.log(3.0),
        device=torch.device("cpu"),
    )

    agg.add(torch.tensor([[0.0], [10.0], [20.0]]))
    assert torch.allclose(agg.current(), torch.tensor([0.0]))

    agg.advance()
    agg.add(torch.tensor([[100.0], [110.0], [120.0]]))

    expected = torch.tensor([(100.0 + 10.0 / 3.0) / (1.0 + 1.0 / 3.0)])
    assert torch.allclose(agg.current(), expected, atol=1e-5)


def test_solution_has_no_repo_only_imports():
    source = Path("demo/solution.py").read_text()

    assert "atec_rl_lab" not in source
    assert "scripts.act" not in source
```

- [ ] **Step 2: Run tests to verify they fail against the current Task D solution**

Run:

```bash
conda run -n atec pytest tests/task_e/test_submission_solution.py -q
```

Expected: FAIL with missing helper attributes such as `_extract_joint_state` or import-time behavior from the current Task D `demo/solution.py`.

- [ ] **Step 3: Add helper interfaces to `demo/solution.py`**

Replace the current Task D-specific `demo/solution.py` content with a Task E module scaffold containing these imports, constants, helper functions, aggregator, and a temporary dependency-injection-friendly `AlgSolution`. Task 2 and Task 3 will extend this same file with the ACT model and checkpoint policy.

```python
"""ATEC Task E submission: self-contained RGB ACT inference."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

_DIR = os.path.dirname(os.path.abspath(__file__))
_ACTION_DIM = 8
_IMAGE_SIZE = 224
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    return torch.as_tensor(value, device=device)


def _extract_joint_state(obs: dict, device: torch.device) -> torch.Tensor:
    proprio = obs.get("proprio") if isinstance(obs, dict) else None
    if proprio is None:
        raise KeyError("obs['proprio'] is required for Task E ACT state")
    state = _to_tensor(proprio, device).to(dtype=torch.float32)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.shape[-1] < _ACTION_DIM:
        raise ValueError(f"proprio has {state.shape[-1]} values, expected at least {_ACTION_DIM}")
    return state[:, :_ACTION_DIM]


def _extract_video_rgb(obs: dict, device: torch.device) -> torch.Tensor:
    image_obs = obs.get("image", {}) if isinstance(obs, dict) else {}
    if image_obs is None:
        image_obs = {}
    for key in ("video_rgb", "ee_rgb", "head_rgb"):
        value = image_obs.get(key) if isinstance(image_obs, dict) else None
        if value is not None:
            rgb = _to_tensor(value, device)
            if rgb.ndim == 3:
                rgb = rgb.unsqueeze(0)
            if rgb.ndim != 4:
                raise ValueError(f"{key} must have 3 or 4 dims, got shape {tuple(rgb.shape)}")
            if rgb.shape[-1] >= 3:
                return rgb[..., :3]
            if rgb.shape[1] >= 3:
                return rgb[:, :3]
            raise ValueError(f"{key} does not contain at least 3 color channels")
    raise KeyError("Task E ACT needs video_rgb, ee_rgb, or head_rgb")


def _prepare_rgb(rgb: Any, device: torch.device) -> torch.Tensor:
    x = _to_tensor(rgb, device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"RGB input must have shape BHWC or BCHW, got {tuple(x.shape)}")
    if x.shape[-1] >= 3:
        x = x[..., :3].permute(0, 3, 1, 2).contiguous()
    elif x.shape[1] >= 3:
        x = x[:, :3].contiguous()
    else:
        raise ValueError(f"RGB input has no 3-channel dimension: {tuple(x.shape)}")
    x = x.to(dtype=torch.float32)
    if x.numel() > 0 and x.max() > 1.5:
        x = x / 255.0
    x = F.interpolate(x, size=(_IMAGE_SIZE, _IMAGE_SIZE), mode="bilinear", align_corners=False)
    mean = torch.tensor(_IMAGENET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(1)


def _fallback_action(obs: dict | None) -> list[float]:
    try:
        state = _extract_joint_state(obs or {}, torch.device("cpu"))[0]
        return [float(v) for v in state.detach().cpu().tolist()[:_ACTION_DIM]]
    except Exception:
        return [0.0] * _ACTION_DIM


class _TemporalAggregator:
    def __init__(self, num_queries: int, action_dim: int, decay: float = 0.01,
                 device: torch.device | str = "cpu"):
        self.num_queries = int(num_queries)
        self.action_dim = int(action_dim)
        self.decay = float(decay)
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self._chunks: list[tuple[int, torch.Tensor]] = []

    def add(self, chunk: torch.Tensor) -> None:
        chunk = chunk.to(device=self.device, dtype=torch.float32)
        if chunk.ndim != 2:
            raise ValueError(f"chunk must have shape (num_queries, action_dim), got {tuple(chunk.shape)}")
        if chunk.shape[1] != self.action_dim:
            raise ValueError(f"chunk action_dim={chunk.shape[1]}, expected {self.action_dim}")
        self._chunks.append((self._step, chunk[:self.num_queries].detach()))
        min_start = self._step - self.num_queries + 1
        self._chunks = [(start, act) for start, act in self._chunks if start >= min_start]

    def current(self) -> torch.Tensor:
        predictions = []
        weights = []
        for start, chunk in self._chunks:
            offset = self._step - start
            if 0 <= offset < chunk.shape[0]:
                predictions.append(chunk[offset])
                age = self._step - start
                weights.append(math.exp(-self.decay * age))
        if not predictions:
            return torch.zeros(self.action_dim, device=self.device, dtype=torch.float32)
        pred = torch.stack(predictions, dim=0)
        weight = torch.tensor(weights, device=self.device, dtype=torch.float32).view(-1, 1)
        return (pred * weight).sum(dim=0) / weight.sum().clamp(min=1e-8)

    def advance(self) -> None:
        self._step += 1


class AlgSolution:
    def __init__(self, policy: Any | None = None, policy_path: str | None = None):
        self._policy = policy
        self._policy_path = policy_path

    def reset(self, **kwargs) -> None:
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()

    def predicts(self, obs: dict, current_score: float):
        if self._policy is None:
            action = _fallback_action(obs)
        else:
            try:
                action = self._policy.predict(obs)
            except Exception:
                action = _fallback_action(obs)
        return {"action": [float(x) for x in action[:_ACTION_DIM]], "giveup": False}
```

- [ ] **Step 4: Run tests to verify Task 1 passes**

Run:

```bash
conda run -n atec pytest tests/task_e/test_submission_solution.py -q
```

Expected: PASS for the preprocessing, fallback, aggregation, and static import tests. The model and checkpoint tests are added in later tasks.

- [ ] **Step 5: Commit Task 1 if commits are authorized**

Run only after explicit commit authorization:

```bash
git add demo/solution.py tests/task_e/test_submission_solution.py
git commit -m "test: add task e submission preprocessing"
```

Expected: commit succeeds. If commits are not authorized, leave the files modified and continue.

---

### Task 2: Embed the self-contained ACT/DETR model

**Files:**
- Modify: `demo/solution.py`
- Modify: `tests/task_e/test_submission_solution.py`

**Interfaces:**
- Consumes: `_prepare_rgb()` output shaped `(B, 1, 3, 224, 224)` and normalized state shaped `(B, 8)`.
- Produces:
  - `_ActConfig` with architecture constants.
  - `_ActAgent(state_dim: int, act_dim: int, cfg: _ActConfig)` with `get_action(obs: dict[str, torch.Tensor]) -> torch.Tensor` returning `(B, 30, 8)`.
  - Model parameter names compatible with checkpoint keys under `agent` and `ema_agent`.

- [ ] **Step 1: Add the model forward shape test**

Append this test to `tests/task_e/test_submission_solution.py`:

```python
def test_act_agent_forward_shape_cpu():
    cfg = solution._ActConfig()
    model = solution._ActAgent(state_dim=8, act_dim=8, cfg=cfg).eval()
    obs = {
        "state": torch.zeros((1, 8), dtype=torch.float32),
        "rgb": torch.zeros((1, 1, 3, 224, 224), dtype=torch.float32),
    }

    with torch.inference_mode():
        action_chunk = model.get_action(obs)

    assert action_chunk.shape == (1, cfg.num_queries, 8)
    assert action_chunk.dtype == torch.float32
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```bash
conda run -n atec pytest tests/task_e/test_submission_solution.py::test_act_agent_forward_shape_cpu -q
```

Expected: FAIL because `_ActConfig` and `_ActAgent` are not defined yet.

- [ ] **Step 3: Add model code to `demo/solution.py`**

Add these implementation units below `_TemporalAggregator` and above `AlgSolution`:

```python
@dataclass(frozen=True)
class _ActConfig:
    position_embedding: str = "sine"
    backbone: str = "resnet18"
    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False
    include_depth: bool = False
    include_rgb: bool = True
    enc_layers: int = 2
    dec_layers: int = 4
    dim_feedforward: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    num_queries: int = 30
    pre_norm: bool = False


class _PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor
        not_mask = torch.ones_like(x[0, [0]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class _FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "num_batches_tracked"
        if key in state_dict:
            del state_dict[key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + 1e-5).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
```

Then add the remaining model classes by copying the source bodies from these exact references into `demo/solution.py`, renaming the classes/functions with a leading underscore and replacing local package imports with the local names above:

```text
source/atec_rl_lab/atec_rl_lab/train/act/act/detr/backbone.py:59-128
source/atec_rl_lab/atec_rl_lab/train/act/act/detr/transformer.py:20-302
source/atec_rl_lab/atec_rl_lab/train/act/act/detr/detr_vae.py:16-124
source/atec_rl_lab/atec_rl_lab/train/act/act/detr/detr_vae.py:127-140
scripts/act/train_task_e.py:208-261
```

Apply these exact adaptations while copying:

```python
# Backbone construction must not download pretrained weights in submission.
backbone = getattr(torchvision.models, name)(
    weights=None,
    replace_stride_with_dilation=[False, False, dilation],
    norm_layer=_FrozenBatchNorm2d,
)
```

```python
# build_position_encoding replacement.
def _build_position_encoding(cfg: _ActConfig):
    n_steps = cfg.hidden_dim // 2
    if cfg.position_embedding in ("v2", "sine"):
        return _PositionEmbeddingSine(n_steps, normalize=True)
    raise ValueError(f"not supported position embedding: {cfg.position_embedding}")
```

```python
# _ActAgent must keep parameter prefix "model." for checkpoint compatibility.
class _ActAgent(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, cfg: _ActConfig):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.include_rgb = cfg.include_rgb
        backbones = [_build_backbone(cfg)] if cfg.include_rgb else None
        transformer = _build_transformer(cfg)
        encoder = _build_encoder(cfg)
        self.model = _DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=state_dim,
            action_dim=act_dim,
            num_queries=cfg.num_queries,
        )

    def get_action(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        a_hat, _ = self.model(obs if self.include_rgb else obs["state"])
        return a_hat
```

The copied code must not leave imports of `IPython`, `NestedTensor`, `is_main_process`, `atec_rl_lab`, or `scripts.act`.

- [ ] **Step 4: Run Task 2 tests**

Run:

```bash
conda run -n atec pytest tests/task_e/test_submission_solution.py::test_act_agent_forward_shape_cpu -q
conda run -n atec pytest tests/task_e/test_submission_solution.py::test_solution_has_no_repo_only_imports -q
```

Expected: both PASS. The model shape test may take several seconds on CPU.

- [ ] **Step 5: Commit Task 2 if commits are authorized**

Run only after explicit commit authorization:

```bash
git add demo/solution.py tests/task_e/test_submission_solution.py
git commit -m "feat: embed task e act model"
```

Expected: commit succeeds. If commits are not authorized, leave the files modified and continue.

---

### Task 3: Add checkpoint loading, `AlgSolution` wiring, and policy artifact

**Files:**
- Modify: `demo/solution.py`
- Modify: `tests/task_e/test_submission_solution.py`
- Create local artifact: `demo/policy.pt`

**Interfaces:**
- Consumes: `policy.pt` checkpoint dict with `norm_stats`, `ema_agent`, and optionally `agent`.
- Produces:
  - `_find_policy_path(explicit_path: str | None = None) -> str`.
  - `_ActInferencePolicy(policy_path: str | None = None, device: torch.device | str | None = None)` with `.predict(obs: dict) -> list[float]` and `.reset() -> None`.
  - `AlgSolution(policy: Any | None = None, policy_path: str | None = None)` that loads `_ActInferencePolicy` by default.

- [ ] **Step 1: Add `AlgSolution` dependency-injection and fallback tests**

Append this test code to `tests/task_e/test_submission_solution.py`:

```python
class _FakePolicy:
    def __init__(self):
        self.reset_called = False

    def reset(self):
        self.reset_called = True

    def predict(self, obs):
        return [0.25] * 8


class _BrokenPolicy:
    def predict(self, obs):
        raise RuntimeError("synthetic inference failure")


def test_algsolution_uses_injected_policy_and_reset():
    policy = _FakePolicy()
    agent = solution.AlgSolution(policy=policy)

    resp = agent.predicts(obs={}, current_score=0.0)
    agent.reset()

    assert resp == {"action": [0.25] * 8, "giveup": False}
    assert policy.reset_called is True


def test_algsolution_returns_fallback_action_when_policy_raises():
    agent = solution.AlgSolution(policy=_BrokenPolicy())
    obs = {"proprio": torch.arange(10, dtype=torch.float32).view(1, 10)}

    resp = agent.predicts(obs=obs, current_score=0.0)

    assert resp == {"action": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "giveup": False}
```

- [ ] **Step 2: Run tests to verify default checkpoint wrapper is still missing**

Run:

```bash
conda run -n atec pytest tests/task_e/test_submission_solution.py::test_algsolution_uses_injected_policy_and_reset tests/task_e/test_submission_solution.py::test_algsolution_returns_fallback_action_when_policy_raises -q
```

Expected: The injection tests may already pass from Task 1. If they pass, continue; they protect the fallback behavior while adding real checkpoint loading.

- [ ] **Step 3: Add checkpoint policy code to `demo/solution.py`**

Replace the temporary `AlgSolution` from Task 1 with this final wrapper, and add `_find_policy_path` plus `_ActInferencePolicy` immediately above it:

```python
def _find_policy_path(explicit_path: str | None = None) -> str:
    candidates: list[str] = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates.extend([
        os.path.join(_DIR, "policy.pt"),
        os.path.join(_DIR, "best_loss.pt"),
        os.path.abspath(os.path.join(
            _DIR,
            "..",
            "runs",
            "act-task-e-rgb-full-order-123-20260624-1925",
            "checkpoints",
            "best_loss.pt",
        )),
    ])
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError("Task E ACT checkpoint not found; expected demo/policy.pt")


class _ActInferencePolicy:
    def __init__(self, policy_path: str | None = None, device: torch.device | str | None = None):
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.policy_path = _find_policy_path(policy_path)
        ckpt = torch.load(self.policy_path, map_location=self.device)
        self.norm_stats = {
            key: value.to(device=self.device, dtype=torch.float32)
            for key, value in ckpt["norm_stats"].items()
        }
        self.state_dim = int(self.norm_stats["state_mean"].shape[-1])
        self.action_dim = int(self.norm_stats["action_mean"].shape[-1])
        self.cfg = _ActConfig()
        self.agent = _ActAgent(self.state_dim, self.action_dim, self.cfg).to(self.device)
        state_dict = ckpt.get("ema_agent") or ckpt["agent"]
        self.agent.load_state_dict(state_dict, strict=True)
        self.agent.eval()
        self.aggregator = _TemporalAggregator(
            num_queries=self.cfg.num_queries,
            action_dim=self.action_dim,
            decay=0.01,
            device=self.device,
        )

    def reset(self) -> None:
        self.aggregator.reset()

    def predict(self, obs: dict) -> list[float]:
        state = _extract_joint_state(obs, self.device)
        rgb = _prepare_rgb(_extract_video_rgb(obs, self.device), self.device)
        state = (state - self.norm_stats["state_mean"]) / self.norm_stats["state_std"].clamp(min=1e-2)
        with torch.inference_mode():
            chunk = self.agent.get_action({"state": state, "rgb": rgb})[0]
        chunk = chunk * self.norm_stats["action_std"].clamp(min=1e-2) + self.norm_stats["action_mean"]
        self.aggregator.add(chunk)
        action = self.aggregator.current()
        self.aggregator.advance()
        return [float(v) for v in action.detach().cpu().tolist()[:_ACTION_DIM]]


class AlgSolution:
    def __init__(self, policy: Any | None = None, policy_path: str | None = None):
        self._policy = policy if policy is not None else _ActInferencePolicy(policy_path=policy_path)
        self._last_error: str | None = None

    def reset(self, **kwargs) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()
        self._last_error = None

    def predicts(self, obs: dict, current_score: float):
        try:
            action = self._policy.predict(obs)
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            action = _fallback_action(obs)
        if len(action) < _ACTION_DIM:
            action = list(action) + [0.0] * (_ACTION_DIM - len(action))
        return {"action": [float(x) for x in action[:_ACTION_DIM]], "giveup": False}
```

- [ ] **Step 4: Copy the trained checkpoint into the submission artifact path**

Run:

```bash
cp runs/act-task-e-rgb-full-order-123-20260624-1925/checkpoints/best_loss.pt demo/policy.pt
ls -lh demo/policy.pt
```

Expected: `demo/policy.pt` exists and has non-zero size.

- [ ] **Step 5: Run unit tests**

Run:

```bash
conda run -n atec pytest tests/task_e/test_submission_solution.py -q
```

Expected: all tests PASS.

- [ ] **Step 6: Run checkpoint load smoke test**

Run:

```bash
conda run -n atec python -c "from demo.solution import AlgSolution; a=AlgSolution(); print(type(a).__name__)"
```

Expected output includes:

```text
AlgSolution
```

- [ ] **Step 7: Run fake-observation inference smoke test**

Run:

```bash
conda run -n atec python - <<'PY'
import torch
from demo.solution import AlgSolution
agent = AlgSolution()
obs = {
    "proprio": torch.zeros((1, 24), dtype=torch.float32),
    "image": {"video_rgb": torch.zeros((1, 480, 640, 3), dtype=torch.uint8)},
}
resp = agent.predicts(obs, 0.0)
print(resp["giveup"], len(resp["action"]), all(isinstance(x, float) for x in resp["action"]))
PY
```

Expected output:

```text
False 8 True
```

- [ ] **Step 8: Commit Task 3 if commits are authorized**

Run only after explicit commit authorization. Do not add `demo/policy.pt` unless the user explicitly wants binary artifacts committed:

```bash
git add demo/solution.py tests/task_e/test_submission_solution.py
git commit -m "feat: load task e act checkpoint"
```

Expected: commit succeeds. If commits are not authorized, leave the files modified and continue.

---

### Task 4: Run local verification and prepare submission notes

**Files:**
- Modify: no code changes expected unless verification exposes a defect.
- Read/verify: `demo/solution.py`, `demo/policy.pt`, `tests/task_e/test_submission_solution.py`.

**Interfaces:**
- Consumes: completed Task 3 implementation.
- Produces: verified local submission file set and exact commands/results for the user.

- [ ] **Step 1: Run Python syntax check**

Run:

```bash
python -m py_compile demo/solution.py
```

Expected: command exits with status 0 and prints no errors.

- [ ] **Step 2: Run focused unit tests**

Run:

```bash
conda run -n atec pytest tests/task_e/test_submission_solution.py -q
```

Expected: all tests PASS.

- [ ] **Step 3: Run all ACT/Task E lightweight tests**

Run:

```bash
conda run -n atec pytest tests/act tests/task_e -q
```

Expected: all tests PASS. If unrelated pre-existing tests fail, record the exact failures and do not claim the whole suite passes.

- [ ] **Step 4: Run local Task E evaluation if Isaac Sim can launch**

Run:

```bash
PYTHONPATH=. conda run -n atec python scripts/play_atec_task.py \
  --task ATEC-TaskE-Piper \
  --num_envs 1 \
  --enable_cameras \
  --debug
```

Expected: the program imports `demo.solution.AlgSolution`, steps the Task E Piper environment, and exits without Python exceptions. Record final `score:` and `elapsed_time:` from stdout. If Isaac Sim cannot launch in this environment, record the launcher error and keep the unit/checkpoint smoke evidence.

- [ ] **Step 5: Confirm submission file set**

Run:

```bash
ls -lh demo/solution.py demo/policy.pt
python - <<'PY'
from pathlib import Path
source = Path('demo/solution.py').read_text()
print('has AlgSolution:', 'class AlgSolution' in source)
print('blocked imports:', any(s in source for s in ['atec_rl_lab', 'scripts.act']))
PY
```

Expected output includes:

```text
has AlgSolution: True
blocked imports: False
```

- [ ] **Step 6: Report verification evidence**

Report these exact items to the user:

```text
Submission files:
- demo/solution.py
- demo/policy.pt

Verification run:
- python -m py_compile demo/solution.py: <pass/fail>
- conda run -n atec pytest tests/task_e/test_submission_solution.py -q: <pass/fail>
- conda run -n atec pytest tests/act tests/task_e -q: <pass/fail or skipped with reason>
- local Task E play command: <score/elapsed or skipped/failed with exact reason>
```

- [ ] **Step 7: Commit verification-related changes if commits are authorized**

If a defect was fixed during verification and commits are authorized, run:

```bash
git add demo/solution.py tests/task_e/test_submission_solution.py
git commit -m "fix: verify task e submission solution"
```

Expected: commit succeeds. If no code changed during verification, no commit is needed.

---

## Self-Review Notes

- Spec coverage: Task 1 covers observation extraction, RGB preprocessing, fallback behavior, and temporal aggregation. Task 2 covers self-contained ACT/DETR model reconstruction and no local repository imports. Task 3 covers checkpoint path, `ema_agent` preference, `AlgSolution`, and `demo/policy.pt`. Task 4 covers syntax, unit, checkpoint, and local evaluation verification.
- Placeholder scan: the plan contains concrete file paths, code snippets, commands, and expected outputs. No open-ended implementation markers are intentionally left.
- Type consistency: helper names used in tests match the produced interfaces; `AlgSolution(policy=...)` dependency injection is preserved while default construction loads `_ActInferencePolicy`; action outputs are always 8-float Python lists.
