# Task E ACT Grasp Data Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Commit steps are included as checkpoints; execute commit commands only if the user has authorized commits in the execution session.

**Goal:** Build a validated Task E ACT data-collection workflow that sweeps grasp offsets separately for the box, bottle, and banana, verifies the optimized full-order state machine, then collects at least 100 successful full pick-and-place demonstrations.

**Architecture:** Keep the current scripted-oracle ACT pipeline, but make its state machine configurable through explicit step tables and per-object grasp offsets. Add pure helper functions for CLI/object-offset resolution, a reusable Task E environment builder, a generic sweep script, and collection-script safety gates for fixed `[1, 2, 3]` order and max-attempt validation.

**Tech Stack:** Python, argparse, Isaac Lab `AppLauncher`, Isaac Lab `ManagerBasedRLEnv`, HDF5 via `h5py`, PyTorch tensors, pytest-style unit tests with lightweight module stubs.

---

## File Structure

Create or modify these files:

- Create: `scripts/act/task_e/options.py`
  - Pure helper functions for parsing object IDs, per-object offsets, sweep JSON, and best-offset selection.
  - No Isaac Lab imports; safe to unit test.

- Create: `scripts/act/task_e/env_setup.py`
  - Reusable `build_task_e_env()` function shared by collection and sweep scripts.
  - Contains Isaac Lab environment construction currently embedded in `collect_demos_task_e.py`.

- Modify: `scripts/act/task_e/config.py`
  - Add `OPTIMIZED_STEPS` and `OBJECT_GRASP_Z_OFFSETS`.
  - Export the new constants through `__all__`.

- Modify: `scripts/act/task_e/state_machine.py`
  - Allow `PickPlaceStateMachine` to accept a custom step table and per-object grasp offsets.
  - Use object-specific offsets during `REACH` and `CLOSE`.

- Modify: `scripts/act/task_e/collector.py`
  - Add `get_objects_in_basket()` for per-object success diagnostics.
  - Keep `check_objects_in_basket()` as a compatibility wrapper.
  - Pass configured steps and offsets into `PickPlaceStateMachine`.

- Modify: `scripts/act/cli_args.py`
  - Add `--full_order_123`, `--optimized_grasp_flow`, `--grasp_z_offsets`, `--grasp_offset_json`, and `--max_attempts`.

- Modify: `scripts/act/collect_demos_task_e.py`
  - Use `build_task_e_env()`.
  - Resolve full-order mode, optimized steps, and offsets.
  - Enforce max attempts for validation runs.
  - Print per-object success diagnostics.

- Create: `scripts/act/sweep_task_e_grasp_offsets.py`
  - Sweep offsets independently for object 1, 2, and 3.
  - Save `best_offsets` and full sweep results to JSON.

- Create: `tests/act/test_task_e_options.py`
  - Unit tests for pure option parsing and best-offset selection.

- Create: `tests/act/test_task_e_state_machine.py`
  - Unit tests for configurable state-machine steps and per-object offsets, using module stubs.

- Create: `tests/act/test_task_e_collector_success.py`
  - Unit tests for per-object basket success reporting, using fake env/scene objects.

---

### Task 1: Add Pure Option Helpers

**Files:**
- Create: `tests/act/test_task_e_options.py`
- Create: `scripts/act/task_e/options.py`

- [ ] **Step 1: Write failing tests for option parsing and best-offset selection**

Create `tests/act/test_task_e_options.py` with:

```python
from pathlib import Path
import json
import sys

_REPO = Path(__file__).resolve().parents[2]
_ACT_DIR = _REPO / "scripts" / "act"
if str(_ACT_DIR) not in sys.path:
    sys.path.insert(0, str(_ACT_DIR))

from task_e.options import (  # noqa: E402
    choose_best_offset,
    load_grasp_offset_json,
    parse_grasp_z_offsets,
    resolve_grasp_z_offsets,
    resolve_pick_objects,
)


def test_parse_grasp_z_offsets_accepts_object_colon_value_pairs():
    offsets = parse_grasp_z_offsets(["1:0.080", "2:0.085", "3:0.075"])

    assert offsets == {1: 0.080, 2: 0.085, 3: 0.075}


def test_parse_grasp_z_offsets_rejects_unknown_object_id():
    try:
        parse_grasp_z_offsets(["4:0.080"])
    except ValueError as exc:
        assert "object id must be one of [1, 2, 3]" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_parse_grasp_z_offsets_rejects_bad_pair_format():
    try:
        parse_grasp_z_offsets(["1=0.080"])
    except ValueError as exc:
        assert "expected OBJECT_ID:OFFSET" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_resolve_pick_objects_forces_full_order():
    assert resolve_pick_objects([3], full_order_123=True) == [1, 2, 3]


def test_resolve_pick_objects_sorts_and_deduplicates_normal_mode():
    assert resolve_pick_objects([3, 1, 3, 2], full_order_123=False) == [1, 2, 3]


def test_load_grasp_offset_json_reads_best_offsets(tmp_path):
    path = tmp_path / "sweep.json"
    path.write_text(
        json.dumps({"best_offsets": {"1": 0.080, "2": 0.085, "3": 0.075}}),
        encoding="utf-8",
    )

    assert load_grasp_offset_json(str(path)) == {1: 0.080, 2: 0.085, 3: 0.075}


def test_resolve_grasp_z_offsets_prefers_cli_over_json(tmp_path):
    path = tmp_path / "sweep.json"
    path.write_text(
        json.dumps({"best_offsets": {"1": 0.070, "2": 0.070, "3": 0.070}}),
        encoding="utf-8",
    )

    offsets = resolve_grasp_z_offsets(
        cli_offsets=["1:0.080", "2:0.085"],
        json_path=str(path),
        defaults={1: 0.090, 2: 0.090, 3: 0.090},
    )

    assert offsets == {1: 0.080, 2: 0.085, 3: 0.090}


def test_choose_best_offset_uses_success_rate_then_middle_tie_breaker():
    rows = [
        {"offset": 0.070, "attempts": 5, "successes": 4, "success_rate": 0.8},
        {"offset": 0.075, "attempts": 5, "successes": 5, "success_rate": 1.0},
        {"offset": 0.080, "attempts": 5, "successes": 5, "success_rate": 1.0},
        {"offset": 0.085, "attempts": 5, "successes": 5, "success_rate": 1.0},
        {"offset": 0.090, "attempts": 5, "successes": 3, "success_rate": 0.6},
    ]

    assert choose_best_offset(rows) == 0.080
```

- [ ] **Step 2: Run tests and verify they fail because the helper module does not exist**

Run:

```bash
pytest tests/act/test_task_e_options.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'task_e.options'`.

- [ ] **Step 3: Implement `scripts/act/task_e/options.py`**

Create `scripts/act/task_e/options.py` with:

```python
"""Pure option helpers for Task E ACT data collection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

FULL_ORDER_OBJECTS = [1, 2, 3]
VALID_OBJECT_IDS = set(FULL_ORDER_OBJECTS)


def _validate_object_id(obj_id: int) -> int:
    if obj_id not in VALID_OBJECT_IDS:
        raise ValueError(f"object id must be one of {FULL_ORDER_OBJECTS}, got {obj_id}")
    return obj_id


def resolve_pick_objects(pick_objects: Iterable[int], full_order_123: bool) -> list[int]:
    """Return the object order requested by CLI options."""
    if full_order_123:
        return FULL_ORDER_OBJECTS.copy()

    resolved = sorted({_validate_object_id(int(obj_id)) for obj_id in pick_objects})
    if not resolved:
        raise ValueError("at least one object id is required")
    return resolved


def parse_grasp_z_offsets(items: Iterable[str] | None) -> dict[int, float]:
    """Parse CLI pairs such as ['1:0.080', '2:0.085']."""
    offsets: dict[int, float] = {}
    if not items:
        return offsets

    for item in items:
        if ":" not in item:
            raise ValueError(f"expected OBJECT_ID:OFFSET, got {item!r}")
        obj_text, offset_text = item.split(":", 1)
        try:
            obj_id = _validate_object_id(int(obj_text))
            offset = float(offset_text)
        except ValueError as exc:
            if "object id" in str(exc):
                raise
            raise ValueError(f"expected OBJECT_ID:OFFSET, got {item!r}") from exc
        offsets[obj_id] = offset
    return offsets


def load_grasp_offset_json(path: str | Path | None) -> dict[int, float]:
    """Load best_offsets from a sweep JSON file."""
    if path is None:
        return {}

    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    raw_offsets = payload.get("best_offsets")
    if not isinstance(raw_offsets, dict):
        raise ValueError(f"{json_path} does not contain a best_offsets object")

    offsets: dict[int, float] = {}
    for obj_text, offset in raw_offsets.items():
        obj_id = _validate_object_id(int(obj_text))
        offsets[obj_id] = float(offset)
    return offsets


def resolve_grasp_z_offsets(
    cli_offsets: Iterable[str] | None,
    json_path: str | Path | None,
    defaults: Mapping[int, float] | None = None,
) -> dict[int, float]:
    """Resolve per-object grasp offsets with CLI values taking precedence."""
    resolved: dict[int, float] = {}
    if defaults:
        resolved.update({int(k): float(v) for k, v in defaults.items()})
    resolved.update(load_grasp_offset_json(json_path))
    resolved.update(parse_grasp_z_offsets(cli_offsets))
    for obj_id in list(resolved):
        _validate_object_id(int(obj_id))
    return resolved


def choose_best_offset(rows: list[dict]) -> float:
    """Choose the best offset from sweep result rows.

    Rows must contain 'offset' and either 'success_rate' or both 'successes' and
    'attempts'. Ties use the middle offset among tied candidates.
    """
    if not rows:
        raise ValueError("cannot choose a best offset from zero rows")

    normalized = []
    for row in rows:
        offset = float(row["offset"])
        if "success_rate" in row:
            rate = float(row["success_rate"])
        else:
            attempts = int(row["attempts"])
            successes = int(row["successes"])
            rate = successes / attempts if attempts > 0 else 0.0
        early_terminated = int(row.get("early_terminations", 0))
        normalized.append((rate, -early_terminated, offset))

    best_rate = max(rate for rate, _, _ in normalized)
    tied = sorted(
        item for item in normalized if item[0] == best_rate
    )
    return tied[len(tied) // 2][2]
```

- [ ] **Step 4: Run helper tests and verify they pass**

Run:

```bash
pytest tests/act/test_task_e_options.py -q
```

Expected: PASS, 8 tests.

- [ ] **Step 5: Commit checkpoint if commits are authorized**

```bash
git add scripts/act/task_e/options.py tests/act/test_task_e_options.py
git commit -m "feat: add task e grasp option helpers"
```

---

### Task 2: Make the State Machine Configurable

**Files:**
- Create: `tests/act/test_task_e_state_machine.py`
- Modify: `scripts/act/task_e/config.py`
- Modify: `scripts/act/task_e/state_machine.py`

- [ ] **Step 1: Write failing state-machine tests**

Create `tests/act/test_task_e_state_machine.py` with:

```python
from pathlib import Path
from types import ModuleType
import importlib
import sys

import torch

_REPO = Path(__file__).resolve().parents[2]
_ACT_DIR = _REPO / "scripts" / "act"
if str(_ACT_DIR) not in sys.path:
    sys.path.insert(0, str(_ACT_DIR))


def _install_stubs():
    env_cfg = ModuleType("atec_rl_lab.tasks.task_e.env_cfg")
    env_cfg.BASKET_CENTER_X = 0.40
    env_cfg.BASKET_CENTER_Y = 0.00
    env_cfg.TABLE_CENTER_X = 0.50
    env_cfg.TABLE_CENTER_Y = 0.00
    env_cfg.TABLE_TOP_Z = 0.20
    env_cfg.TABLE_HALF_X = 0.30
    env_cfg.BASKET_EXCL_HALF_X = 0.20
    env_cfg.BASKET_EXCL_HALF_Y = 0.11

    sys.modules.setdefault("atec_rl_lab", ModuleType("atec_rl_lab"))
    sys.modules.setdefault("atec_rl_lab.tasks", ModuleType("atec_rl_lab.tasks"))
    sys.modules.setdefault("atec_rl_lab.tasks.task_e", ModuleType("atec_rl_lab.tasks.task_e"))
    sys.modules["atec_rl_lab.tasks.task_e.env_cfg"] = env_cfg

    math_mod = ModuleType("isaaclab.utils.math")
    math_mod.matrix_from_quat = lambda quat: torch.eye(3).repeat(quat.shape[0], 1, 1)
    math_mod.quat_from_matrix = lambda matrix: torch.tensor(
        [[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32
    ).repeat(matrix.shape[0], 1)

    sys.modules.setdefault("isaaclab", ModuleType("isaaclab"))
    sys.modules.setdefault("isaaclab.utils", ModuleType("isaaclab.utils"))
    sys.modules["isaaclab.utils.math"] = math_mod


_install_stubs()
state_machine = importlib.import_module("task_e.state_machine")
config = importlib.import_module("task_e.config")
PickPlaceStateMachine = state_machine.PickPlaceStateMachine


def _single_step_table():
    return {state: 1 for state in config.STATE_ORDER}


def test_state_machine_uses_per_object_grasp_offset_for_reach_and_close():
    sm = PickPlaceStateMachine(
        [1],
        "cpu",
        steps=_single_step_table(),
        grasp_z_offsets={1: 0.123},
    )

    obj_pos = torch.tensor([1.0, 2.0, 0.30])
    sm.tick(obj_pos)  # INIT
    sm.tick(obj_pos)  # PRE_GRASP caches object position

    reach_pos, _, reach_gripper = sm.tick(torch.tensor([9.0, 9.0, 0.90]))
    close_pos, _, close_gripper = sm.tick(torch.tensor([8.0, 8.0, 0.80]))

    assert torch.allclose(reach_pos, torch.tensor([1.0, 2.0, 0.423]))
    assert reach_gripper == "open"
    assert torch.allclose(close_pos, torch.tensor([1.0, 2.0, 0.423]))
    assert close_gripper == "close"


def test_state_machine_falls_back_to_global_grasp_offset():
    sm = PickPlaceStateMachine([2], "cpu", steps=_single_step_table())

    obj_pos = torch.tensor([1.0, 2.0, 0.30])
    sm.tick(obj_pos)  # INIT
    sm.tick(obj_pos)  # PRE_GRASP
    reach_pos, _, _ = sm.tick(obj_pos)  # REACH

    expected_z = 0.30 + config.GRASP_Z_OFFSET
    assert torch.allclose(reach_pos, torch.tensor([1.0, 2.0, expected_z]))


def test_state_machine_advances_to_next_object_after_retract():
    sm = PickPlaceStateMachine([1, 2], "cpu", steps=_single_step_table())
    obj_pos = torch.tensor([1.0, 2.0, 0.30])

    visited = []
    while not sm.done:
        visited.append((sm.state, sm.current_object_key))
        sm.tick(obj_pos)

    assert ("PRE_GRASP", "object_1") in visited
    assert ("RETRACT", "object_1") in visited
    assert ("PRE_GRASP", "object_2") in visited
    assert ("RETRACT", "object_2") in visited
```

- [ ] **Step 2: Run tests and verify they fail because constructor args are not supported**

Run:

```bash
pytest tests/act/test_task_e_state_machine.py -q
```

Expected: FAIL with `TypeError` mentioning unexpected keyword argument `steps` or `grasp_z_offsets`.

- [ ] **Step 3: Add optimized config constants**

Modify `scripts/act/task_e/config.py`.

Add these names to `__all__` near the existing state-machine exports:

```python
    "STEPS", "OPTIMIZED_STEPS", "STATE_ORDER", "OBJECT_GRASP_Z_OFFSETS",
```

Replace the existing state-machine block with this complete block:

```python
# ------------------------------------------------------------------ #
# State-machine
# ------------------------------------------------------------------ #
STEPS: dict[str, int] = {
    "INIT":         100,
    "PRE_GRASP":    200,
    "REACH":        100,
    "CLOSE":         40,
    "LIFT":         160,
    "TRANSPORT":    200,
    "PLACE":         80,
    "OPEN":          60,
    "LIFT_RETRACT":  80,
    "RETRACT":       80,
}

OPTIMIZED_STEPS: dict[str, int] = {
    "INIT":          60,
    "PRE_GRASP":   120,
    "REACH":       100,
    "CLOSE":        60,
    "LIFT":        120,
    "TRANSPORT":   150,
    "PLACE":        70,
    "OPEN":         50,
    "LIFT_RETRACT": 60,
    "RETRACT":      70,
}

STATE_ORDER = ["INIT", "PRE_GRASP", "REACH", "CLOSE", "LIFT",
               "TRANSPORT", "PLACE", "OPEN", "LIFT_RETRACT", "RETRACT"]
```

After the `GRASP_Z_OFFSET` definition, add:

```python
OBJECT_GRASP_Z_OFFSETS: dict[int, float] = {
    1: GRASP_Z_OFFSET,
    2: GRASP_Z_OFFSET,
    3: GRASP_Z_OFFSET,
}
```

- [ ] **Step 4: Update `PickPlaceStateMachine` to accept steps and offsets**

Modify `scripts/act/task_e/state_machine.py`.

Change the import block to:

```python
from collections.abc import Mapping

import torch
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

from .config import (
    STEPS, STATE_ORDER,
    CARRY_Z, PLACE_HEIGHT,
    RETRACT_POS_X, RETRACT_POS_Y,
    GRASP_Z_OFFSET,
    BASKET_CENTER_X, BASKET_CENTER_Y,
    DEFAULT_PLACE_QUAT_W,
)
```

Replace the class docstring and constructor with:

```python
class PickPlaceStateMachine:
    """Finite state machine that sequences pick-and-place for multiple objects.

    States (in order): INIT → PRE_GRASP → REACH → CLOSE → LIFT →
                       TRANSPORT → PLACE → OPEN → LIFT_RETRACT → RETRACT →
                       (next object or done)
    """

    def __init__(
        self,
        object_indices: list[int],
        device: str,
        steps: Mapping[str, int] | None = None,
        grasp_z_offsets: Mapping[int, float] | None = None,
    ):
        self._obj_indices = object_indices
        self._device      = device
        self._steps       = dict(steps) if steps is not None else STEPS
        self._grasp_z_offsets = {
            int(k): float(v) for k, v in (grasp_z_offsets or {}).items()
        }
        self._grasp_quat_cache: dict[int, torch.Tensor] = {}
        self.reset()
```

Replace the transition condition in `tick()`:

```python
if self._count >= STEPS[s]:
```

with:

```python
if self._count >= self._steps[s]:
```

Replace the `REACH` and `CLOSE` branches in `_get_target_pos_gripper()` with:

```python
        elif s == "REACH":
            p = obj_pos.clone(); p[2] += self._get_grasp_z_offset()
            return p, "open"
        elif s == "CLOSE":
            p = obj_pos.clone(); p[2] += self._get_grasp_z_offset()
            return p, "close"
```

Add this helper before `_get_target_quat()`:

```python
    def _get_grasp_z_offset(self) -> float:
        cur_idx = self._obj_indices[min(self._ptr, len(self._obj_indices) - 1)]
        return self._grasp_z_offsets.get(cur_idx, GRASP_Z_OFFSET)
```

- [ ] **Step 5: Run state-machine tests**

Run:

```bash
pytest tests/act/test_task_e_state_machine.py -q
```

Expected: PASS, 3 tests.

- [ ] **Step 6: Run option tests to catch regressions**

Run:

```bash
pytest tests/act/test_task_e_options.py tests/act/test_task_e_state_machine.py -q
```

Expected: PASS, 11 tests.

- [ ] **Step 7: Commit checkpoint if commits are authorized**

```bash
git add scripts/act/task_e/config.py scripts/act/task_e/state_machine.py tests/act/test_task_e_state_machine.py
git commit -m "feat: configure task e pick place state machine"
```

---

### Task 3: Add Per-Object Success Diagnostics to Collector

**Files:**
- Create: `tests/act/test_task_e_collector_success.py`
- Modify: `scripts/act/task_e/collector.py`

- [ ] **Step 1: Write failing collector success tests**

Create `tests/act/test_task_e_collector_success.py` with:

```python
from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib
import sys

import torch

_REPO = Path(__file__).resolve().parents[2]
_ACT_DIR = _REPO / "scripts" / "act"
if str(_ACT_DIR) not in sys.path:
    sys.path.insert(0, str(_ACT_DIR))


def _install_stubs():
    env_cfg = ModuleType("atec_rl_lab.tasks.task_e.env_cfg")
    env_cfg.TABLE_CENTER_X = 0.50
    env_cfg.TABLE_CENTER_Y = 0.00
    env_cfg.TABLE_TOP_Z = 0.20
    env_cfg.TABLE_HALF_X = 0.30
    env_cfg.BASKET_CENTER_X = 0.40
    env_cfg.BASKET_CENTER_Y = 0.00
    env_cfg.BASKET_EXCL_HALF_X = 0.20
    env_cfg.BASKET_EXCL_HALF_Y = 0.11

    sys.modules.setdefault("atec_rl_lab", ModuleType("atec_rl_lab"))
    sys.modules.setdefault("atec_rl_lab.tasks", ModuleType("atec_rl_lab.tasks"))
    sys.modules.setdefault("atec_rl_lab.tasks.task_e", ModuleType("atec_rl_lab.tasks.task_e"))
    sys.modules["atec_rl_lab.tasks.task_e.env_cfg"] = env_cfg

    utils_mod = ModuleType("atec_rl_lab.utils")
    utils_mod.CartesianController = object
    sys.modules["atec_rl_lab.utils"] = utils_mod

    envs_mod = ModuleType("isaaclab.envs")
    envs_mod.ManagerBasedRLEnv = object
    sys.modules.setdefault("isaaclab", ModuleType("isaaclab"))
    sys.modules["isaaclab.envs"] = envs_mod

    math_mod = ModuleType("isaaclab.utils.math")
    math_mod.matrix_from_quat = lambda quat: torch.eye(3).repeat(quat.shape[0], 1, 1)
    math_mod.quat_from_matrix = lambda matrix: torch.tensor(
        [[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32
    ).repeat(matrix.shape[0], 1)
    sys.modules.setdefault("isaaclab.utils", ModuleType("isaaclab.utils"))
    sys.modules["isaaclab.utils.math"] = math_mod


_install_stubs()
collector = importlib.import_module("task_e.collector")


class _FakeRigidObject:
    def __init__(self, pos):
        self.data = SimpleNamespace(root_pos_w=torch.tensor([pos], dtype=torch.float32))


def _fake_env(positions):
    rigid_objects = {
        f"object_{obj_id}": _FakeRigidObject(pos)
        for obj_id, pos in positions.items()
    }
    scene = SimpleNamespace(rigid_objects=rigid_objects)
    return SimpleNamespace(unwrapped=SimpleNamespace(scene=scene))


def test_get_objects_in_basket_reports_each_requested_object():
    env = _fake_env({
        1: [0.40, 0.00, 0.22],
        2: [0.90, 0.00, 0.22],
        3: [0.40, 0.00, 0.50],
    })

    result = collector.get_objects_in_basket(env, [1, 2, 3])

    assert result == {1: True, 2: False, 3: False}


def test_check_objects_in_basket_remains_all_success_wrapper():
    success_env = _fake_env({1: [0.40, 0.00, 0.22], 2: [0.40, 0.00, 0.22]})
    fail_env = _fake_env({1: [0.40, 0.00, 0.22], 2: [0.90, 0.00, 0.22]})

    assert collector.check_objects_in_basket(success_env, [1, 2]) is True
    assert collector.check_objects_in_basket(fail_env, [1, 2]) is False
```

- [ ] **Step 2: Run tests and verify they fail because `get_objects_in_basket` is missing**

Run:

```bash
pytest tests/act/test_task_e_collector_success.py -q
```

Expected: FAIL with `AttributeError: module 'task_e.collector' has no attribute 'get_objects_in_basket'`.

- [ ] **Step 3: Implement per-object success diagnostics**

Modify `scripts/act/task_e/collector.py`.

Replace `check_objects_in_basket()` with this block:

```python
_BASKET_MAX_Z = TABLE_TOP_Z + 0.1   # object must be below this to count as inside


def get_objects_in_basket(env: ManagerBasedRLEnv, pick_objects: list[int]) -> dict[int, bool]:
    """Return per-object basket success for the requested objects."""
    result: dict[int, bool] = {}
    for obj_idx in pick_objects:
        pos = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"].data.root_pos_w[0]
        in_basket = (
            abs(pos[0].item() - BASKET_CENTER_X) <= BASKET_IN_X and
            abs(pos[1].item() - BASKET_CENTER_Y) <= BASKET_IN_Y and
            pos[2].item() <= _BASKET_MAX_Z
        )
        result[obj_idx] = bool(in_basket)
    return result


def check_objects_in_basket(env: ManagerBasedRLEnv, pick_objects: list[int]) -> bool:
    """Return True only if every picked object is inside the basket region and settled."""
    return all(get_objects_in_basket(env, pick_objects).values())
```

Update `collect_one_demo()` signature to accept configurable steps and offsets:

```python
def collect_one_demo(
    env:         ManagerBasedRLEnv,
    robot,
    ik_ctrl:     CartesianController,
    arm_ids:     list[int],
    gripper_ids: list[int],
    pick_objects: list[int],
    device:      str,
    default_jpos: torch.Tensor,
    rng:         np.random.Generator,
    camera=None,
    steps: dict[str, int] | None = None,
    grasp_z_offsets: dict[int, float] | None = None,
) -> dict | None:
```

Replace the state-machine construction:

```python
sm = PickPlaceStateMachine(pick_objects, device)
```

with:

```python
sm = PickPlaceStateMachine(
    pick_objects,
    device,
    steps=steps,
    grasp_z_offsets=grasp_z_offsets,
)
```

- [ ] **Step 4: Run collector success tests**

Run:

```bash
pytest tests/act/test_task_e_collector_success.py -q
```

Expected: PASS, 2 tests.

- [ ] **Step 5: Run all Task E ACT unit tests**

Run:

```bash
pytest tests/act/test_task_e_options.py tests/act/test_task_e_state_machine.py tests/act/test_task_e_collector_success.py -q
```

Expected: PASS, 13 tests.

- [ ] **Step 6: Commit checkpoint if commits are authorized**

```bash
git add scripts/act/task_e/collector.py tests/act/test_task_e_collector_success.py
git commit -m "feat: report task e per-object basket success"
```

---

### Task 4: Add CLI Flags and Reusable Environment Setup

**Files:**
- Modify: `scripts/act/cli_args.py`
- Create: `scripts/act/task_e/env_setup.py`
- Modify: `scripts/act/collect_demos_task_e.py`

- [ ] **Step 1: Extend collection CLI arguments**

Modify `scripts/act/cli_args.py` by adding these parser arguments at the end of `add_collect_demo_args()` after `--only_success`:

```python
    parser.add_argument(
        "--full_order_123", action="store_true", default=False,
        help="Force Task E collection order to object_1, object_2, object_3 and require full-order success.",
    )
    parser.add_argument(
        "--optimized_grasp_flow", action="store_true", default=False,
        help="Use optimized pick-place dwell times for faster Task E grasp data collection.",
    )
    parser.add_argument(
        "--grasp_z_offsets", type=str, nargs="*", default=None,
        metavar="OBJECT_ID:OFFSET",
        help="Per-object grasp height offsets, for example: 1:0.080 2:0.085 3:0.075.",
    )
    parser.add_argument(
        "--grasp_offset_json", type=str, default=None,
        help="Path to sweep JSON containing best_offsets for object-specific grasp heights.",
    )
    parser.add_argument(
        "--max_attempts", type=int, default=None,
        help="Maximum attempts before exiting non-zero if num_demos successful demos are not collected.",
    )
```

- [ ] **Step 2: Create reusable environment builder**

Create `scripts/act/task_e/env_setup.py` with:

```python
"""Environment setup helpers for Task E ACT collection scripts."""

from __future__ import annotations

import time

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv

from atec_rl_lab.tasks.task_e.env_cfg import TaskEEnvPiperCfg

from .config import ACT_DAMPING, ACT_EFFORT_LIMIT, ACT_STIFFNESS, ACT_VEL_LIMIT


def build_task_e_env(pick_objects: list[int], need_camera: bool) -> ManagerBasedRLEnv:
    """Build the single-env Task E Piper environment used for ACT collection."""
    cfg = TaskEEnvPiperCfg()
    cfg.seed = int(time.time_ns() % (2**31))
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 40.0 * len(pick_objects) + 10.0
    cfg.scene.robot.actuators["default"] = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        effort_limit=ACT_EFFORT_LIMIT,
        velocity_limit=ACT_VEL_LIMIT,
        stiffness=ACT_STIFFNESS,
        damping=ACT_DAMPING,
    )
    _ = need_camera  # Camera is already configured by TaskEEnvPiperCfg when cameras are enabled.
    return ManagerBasedRLEnv(cfg)
```

- [ ] **Step 3: Update `collect_demos_task_e.py` imports**

Modify `scripts/act/collect_demos_task_e.py`.

Remove these imports:

```python
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
```

Replace the config import block:

```python
from task_e.config import (
    EE_BODY_NAME, ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES,
    ACT_STIFFNESS, ACT_DAMPING, ACT_EFFORT_LIMIT, ACT_VEL_LIMIT,
    CAM_H, CAM_W, CAM_POS, CAM_ROT,
)
from task_e.collector import check_objects_in_basket, collect_one_demo
```

with:

```python
from task_e.config import (
    EE_BODY_NAME, ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES,
    OBJECT_GRASP_Z_OFFSETS, OPTIMIZED_STEPS,
)
from task_e.collector import collect_one_demo, get_objects_in_basket
from task_e.env_setup import build_task_e_env
from task_e.options import resolve_grasp_z_offsets, resolve_pick_objects
```

Delete the local `build_env()` function from `collect_demos_task_e.py`.

- [ ] **Step 4: Replace `main()` in `collect_demos_task_e.py`**

Replace the full `main()` function with:

```python
def main() -> None:
    try:
        pick_objects = resolve_pick_objects(args_cli.pick_objects, args_cli.full_order_123)
        grasp_z_offsets = resolve_grasp_z_offsets(
            args_cli.grasp_z_offsets,
            args_cli.grasp_offset_json,
            defaults=OBJECT_GRASP_Z_OFFSETS,
        )
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc

    if args_cli.full_order_123 and not args_cli.only_success:
        print("[INFO] --full_order_123 enables --only_success for full-order datasets.")
        args_cli.only_success = True

    steps = OPTIMIZED_STEPS if args_cli.optimized_grasp_flow else None
    need_camera = args_cli.save_video or args_cli.save_images

    print(f"[INFO] pick_objects={pick_objects}")
    print(f"[INFO] grasp_z_offsets={grasp_z_offsets}")
    print(f"[INFO] optimized_grasp_flow={args_cli.optimized_grasp_flow}")

    env    = build_task_e_env(pick_objects, need_camera)
    dev    = env.unwrapped.device
    camera = env.unwrapped.scene["video_cam"] if need_camera else None

    robot = env.unwrapped.scene.articulations["robot"]
    arm_ids,     _ = robot.find_joints(ARM_JOINT_NAMES)
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
    ik_ctrl = CartesianController(
        robot=robot, ee_body_name=EE_BODY_NAME,
        arm_joint_names=ARM_JOINT_NAMES,
        num_envs=1, device=dev,
        command_type="pose",
        lambda_val=0.05,
        max_joint_delta=0.2,
    )
    default_jpos = robot.data.default_joint_pos.clone()

    video_dir = None
    imageio   = None
    if args_cli.save_video:
        video_dir = args_cli.video_dir or os.path.join(args_cli.output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        import imageio as _io
        imageio = _io

    traj_path, _ = init_output(args_cli.output_dir)
    rng = np.random.default_rng()

    n_ok = 0
    attempt = 0
    while n_ok < args_cli.num_demos:
        if args_cli.max_attempts is not None and attempt >= args_cli.max_attempts:
            print(
                f"[ERROR] Reached --max_attempts={args_cli.max_attempts} "
                f"with {n_ok}/{args_cli.num_demos} successful demos."
            )
            env.close()
            raise SystemExit(1)

        attempt += 1
        print(f"\n[INFO] Demo {n_ok + 1}/{args_cli.num_demos}  (attempt {attempt})")
        print(f"[INFO] order={pick_objects} offsets={grasp_z_offsets}")

        data = collect_one_demo(
            env, robot, ik_ctrl,
            arm_ids, gripper_ids,
            pick_objects, dev,
            default_jpos=default_jpos,
            rng=rng,
            camera=camera,
            steps=steps,
            grasp_z_offsets=grasp_z_offsets,
        )
        if data is None:
            print("[WARN] Early termination — skipping.")
            continue

        success_map = get_objects_in_basket(env, pick_objects)
        print(f"[INFO] success: {success_map}")
        if args_cli.only_success and not all(success_map.values()):
            print("[WARN] Objects not in basket — skipping (--only_success).")
            continue

        save_traj(traj_path, n_ok, data, args_cli.save_images)

        T     = len(data["qpos"])
        notes = [f"{T} steps"]
        if args_cli.save_video and "frames" in data:
            vp = os.path.join(video_dir, f"demo_{n_ok:04d}.mp4")
            imageio.mimwrite(vp, data["frames"], fps=50, quality=7)
            notes.append(f"video → {vp}")
        if args_cli.save_images and "frames" in data:
            notes.append("images saved")
        print(f"[INFO] traj_{n_ok}: {', '.join(notes)}")
        n_ok += 1

    print(f"\n[INFO] Collected {n_ok} demos → {traj_path}")
    env.close()
```

- [ ] **Step 5: Run unit tests**

Run:

```bash
pytest tests/act/test_task_e_options.py tests/act/test_task_e_state_machine.py tests/act/test_task_e_collector_success.py -q
```

Expected: PASS, 13 tests.

- [ ] **Step 6: Compile changed scripts**

Run:

```bash
python -m py_compile \
  scripts/act/cli_args.py \
  scripts/act/task_e/env_setup.py \
  scripts/act/collect_demos_task_e.py
```

Expected: command exits with status 0 and prints no syntax errors.

- [ ] **Step 7: Commit checkpoint if commits are authorized**

```bash
git add scripts/act/cli_args.py scripts/act/task_e/env_setup.py scripts/act/collect_demos_task_e.py
git commit -m "feat: add task e full-order collection controls"
```

---

### Task 5: Add Generic Per-Object Grasp Offset Sweep Script

**Files:**
- Create: `scripts/act/sweep_task_e_grasp_offsets.py`

- [ ] **Step 1: Create the sweep script**

Create `scripts/act/sweep_task_e_grasp_offsets.py` with:

```python
"""Sweep per-object grasp offsets for Task E ACT demonstration collection.

Example:
python scripts/act/sweep_task_e_grasp_offsets.py \
    --objects 1 2 3 \
    --offsets 0.070 0.075 0.080 0.085 0.090 0.095 \
    --attempts_per_offset 5 \
    --optimized_grasp_flow \
    --headless \
    --enable_cameras \
    --output datasets/atec_task_e/grasp_offset_sweep.json
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher
from cli_args import add_collect_demo_args

parser = argparse.ArgumentParser(description="Sweep Task E per-object grasp offsets.")
parser.add_argument("--objects", type=int, nargs="+", default=[1, 2, 3])
parser.add_argument(
    "--offsets", type=float, nargs="+",
    default=[0.070, 0.075, 0.080, 0.085, 0.090, 0.095],
)
parser.add_argument("--attempts_per_offset", type=int, default=5)
parser.add_argument("--output", type=str, default="datasets/atec_task_e/grasp_offset_sweep.json")
# Reuse these flags so AppLauncher accepts the same camera/headless options as collection.
add_collect_demo_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Sweeps do not save image datasets, but allowing --enable_cameras keeps commands consistent.
if args_cli.save_video or args_cli.save_images:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np

from atec_rl_lab.utils import CartesianController

from task_e.config import (
    EE_BODY_NAME, ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES,
    OBJECT_GRASP_Z_OFFSETS, OPTIMIZED_STEPS,
)
from task_e.collector import collect_one_demo, get_objects_in_basket
from task_e.env_setup import build_task_e_env
from task_e.options import choose_best_offset, resolve_pick_objects


def _make_controller(env):
    dev = env.unwrapped.device
    robot = env.unwrapped.scene.articulations["robot"]
    arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
    ik_ctrl = CartesianController(
        robot=robot,
        ee_body_name=EE_BODY_NAME,
        arm_joint_names=ARM_JOINT_NAMES,
        num_envs=1,
        device=dev,
        command_type="pose",
        lambda_val=0.05,
        max_joint_delta=0.2,
    )
    return dev, robot, arm_ids, gripper_ids, ik_ctrl, robot.data.default_joint_pos.clone()


def _sweep_object(obj_id: int, offsets: list[float], attempts_per_offset: int) -> tuple[list[dict], float]:
    print(f"\n[INFO] Sweeping object_{obj_id}")
    env = build_task_e_env([obj_id], need_camera=False)
    dev, robot, arm_ids, gripper_ids, ik_ctrl, default_jpos = _make_controller(env)
    rng = np.random.default_rng()
    steps = OPTIMIZED_STEPS if args_cli.optimized_grasp_flow else None

    rows: list[dict] = []
    for offset in offsets:
        successes = 0
        early_terminations = 0
        print(f"[INFO] object_{obj_id} offset={offset:.3f}")
        for attempt in range(1, attempts_per_offset + 1):
            offsets_for_attempt = dict(OBJECT_GRASP_Z_OFFSETS)
            offsets_for_attempt[obj_id] = float(offset)
            data = collect_one_demo(
                env, robot, ik_ctrl,
                arm_ids, gripper_ids,
                [obj_id], dev,
                default_jpos=default_jpos,
                rng=rng,
                camera=None,
                steps=steps,
                grasp_z_offsets=offsets_for_attempt,
            )
            if data is None:
                early_terminations += 1
                print(f"  attempt {attempt}/{attempts_per_offset}: early termination")
                continue

            success = get_objects_in_basket(env, [obj_id])[obj_id]
            successes += int(success)
            print(f"  attempt {attempt}/{attempts_per_offset}: success={success}")

        row = {
            "object_id": obj_id,
            "offset": float(offset),
            "attempts": attempts_per_offset,
            "successes": successes,
            "success_rate": successes / attempts_per_offset if attempts_per_offset else 0.0,
            "early_terminations": early_terminations,
        }
        rows.append(row)
        print(
            f"[INFO] object_{obj_id} offset={offset:.3f}: "
            f"{successes}/{attempts_per_offset} "
            f"({row['success_rate'] * 100:.1f}%)"
        )

    best = choose_best_offset(rows)
    print(f"[INFO] object_{obj_id} best offset={best:.3f}")
    env.close()
    return rows, best


def main() -> None:
    try:
        objects = resolve_pick_objects(args_cli.objects, full_order_123=False)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc

    payload = {
        "objects": objects,
        "offsets": [float(v) for v in args_cli.offsets],
        "attempts_per_offset": int(args_cli.attempts_per_offset),
        "optimized_grasp_flow": bool(args_cli.optimized_grasp_flow),
        "best_offsets": {},
        "results": {},
    }

    for obj_id in objects:
        rows, best = _sweep_object(obj_id, payload["offsets"], payload["attempts_per_offset"])
        payload["results"][str(obj_id)] = rows
        payload["best_offsets"][str(obj_id)] = best

    output = args_cli.output
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    print("\n[INFO] Sweep complete")
    print(f"[INFO] best_offsets={payload['best_offsets']}")
    print(f"[INFO] saved → {output}")


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Compile the sweep script**

Run:

```bash
python -m py_compile scripts/act/sweep_task_e_grasp_offsets.py
```

Expected: command exits with status 0 and prints no syntax errors.

- [ ] **Step 3: Run all lightweight tests**

Run:

```bash
pytest tests/act/test_task_e_options.py tests/act/test_task_e_state_machine.py tests/act/test_task_e_collector_success.py -q
```

Expected: PASS, 13 tests.

- [ ] **Step 4: Commit checkpoint if commits are authorized**

```bash
git add scripts/act/sweep_task_e_grasp_offsets.py
git commit -m "feat: add task e grasp offset sweep"
```

---

### Task 6: Validate Workflow Commands and Update Operator Notes

**Files:**
- Modify: `docs/superpowers/specs/2026-06-22-task-e-act-grasp-data-optimization-design.md`
- Modify: `example.md`

- [ ] **Step 1: Add final command block to `example.md`**

In `example.md`, after the existing Task E ACT collection command block, add this section:

```markdown
#### Optimized full-order Task E data collection

First sweep grasp offsets separately for the box, bottle, and banana:

```bash
python scripts/act/sweep_task_e_grasp_offsets.py \
    --objects 1 2 3 \
    --offsets 0.070 0.075 0.080 0.085 0.090 0.095 \
    --attempts_per_offset 5 \
    --optimized_grasp_flow \
    --headless \
    --enable_cameras \
    --output datasets/atec_task_e/grasp_offset_sweep.json
```

Then validate the full `object_1 → object_2 → object_3` state-machine flow before collecting the final dataset:

```bash
python scripts/act/collect_demos_task_e.py \
    --full_order_123 \
    --num_demos 5 \
    --max_attempts 10 \
    --only_success \
    --headless \
    --enable_cameras \
    --optimized_grasp_flow \
    --grasp_offset_json datasets/atec_task_e/grasp_offset_sweep.json \
    --output_dir datasets/atec_task_e/validation_full_order
```

Only after validation succeeds, collect the final 100 full-order demonstrations:

```bash
python scripts/act/collect_demos_task_e.py \
    --full_order_123 \
    --num_demos 100 \
    --only_success \
    --headless \
    --enable_cameras \
    --save_images \
    --optimized_grasp_flow \
    --grasp_offset_json datasets/atec_task_e/grasp_offset_sweep.json \
    --output_dir datasets/atec_task_e/final_100
```
```

- [ ] **Step 2: Add implementation note to the design spec**

Append this section to `docs/superpowers/specs/2026-06-22-task-e-act-grasp-data-optimization-design.md`:

```markdown
## Implementation Notes

The implementation adds a validation gate in the operator workflow rather than automatically starting the final collection after sweep. Operators must run the full-order validation command and confirm it collects successful `object_1 → object_2 → object_3` trajectories before running the final 100-demo command.

The final dataset should be collected into `datasets/atec_task_e/final_100/` so validation runs do not overwrite training data.
```

- [ ] **Step 3: Run documentation grep checks**

Run:

```bash
grep -R "sweep_task_e_grasp_offsets.py" -n example.md docs/superpowers/specs/2026-06-22-task-e-act-grasp-data-optimization-design.md
grep -R "validation_full_order" -n example.md docs/superpowers/specs/2026-06-22-task-e-act-grasp-data-optimization-design.md
grep -R "final_100" -n example.md docs/superpowers/specs/2026-06-22-task-e-act-grasp-data-optimization-design.md
```

Expected: each command prints at least one matching line from `example.md` and at least one matching line from the design spec.

- [ ] **Step 4: Run all lightweight tests and compile all touched scripts**

Run:

```bash
pytest tests/act/test_task_e_options.py tests/act/test_task_e_state_machine.py tests/act/test_task_e_collector_success.py -q
python -m py_compile \
  scripts/act/cli_args.py \
  scripts/act/task_e/options.py \
  scripts/act/task_e/env_setup.py \
  scripts/act/task_e/config.py \
  scripts/act/task_e/state_machine.py \
  scripts/act/task_e/collector.py \
  scripts/act/collect_demos_task_e.py \
  scripts/act/sweep_task_e_grasp_offsets.py
```

Expected: pytest passes all 13 tests; py_compile exits with status 0.

- [ ] **Step 5: Commit checkpoint if commits are authorized**

```bash
git add example.md docs/superpowers/specs/2026-06-22-task-e-act-grasp-data-optimization-design.md
git commit -m "docs: document optimized task e data collection"
```

---

### Task 7: Manual Isaac Lab Validation Gate

**Files:**
- No code files changed in this task.
- Artifacts produced under `datasets/atec_task_e/`.

- [ ] **Step 1: Run per-object sweep**

Run:

```bash
python scripts/act/sweep_task_e_grasp_offsets.py \
    --objects 1 2 3 \
    --offsets 0.070 0.075 0.080 0.085 0.090 0.095 \
    --attempts_per_offset 5 \
    --optimized_grasp_flow \
    --headless \
    --enable_cameras \
    --output datasets/atec_task_e/grasp_offset_sweep.json
```

Expected:

- Command completes without Python exceptions.
- Output prints a best offset for object 1, object 2, and object 3.
- `datasets/atec_task_e/grasp_offset_sweep.json` exists.
- JSON contains `best_offsets` with keys `"1"`, `"2"`, and `"3"`.

- [ ] **Step 2: Inspect sweep JSON**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path('datasets/atec_task_e/grasp_offset_sweep.json')
payload = json.loads(path.read_text())
print(payload['best_offsets'])
assert set(payload['best_offsets']) == {'1', '2', '3'}
for obj_id, rows in payload['results'].items():
    assert rows, obj_id
    print(obj_id, max(row['success_rate'] for row in rows))
PY
```

Expected: prints three best offsets and one max success rate per object; no assertion fails.

- [ ] **Step 3: Run full-order validation before final collection**

Run:

```bash
python scripts/act/collect_demos_task_e.py \
    --full_order_123 \
    --num_demos 5 \
    --max_attempts 10 \
    --only_success \
    --headless \
    --enable_cameras \
    --optimized_grasp_flow \
    --grasp_offset_json datasets/atec_task_e/grasp_offset_sweep.json \
    --output_dir datasets/atec_task_e/validation_full_order
```

Expected:

- Command exits with status 0.
- Logs show `pick_objects=[1, 2, 3]`.
- Logs show per-object success maps where `{1: True, 2: True, 3: True}` for saved trajectories.
- `datasets/atec_task_e/validation_full_order/trajectory.hdf5` exists.
- The command collects 5 successful demos within 10 attempts.

- [ ] **Step 4: If validation fails, tune before collecting final data**

If the validation command exits non-zero or fails to collect 5 successes within 10 attempts:

1. Identify the failing object from the printed success maps.
2. Rerun the sweep for only that object with a narrower offset range around the best prior value. Example for object 1 around `0.080`:

```bash
python scripts/act/sweep_task_e_grasp_offsets.py \
    --objects 1 \
    --offsets 0.074 0.076 0.078 0.080 0.082 0.084 0.086 \
    --attempts_per_offset 10 \
    --optimized_grasp_flow \
    --headless \
    --enable_cameras \
    --output datasets/atec_task_e/grasp_offset_sweep_object_1_refined.json
```

3. Merge the refined best offset into the main JSON using this command, replacing `1` and the refined file if a different object failed:

```bash
python - <<'PY'
import json
from pathlib import Path
main = Path('datasets/atec_task_e/grasp_offset_sweep.json')
refined = Path('datasets/atec_task_e/grasp_offset_sweep_object_1_refined.json')
main_payload = json.loads(main.read_text())
refined_payload = json.loads(refined.read_text())
main_payload['best_offsets']['1'] = refined_payload['best_offsets']['1']
main_payload['results']['1_refined'] = refined_payload['results']['1']
main.write_text(json.dumps(main_payload, indent=2, sort_keys=True))
print(main_payload['best_offsets'])
PY
```

4. Rerun Step 3 full-order validation.

- [ ] **Step 5: Collect final 100 demos only after validation passes**

Run this only after Step 3 passes:

```bash
python scripts/act/collect_demos_task_e.py \
    --full_order_123 \
    --num_demos 100 \
    --only_success \
    --headless \
    --enable_cameras \
    --save_images \
    --optimized_grasp_flow \
    --grasp_offset_json datasets/atec_task_e/grasp_offset_sweep.json \
    --output_dir datasets/atec_task_e/final_100
```

Expected:

- Command exits with status 0.
- `datasets/atec_task_e/final_100/trajectory.hdf5` exists.
- Logs show 100 saved successful trajectories.
- Saved trajectories include RGB frames because `--save_images` is enabled.

- [ ] **Step 6: Verify final HDF5 trajectory count**

Run:

```bash
python - <<'PY'
import h5py
path = 'datasets/atec_task_e/final_100/trajectory.hdf5'
with h5py.File(path, 'r') as f:
    keys = sorted(f.keys(), key=lambda k: int(k.split('_')[1]))
    print(len(keys), keys[:3], keys[-3:])
    assert len(keys) == 100
    first = f[keys[0]]
    assert 'obs' in first
    assert 'actions' in first
    assert 'images/rgb' in first
    print(first['obs'].shape, first['actions'].shape, first['images/rgb'].shape)
PY
```

Expected: prints `100`; no assertion fails.

---

## Self-Review Notes

Spec coverage:

- Per-object offset sweep is implemented by Task 5 and validated in Task 7.
- Full-order `[1, 2, 3]` collection is implemented by Task 4 and validated in Task 7.
- The gate that confirms the adjusted state machine can place all three objects before final collection is covered by Task 7 Step 3.
- Optimized dwell times are implemented by Task 2 and exposed through Task 4.
- Per-object offsets in the state machine are implemented by Task 2.
- Per-object success logging is implemented by Task 3 and used by Task 4/5.
- Separate sweep, validation, and final output paths are covered by Task 6 and Task 7.

Placeholder scan:

- The plan contains no `TBD`, no `TODO`, and no undefined implementation step.
- Every command includes expected output.
- Every new helper function used by later tasks is defined before it is used.

Type consistency:

- Offset mappings are consistently `dict[int, float]` in Python and string-keyed under JSON `best_offsets`.
- Object order is consistently `[1, 2, 3]` for full-order mode.
- The state-machine constructor accepts `steps` and `grasp_z_offsets`, and the collector passes the same names.
