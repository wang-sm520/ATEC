# Task A G1 Corridor Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Task A-specific Unitree G1 AMP corridor training environment that improves straight walking and stair traversal while preserving the strict 79-dimensional G1 AMP motion format.

**Architecture:** Keep the existing broad `ATEC-Isaac-AMP-Unitree-G1-TaskA-v0` environment unchanged and add a new `ATEC-Isaac-AMP-Unitree-G1-TaskA-Corridor-v0` environment. The new environment uses expanded Task A corridor terrain sequences, a corridor-aware reset helper, a centerline penalty reward, and strict AMP motion validation for upstairs/downstairs data.

**Tech Stack:** Python, IsaacLab config classes, RSL-RL AMP/PPO, PyTorch tensors for reward/reset helpers, pytest for lightweight unit tests, GMR retargeting scripts for motion data conversion.

---

## File structure

Create or modify these files:

- Create `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/task_a_corridor_terrain.py`
  - Pure-Python constants/helpers for Task A corridor terrain ordering.
  - No IsaacLab imports, so it can be tested in the local pytest environment.

- Create `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/task_a_corridor_env_cfg.py`
  - Defines `UnitreeG1AMPTaskACorridorEnvCfg`.
  - Builds `15 x N` Task A-style corridor terrain.
  - Sets narrow-but-nonzero command ranges, yaw reset range, and centerline reward weight.

- Modify `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/__init__.py`
  - Registers `ATEC-Isaac-AMP-Unitree-G1-TaskA-Corridor-v0`.

- Modify `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/rough_env_cfg.py`
  - Adds a zero-weight `corridor_centerline_l1` reward term to `G1AMPRewardsCfg` so the corridor environment can activate it.

- Modify `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/utils.py`
  - Adds shared helpers to map env ids to Task A corridor start origins.

- Modify `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/events.py`
  - Adds `reset_root_state_task_a_corridor`, a reset event that spawns envs at row 0 of their assigned corridor.

- Modify `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/rewards.py`
  - Adds `corridor_centerline_l1` using the same corridor mapping as reset.

- Create `motion_data/validate_amp_motion_g1.py`
  - Strict validator for G1 AMP JSON files.
  - Requires `Frames` to be exactly `(N, 79)` and finite.

- Modify `scripts/rsl_rl/play_amp.py`
  - Changes fallback `amp_obs_dim` from 73 to 79.

- Create/modify tests:
  - `tests/test_task_a_corridor_training.py`
  - `tests/test_validate_amp_motion_g1.py`
  - `tests/test_rsl_rl_script_compat.py`
  - `tests/test_feet_min_clearance.py`

---

### Task 1: Add pure corridor terrain sequence helper

**Files:**
- Create: `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/task_a_corridor_terrain.py`
- Test: `tests/test_task_a_corridor_training.py`

- [ ] **Step 1: Write failing tests for row-major corridor sequence expansion**

Create `tests/test_task_a_corridor_training.py` with this content:

```python
from pathlib import Path
import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import torch

_REPO = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPO / "source" / "atec_rl_lab"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

_TERRAIN_HELPER_PATH = (
    _SOURCE_ROOT
    / "atec_rl_lab"
    / "train"
    / "locomotion"
    / "velocity"
    / "config"
    / "humanoid"
    / "unitree_g1"
    / "task_a_corridor_terrain.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_expand_task_a_corridor_sequence_repeats_each_task_a_row_across_columns():
    terrain = _load_module("task_a_corridor_terrain_under_test", _TERRAIN_HELPER_PATH)

    expanded = terrain.expand_task_a_corridor_sequence(num_cols=3)
    rows = [expanded[row * 3 : (row + 1) * 3] for row in range(len(terrain.TASK_A_CORRIDOR_SEQUENCE))]

    assert len(expanded) == 45
    assert rows[0] == ["flat", "flat", "flat"]
    assert rows[1] == ["flat", "flat", "flat"]
    assert rows[2] == ["random_rough", "random_rough", "random_rough"]
    assert rows[6] == ["hf_pyramid_slope", "hf_pyramid_slope", "hf_pyramid_slope"]
    assert rows[7] == ["hf_pyramid_slope_inv", "hf_pyramid_slope_inv", "hf_pyramid_slope_inv"]
    assert rows[10] == ["pyramid_stairs", "pyramid_stairs", "pyramid_stairs"]
    assert rows[11] == ["pyramid_stairs_inv", "pyramid_stairs_inv", "pyramid_stairs_inv"]
    assert rows[14] == ["flat", "flat", "flat"]


def test_expand_task_a_corridor_sequence_rejects_non_positive_columns():
    terrain = _load_module("task_a_corridor_terrain_under_test_invalid", _TERRAIN_HELPER_PATH)

    try:
        terrain.expand_task_a_corridor_sequence(num_cols=0)
    except ValueError as exc:
        assert "num_cols must be positive" in str(exc)
    else:
        raise AssertionError("expected ValueError for num_cols=0")
```

- [ ] **Step 2: Run the new terrain helper tests and verify they fail**

Run:

```bash
pytest tests/test_task_a_corridor_training.py::test_expand_task_a_corridor_sequence_repeats_each_task_a_row_across_columns tests/test_task_a_corridor_training.py::test_expand_task_a_corridor_sequence_rejects_non_positive_columns -q
```

Expected: FAIL because `task_a_corridor_terrain.py` does not exist.

- [ ] **Step 3: Implement the pure terrain helper**

Create `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/task_a_corridor_terrain.py`:

```python
"""Pure helpers for building Task A-style parallel corridor terrain."""

from __future__ import annotations

TASK_A_CORRIDOR_SEQUENCE: list[str] = [
    "flat",
    "flat",
    "random_rough",
    "random_rough",
    "random_rough",
    "random_rough",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "flat",
]

TASK_A_CORRIDOR_NUM_ROWS = len(TASK_A_CORRIDOR_SEQUENCE)


def expand_task_a_corridor_sequence(num_cols: int) -> list[str]:
    """Return a row-major terrain sequence for ``num_cols`` parallel Task A corridors.

    ``BetterTerrainGenerator`` increments its cell counter in row-major order for the
    generated terrain grid, matching the existing code's ``flat_idx = row * num_cols + col``
    convention in ``mdp/utils.py``. Repeating each row terrain across all columns makes every
    column a full Task A corridor.
    """
    if num_cols <= 0:
        raise ValueError(f"num_cols must be positive, got {num_cols}")

    expanded: list[str] = []
    for terrain_name in TASK_A_CORRIDOR_SEQUENCE:
        expanded.extend([terrain_name] * num_cols)
    return expanded
```

- [ ] **Step 4: Run the terrain helper tests and verify they pass**

Run:

```bash
pytest tests/test_task_a_corridor_training.py::test_expand_task_a_corridor_sequence_repeats_each_task_a_row_across_columns tests/test_task_a_corridor_training.py::test_expand_task_a_corridor_sequence_rejects_non_positive_columns -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/task_a_corridor_terrain.py tests/test_task_a_corridor_training.py
git commit -m "feat: add task a corridor terrain sequence helper"
```

If this environment still lacks `git`, record the changed files and commit from a shell that has Git installed.

---

### Task 2: Add corridor origin helper and centerline reward

**Files:**
- Modify: `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/utils.py`
- Modify: `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/rewards.py`
- Modify: `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/rough_env_cfg.py`
- Test: `tests/test_task_a_corridor_training.py`
- Test: `tests/test_feet_min_clearance.py`

- [ ] **Step 1: Add failing tests for corridor origin mapping**

Append this test to `tests/test_task_a_corridor_training.py`:

```python
def test_task_a_corridor_start_origins_maps_env_ids_to_row_zero_columns():
    from atec_rl_lab.train.locomotion.velocity.mdp.utils import task_a_corridor_start_origins

    terrain_origins = torch.tensor(
        [
            [[-140.0, -20.0, 0.0], [-140.0, 0.0, 0.0], [-140.0, 20.0, 0.0]],
            [[-120.0, -20.0, 0.0], [-120.0, 0.0, 0.0], [-120.0, 20.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    env = SimpleNamespace(
        num_envs=5,
        device="cpu",
        scene=SimpleNamespace(
            terrain=SimpleNamespace(terrain_origins=terrain_origins),
            env_origins=torch.zeros(5, 3),
        ),
    )

    origins = task_a_corridor_start_origins(env)

    expected = torch.tensor(
        [
            [-140.0, -20.0, 0.0],
            [-140.0, 0.0, 0.0],
            [-140.0, 20.0, 0.0],
            [-140.0, -20.0, 0.0],
            [-140.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(origins, expected)


def test_task_a_corridor_start_origins_supports_subset_env_ids():
    from atec_rl_lab.train.locomotion.velocity.mdp.utils import task_a_corridor_start_origins

    terrain_origins = torch.tensor(
        [[[-140.0, -10.0, 0.0], [-140.0, 10.0, 0.0]]],
        dtype=torch.float32,
    )
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        scene=SimpleNamespace(
            terrain=SimpleNamespace(terrain_origins=terrain_origins),
            env_origins=torch.zeros(4, 3),
        ),
    )

    origins = task_a_corridor_start_origins(env, env_ids=torch.tensor([1, 2, 3]))

    expected = torch.tensor(
        [[-140.0, 10.0, 0.0], [-140.0, -10.0, 0.0], [-140.0, 10.0, 0.0]],
        dtype=torch.float32,
    )
    assert torch.allclose(origins, expected)
```

- [ ] **Step 2: Add failing tests for centerline reward**

In `tests/test_feet_min_clearance.py`, after this existing line:

```python
feet_min_clearance = _rewards.feet_min_clearance
```

add:

```python
corridor_centerline_l1 = _rewards.corridor_centerline_l1
```

Append this test to the same file:

```python
def test_corridor_centerline_l1_uses_row_zero_corridor_centers_and_clips_error():
    terrain_origins = torch.tensor(
        [[[-140.0, -10.0, 0.0], [-140.0, 10.0, 0.0]]],
        dtype=torch.float32,
    )
    scene = _Scene()
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor(
                [
                    [0.0, -10.0, 0.8],
                    [0.0, 11.25, 0.8],
                    [0.0, 99.0, 0.8],
                ],
                dtype=torch.float32,
            )
        )
    )
    scene.terrain = SimpleNamespace(terrain_origins=terrain_origins)
    scene.env_origins = torch.zeros(3, 3)
    env = SimpleNamespace(num_envs=3, device="cpu", scene=scene)

    penalty = corridor_centerline_l1(env, y_clip=2.0, asset_cfg=SceneEntityCfg("robot"))

    assert torch.allclose(penalty, torch.tensor([0.0, 1.25, 2.0]))
```

- [ ] **Step 3: Run the new origin/reward tests and verify they fail**

Run:

```bash
pytest tests/test_task_a_corridor_training.py::test_task_a_corridor_start_origins_maps_env_ids_to_row_zero_columns tests/test_task_a_corridor_training.py::test_task_a_corridor_start_origins_supports_subset_env_ids tests/test_feet_min_clearance.py::test_corridor_centerline_l1_uses_row_zero_corridor_centers_and_clips_error -q
```

Expected: FAIL because `task_a_corridor_start_origins` and `corridor_centerline_l1` do not exist yet.

- [ ] **Step 4: Implement the corridor origin helper**

Append this function to `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/utils.py`:

```python

def task_a_corridor_start_origins(env: ManagerBasedEnv, env_ids: torch.Tensor | None = None, device=None) -> torch.Tensor:
    """Return row-0 Task A corridor start origins for env ids.

    Env ``i`` is assigned to column ``i % num_cols``. The returned origins are always from
    terrain row 0 so resets start at the beginning of each 300 m corridor instead of the
    current terrain tile or curriculum level.
    """
    if device is None:
        device = getattr(env, "device", "cpu")

    if env_ids is None:
        ids = torch.arange(env.num_envs, device=device, dtype=torch.long)
    else:
        ids = env_ids.to(device=device, dtype=torch.long)

    terrain = getattr(env.scene, "terrain", None)
    terrain_origins = getattr(terrain, "terrain_origins", None)
    if terrain_origins is None:
        env_origins = env.scene.env_origins
        if not torch.is_tensor(env_origins):
            env_origins = torch.as_tensor(env_origins, dtype=torch.float32, device=device)
        else:
            env_origins = env_origins.to(device=device)
        return env_origins[ids]

    if not torch.is_tensor(terrain_origins):
        origins = torch.as_tensor(terrain_origins, dtype=torch.float32, device=device)
    else:
        origins = terrain_origins.to(device=device)

    if origins.ndim != 3 or origins.shape[0] < 1 or origins.shape[1] < 1 or origins.shape[2] != 3:
        raise ValueError(f"expected terrain_origins shape (rows, cols, 3), got {tuple(origins.shape)}")

    num_cols = origins.shape[1]
    col_ids = ids % num_cols
    return origins[0, col_ids, :]
```

- [ ] **Step 5: Implement the centerline reward**

In `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/rewards.py`, add this import near the other imports:

```python
from .utils import task_a_corridor_start_origins
```

Then add this function after `track_ang_vel_z_world_exp`:

```python

def corridor_centerline_l1(
    env: ManagerBasedRLEnv,
    y_clip: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral distance from each env's assigned Task A corridor centerline."""
    asset: RigidObject = env.scene[asset_cfg.name]
    start_origins = task_a_corridor_start_origins(env, device=asset.data.root_pos_w.device)
    y_error = asset.data.root_pos_w[:, 1] - start_origins[:, 1]
    return torch.clamp(torch.abs(y_error), max=float(y_clip))
```

- [ ] **Step 6: Add the zero-weight reward term to G1 AMP rewards**

In `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/rough_env_cfg.py`, inside `G1AMPRewardsCfg`, immediately after the existing `lin_vel_y_l2` term, add:

```python
    corridor_centerline_l1 = RewTerm(
        func=mdp.corridor_centerline_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "y_clip": 3.0},
    )
```

- [ ] **Step 7: Run the origin/reward tests and verify they pass**

Run:

```bash
pytest tests/test_task_a_corridor_training.py::test_task_a_corridor_start_origins_maps_env_ids_to_row_zero_columns tests/test_task_a_corridor_training.py::test_task_a_corridor_start_origins_supports_subset_env_ids tests/test_feet_min_clearance.py::test_corridor_centerline_l1_uses_row_zero_corridor_centers_and_clips_error -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 2**

Run:

```bash
git add source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/utils.py source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/rewards.py source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/rough_env_cfg.py tests/test_task_a_corridor_training.py tests/test_feet_min_clearance.py
git commit -m "feat: add task a corridor centerline reward"
```

If this environment still lacks `git`, record the changed files and commit from a shell that has Git installed.

---

### Task 3: Add corridor-specific reset event

**Files:**
- Modify: `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/events.py`
- Test: `tests/test_task_a_corridor_training.py`

- [ ] **Step 1: Add IsaacLab stubs and failing reset test**

Append these helper stubs and test to `tests/test_task_a_corridor_training.py`:

```python
def _install_event_import_stubs():
    isaaclab = ModuleType("isaaclab")
    sys.modules.setdefault("isaaclab", isaaclab)

    assets = ModuleType("isaaclab.assets")
    assets.Articulation = object
    assets.RigidObject = object
    sys.modules["isaaclab.assets"] = assets

    managers = ModuleType("isaaclab.managers")

    class SceneEntityCfg:
        def __init__(self, name, body_names=None, body_ids=None, joint_names=None, joint_ids=None):
            self.name = name
            self.body_names = body_names
            self.body_ids = body_ids or []
            self.joint_names = joint_names
            self.joint_ids = joint_ids or []

    managers.SceneEntityCfg = SceneEntityCfg
    sys.modules["isaaclab.managers"] = managers

    envs = ModuleType("isaaclab.envs")
    envs.ManagerBasedEnv = object
    sys.modules["isaaclab.envs"] = envs

    math_mod = ModuleType("isaaclab.utils.math")

    def sample_uniform(low, high, shape, device=None):
        return torch.zeros(shape, device=device)

    def quat_from_euler_xyz(roll, pitch, yaw):
        return torch.stack(
            [torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll), torch.zeros_like(roll)],
            dim=-1,
        )

    def quat_mul(q1, q2):
        return q1

    math_mod.sample_uniform = sample_uniform
    math_mod.quat_from_euler_xyz = quat_from_euler_xyz
    math_mod.quat_mul = quat_mul

    utils = ModuleType("isaaclab.utils")
    utils.math = math_mod
    sys.modules["isaaclab.utils"] = utils
    sys.modules["isaaclab.utils.math"] = math_mod
    return SceneEntityCfg


class _FakeAsset:
    def __init__(self):
        self.device = "cpu"
        self.data = SimpleNamespace(
            default_root_state=torch.tensor(
                [
                    [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
        self.pose_env_ids = None
        self.velocity_env_ids = None
        self.written_pose = None
        self.written_velocity = None

    def write_root_pose_to_sim(self, pose, env_ids):
        self.written_pose = pose
        self.pose_env_ids = env_ids

    def write_root_velocity_to_sim(self, velocity, env_ids):
        self.written_velocity = velocity
        self.velocity_env_ids = env_ids


def test_reset_root_state_task_a_corridor_uses_row_zero_origins():
    SceneEntityCfg = _install_event_import_stubs()
    from atec_rl_lab.train.locomotion.velocity.mdp.events import reset_root_state_task_a_corridor

    asset = _FakeAsset()
    terrain_origins = torch.tensor(
        [[[-140.0, -10.0, 0.0], [-140.0, 10.0, 0.0]]],
        dtype=torch.float32,
    )
    scene = {"robot": asset}
    scene = SimpleNamespace(
        __getitem__=scene.__getitem__,
        terrain=SimpleNamespace(terrain_origins=terrain_origins),
        env_origins=torch.zeros(3, 3),
    )
    env = SimpleNamespace(num_envs=3, device="cpu", scene=scene)

    reset_root_state_task_a_corridor(
        env,
        env_ids=torch.tensor([0, 1, 2]),
        pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        velocity_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        asset_cfg=SceneEntityCfg("robot"),
    )

    expected_positions = torch.tensor(
        [[-140.0, -10.0, 0.8], [-140.0, 10.0, 0.8], [-140.0, -10.0, 0.8]],
        dtype=torch.float32,
    )
    assert torch.allclose(asset.written_pose[:, :3], expected_positions)
    assert torch.equal(asset.pose_env_ids, torch.tensor([0, 1, 2]))
    assert torch.allclose(asset.written_velocity, torch.zeros(3, 6))
```

- [ ] **Step 2: Run the reset test and verify it fails**

Run:

```bash
pytest tests/test_task_a_corridor_training.py::test_reset_root_state_task_a_corridor_uses_row_zero_origins -q
```

Expected: FAIL because `reset_root_state_task_a_corridor` does not exist.

- [ ] **Step 3: Implement the corridor reset event**

In `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/events.py`, change the utils import from:

```python
from .utils import is_env_assigned_to_terrain
```

to:

```python
from .utils import is_env_assigned_to_terrain, task_a_corridor_start_origins
```

Then add this function after `reset_root_state_uniform`:

```python

def reset_root_state_task_a_corridor(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset robot roots at row-0 starts of their assigned Task A corridors."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)
    else:
        env_ids = env_ids.to(device=asset.device, dtype=torch.long)

    root_states = asset.data.default_root_state[env_ids].clone()
    start_origins = task_a_corridor_start_origins(env, env_ids=env_ids, device=asset.device)

    pose_ranges = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_ranges_tensor = torch.tensor(pose_ranges, device=asset.device)
    pose_samples = math_utils.sample_uniform(
        pose_ranges_tensor[:, 0], pose_ranges_tensor[:, 1], (len(env_ids), 6), device=asset.device
    )

    positions = root_states[:, 0:3] + start_origins + pose_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    velocity_ranges = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    velocity_ranges_tensor = torch.tensor(velocity_ranges, device=asset.device)
    velocity_samples = math_utils.sample_uniform(
        velocity_ranges_tensor[:, 0], velocity_ranges_tensor[:, 1], (len(env_ids), 6), device=asset.device
    )
    velocities = root_states[:, 7:13] + velocity_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
```

- [ ] **Step 4: Run the reset test and verify it passes**

Run:

```bash
pytest tests/test_task_a_corridor_training.py::test_reset_root_state_task_a_corridor_uses_row_zero_origins -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/events.py tests/test_task_a_corridor_training.py
git commit -m "feat: reset g1 at task a corridor starts"
```

If this environment still lacks `git`, record the changed files and commit from a shell that has Git installed.

---

### Task 4: Add the Task A corridor environment config and registration

**Files:**
- Create: `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/task_a_corridor_env_cfg.py`
- Modify: `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/__init__.py`
- Test: `tests/test_task_a_corridor_training.py`

- [ ] **Step 1: Add source-level tests for the new env config and registration**

Append these tests to `tests/test_task_a_corridor_training.py`:

```python
def test_task_a_corridor_env_cfg_contains_expected_task_a_training_controls():
    path = (
        _SOURCE_ROOT
        / "atec_rl_lab"
        / "train"
        / "locomotion"
        / "velocity"
        / "config"
        / "humanoid"
        / "unitree_g1"
        / "task_a_corridor_env_cfg.py"
    )
    src = path.read_text(encoding="utf-8")

    assert "class UnitreeG1AMPTaskACorridorEnvCfg" in src
    assert "expand_task_a_corridor_sequence" in src
    assert "reset_root_state_task_a_corridor" in src
    assert "corridor_centerline_l1.weight = -1.0" in src
    assert "lin_vel_x = (0.6, 1.6)" in src
    assert "lin_vel_y = (-0.10, 0.10)" in src
    assert "ang_vel_z = (-0.25, 0.25)" in src
    assert "heading = (-0.10, 0.10)" in src
    assert "terrain_levels = None" in src


def test_task_a_corridor_env_is_registered():
    init_path = (
        _SOURCE_ROOT
        / "atec_rl_lab"
        / "train"
        / "locomotion"
        / "velocity"
        / "config"
        / "humanoid"
        / "unitree_g1"
        / "__init__.py"
    )
    src = init_path.read_text(encoding="utf-8")

    assert 'id="ATEC-Isaac-AMP-Unitree-G1-TaskA-Corridor-v0"' in src
    assert "task_a_corridor_env_cfg:UnitreeG1AMPTaskACorridorEnvCfg" in src
```

- [ ] **Step 2: Run the env config tests and verify they fail**

Run:

```bash
pytest tests/test_task_a_corridor_training.py::test_task_a_corridor_env_cfg_contains_expected_task_a_training_controls tests/test_task_a_corridor_training.py::test_task_a_corridor_env_is_registered -q
```

Expected: FAIL because the env config file and registration do not exist yet.

- [ ] **Step 3: Implement the corridor env config**

Create `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/task_a_corridor_env_cfg.py`:

```python
"""Task A corridor-specialized training env for Unitree G1 AMP."""

from __future__ import annotations

import copy

import isaaclab.terrains as terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.velocity.mdp as mdp
from atec_rl_lab.tasks.task_base import BetterTerrainGenerator, BetterTerrainGeneratorCfg, BetterTerrainImporter

from .rough_env_cfg import UnitreeG1AMPRoughEnvCfg
from .task_a_corridor_terrain import TASK_A_CORRIDOR_NUM_ROWS, expand_task_a_corridor_sequence


@configclass
class UnitreeG1AMPTaskACorridorEnvCfg(UnitreeG1AMPRoughEnvCfg):
    """G1 AMP training env with parallel Task A evaluation-style corridors."""

    task_a_num_cols: int = 20

    def __post_init__(self) -> None:
        super().__post_init__()

        physics_material = copy.deepcopy(self.scene.terrain.physics_material)
        visual_material = copy.deepcopy(self.scene.terrain.visual_material)

        self.scene.terrain = TerrainImporterCfg(
            class_type=BetterTerrainImporter,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=BetterTerrainGeneratorCfg(
                class_type=BetterTerrainGenerator,
                seed=0,
                size=(20.0, 20.0),
                border_width=0.0,
                num_rows=TASK_A_CORRIDOR_NUM_ROWS,
                num_cols=self.task_a_num_cols,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                curriculum=False,
                terrain_sequence=expand_task_a_corridor_sequence(self.task_a_num_cols),
                sub_terrains={
                    "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
                    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                        proportion=0.1,
                        noise_range=(0.02, 0.10),
                        noise_step=0.02,
                        border_width=0.25,
                    ),
                    "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                        proportion=0.2,
                        slope_range=(0.39, 0.40),
                        platform_width=2.5,
                        border_width=0.25,
                    ),
                    "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                        proportion=0.2,
                        slope_range=(0.39, 0.40),
                        platform_width=2.5,
                        border_width=0.25,
                    ),
                    "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                        proportion=0.2,
                        step_height_range=(0.05, 0.20),
                        step_width=0.3,
                        platform_width=3.0,
                        border_width=1.0,
                        holes=False,
                    ),
                    "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                        proportion=0.2,
                        step_height_range=(0.05, 0.20),
                        step_width=0.3,
                        platform_width=3.0,
                        border_width=1.0,
                        holes=False,
                    ),
                },
            ),
            max_init_terrain_level=0,
            collision_group=-1,
            physics_material=physics_material,
            visual_material=visual_material,
            debug_vis=False,
        )

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        if self.scene.height_scanner_base is not None:
            self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        self.events.randomize_reset_base = EventTerm(
            func=mdp.reset_root_state_task_a_corridor,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.25, 0.25), "yaw": (-0.10, 0.10)},
                "velocity_range": {
                    "x": (-0.2, 0.2),
                    "y": (-0.1, 0.1),
                    "z": (-0.1, 0.1),
                    "roll": (-0.1, 0.1),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.1, 0.1),
                },
            },
        )

        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 1.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.10, 0.10)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.25, 0.25)
        self.commands.base_velocity.ranges.heading = (-0.10, 0.10)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 0.5

        self.rewards.corridor_centerline_l1.weight = -1.0
        self.rewards.corridor_centerline_l1.params["y_clip"] = 3.0
        self.rewards.lin_vel_y_l2.weight = -0.5

        if self.__class__.__name__ == "UnitreeG1AMPTaskACorridorEnvCfg":
            self.disable_zero_weight_rewards()
```

- [ ] **Step 4: Register the corridor env**

Append this block to `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/__init__.py`:

```python

gym.register(
    id="ATEC-Isaac-AMP-Unitree-G1-TaskA-Corridor-v0",
    entry_point="atec_rl_lab.tasks.task_base:G1AMPGaitEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_a_corridor_env_cfg:UnitreeG1AMPTaskACorridorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:UnitreeG1AMPRoughRunnerCfg",
    },
)
```

- [ ] **Step 5: Run env config tests and verify they pass**

Run:

```bash
pytest tests/test_task_a_corridor_training.py::test_task_a_corridor_env_cfg_contains_expected_task_a_training_controls tests/test_task_a_corridor_training.py::test_task_a_corridor_env_is_registered -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/task_a_corridor_env_cfg.py source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/__init__.py tests/test_task_a_corridor_training.py
git commit -m "feat: add task a g1 corridor training env"
```

If this environment still lacks `git`, record the changed files and commit from a shell that has Git installed.

---

### Task 5: Add strict G1 AMP motion JSON validator

**Files:**
- Create: `motion_data/validate_amp_motion_g1.py`
- Test: `tests/test_validate_amp_motion_g1.py`

- [ ] **Step 1: Write failing validator tests**

Create `tests/test_validate_amp_motion_g1.py`:

```python
from pathlib import Path
import importlib.util
import json

_REPO = Path(__file__).resolve().parents[1]
_VALIDATOR_PATH = _REPO / "motion_data" / "validate_amp_motion_g1.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_amp_motion_g1_under_test", _VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_motion(path: Path, width: int, frame_duration: float = 0.02, motion_weight: float = 1.0, value: float = 0.1):
    frame = [value] * width
    path.write_text(
        json.dumps(
            {
                "LoopMode": "Wrap",
                "FrameDuration": frame_duration,
                "MotionWeight": motion_weight,
                "Frames": [frame, frame],
            }
        ),
        encoding="utf-8",
    )


def test_validate_accepts_exact_79_dim_motion(tmp_path):
    validator = _load_validator()
    path = tmp_path / "good.json"
    _write_motion(path, width=79)

    result = validator.validate_motion_file(path)

    assert result["frames"] == 2
    assert result["width"] == 79
    assert result["frame_duration"] == 0.02
    assert result["motion_weight"] == 1.0


def test_validate_rejects_73_dim_motion(tmp_path):
    validator = _load_validator()
    path = tmp_path / "old_73.json"
    _write_motion(path, width=73)

    try:
        validator.validate_motion_file(path)
    except validator.MotionValidationError as exc:
        assert "expected frame width 79, got 73" in str(exc)
    else:
        raise AssertionError("expected MotionValidationError for 73-dim motion")


def test_validate_rejects_wider_than_79_motion(tmp_path):
    validator = _load_validator()
    path = tmp_path / "wide_80.json"
    _write_motion(path, width=80)

    try:
        validator.validate_motion_file(path)
    except validator.MotionValidationError as exc:
        assert "expected frame width 79, got 80" in str(exc)
    else:
        raise AssertionError("expected MotionValidationError for 80-dim motion")


def test_validate_rejects_nan_motion(tmp_path):
    validator = _load_validator()
    path = tmp_path / "nan.json"
    _write_motion(path, width=79, value=float("nan"))

    try:
        validator.validate_motion_file(path)
    except validator.MotionValidationError as exc:
        assert "non-finite value" in str(exc)
    else:
        raise AssertionError("expected MotionValidationError for NaN motion")
```

- [ ] **Step 2: Run validator tests and verify they fail**

Run:

```bash
pytest tests/test_validate_amp_motion_g1.py -q
```

Expected: FAIL because `motion_data/validate_amp_motion_g1.py` does not exist.

- [ ] **Step 3: Implement the validator**

Create `motion_data/validate_amp_motion_g1.py`:

```python
"""Strict validator for Unitree G1 AMP motion JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

G1_AMP_FRAME_DIM = 79


class MotionValidationError(ValueError):
    """Raised when a G1 AMP motion file does not match the expected schema."""


def _require_finite_number(value: Any, path: Path, frame_idx: int, value_idx: int) -> float:
    if not isinstance(value, (int, float)):
        raise MotionValidationError(f"{path}: frame {frame_idx} value {value_idx} is not numeric: {value!r}")
    value_f = float(value)
    if not math.isfinite(value_f):
        raise MotionValidationError(f"{path}: frame {frame_idx} value {value_idx} is non-finite value: {value!r}")
    return value_f


def validate_motion_file(path: str | Path, frame_dim: int = G1_AMP_FRAME_DIM) -> dict[str, float | int | str]:
    """Validate one G1 AMP JSON file and return a small summary."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("Frames")
    if not isinstance(frames, list) or len(frames) == 0:
        raise MotionValidationError(f"{path}: Frames must be a non-empty list")

    frame_duration = float(data.get("FrameDuration", 0.0))
    if frame_duration <= 0.0 or not math.isfinite(frame_duration):
        raise MotionValidationError(f"{path}: FrameDuration must be positive and finite, got {frame_duration}")

    motion_weight = float(data.get("MotionWeight", 0.0))
    if motion_weight <= 0.0 or not math.isfinite(motion_weight):
        raise MotionValidationError(f"{path}: MotionWeight must be positive and finite, got {motion_weight}")

    width = None
    for frame_idx, frame in enumerate(frames):
        if not isinstance(frame, list):
            raise MotionValidationError(f"{path}: frame {frame_idx} is not a list")
        if width is None:
            width = len(frame)
            if width != frame_dim:
                raise MotionValidationError(f"{path}: expected frame width {frame_dim}, got {width}")
        elif len(frame) != width:
            raise MotionValidationError(f"{path}: frame {frame_idx} width {len(frame)} differs from first width {width}")
        for value_idx, value in enumerate(frame):
            _require_finite_number(value, path, frame_idx, value_idx)

    return {
        "path": str(path),
        "frames": len(frames),
        "width": int(width),
        "frame_duration": frame_duration,
        "motion_weight": motion_weight,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Unitree G1 79-dim AMP motion JSON files.")
    parser.add_argument("motion_files", nargs="+", type=Path)
    args = parser.parse_args(argv)

    for path in args.motion_files:
        summary = validate_motion_file(path)
        print(
            f"OK {summary['path']}: frames={summary['frames']} width={summary['width']} "
            f"dt={summary['frame_duration']} weight={summary['motion_weight']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run validator tests and verify they pass**

Run:

```bash
pytest tests/test_validate_amp_motion_g1.py -q
```

Expected: PASS.

- [ ] **Step 5: Run validator on existing G1 motion files**

Run:

```bash
python motion_data/validate_amp_motion_g1.py motion_data/g1_29dof/*.json
```

Expected: every existing file prints `OK ... width=79`. If any file fails, do not include it in `--amp_motion_files` until converted or removed from the training set.

- [ ] **Step 6: Commit Task 5**

Run:

```bash
git add motion_data/validate_amp_motion_g1.py tests/test_validate_amp_motion_g1.py
git commit -m "feat: validate g1 amp motion dimensions"
```

If this environment still lacks `git`, record the changed files and commit from a shell that has Git installed.

---

### Task 6: Fix play/export AMP dimension fallback

**Files:**
- Modify: `scripts/rsl_rl/play_amp.py`
- Modify: `tests/test_rsl_rl_script_compat.py`

- [ ] **Step 1: Add failing regression test for the 79-dim fallback**

Append this test to `tests/test_rsl_rl_script_compat.py`:

```python

def test_play_amp_script_uses_g1_amp_obs_dim_79_fallback():
    src = _source("scripts/rsl_rl/play_amp.py")

    assert 'setdefault("amp_obs_dim", 79)' in src
    assert 'setdefault("amp_obs_dim", 73)' not in src
```

- [ ] **Step 2: Run the regression test and verify it fails**

Run:

```bash
pytest tests/test_rsl_rl_script_compat.py::test_play_amp_script_uses_g1_amp_obs_dim_79_fallback -q
```

Expected: FAIL because `play_amp.py` still uses `setdefault("amp_obs_dim", 73)`.

- [ ] **Step 3: Change `play_amp.py` fallback to 79**

In `scripts/rsl_rl/play_amp.py`, replace:

```python
    agent_dict["algorithm"].setdefault("amp_obs_dim", 73)
```

with:

```python
    agent_dict["algorithm"].setdefault("amp_obs_dim", 79)
```

- [ ] **Step 4: Run the regression test and verify it passes**

Run:

```bash
pytest tests/test_rsl_rl_script_compat.py::test_play_amp_script_uses_g1_amp_obs_dim_79_fallback -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 6**

Run:

```bash
git add scripts/rsl_rl/play_amp.py tests/test_rsl_rl_script_compat.py
git commit -m "fix: use g1 amp obs dim in play_amp fallback"
```

If this environment still lacks `git`, record the changed files and commit from a shell that has Git installed.

---

### Task 7: Convert upstairs/downstairs source data to strict 79-dim AMP JSON

**Files:**
- Generate: `motion_data/g1_29dof/upstairs01_stageii.json`
- Generate: `motion_data/g1_29dof/downstairs01_stageii.json`
- Generate but do not commit unless explicitly requested: `motion_data/g1_29dof/retargeted/upstairs01_stageii.pkl`
- Generate but do not commit unless explicitly requested: `motion_data/g1_29dof/retargeted/downstairs01_stageii.pkl`

- [ ] **Step 1: Verify GMR source files exist**

Run:

```bash
ls -lh /home/air/wsm/GMR/data/upstairs01_stageii.npz /home/air/wsm/GMR/data/downstairs01_stageii.npz
```

Expected: both files are listed with nonzero size.

- [ ] **Step 2: Create output directory**

Run:

```bash
mkdir -p motion_data/g1_29dof/retargeted
```

Expected: command exits with code 0.

- [ ] **Step 3: Retarget upstairs source data to Unitree G1 pkl**

Run:

```bash
conda run -n gmr python /home/air/wsm/GMR/scripts/smplx_to_robot.py \
  --smplx_file /home/air/wsm/GMR/data/upstairs01_stageii.npz \
  --robot unitree_g1 \
  --save_path /home/air/wang-sm/ATEC/motion_data/g1_29dof/retargeted/upstairs01_stageii.pkl \
  --rate_limit
```

Expected: command prints `Saved to .../upstairs01_stageii.pkl` or the GMR script's equivalent success message.

- [ ] **Step 4: Retarget downstairs source data to Unitree G1 pkl**

Run:

```bash
conda run -n gmr python /home/air/wsm/GMR/scripts/smplx_to_robot.py \
  --smplx_file /home/air/wsm/GMR/data/downstairs01_stageii.npz \
  --robot unitree_g1 \
  --save_path /home/air/wang-sm/ATEC/motion_data/g1_29dof/retargeted/downstairs01_stageii.pkl \
  --rate_limit
```

Expected: command prints `Saved to .../downstairs01_stageii.pkl` or the GMR script's equivalent success message.

- [ ] **Step 5: Convert upstairs pkl to 79-dim AMP JSON**

Run:

```bash
conda run -n gmr python /home/air/wang-sm/ATEC/motion_data/gmr_to_amp_json.py \
  --pkl /home/air/wang-sm/ATEC/motion_data/g1_29dof/retargeted/upstairs01_stageii.pkl \
  --output /home/air/wang-sm/ATEC/motion_data/g1_29dof/upstairs01_stageii.json \
  --motion_weight 2.0 \
  --gmr_root /home/air/wsm/GMR
```

Expected: output includes `Wrote ... -> .../upstairs01_stageii.json` and reports `frames.shape[1] == 79` via the script assertion.

- [ ] **Step 6: Convert downstairs pkl to 79-dim AMP JSON**

Run:

```bash
conda run -n gmr python /home/air/wang-sm/ATEC/motion_data/gmr_to_amp_json.py \
  --pkl /home/air/wang-sm/ATEC/motion_data/g1_29dof/retargeted/downstairs01_stageii.pkl \
  --output /home/air/wang-sm/ATEC/motion_data/g1_29dof/downstairs01_stageii.json \
  --motion_weight 2.0 \
  --gmr_root /home/air/wsm/GMR
```

Expected: output includes `Wrote ... -> .../downstairs01_stageii.json` and reports `frames.shape[1] == 79` via the script assertion.

- [ ] **Step 7: Validate generated stair JSON files strictly**

Run:

```bash
python motion_data/validate_amp_motion_g1.py \
  motion_data/g1_29dof/upstairs01_stageii.json \
  motion_data/g1_29dof/downstairs01_stageii.json
```

Expected:

```text
OK motion_data/g1_29dof/upstairs01_stageii.json: frames=<N> width=79 dt=<positive> weight=2.0
OK motion_data/g1_29dof/downstairs01_stageii.json: frames=<N> width=79 dt=<positive> weight=2.0
```

- [ ] **Step 8: Decide whether to version generated JSON files**

If the repository already versions `motion_data/g1_29dof/*.json`, commit only the JSON files and leave pkl files uncommitted:

```bash
git add motion_data/g1_29dof/upstairs01_stageii.json motion_data/g1_29dof/downstairs01_stageii.json
git commit -m "data: add g1 upstairs and downstairs amp motions"
```

If this environment still lacks `git`, record the generated JSON paths and commit from a shell that has Git installed.

---

### Task 8: Run targeted tests and training smoke test

**Files:**
- No new code files expected.

- [ ] **Step 1: Run lightweight pytest suite for this feature**

Run:

```bash
pytest \
  tests/test_task_a_corridor_training.py \
  tests/test_validate_amp_motion_g1.py \
  tests/test_rsl_rl_script_compat.py::test_play_amp_script_uses_g1_amp_obs_dim_79_fallback \
  tests/test_feet_min_clearance.py::test_corridor_centerline_l1_uses_row_zero_corridor_centers_and_clips_error \
  -q
```

Expected: PASS.

- [ ] **Step 2: Validate all selected G1 AMP motion files**

Run:

```bash
python motion_data/validate_amp_motion_g1.py motion_data/g1_29dof/*.json
```

Expected: every file prints `OK ... width=79`. If any file fails, exclude it from `--amp_motion_files` and fix the source data before full training.

- [ ] **Step 3: Run corridor env training smoke test**

Run in the IsaacLab-capable environment:

```bash
python scripts/rsl_rl/train_amp.py \
  --task ATEC-Isaac-AMP-Unitree-G1-TaskA-Corridor-v0 \
  --num_envs 64 \
  --max_iterations 5 \
  --amp_motion_files motion_data/g1_29dof/*.json \
  --amp_obs_dim 79
```

Expected:

```text
MotionLoaderG1 loads all selected motion files
AMP is enabled
no 73/79 dimension mismatch
terrain generation succeeds
rollout and update complete for 5 iterations
```

- [ ] **Step 4: Export/play smoke test for the latest checkpoint**

Run after Step 3 creates a checkpoint:

```bash
python scripts/rsl_rl/play_amp.py \
  --task ATEC-Isaac-AMP-Unitree-G1-TaskA-Corridor-v0 \
  --num_envs 4 \
  --amp_obs_dim 79
```

Expected:

```text
checkpoint loads
TorchScript policy exports to logs/rsl_rl_amp/<experiment>/<run>/exported/policy.pt
no AMPPPO construction error from amp_obs_dim
```

- [ ] **Step 5: Commit final verification metadata if scripts or docs changed**

If only generated logs changed, do not commit logs. If you added a short note to docs, commit it:

```bash
git add docs/superpowers/specs/2026-06-16-task-a-g1-corridor-training-design.md docs/superpowers/plans/2026-06-16-task-a-g1-corridor-training.md
git commit -m "docs: document task a corridor training implementation"
```

If this environment still lacks `git`, record the verification commands and outputs for the final report.

---

## Self-review notes

Spec coverage:

- New environment without overwriting `ATEC-Isaac-AMP-Unitree-G1-TaskA-v0`: Task 4.
- Parallel Task A corridor terrain sequence: Tasks 1 and 4.
- Start-of-corridor reset instead of geometric center reset: Tasks 2, 3, and 4.
- Straight-walking command ranges with nonzero lateral/yaw recovery: Task 4.
- Centerline penalty relative to each corridor centerline, not global `y=0`: Task 2.
- Strict 79-dimensional AMP motion data validation: Task 5.
- Upstairs/downstairs GMR retarget and JSON conversion: Task 7.
- `play_amp.py` 73-to-79 fallback fix: Task 6.
- Smoke testing and evaluation setup: Task 8.

No placeholders are intentionally left in the task steps. Generated pkl files are explicitly not committed unless requested; generated JSON files are committed only if the repository's existing data-versioning practice allows it.
