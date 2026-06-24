# B2Piper Pico Arm Teleop Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a simulation-only `ATEC-TaskB-B2Piper` collector where one Pico controller teleoperates only the Piper arm and saves only arm/EE/Pico data.

**Architecture:** Keep testable arm teleop math in `atec_rl_lab.teleop`, keep the Pico SDK adapter isolated behind a small reader API, and keep the Isaac Sim/AppLauncher collector as a thin script that wires env, IK, Pico input, action packing, and HDF5 recording. The simulator still receives 20-dimensional B2Piper actions, but the first 12 B2 leg entries are always zero and the saved dataset exposes only the 8 Piper action entries plus arm state and controller signals.

**Tech Stack:** Python 3.11, NumPy, PyTorch, h5py, Isaac Lab `ManagerBasedRLEnv`, ATEC `TaskBEnvB2Cfg`, ATEC `CartesianController`, optional `xrobotoolkit_sdk`.

## Global Constraints

- Simulation target is `ATEC-TaskB-B2Piper`.
- Do not teleoperate B2 legs, body pose, base velocity, or locomotion.
- Do not record B2 leg state, base velocity, base pose, or locomotion commands as primary dataset fields.
- B2 leg action entries must remain zero: `full_action[0:12] = 0.0`.
- Piper arm action entries must occupy `full_action[12:20]`.
- Saved policy/action data must use only the 8-dimensional Piper action.
- Environment action scale is `0.5`.
- Piper joints are `arm_joint1` through `arm_joint8`.
- IK controls `arm_joint1` through `arm_joint6`.
- Gripper controls `arm_joint7` and `arm_joint8`.
- First version uses stable calibrated end-effector orientation, not full controller 6D orientation following.
- Do not train a policy.
- Do not integrate real B2Piper hardware.
- Do not replace current Task B solution or active `demo/solution.py`.

---

## File Structure

- Create `source/atec_rl_lab/atec_rl_lab/teleop/__init__.py`
  - Public exports for arm-only teleop utilities.
- Create `source/atec_rl_lab/atec_rl_lab/teleop/b2piper_arm.py`
  - Pure NumPy utilities for action packing, gripper interpolation, workspace clamping, coordinate conversion, and relative end-effector target mapping.
- Create `source/atec_rl_lab/atec_rl_lab/teleop/arm_hdf5.py`
  - HDF5 output initialization and one-trajectory append helpers.
- Create `source/atec_rl_lab/atec_rl_lab/teleop/pico.py`
  - Pico SDK adapter that reads one controller into a normalized `ControllerSample`.
- Create `scripts/teleop/collect_task_b_b2piper_arm_pico.py`
  - Isaac Sim collection entry point.
- Create `tests/test_b2piper_arm_teleop_utils.py`
  - Unit tests for pure action/mapping utilities.
- Create `tests/test_arm_hdf5_recorder.py`
  - Unit tests for dataset shape, metadata, and absence of leg datasets.
- Create `tests/test_pico_reader_adapter.py`
  - Unit tests for Pico SDK adapter using fake SDK objects.

---

### Task 1: Arm-Only Teleop Math Utilities

**Files:**
- Create: `source/atec_rl_lab/atec_rl_lab/teleop/__init__.py`
- Create: `source/atec_rl_lab/atec_rl_lab/teleop/b2piper_arm.py`
- Test: `tests/test_b2piper_arm_teleop_utils.py`

**Interfaces:**
- Consumes: none.
- Produces:
  - `ArmTeleopConfig`
  - `ArmTeleopMapper`
  - `CalibrationState`
  - `unity_to_robot_position(pos_xyz: np.ndarray) -> np.ndarray`
  - `lerp_gripper(trigger: float) -> np.ndarray`
  - `piper_target_to_action(target_arm_q: np.ndarray, default_arm_q: np.ndarray, action_scale: float = 0.5) -> np.ndarray`
  - `pack_b2piper_action(piper_action_8: np.ndarray) -> np.ndarray`
  - `clamp_workspace(pos: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray`
  - `limit_step(previous: np.ndarray, desired: np.ndarray, max_step: float) -> np.ndarray`

- [ ] **Step 1: Write failing tests for arm action packing, gripper interpolation, workspace clamping, and calibration**

Create `tests/test_b2piper_arm_teleop_utils.py`:

```python
import numpy as np

from atec_rl_lab.teleop.b2piper_arm import (
    ArmTeleopConfig,
    ArmTeleopMapper,
    clamp_workspace,
    lerp_gripper,
    limit_step,
    pack_b2piper_action,
    piper_target_to_action,
    unity_to_robot_position,
)


def test_pack_b2piper_action_keeps_leg_entries_zero():
    piper = np.arange(8, dtype=np.float32) + 1.0

    full = pack_b2piper_action(piper)

    assert full.shape == (20,)
    np.testing.assert_allclose(full[:12], np.zeros(12, dtype=np.float32))
    np.testing.assert_allclose(full[12:], piper)


def test_piper_target_to_action_uses_default_offset_and_scale():
    default = np.array([0.2, -0.2, 0.0, 0.5, -0.5, 0.1, 0.035, -0.035], dtype=np.float32)
    target = default + np.array([0.1, -0.1, 0.0, 0.25, -0.25, 0.05, -0.05, 0.05], dtype=np.float32)

    action = piper_target_to_action(target, default, action_scale=0.5)

    expected = (target - default) / 0.5
    np.testing.assert_allclose(action, expected.astype(np.float32))


def test_lerp_gripper_clamps_trigger_and_interpolates():
    np.testing.assert_allclose(lerp_gripper(-2.0), np.array([0.035, -0.035], dtype=np.float32))
    np.testing.assert_allclose(lerp_gripper(2.0), np.array([-0.015, 0.015], dtype=np.float32))
    np.testing.assert_allclose(lerp_gripper(0.5), np.array([0.010, -0.010], dtype=np.float32))


def test_workspace_clamp_and_step_limit():
    lower = np.array([0.15, -0.45, -0.30], dtype=np.float32)
    upper = np.array([0.85, 0.45, 0.35], dtype=np.float32)
    pos = np.array([2.0, -2.0, 0.0], dtype=np.float32)

    clamped = clamp_workspace(pos, lower, upper)

    np.testing.assert_allclose(clamped, np.array([0.85, -0.45, 0.0], dtype=np.float32))

    previous = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    desired = np.array([0.10, -0.10, 0.01], dtype=np.float32)
    limited = limit_step(previous, desired, max_step=0.03)
    np.testing.assert_allclose(limited, np.array([0.03, -0.03, 0.01], dtype=np.float32))


def test_unity_to_robot_position_matches_sonic_convention():
    unity = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    robot = unity_to_robot_position(unity)

    np.testing.assert_allclose(robot, np.array([-1.0, 3.0, 2.0], dtype=np.float32))


def test_mapper_calibration_has_no_jump_then_tracks_relative_motion():
    cfg = ArmTeleopConfig(
        translation_scale=1.0,
        max_target_step_m=0.50,
        workspace_lower=np.array([0.15, -0.45, -0.30], dtype=np.float32),
        workspace_upper=np.array([0.85, 0.45, 0.35], dtype=np.float32),
    )
    mapper = ArmTeleopMapper(cfg)
    controller0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ee0 = np.array([0.40, 0.0, 0.10], dtype=np.float32)
    quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    mapper.calibrate(controller0, ee0, quat0)
    target0 = mapper.update(controller0, enabled=True)
    target1 = mapper.update(np.array([0.10, -0.10, 0.05], dtype=np.float32), enabled=True)
    held = mapper.update(np.array([0.40, 0.40, 0.40], dtype=np.float32), enabled=False)

    np.testing.assert_allclose(target0.pos_b, ee0)
    np.testing.assert_allclose(target0.quat_b, quat0)
    np.testing.assert_allclose(target1.pos_b, np.array([0.50, -0.10, 0.15], dtype=np.float32))
    np.testing.assert_allclose(held.pos_b, target1.pos_b)
```

- [ ] **Step 2: Run tests and verify they fail because the module is missing**

Run:

```bash
cd /home/air/wang-sm/ATEC
PYTHONPATH=source/atec_rl_lab pytest tests/test_b2piper_arm_teleop_utils.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'atec_rl_lab.teleop'`.

- [ ] **Step 3: Implement arm-only teleop utilities**

Create `source/atec_rl_lab/atec_rl_lab/teleop/__init__.py`:

```python
"""Teleoperation helpers for ATEC simulation tools."""

from .b2piper_arm import (
    ArmTeleopConfig,
    ArmTeleopMapper,
    CalibrationState,
    EeTarget,
    clamp_workspace,
    lerp_gripper,
    limit_step,
    pack_b2piper_action,
    piper_target_to_action,
    unity_to_robot_position,
)

__all__ = [
    "ArmTeleopConfig",
    "ArmTeleopMapper",
    "CalibrationState",
    "EeTarget",
    "clamp_workspace",
    "lerp_gripper",
    "limit_step",
    "pack_b2piper_action",
    "piper_target_to_action",
    "unity_to_robot_position",
]
```

Create `source/atec_rl_lab/atec_rl_lab/teleop/b2piper_arm.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


ACTION_SCALE = 0.5
LEG_ACTION_DIM = 12
ARM_ACTION_DIM = 8
FULL_ACTION_DIM = LEG_ACTION_DIM + ARM_ACTION_DIM
GRIPPER_OPEN = np.array([0.035, -0.035], dtype=np.float32)
GRIPPER_CLOSE = np.array([-0.015, 0.015], dtype=np.float32)


@dataclass(frozen=True)
class ArmTeleopConfig:
    translation_scale: float = 1.0
    max_target_step_m: float = 0.03
    workspace_lower: np.ndarray = field(
        default_factory=lambda: np.array([0.15, -0.45, -0.30], dtype=np.float32)
    )
    workspace_upper: np.ndarray = field(
        default_factory=lambda: np.array([0.85, 0.45, 0.35], dtype=np.float32)
    )
    axis_map: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))


@dataclass(frozen=True)
class CalibrationState:
    controller_ref_robot: np.ndarray
    ee_ref_pos_b: np.ndarray
    ee_ref_quat_b: np.ndarray


@dataclass(frozen=True)
class EeTarget:
    pos_b: np.ndarray
    quat_b: np.ndarray


def _as_float32_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {arr.shape}")
    return arr


def unity_to_robot_position(pos_xyz: np.ndarray) -> np.ndarray:
    pos = _as_float32_vector(pos_xyz, 3, "pos_xyz")
    return np.array([-pos[0], pos[2], pos[1]], dtype=np.float32)


def lerp_gripper(trigger: float) -> np.ndarray:
    alpha = float(np.clip(trigger, 0.0, 1.0))
    return ((1.0 - alpha) * GRIPPER_OPEN + alpha * GRIPPER_CLOSE).astype(np.float32)


def piper_target_to_action(
    target_arm_q: np.ndarray,
    default_arm_q: np.ndarray,
    action_scale: float = ACTION_SCALE,
) -> np.ndarray:
    target = _as_float32_vector(target_arm_q, ARM_ACTION_DIM, "target_arm_q")
    default = _as_float32_vector(default_arm_q, ARM_ACTION_DIM, "default_arm_q")
    return ((target - default) / float(action_scale)).astype(np.float32)


def pack_b2piper_action(piper_action_8: np.ndarray) -> np.ndarray:
    piper = _as_float32_vector(piper_action_8, ARM_ACTION_DIM, "piper_action_8")
    full = np.zeros(FULL_ACTION_DIM, dtype=np.float32)
    full[LEG_ACTION_DIM:] = piper
    return full


def clamp_workspace(pos: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    p = _as_float32_vector(pos, 3, "pos")
    lo = _as_float32_vector(lower, 3, "lower")
    hi = _as_float32_vector(upper, 3, "upper")
    return np.minimum(np.maximum(p, lo), hi).astype(np.float32)


def limit_step(previous: np.ndarray, desired: np.ndarray, max_step: float) -> np.ndarray:
    prev = _as_float32_vector(previous, 3, "previous")
    des = _as_float32_vector(desired, 3, "desired")
    delta = np.clip(des - prev, -float(max_step), float(max_step))
    return (prev + delta).astype(np.float32)


class ArmTeleopMapper:
    def __init__(self, config: ArmTeleopConfig | None = None):
        self.config = config or ArmTeleopConfig()
        self.calibration: CalibrationState | None = None
        self._last_target: EeTarget | None = None

    @property
    def is_calibrated(self) -> bool:
        return self.calibration is not None

    def calibrate(
        self,
        controller_pos_robot: np.ndarray,
        ee_pos_b: np.ndarray,
        ee_quat_b: np.ndarray,
    ) -> None:
        controller = _as_float32_vector(controller_pos_robot, 3, "controller_pos_robot")
        ee_pos = clamp_workspace(ee_pos_b, self.config.workspace_lower, self.config.workspace_upper)
        ee_quat = _as_float32_vector(ee_quat_b, 4, "ee_quat_b")
        norm = np.linalg.norm(ee_quat)
        if norm <= 0.0:
            raise ValueError("ee_quat_b must be non-zero")
        ee_quat = (ee_quat / norm).astype(np.float32)
        self.calibration = CalibrationState(controller, ee_pos, ee_quat)
        self._last_target = EeTarget(ee_pos, ee_quat)

    def reset(self) -> None:
        self.calibration = None
        self._last_target = None

    def update(self, controller_pos_robot: np.ndarray, enabled: bool) -> EeTarget:
        if self.calibration is None:
            raise RuntimeError("ArmTeleopMapper must be calibrated before update()")
        if not enabled and self._last_target is not None:
            return self._last_target

        controller = _as_float32_vector(controller_pos_robot, 3, "controller_pos_robot")
        delta_robot = controller - self.calibration.controller_ref_robot
        mapped_delta = self.config.axis_map.astype(np.float32) @ delta_robot
        desired = self.calibration.ee_ref_pos_b + mapped_delta * float(self.config.translation_scale)
        desired = clamp_workspace(desired, self.config.workspace_lower, self.config.workspace_upper)

        previous = self._last_target.pos_b if self._last_target is not None else self.calibration.ee_ref_pos_b
        limited = limit_step(previous, desired, self.config.max_target_step_m)
        self._last_target = EeTarget(limited, self.calibration.ee_ref_quat_b)
        return self._last_target
```

- [ ] **Step 4: Run utility tests and verify they pass**

Run:

```bash
cd /home/air/wang-sm/ATEC
PYTHONPATH=source/atec_rl_lab pytest tests/test_b2piper_arm_teleop_utils.py -q
```

Expected: PASS, 6 tests.

- [ ] **Step 5: Commit Task 1**

```bash
cd /home/air/wang-sm/ATEC
git add source/atec_rl_lab/atec_rl_lab/teleop/__init__.py \
  source/atec_rl_lab/atec_rl_lab/teleop/b2piper_arm.py \
  tests/test_b2piper_arm_teleop_utils.py
git commit -m "Add B2Piper arm teleop utilities"
```

---

### Task 2: Arm-Only HDF5 Recorder

**Files:**
- Create: `source/atec_rl_lab/atec_rl_lab/teleop/arm_hdf5.py`
- Modify: `source/atec_rl_lab/atec_rl_lab/teleop/__init__.py`
- Test: `tests/test_arm_hdf5_recorder.py`

**Interfaces:**
- Consumes:
  - NumPy arrays from collection loop.
- Produces:
  - `ArmTrajectory`
  - `init_arm_hdf5(output_dir: str, metadata: dict[str, Any]) -> str`
  - `append_arm_trajectory(traj_path: str, traj_idx: int, traj: ArmTrajectory) -> None`

- [ ] **Step 1: Write failing tests for arm-only HDF5 output**

Create `tests/test_arm_hdf5_recorder.py`:

```python
import h5py
import numpy as np

from atec_rl_lab.teleop.arm_hdf5 import ArmTrajectory, append_arm_trajectory, init_arm_hdf5


def _sample_traj(length: int = 3) -> ArmTrajectory:
    return ArmTrajectory(
        arm_qpos=np.ones((length, 8), dtype=np.float32),
        arm_qvel=np.ones((length, 8), dtype=np.float32) * 2.0,
        arm_action=np.ones((length, 8), dtype=np.float32) * 3.0,
        ee_pos=np.ones((length, 3), dtype=np.float32) * 4.0,
        ee_quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (length, 1)),
        ee_target_pos=np.ones((length, 3), dtype=np.float32) * 5.0,
        ee_target_quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (length, 1)),
        controller_pos=np.ones((length, 3), dtype=np.float32) * 6.0,
        controller_quat=np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (length, 1)),
        controller_buttons=np.ones((length, 4), dtype=np.float32) * 7.0,
        score=np.arange(length, dtype=np.float32),
    )


def test_init_arm_hdf5_writes_metadata(tmp_path):
    path = init_arm_hdf5(
        str(tmp_path),
        {
            "env_id": "ATEC-TaskB-B2Piper",
            "arm_joint_names": ["arm_joint1", "arm_joint2"],
            "action_scale": 0.5,
        },
    )

    with h5py.File(path, "r") as f:
        assert f.attrs["env_id"] == "ATEC-TaskB-B2Piper"
        assert f.attrs["action_scale"] == 0.5
        assert "arm_joint_names_json" in f.attrs


def test_append_arm_trajectory_saves_only_arm_fields(tmp_path):
    path = init_arm_hdf5(str(tmp_path), {"env_id": "ATEC-TaskB-B2Piper"})

    append_arm_trajectory(path, 0, _sample_traj())

    with h5py.File(path, "r") as f:
        grp = f["traj_0000"]
        assert grp["obs/arm_qpos"].shape == (3, 8)
        assert grp["obs/arm_qvel"].shape == (3, 8)
        assert grp["actions/arm_action"].shape == (3, 8)
        assert grp["ee_pos"].shape == (3, 3)
        assert grp["controller_buttons"].shape == (3, 4)
        assert "leg_qpos" not in grp
        assert "base_pose" not in grp
        assert "locomotion" not in grp
```

- [ ] **Step 2: Run tests and verify they fail because `arm_hdf5` is missing**

Run:

```bash
cd /home/air/wang-sm/ATEC
PYTHONPATH=source/atec_rl_lab pytest tests/test_arm_hdf5_recorder.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'atec_rl_lab.teleop.arm_hdf5'`.

- [ ] **Step 3: Implement HDF5 recorder**

Create `source/atec_rl_lab/atec_rl_lab/teleop/arm_hdf5.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import os

import h5py
import numpy as np


@dataclass(frozen=True)
class ArmTrajectory:
    arm_qpos: np.ndarray
    arm_qvel: np.ndarray
    arm_action: np.ndarray
    ee_pos: np.ndarray
    ee_quat: np.ndarray
    ee_target_pos: np.ndarray
    ee_target_quat: np.ndarray
    controller_pos: np.ndarray
    controller_quat: np.ndarray
    controller_buttons: np.ndarray
    score: np.ndarray | None = None


def _store_attr(group: h5py.File, key: str, value: Any) -> None:
    if isinstance(value, (str, int, float, bool, np.number)):
        group.attrs[key] = value
    else:
        group.attrs[f"{key}_json"] = json.dumps(value)


def init_arm_hdf5(output_dir: str, metadata: dict[str, Any]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    traj_path = os.path.join(output_dir, "trajectory.hdf5")
    with h5py.File(traj_path, "w") as f:
        for key, value in metadata.items():
            _store_attr(f, key, value)
    return traj_path


def _require_shape(name: str, array: np.ndarray, trailing_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 1 + len(trailing_shape) or arr.shape[1:] != trailing_shape:
        raise ValueError(f"{name} must have shape (T, {', '.join(map(str, trailing_shape))}), got {arr.shape}")
    return arr


def _trajectory_arrays(traj: ArmTrajectory) -> dict[str, np.ndarray]:
    arrays = {
        "obs/arm_qpos": _require_shape("arm_qpos", traj.arm_qpos, (8,)),
        "obs/arm_qvel": _require_shape("arm_qvel", traj.arm_qvel, (8,)),
        "actions/arm_action": _require_shape("arm_action", traj.arm_action, (8,)),
        "ee_pos": _require_shape("ee_pos", traj.ee_pos, (3,)),
        "ee_quat": _require_shape("ee_quat", traj.ee_quat, (4,)),
        "ee_target_pos": _require_shape("ee_target_pos", traj.ee_target_pos, (3,)),
        "ee_target_quat": _require_shape("ee_target_quat", traj.ee_target_quat, (4,)),
        "controller_pos": _require_shape("controller_pos", traj.controller_pos, (3,)),
        "controller_quat": _require_shape("controller_quat", traj.controller_quat, (4,)),
        "controller_buttons": np.asarray(traj.controller_buttons, dtype=np.float32),
    }
    if arrays["controller_buttons"].ndim != 2:
        raise ValueError(f"controller_buttons must have shape (T, N), got {arrays['controller_buttons'].shape}")
    if traj.score is not None:
        arrays["score"] = np.asarray(traj.score, dtype=np.float32).reshape(-1)
    lengths = {name: value.shape[0] for name, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"all trajectory arrays must share T, got {lengths}")
    return arrays


def append_arm_trajectory(traj_path: str, traj_idx: int, traj: ArmTrajectory) -> None:
    arrays = _trajectory_arrays(traj)
    with h5py.File(traj_path, "a") as f:
        group_name = f"traj_{traj_idx:04d}"
        if group_name in f:
            raise ValueError(f"{group_name} already exists in {traj_path}")
        grp = f.create_group(group_name)
        for name, array in arrays.items():
            grp.create_dataset(name, data=array, compression="gzip")
```

Update `source/atec_rl_lab/atec_rl_lab/teleop/__init__.py`:

```python
from .arm_hdf5 import ArmTrajectory, append_arm_trajectory, init_arm_hdf5
```

Add these entries to `__all__`:

```python
"ArmTrajectory",
"append_arm_trajectory",
"init_arm_hdf5",
```

- [ ] **Step 4: Run HDF5 tests and existing utility tests**

Run:

```bash
cd /home/air/wang-sm/ATEC
PYTHONPATH=source/atec_rl_lab pytest tests/test_arm_hdf5_recorder.py tests/test_b2piper_arm_teleop_utils.py -q
```

Expected: PASS, 8 tests.

- [ ] **Step 5: Commit Task 2**

```bash
cd /home/air/wang-sm/ATEC
git add source/atec_rl_lab/atec_rl_lab/teleop/__init__.py \
  source/atec_rl_lab/atec_rl_lab/teleop/arm_hdf5.py \
  tests/test_arm_hdf5_recorder.py
git commit -m "Add arm-only HDF5 recorder"
```

---

### Task 3: Pico Controller Reader Adapter

**Files:**
- Create: `source/atec_rl_lab/atec_rl_lab/teleop/pico.py`
- Modify: `source/atec_rl_lab/atec_rl_lab/teleop/__init__.py`
- Test: `tests/test_pico_reader_adapter.py`

**Interfaces:**
- Consumes:
  - `unity_to_robot_position(pos_xyz: np.ndarray) -> np.ndarray`
- Produces:
  - `ControllerSample`
  - `read_controller_sample(xrt_module: Any, side: str) -> ControllerSample | None`
  - `PicoControllerReader`

- [ ] **Step 1: Write failing fake-SDK tests**

Create `tests/test_pico_reader_adapter.py`:

```python
from types import SimpleNamespace

import numpy as np

from atec_rl_lab.teleop.pico import PicoControllerReader, read_controller_sample


class FakeXrt:
    def __init__(self):
        self.right_pose = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        self.left_pose = [-1.0, 4.0, 5.0, 0.1, 0.2, 0.3, 0.9]

    def get_right_controller_pose(self):
        return self.right_pose

    def get_left_controller_pose(self):
        return self.left_pose

    def get_right_trigger(self):
        return 0.75

    def get_left_trigger(self):
        return 0.25

    def get_right_grip(self):
        return 1.0

    def get_left_grip(self):
        return 0.0

    def get_A_button(self):
        return True

    def get_X_button(self):
        return False

    def get_time_stamp_ns(self):
        return 123456


def test_read_right_controller_sample_normalizes_pose_and_buttons():
    sample = read_controller_sample(FakeXrt(), "right")

    assert sample is not None
    np.testing.assert_allclose(sample.pos_robot, np.array([-1.0, 3.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(sample.quat_xyzw, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    assert sample.trigger == 0.75
    assert sample.grip == 1.0
    assert sample.primary is True
    assert sample.timestamp_ns == 123456


def test_read_left_controller_sample_uses_left_buttons():
    sample = read_controller_sample(FakeXrt(), "left")

    assert sample is not None
    np.testing.assert_allclose(sample.pos_robot, np.array([1.0, 5.0, 4.0], dtype=np.float32))
    assert sample.trigger == 0.25
    assert sample.grip == 0.0
    assert sample.primary is False


def test_reader_returns_none_when_pose_is_invalid():
    fake = FakeXrt()
    fake.right_pose = [1.0, 2.0]

    sample = read_controller_sample(fake, "right")

    assert sample is None


def test_reader_class_delegates_to_module():
    fake = FakeXrt()
    reader = PicoControllerReader(fake, side="right")

    sample = reader.read()

    assert sample is not None
    assert sample.enabled is True
    np.testing.assert_allclose(sample.buttons_vector(), np.array([0.75, 1.0, 1.0, 1.0], dtype=np.float32))
```

- [ ] **Step 2: Run tests and verify they fail because `pico.py` is missing**

Run:

```bash
cd /home/air/wang-sm/ATEC
PYTHONPATH=source/atec_rl_lab pytest tests/test_pico_reader_adapter.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'atec_rl_lab.teleop.pico'`.

- [ ] **Step 3: Implement Pico adapter**

Create `source/atec_rl_lab/atec_rl_lab/teleop/pico.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .b2piper_arm import unity_to_robot_position


@dataclass(frozen=True)
class ControllerSample:
    pos_robot: np.ndarray
    quat_xyzw: np.ndarray
    trigger: float
    grip: float
    primary: bool
    timestamp_ns: int

    @property
    def enabled(self) -> bool:
        return self.grip > 0.5

    def buttons_vector(self) -> np.ndarray:
        return np.array(
            [
                float(self.trigger),
                float(self.grip),
                1.0 if self.primary else 0.0,
                1.0 if self.enabled else 0.0,
            ],
            dtype=np.float32,
        )


_POSE_FUNC = {
    "left": "get_left_controller_pose",
    "right": "get_right_controller_pose",
}
_TRIGGER_FUNC = {
    "left": "get_left_trigger",
    "right": "get_right_trigger",
}
_GRIP_FUNC = {
    "left": "get_left_grip",
    "right": "get_right_grip",
}
_PRIMARY_FUNC = {
    "left": "get_X_button",
    "right": "get_A_button",
}


def _call_or_default(module: Any, func_name: str, default: Any) -> Any:
    func = getattr(module, func_name, None)
    if not callable(func):
        return default
    try:
        return func()
    except Exception:
        return default


def _pose_to_arrays(raw_pose: Any) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        arr = np.asarray(raw_pose, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.shape[0] < 7:
        return None
    pos_unity = arr[:3]
    quat_xyzw = arr[3:7]
    quat_norm = float(np.linalg.norm(quat_xyzw))
    if quat_norm <= 0.0:
        return None
    return unity_to_robot_position(pos_unity), (quat_xyzw / quat_norm).astype(np.float32)


def read_controller_sample(xrt_module: Any, side: str) -> ControllerSample | None:
    if side not in _POSE_FUNC:
        raise ValueError("side must be 'left' or 'right'")
    raw_pose = _call_or_default(xrt_module, _POSE_FUNC[side], None)
    parsed = _pose_to_arrays(raw_pose)
    if parsed is None:
        return None
    pos_robot, quat_xyzw = parsed
    trigger = float(np.clip(_call_or_default(xrt_module, _TRIGGER_FUNC[side], 0.0), 0.0, 1.0))
    grip = float(np.clip(_call_or_default(xrt_module, _GRIP_FUNC[side], 0.0), 0.0, 1.0))
    primary = bool(_call_or_default(xrt_module, _PRIMARY_FUNC[side], False))
    timestamp_ns = int(_call_or_default(xrt_module, "get_time_stamp_ns", 0))
    return ControllerSample(pos_robot, quat_xyzw, trigger, grip, primary, timestamp_ns)


class PicoControllerReader:
    def __init__(self, xrt_module: Any, side: str = "right"):
        if side not in _POSE_FUNC:
            raise ValueError("side must be 'left' or 'right'")
        self.xrt = xrt_module
        self.side = side

    def read(self) -> ControllerSample | None:
        return read_controller_sample(self.xrt, self.side)
```

Update `source/atec_rl_lab/atec_rl_lab/teleop/__init__.py`:

```python
from .pico import ControllerSample, PicoControllerReader, read_controller_sample
```

Add these entries to `__all__`:

```python
"ControllerSample",
"PicoControllerReader",
"read_controller_sample",
```

- [ ] **Step 4: Run Pico adapter tests and prior unit tests**

Run:

```bash
cd /home/air/wang-sm/ATEC
PYTHONPATH=source/atec_rl_lab pytest \
  tests/test_pico_reader_adapter.py \
  tests/test_b2piper_arm_teleop_utils.py \
  tests/test_arm_hdf5_recorder.py \
  -q
```

Expected: PASS, 12 tests.

- [ ] **Step 5: Commit Task 3**

```bash
cd /home/air/wang-sm/ATEC
git add source/atec_rl_lab/atec_rl_lab/teleop/__init__.py \
  source/atec_rl_lab/atec_rl_lab/teleop/pico.py \
  tests/test_pico_reader_adapter.py
git commit -m "Add Pico controller reader adapter"
```

---

### Task 4: Isaac Sim Arm-Only Collection Script

**Files:**
- Create: `scripts/teleop/collect_task_b_b2piper_arm_pico.py`

**Interfaces:**
- Consumes:
  - `ArmTeleopConfig`
  - `ArmTeleopMapper`
  - `ArmTrajectory`
  - `PicoControllerReader`
  - `append_arm_trajectory`
  - `init_arm_hdf5`
  - `lerp_gripper`
  - `pack_b2piper_action`
  - `piper_target_to_action`
- Produces:
  - CLI script that records arm-only trajectories to `OUTPUT_DIR/trajectory.hdf5`.

- [ ] **Step 1: Write a compile smoke test command in the task notes**

No pytest import test is used for this script because importing it launches Isaac Sim through `AppLauncher`. The automated check is Python compilation:

```bash
cd /home/air/wang-sm/ATEC
python -m py_compile scripts/teleop/collect_task_b_b2piper_arm_pico.py
```

Expected before the file exists: FAIL with `No such file or directory`.

- [ ] **Step 2: Create the collection script**

Create `scripts/teleop/collect_task_b_b2piper_arm_pico.py`:

```python
"""Collect arm-only Pico teleoperation demos in ATEC-TaskB-B2Piper.

Example:
    PYTHONPATH=source/atec_rl_lab python scripts/teleop/collect_task_b_b2piper_arm_pico.py \
        --num_episodes 1 \
        --max_steps_per_episode 500 \
        --output_dir datasets/task_b_b2piper_arm_pico_smoke \
        --headless
"""

from __future__ import annotations

import argparse
import os
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect B2Piper arm-only Pico teleop demos.")
parser.add_argument("--num_episodes", type=int, default=1)
parser.add_argument("--max_steps_per_episode", type=int, default=1000)
parser.add_argument("--output_dir", type=str, default="datasets/task_b_b2piper_arm_pico")
parser.add_argument("--controller", choices=["left", "right"], default="right")
parser.add_argument("--translation_scale", type=float, default=1.0)
parser.add_argument("--save_images", action="store_true", default=False)
parser.add_argument("--wait_for_pico", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.save_images:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

from atec_rl_lab.tasks.task_b import TaskBEnvB2Cfg  # noqa: E402
from atec_rl_lab.utils import CartesianController  # noqa: E402
from atec_rl_lab.teleop import (  # noqa: E402
    ArmTeleopConfig,
    ArmTeleopMapper,
    ArmTrajectory,
    PicoControllerReader,
    append_arm_trajectory,
    init_arm_hdf5,
    lerp_gripper,
    pack_b2piper_action,
    piper_target_to_action,
)

try:
    import xrobotoolkit_sdk as xrt  # noqa: E402
except ImportError:
    xrt = None


ENV_ID = "ATEC-TaskB-B2Piper"
EE_BODY_NAME = "gripper_base"
ARM_JOINT_NAMES = [f"arm_joint{i}" for i in range(1, 7)]
GRIPPER_JOINT_NAMES = ["arm_joint7", "arm_joint8"]
ALL_ARM_JOINT_NAMES = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES
ACTION_SCALE = 0.5


def build_env() -> ManagerBasedRLEnv:
    cfg = TaskBEnvB2Cfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = max(10.0, float(args_cli.max_steps_per_episode) * 0.02 + 2.0)
    return ManagerBasedRLEnv(cfg)


def current_ee_pose_b(robot, ee_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    root_pose_w = robot.data.root_pose_w
    ee_pose_w = robot.data.body_pose_w[:, ee_idx]
    return subtract_frame_transforms(
        root_pose_w[:, :3],
        root_pose_w[:, 3:],
        ee_pose_w[:, :3],
        ee_pose_w[:, 3:],
    )


def safe_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32)


def make_metadata() -> dict:
    return {
        "env_id": ENV_ID,
        "robot_joint_order": "12 B2 leg joints followed by arm_joint1..arm_joint8",
        "arm_joint_names": ALL_ARM_JOINT_NAMES,
        "action_scale": ACTION_SCALE,
        "controller_side": args_cli.controller,
        "controller_quat_order": "xyzw",
        "controller_to_robot_axis_map": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "translation_scale": args_cli.translation_scale,
        "workspace_bounds": {"lower": [0.15, -0.45, -0.30], "upper": [0.85, 0.45, 0.35]},
        "orientation_mode": "hold_calibrated_ee_quat",
        "calibration_policy": "calibrate on first enabled sample and on primary button",
    }


def wait_for_sample(reader: PicoControllerReader):
    while simulation_app.is_running():
        sample = reader.read()
        if sample is not None:
            return sample
        print("[INFO] Waiting for Pico controller sample...")
        time.sleep(0.25)
        if not args_cli.wait_for_pico:
            return None
    return None


def run_collection() -> None:
    if xrt is None:
        raise RuntimeError("xrobotoolkit_sdk is required for Pico teleoperation collection")
    xrt.init()

    env = build_env()
    device = env.unwrapped.device
    robot = env.unwrapped.scene.articulations["robot"]
    arm_ids, arm_names = robot.find_joints(ARM_JOINT_NAMES)
    gripper_ids, gripper_names = robot.find_joints(GRIPPER_JOINT_NAMES)
    if len(arm_ids) != 6 or len(gripper_ids) != 2:
        raise RuntimeError(f"Expected 6 arm and 2 gripper joints, got {arm_names} and {gripper_names}")

    ee_body_ids, ee_body_names = robot.find_bodies(EE_BODY_NAME)
    if len(ee_body_ids) != 1:
        raise RuntimeError(f"Expected one EE body '{EE_BODY_NAME}', got {ee_body_names}")
    ee_idx = ee_body_ids[0]

    ik_ctrl = CartesianController(
        robot=robot,
        ee_body_name=EE_BODY_NAME,
        arm_joint_names=ARM_JOINT_NAMES,
        num_envs=1,
        device=device,
        command_type="pose",
        lambda_val=0.08,
        max_joint_delta=0.08,
    )
    reader = PicoControllerReader(xrt, side=args_cli.controller)
    teleop_cfg = ArmTeleopConfig(translation_scale=args_cli.translation_scale)
    mapper = ArmTeleopMapper(teleop_cfg)
    traj_path = init_arm_hdf5(args_cli.output_dir, make_metadata())

    for episode_idx in range(args_cli.num_episodes):
        obs, _ = env.reset()
        del obs
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos,
            torch.zeros_like(robot.data.default_joint_vel),
        )
        env.unwrapped.scene.write_data_to_sim()
        env.unwrapped.sim.forward()
        robot.update(dt=env.unwrapped.physics_dt)
        ik_ctrl.reset()

        default_arm_q = safe_numpy(robot.data.default_joint_pos[0, arm_ids + gripper_ids])
        sample = wait_for_sample(reader)
        if sample is None:
            print("[WARN] No Pico sample available; stopping collection.")
            break

        ee_pos_b, ee_quat_b = current_ee_pose_b(robot, ee_idx)
        mapper.calibrate(sample.pos_robot, safe_numpy(ee_pos_b[0]), safe_numpy(ee_quat_b[0]))

        buffers = {
            "arm_qpos": [],
            "arm_qvel": [],
            "arm_action": [],
            "ee_pos": [],
            "ee_quat": [],
            "ee_target_pos": [],
            "ee_target_quat": [],
            "controller_pos": [],
            "controller_quat": [],
            "controller_buttons": [],
            "score": [],
        }
        last_safe_arm_target = safe_numpy(robot.data.joint_pos[0, arm_ids + gripper_ids])
        total_score = 0.0

        for step_idx in range(args_cli.max_steps_per_episode):
            if not simulation_app.is_running():
                break
            sample = reader.read()
            if sample is None:
                ee_target = mapper.update(mapper.calibration.controller_ref_robot, enabled=False)
                trigger = 0.0
                buttons = np.zeros(4, dtype=np.float32)
                controller_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                controller_pos = mapper.calibration.controller_ref_robot
            else:
                if sample.primary:
                    ee_pos_b, ee_quat_b = current_ee_pose_b(robot, ee_idx)
                    mapper.calibrate(sample.pos_robot, safe_numpy(ee_pos_b[0]), safe_numpy(ee_quat_b[0]))
                ee_target = mapper.update(sample.pos_robot, enabled=sample.enabled)
                trigger = sample.trigger
                buttons = sample.buttons_vector()
                controller_quat = sample.quat_xyzw
                controller_pos = sample.pos_robot

            ee_pos_tensor = torch.tensor([ee_target.pos_b], dtype=torch.float32, device=device)
            ee_quat_tensor = torch.tensor([ee_target.quat_b], dtype=torch.float32, device=device)
            try:
                arm6_target = ik_ctrl.compute_base(ee_pos_tensor, ee_quat_tensor)
                arm6_np = safe_numpy(arm6_target[0])
                if not np.isfinite(arm6_np).all():
                    raise FloatingPointError("IK produced non-finite target")
                gripper_np = lerp_gripper(trigger)
                arm_target = np.concatenate([arm6_np, gripper_np]).astype(np.float32)
                last_safe_arm_target = arm_target
            except Exception as exc:
                print(f"[WARN] Holding previous arm target after IK failure: {exc}")
                arm_target = last_safe_arm_target

            piper_action = piper_target_to_action(arm_target, default_arm_q, ACTION_SCALE)
            full_action_np = pack_b2piper_action(piper_action)
            full_action = torch.tensor(full_action_np, dtype=torch.float32, device=device).view(1, -1)

            ee_pose_w = robot.data.body_pose_w[:, ee_idx]
            buffers["arm_qpos"].append(safe_numpy(robot.data.joint_pos[0, arm_ids + gripper_ids]))
            buffers["arm_qvel"].append(safe_numpy(robot.data.joint_vel[0, arm_ids + gripper_ids]))
            buffers["arm_action"].append(piper_action)
            buffers["ee_pos"].append(safe_numpy(ee_pose_w[0, :3]))
            buffers["ee_quat"].append(safe_numpy(ee_pose_w[0, 3:]))
            buffers["ee_target_pos"].append(ee_target.pos_b.astype(np.float32))
            buffers["ee_target_quat"].append(ee_target.quat_b.astype(np.float32))
            buffers["controller_pos"].append(controller_pos.astype(np.float32))
            buffers["controller_quat"].append(controller_quat.astype(np.float32))
            buffers["controller_buttons"].append(buttons.astype(np.float32))
            buffers["score"].append(np.float32(total_score))

            _, reward, terminated, truncated, info = env.step(full_action)
            del info
            robot.update(dt=env.unwrapped.physics_dt)
            if isinstance(reward, torch.Tensor):
                total_score += float(reward.mean().item())
            else:
                total_score += float(reward)
            if bool(terminated.any()) or bool(truncated.any()):
                print(f"[INFO] Episode {episode_idx} ended at step {step_idx}")
                break

        if buffers["arm_action"]:
            traj = ArmTrajectory(
                arm_qpos=np.stack(buffers["arm_qpos"]),
                arm_qvel=np.stack(buffers["arm_qvel"]),
                arm_action=np.stack(buffers["arm_action"]),
                ee_pos=np.stack(buffers["ee_pos"]),
                ee_quat=np.stack(buffers["ee_quat"]),
                ee_target_pos=np.stack(buffers["ee_target_pos"]),
                ee_target_quat=np.stack(buffers["ee_target_quat"]),
                controller_pos=np.stack(buffers["controller_pos"]),
                controller_quat=np.stack(buffers["controller_quat"]),
                controller_buttons=np.stack(buffers["controller_buttons"]),
                score=np.asarray(buffers["score"], dtype=np.float32),
            )
            append_arm_trajectory(traj_path, episode_idx, traj)
            print(f"[INFO] Saved traj_{episode_idx:04d} with {len(buffers['arm_action'])} steps")

    env.close()
    xrt.close()


if __name__ == "__main__":
    try:
        run_collection()
    finally:
        simulation_app.close()
```

- [ ] **Step 3: Compile the script**

Run:

```bash
cd /home/air/wang-sm/ATEC
python -m py_compile scripts/teleop/collect_task_b_b2piper_arm_pico.py
```

Expected: PASS with no output.

- [ ] **Step 4: Run all unit tests from earlier tasks**

Run:

```bash
cd /home/air/wang-sm/ATEC
PYTHONPATH=source/atec_rl_lab pytest \
  tests/test_b2piper_arm_teleop_utils.py \
  tests/test_arm_hdf5_recorder.py \
  tests/test_pico_reader_adapter.py \
  -q
```

Expected: PASS, 12 tests.

- [ ] **Step 5: Commit Task 4**

```bash
cd /home/air/wang-sm/ATEC
git add scripts/teleop/collect_task_b_b2piper_arm_pico.py
git commit -m "Add B2Piper Pico arm collection script"
```

---

### Task 5: Manual Smoke Test And Dataset Inspection

**Files:**
- Modify: none.

**Interfaces:**
- Consumes:
  - `scripts/teleop/collect_task_b_b2piper_arm_pico.py`
  - `datasets/task_b_b2piper_arm_pico_smoke/trajectory.hdf5`
- Produces:
  - Manual verification evidence that one trajectory has arm-only datasets and no leg datasets.

- [ ] **Step 1: Launch one short collection run**

Run with Pico connected:

```bash
cd /home/air/wang-sm/ATEC
PYTHONPATH=source/atec_rl_lab python scripts/teleop/collect_task_b_b2piper_arm_pico.py \
  --num_episodes 1 \
  --max_steps_per_episode 300 \
  --output_dir datasets/task_b_b2piper_arm_pico_smoke \
  --wait_for_pico \
  --headless
```

Expected:

```text
[INFO] Saved traj_0000 with ... steps
```

- [ ] **Step 2: Inspect the HDF5 structure**

Run:

```bash
cd /home/air/wang-sm/ATEC
python - <<'PY'
import h5py

path = "datasets/task_b_b2piper_arm_pico_smoke/trajectory.hdf5"
with h5py.File(path, "r") as f:
    print("attrs:", sorted(f.attrs.keys()))
    grp = f["traj_0000"]
    names = []
    grp.visit(names.append)
    print("datasets:", sorted(names))
    assert grp["obs/arm_qpos"].shape[1] == 8
    assert grp["actions/arm_action"].shape[1] == 8
    assert "leg_qpos" not in grp
    assert "base_pose" not in grp
    assert "locomotion" not in grp
print("arm-only dataset ok")
PY
```

Expected:

```text
arm-only dataset ok
```

- [ ] **Step 3: Verify B2 leg action packing remains zero**

Run:

```bash
cd /home/air/wang-sm/ATEC
PYTHONPATH=source/atec_rl_lab python - <<'PY'
import numpy as np
from atec_rl_lab.teleop import pack_b2piper_action

full = pack_b2piper_action(np.ones(8, dtype=np.float32))
assert np.allclose(full[:12], 0.0)
assert np.allclose(full[12:], 1.0)
print("leg action packing ok")
PY
```

Expected:

```text
leg action packing ok
```

- [ ] **Step 4: Commit smoke-test notes only if a notes file is created**

If implementation adds a notes file such as `docs/task_b_b2piper_pico_arm_smoke.md`, commit it:

```bash
cd /home/air/wang-sm/ATEC
git add docs/task_b_b2piper_pico_arm_smoke.md
git commit -m "Document B2Piper Pico arm smoke test"
```

If no notes file is created, do not commit generated `datasets/` output.

---

## Self-Review

Spec coverage:

- Simulation-only `ATEC-TaskB-B2Piper`: Task 4 builds the collector around `TaskBEnvB2Cfg`.
- B2 legs not controlled and not recorded: Task 1 enforces zero first 12 action entries; Task 2 only writes arm datasets; Task 5 inspects absence of leg/base/locomotion fields.
- Pico controller input: Task 3 implements a Pico SDK adapter and fake-SDK tests.
- Relative calibration and workspace bounds: Task 1 implements mapper calibration, clamping, and step limiting.
- IK and 20D env action construction: Task 4 wires `CartesianController.compute_base()` and action packing.
- HDF5 format and metadata: Task 2 implements required fields and file attrs.
- Error handling for Pico absence and IK failure: Task 4 waits or exits for missing samples and holds previous safe target on IK failure.
- Testing and manual verification: Tasks 1-3 provide unit tests; Task 4 compiles the script; Task 5 provides smoke-test and HDF5 inspection commands.

Placeholder scan:

- The plan contains no unresolved markers or vague catch-all steps.
- Every task has exact files, commands, and expected outcomes.

Type consistency:

- `ControllerSample.pos_robot` matches `ArmTeleopMapper.update(controller_pos_robot, enabled)`.
- `ArmTrajectory.arm_action` is consistently 8-dimensional.
- `pack_b2piper_action()` consistently returns a 20-dimensional action with zero leg entries.
- `piper_target_to_action()` consistently consumes target and default 8-dimensional arm joint vectors.
