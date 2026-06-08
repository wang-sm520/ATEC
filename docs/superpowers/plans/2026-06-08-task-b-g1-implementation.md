# Task B G1 Visual Contact Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a G1 Task B baseline that searches with RGB-D, approaches detected objects, earns contact score with the hands, and attempts conservative push/sweep placement near the target circle.

**Architecture:** Keep the submission logic self-contained in `demo/solution_task_b_g1.py` so the current Task D `demo/solution.py` is not disturbed. Reuse the existing G1 velocity-policy bridge pattern for locomotion, add pure-Python odometry/planner/perception modules inside the Task B solution file, and add separate Isaac probe/evaluation scripts for sensor calibration and local scoring.

**Tech Stack:** Python 3.12, PyTorch, IsaacLab/Gymnasium for probe and evaluation scripts, `unittest` for non-Isaac tests, optional Pillow for saving debug PNGs when available.

---

## Scope and file map

The approved design covers one baseline subsystem: Task B G1 visual search/contact/push. It does not need to be split into separate implementation plans.

Files to create:

- `demo/solution_task_b_g1.py` — self-contained Task B G1 `AlgSolution` with locomotion bridge, odometry, RGB-D perception, planner, and local hand override.
- `tests/task_b/test_solution_task_b_g1.py` — fast non-Isaac tests for geometry, policy input shape, RGB-D synthetic detection, planner transitions, hand override, and `AlgSolution` reset/predict glue.
- `scripts/probe_task_b_g1_camera.py` — Isaac script to print image keys/shapes/ranges and save Task B G1 head RGB-D snapshots.
- `scripts/probe_task_b_g1_objects.py` — Isaac script that records env.scene object truth and compares it with submission-style RGB-D detections for calibration only.
- `scripts/eval_task_b_g1.py` — Isaac local evaluator that imports `demo.solution_task_b_g1.AlgSolution` directly and runs `ATEC-TaskB-G1` without overwriting the current Task D `demo/solution.py`.

Files to read for reference while implementing:

- `docs/superpowers/specs/2026-06-08-task-b-g1-design.md` — approved design.
- `source/atec_rl_lab/atec_rl_lab/train/task_d/policy_bridge.py` — existing reusable G1 velocity bridge pattern.
- `demo/solution.py` — current self-contained Task D policy bridge and odometry style.
- `source/atec_rl_lab/atec_rl_lab/assets/robots/g1/g1_29dof_dex1.py` — G1 joint order and camera links.
- `source/atec_rl_lab/atec_rl_lab/tasks/task_b/env_cfg.py` — target center, object count, scoring params.
- `source/atec_rl_lab/atec_rl_lab/tasks/task_base/envs_base_cfg.py:193-228` — image observation keys.

Commit strategy:

- Commit after each task that passes its listed verification command.
- Do not overwrite `demo/solution.py` in this plan. When this baseline is good enough for submission, make a separate packaging change that copies `solution_task_b_g1.py` to `solution.py` in a dedicated branch.

---

### Task 1: Add test scaffold and pure helper expectations

**Files:**
- Create: `tests/task_b/test_solution_task_b_g1.py`
- Create: `demo/solution_task_b_g1.py`

- [ ] **Step 1: Write the failing helper tests**

Create `tests/task_b/test_solution_task_b_g1.py` with this initial content:

```python
import math
import os
import sys
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment-specific
    torch = None

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from demo import solution_task_b_g1 as sol  # noqa: E402


class GeometryHelperTest(unittest.TestCase):
    def test_wrap_to_pi_keeps_angles_in_closed_range(self):
        self.assertAlmostEqual(sol._wrap_to_pi(0.0), 0.0)
        self.assertAlmostEqual(sol._wrap_to_pi(3.0 * math.pi), math.pi)
        self.assertAlmostEqual(sol._wrap_to_pi(-3.0 * math.pi), math.pi)
        self.assertLessEqual(sol._wrap_to_pi(123.4), math.pi)
        self.assertGreaterEqual(sol._wrap_to_pi(123.4), -math.pi)

    def test_clamp_limits_value(self):
        self.assertEqual(sol._clamp(2.0, -1.0, 1.0), 1.0)
        self.assertEqual(sol._clamp(-2.0, -1.0, 1.0), -1.0)
        self.assertEqual(sol._clamp(0.25, -1.0, 1.0), 0.25)

    def test_pose_distance_and_bearing(self):
        pose = sol.Pose2D(x=-10.0, y=-10.0, yaw=0.0)
        self.assertAlmostEqual(pose.distance_to((-9.0, -10.0)), 1.0)
        self.assertAlmostEqual(pose.bearing_to((-10.0, -9.0)), math.pi / 2.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the helper tests to verify they fail**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py -q
```

Expected: FAIL during import with `ImportError` or `ModuleNotFoundError` because `demo/solution_task_b_g1.py` does not exist yet.

- [ ] **Step 3: Add the minimal helper implementation**

Create `demo/solution_task_b_g1.py` with this initial content:

```python
"""ATEC Task B G1 baseline: visual search, contact scoring, conservative push.

This file is intentionally self-contained for submission packaging. It does not
import atec_rl_lab modules at runtime; Isaac-only helpers live in scripts/.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - the eval image provides torch.
    torch = None


_DIR = os.path.dirname(os.path.abspath(__file__))
_POLICY_PATH = None
for _name in ("policy_a.pt", "policy.pt"):
    _candidate = os.path.join(_DIR, _name)
    if os.path.exists(_candidate):
        _POLICY_PATH = _candidate
        break


def _wrap_to_pi(angle: float) -> float:
    wrapped = (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if math.isclose(wrapped, -math.pi, abs_tol=1e-12) else wrapped


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _as_float(value: Any) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    def distance_to(self, xy: tuple[float, float]) -> float:
        return math.hypot(float(xy[0]) - self.x, float(xy[1]) - self.y)

    def bearing_to(self, xy: tuple[float, float]) -> float:
        return math.atan2(float(xy[1]) - self.y, float(xy[0]) - self.x)


@dataclass(frozen=True)
class Detection:
    track_id: int
    label: str
    rel_x: float
    rel_y: float
    distance: float
    confidence: float
    world_x: float
    world_y: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class PlannerOutput:
    phase: str
    command: tuple[float, float, float]
    arm_mode: str
    target_world: tuple[float, float] | None = None


class AlgSolution:
    """Temporary shell. Later tasks replace this with the full controller."""

    def reset(self, **kwargs) -> None:
        return None

    def predicts(self, obs: dict, current_score: float):
        proprio = obs["proprio"]
        action_dim = (int(proprio.shape[-1]) - 12) // 3 if hasattr(proprio, "shape") else 33
        return {"action": [0.0] * action_dim, "giveup": False}
```

- [ ] **Step 4: Run the helper tests to verify they pass**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py -q
```

Expected: PASS with 3 tests.

- [ ] **Step 5: Commit**

```bash
git add demo/solution_task_b_g1.py tests/task_b/test_solution_task_b_g1.py
git commit -m "test: scaffold Task B G1 solution helpers"
```

---

### Task 2: Implement G1 locomotion policy bridge

**Files:**
- Modify: `tests/task_b/test_solution_task_b_g1.py`
- Modify: `demo/solution_task_b_g1.py`

- [ ] **Step 1: Add failing bridge tests**

Append these tests to `tests/task_b/test_solution_task_b_g1.py` before the `if __name__ == "__main__"` block:

```python
@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class G1VelocityPolicyBridgeTest(unittest.TestCase):
    class FakePolicy:
        def __init__(self):
            self.calls = []

        def eval(self):
            return self

        def __call__(self, policy_input):
            self.calls.append(policy_input.detach().clone())
            return torch.zeros((policy_input.shape[0], sol.G1VelocityPolicyBridge.ACTION_DIM_BODY))

    def make_proprio(self, action_dim=33):
        proprio = torch.zeros((1, 12 + 3 * action_dim), dtype=torch.float32)
        proprio[0, 9:12] = torch.tensor([0.0, 0.0, -1.0])
        return proprio

    def test_policy_input_dim_is_960(self):
        bridge = sol.G1VelocityPolicyBridge(policy_path="missing.pt", device="cpu")
        self.assertEqual(bridge.policy_input_dim, 960)

    def test_act_builds_term_major_history_and_returns_full_action_dim(self):
        bridge = sol.G1VelocityPolicyBridge(policy_path="missing.pt", device="cpu")
        fake = self.FakePolicy()
        bridge.policy = fake
        action = bridge.act(self.make_proprio(action_dim=33), (0.25, 0.0, 0.1))
        self.assertEqual(len(action), 33)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(tuple(fake.calls[0].shape), (1, 960))
        cmd_slice = fake.calls[0][0, 30 - 3:30]
        self.assertTrue(torch.allclose(cmd_slice, torch.tensor([0.25, 0.0, 0.1])))

    def test_reset_clears_history_buffers(self):
        bridge = sol.G1VelocityPolicyBridge(policy_path="missing.pt", device="cpu")
        bridge.policy = self.FakePolicy()
        bridge.act(self.make_proprio(action_dim=33), (0.25, 0.0, 0.1))
        self.assertGreater(float(bridge._buf_cmd.abs().sum()), 0.0)
        bridge.reset()
        self.assertEqual(float(bridge._buf_cmd.abs().sum()), 0.0)
```

- [ ] **Step 2: Run the bridge tests to verify they fail**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py::G1VelocityPolicyBridgeTest -q
```

Expected: FAIL with `AttributeError: module 'demo.solution_task_b_g1' has no attribute 'G1VelocityPolicyBridge'`.

- [ ] **Step 3: Add the bridge implementation**

Insert this class in `demo/solution_task_b_g1.py` after the dataclasses and before `AlgSolution`:

```python
class G1VelocityPolicyBridge:
    BODY_29_IDX = list(range(29))
    ACTION_DIM_BODY = 29

    HISTORY_LEN = 10
    DIM_ANG_VEL = 3
    DIM_CMD = 3
    DIM_GRAVITY = 3
    DIM_JP = 29
    DIM_JV = 29
    DIM_LASTACT = 29
    POLICY_INPUT_DIM = HISTORY_LEN * (
        DIM_ANG_VEL + DIM_CMD + DIM_GRAVITY + DIM_JP + DIM_JV + DIM_LASTACT
    )

    TRAINING_ACTION_SCALE_29 = (
        0.231, 0.231, 0.231, 0.231, 0.213, 0.213,
        0.231, 0.231, 0.231, 0.231, 0.213, 0.213,
        0.154, 0.213, 0.213,
        0.373, 0.373, 0.213, 0.373,
        0.23, 0.23, 0.23,
        0.373, 0.373, 0.213, 0.373,
        0.23, 0.23, 0.23,
    )
    EVAL_ACTION_SCALE = 0.5

    def __init__(self, policy_path: str | None, device: str = "cuda"):
        if torch is None:
            raise RuntimeError("torch is required for G1VelocityPolicyBridge")
        self.policy_path = policy_path
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.policy = None
        self.action_scale_ratio = torch.tensor(
            [scale / self.EVAL_ACTION_SCALE for scale in self.TRAINING_ACTION_SCALE_29],
            device=self.device,
            dtype=torch.float32,
        ).view(1, self.ACTION_DIM_BODY)
        self.reset()

    @property
    def policy_input_dim(self) -> int:
        return self.POLICY_INPUT_DIM

    def _make_buffer(self, dim: int):
        return torch.zeros((self.HISTORY_LEN, dim), device=self.device, dtype=torch.float32)

    def reset(self) -> None:
        self._buf_ang_vel = self._make_buffer(self.DIM_ANG_VEL)
        self._buf_cmd = self._make_buffer(self.DIM_CMD)
        self._buf_gravity = self._make_buffer(self.DIM_GRAVITY)
        self._buf_jp = self._make_buffer(self.DIM_JP)
        self._buf_jv = self._make_buffer(self.DIM_JV)
        self._buf_lastact = self._make_buffer(self.DIM_LASTACT)

    @staticmethod
    def _push(buf, row) -> None:
        buf[:-1] = buf[1:].clone()
        buf[-1] = row.reshape(-1)

    def _load_policy(self):
        if self.policy is None:
            if self.policy_path is None:
                raise FileNotFoundError("policy_a.pt or policy.pt is required next to solution_task_b_g1.py")
            self.policy = torch.jit.load(self.policy_path, map_location=self.device)
            self.policy.eval()
        return self.policy

    def _coerce_command(self, command: Sequence[float]):
        out = torch.as_tensor(list(command), device=self.device, dtype=torch.float32).reshape(-1)
        if out.numel() != self.DIM_CMD:
            raise ValueError(f"velocity command must have 3 values, got {out.numel()}")
        return out

    def _build_policy_input(self, proprio, command: Sequence[float]):
        proprio = proprio.to(device=self.device, dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2 or proprio.shape[0] != 1:
            raise ValueError(f"proprio must have shape (1, N) or (N,), got {tuple(proprio.shape)}")

        full_action_dim = (int(proprio.shape[-1]) - 12) // 3
        if 12 + 3 * full_action_dim != int(proprio.shape[-1]):
            raise ValueError(f"invalid ATEC proprio layout: {tuple(proprio.shape)}")
        if full_action_dim < self.ACTION_DIM_BODY:
            raise ValueError(f"G1 policy requires at least 29 action joints, got {full_action_dim}")

        base_ang_vel = proprio[0, 3:6]
        projected_gravity = proprio[0, 9:12]
        cmd = self._coerce_command(command)

        jp_start = 12
        jv_start = jp_start + full_action_dim
        act_start = jv_start + full_action_dim
        joint_pos_body = proprio[0, jp_start:jp_start + self.ACTION_DIM_BODY]
        joint_vel_body = proprio[0, jv_start:jv_start + self.ACTION_DIM_BODY]
        last_action_body = proprio[0, act_start:act_start + self.ACTION_DIM_BODY] / self.action_scale_ratio.reshape(-1)

        self._push(self._buf_ang_vel, base_ang_vel)
        self._push(self._buf_cmd, cmd)
        self._push(self._buf_gravity, projected_gravity)
        self._push(self._buf_jp, joint_pos_body)
        self._push(self._buf_jv, joint_vel_body)
        self._push(self._buf_lastact, last_action_body)

        policy_input = torch.cat(
            [
                self._buf_ang_vel.reshape(-1),
                self._buf_cmd.reshape(-1),
                self._buf_gravity.reshape(-1),
                self._buf_jp.reshape(-1),
                self._buf_jv.reshape(-1),
                self._buf_lastact.reshape(-1),
            ],
            dim=-1,
        ).unsqueeze(0)
        return policy_input, full_action_dim

    def act(self, proprio, command: Sequence[float]) -> list[float]:
        policy_input, full_action_dim = self._build_policy_input(proprio, command)
        with torch.inference_mode():
            action_body = self._load_policy()(policy_input)
        if not isinstance(action_body, torch.Tensor):
            action_body = torch.as_tensor(action_body, device=self.device, dtype=torch.float32)
        if action_body.ndim == 1:
            action_body = action_body.unsqueeze(0)
        action_body = action_body.to(device=self.device, dtype=torch.float32)[:, : self.ACTION_DIM_BODY]
        if action_body.shape[-1] != self.ACTION_DIM_BODY:
            raise ValueError(f"policy returned {action_body.shape[-1]} actions, expected 29")
        action_body = action_body * self.action_scale_ratio
        action_full = torch.zeros((1, full_action_dim), device=self.device, dtype=torch.float32)
        action_full[:, self.BODY_29_IDX] = action_body
        return action_full[0].detach().cpu().tolist()
```

- [ ] **Step 4: Run the bridge tests to verify they pass**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py::G1VelocityPolicyBridgeTest -q
```

Expected: PASS with 3 tests.

- [ ] **Step 5: Run all current Task B unit tests**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py -q
```

Expected: PASS with 6 tests.

- [ ] **Step 6: Commit**

```bash
git add demo/solution_task_b_g1.py tests/task_b/test_solution_task_b_g1.py
git commit -m "feat: add Task B G1 locomotion bridge"
```

---

### Task 3: Implement dead-reckoning odometry

**Files:**
- Modify: `tests/task_b/test_solution_task_b_g1.py`
- Modify: `demo/solution_task_b_g1.py`

- [ ] **Step 1: Add failing odometry tests**

Append these tests before the `if __name__ == "__main__"` block:

```python
class DeadReckoningOdometryTest(unittest.TestCase):
    def make_row(self, vx=0.0, vy=0.0, yaw_rate=0.0):
        row = [0.0] * (12 + 3 * 33)
        row[0] = vx
        row[1] = vy
        row[3:6] = [0.0, 0.0, yaw_rate]
        row[9:12] = [0.0, 0.0, -1.0]
        return row

    def test_integrates_body_velocity_in_world_frame(self):
        odom = sol.DeadReckoningOdometry(dt=0.02, x0=-10.0, y0=-10.0)
        pose = odom.update(self.make_row(vx=1.0, vy=0.0))
        self.assertAlmostEqual(pose.x, -9.98, places=5)
        self.assertAlmostEqual(pose.y, -10.0, places=5)
        self.assertAlmostEqual(pose.yaw, 0.0, places=5)

    def test_integrates_yaw_rate_projected_on_up_axis(self):
        odom = sol.DeadReckoningOdometry(dt=0.02, x0=-10.0, y0=-10.0)
        pose = odom.update(self.make_row(yaw_rate=1.0))
        self.assertAlmostEqual(pose.yaw, 0.02, places=5)

    def test_reset_restores_initial_pose(self):
        odom = sol.DeadReckoningOdometry(dt=0.02, x0=-10.0, y0=-10.0)
        odom.update(self.make_row(vx=1.0, yaw_rate=1.0))
        pose = odom.reset()
        self.assertEqual(pose, sol.Pose2D(-10.0, -10.0, 0.0))
```

- [ ] **Step 2: Run the odometry tests to verify they fail**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py::DeadReckoningOdometryTest -q
```

Expected: FAIL with `AttributeError` because `DeadReckoningOdometry` does not exist.

- [ ] **Step 3: Add odometry implementation**

Insert this class after `G1VelocityPolicyBridge` in `demo/solution_task_b_g1.py`:

```python
class DeadReckoningOdometry:
    def __init__(self, dt: float = 0.02, x0: float = -10.0, y0: float = -10.0, yaw0: float = 0.0):
        self.dt = float(dt)
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.yaw0 = float(yaw0)
        self.reset()

    def reset(self) -> Pose2D:
        self.x = self.x0
        self.y = self.y0
        self.yaw = self.yaw0
        self.vx_b = 0.0
        self.vy_b = 0.0
        return self.pose

    @property
    def pose(self) -> Pose2D:
        return Pose2D(self.x, self.y, self.yaw)

    @staticmethod
    def _normalized(v: list[float]) -> list[float]:
        n = math.sqrt(sum(c * c for c in v))
        return [0.0, 0.0, 1.0] if n <= 1e-8 else [c / n for c in v]

    def update(self, proprio_row: Sequence[float]) -> Pose2D:
        lin = [_as_float(proprio_row[i]) for i in range(0, 3)]
        ang = [_as_float(proprio_row[i]) for i in range(3, 6)]
        grav = [_as_float(proprio_row[i]) for i in range(9, 12)]
        up = self._normalized([-grav[0], -grav[1], -grav[2]])
        yaw_rate = sum(a * u for a, u in zip(ang, up))

        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        self.x += (c * lin[0] - s * lin[1]) * self.dt
        self.y += (s * lin[0] + c * lin[1]) * self.dt
        self.yaw = _wrap_to_pi(self.yaw + yaw_rate * self.dt)
        self.vx_b = lin[0]
        self.vy_b = lin[1]
        return self.pose
```

- [ ] **Step 4: Run the odometry tests to verify they pass**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py::DeadReckoningOdometryTest -q
```

Expected: PASS with 3 tests.

- [ ] **Step 5: Run all current Task B unit tests**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py -q
```

Expected: PASS with 9 tests.

- [ ] **Step 6: Commit**

```bash
git add demo/solution_task_b_g1.py tests/task_b/test_solution_task_b_g1.py
git commit -m "feat: add Task B dead-reckoning odometry"
```

---

### Task 4: Implement planner state machine

**Files:**
- Modify: `tests/task_b/test_solution_task_b_g1.py`
- Modify: `demo/solution_task_b_g1.py`

- [ ] **Step 1: Add failing planner tests**

Append these tests before the `if __name__ == "__main__"` block:

```python
class TaskBPlannerTest(unittest.TestCase):
    def detection(self, track_id=1, world=(-9.0, -10.0), rel=(1.0, 0.0), distance=1.0):
        return sol.Detection(
            track_id=track_id,
            label="object",
            rel_x=rel[0],
            rel_y=rel[1],
            distance=distance,
            confidence=0.9,
            world_x=world[0],
            world_y=world[1],
            bbox=(10, 10, 20, 20),
        )

    def test_search_drives_toward_first_waypoint_without_detections(self):
        planner = sol.TaskBPlanner()
        out = planner.step(sol.Pose2D(-10.0, -10.0, 0.0), [], current_score=0.0)
        self.assertEqual(out.phase, "search")
        self.assertEqual(out.arm_mode, "stow")
        self.assertEqual(len(out.command), 3)

    def test_detection_interrupts_search_and_enters_approach(self):
        planner = sol.TaskBPlanner()
        out = planner.step(sol.Pose2D(-10.0, -10.0, 0.0), [self.detection()], current_score=0.0)
        self.assertEqual(out.phase, "approach_object")
        self.assertEqual(out.target_world, (-9.0, -10.0))

    def test_close_detection_enters_touch_phase(self):
        planner = sol.TaskBPlanner()
        planner.step(sol.Pose2D(-10.0, -10.0, 0.0), [self.detection()], current_score=0.0)
        out = planner.step(
            sol.Pose2D(-9.45, -10.0, 0.0),
            [self.detection(world=(-9.0, -10.0), rel=(0.45, 0.0), distance=0.45)],
            current_score=0.0,
        )
        self.assertEqual(out.phase, "touch_object")
        self.assertEqual(out.arm_mode, "left_touch")

    def test_score_delta_marks_contact_and_verifies_next(self):
        planner = sol.TaskBPlanner()
        planner.step(sol.Pose2D(-10.0, -10.0, 0.0), [self.detection()], current_score=0.0)
        planner.step(sol.Pose2D(-9.45, -10.0, 0.0), [self.detection(distance=0.45)], current_score=0.0)
        out = planner.step(sol.Pose2D(-9.35, -10.0, 0.0), [self.detection(distance=0.35)], current_score=1.0)
        self.assertEqual(out.phase, "verify_or_next")
        self.assertIn(1, planner.touched_track_ids)
```

- [ ] **Step 2: Run planner tests to verify they fail**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py::TaskBPlannerTest -q
```

Expected: FAIL with `AttributeError` because `TaskBPlanner` does not exist.

- [ ] **Step 3: Add planner implementation**

Insert this class after `DeadReckoningOdometry` in `demo/solution_task_b_g1.py`:

```python
class TaskBPlanner:
    TARGET_CENTER = (-3.0, -10.0)
    SEARCH_WAYPOINTS = (
        (-14.0, -14.0, 0.0),
        (-6.0, -14.0, 0.0),
        (-6.0, -12.0, math.pi),
        (-14.0, -12.0, math.pi),
        (-14.0, -10.0, 0.0),
        (-6.0, -10.0, 0.0),
        (-6.0, -8.0, math.pi),
        (-14.0, -8.0, math.pi),
        (-14.0, -6.0, 0.0),
        (-6.0, -6.0, 0.0),
    )

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.phase = "search"
        self.waypoint_idx = 0
        self.active_detection: Detection | None = None
        self.touched_track_ids: set[int] = set()
        self.placed_track_ids: set[int] = set()
        self.prev_score = 0.0
        self.phase_steps = 0

    def step(self, pose: Pose2D, detections: list[Detection], current_score: float) -> PlannerOutput:
        score_delta = float(current_score) - self.prev_score
        self.prev_score = float(current_score)
        if score_delta > 0.0 and self.active_detection is not None:
            self.touched_track_ids.add(self.active_detection.track_id)
            if self.phase in ("touch_object", "push_to_goal"):
                self.phase = "verify_or_next"
                self.phase_steps = 0

        if self.phase == "search":
            fresh = self._choose_detection(detections)
            if fresh is not None:
                self.active_detection = fresh
                self.phase = "approach_object"
                self.phase_steps = 0
                return self._approach_output(pose, fresh)
            return self._search_output(pose)

        if self.phase == "approach_object":
            det = self._refresh_active_detection(detections)
            if det is None:
                self.phase = "search"
                self.active_detection = None
                self.phase_steps = 0
                return self._search_output(pose)
            self.active_detection = det
            if det.distance <= 0.55:
                self.phase = "touch_object"
                self.phase_steps = 0
                return PlannerOutput("touch_object", self._face_and_creep(det), "left_touch", (det.world_x, det.world_y))
            return self._approach_output(pose, det)

        if self.phase == "touch_object":
            det = self._refresh_active_detection(detections) or self.active_detection
            self.phase_steps += 1
            if det is None or self.phase_steps > 120:
                self.phase = "verify_or_next"
                self.phase_steps = 0
                return PlannerOutput("verify_or_next", (0.0, 0.0, 0.0), "stow", None)
            if self._near_target((det.world_x, det.world_y), max_distance=4.0):
                if self.phase_steps > 40:
                    self.phase = "push_to_goal"
                    self.phase_steps = 0
                    return self._push_output(pose, det)
            return PlannerOutput("touch_object", self._face_and_creep(det), "left_touch", (det.world_x, det.world_y))

        if self.phase == "push_to_goal":
            det = self._refresh_active_detection(detections) or self.active_detection
            self.phase_steps += 1
            if det is None or self.phase_steps > 180:
                self.phase = "verify_or_next"
                self.phase_steps = 0
                return PlannerOutput("verify_or_next", (0.0, 0.0, 0.0), "stow", None)
            return self._push_output(pose, det)

        if self.phase == "verify_or_next":
            self.phase_steps += 1
            if self.phase_steps >= 20:
                self.active_detection = None
                self.phase = "search"
                self.phase_steps = 0
                return self._search_output(pose)
            return PlannerOutput("verify_or_next", (0.0, 0.0, 0.0), "stow", None)

        self.phase = "search"
        self.active_detection = None
        self.phase_steps = 0
        return self._search_output(pose)

    def _choose_detection(self, detections: list[Detection]) -> Detection | None:
        candidates = [d for d in detections if d.track_id not in self.touched_track_ids and d.confidence >= 0.2]
        if not candidates:
            return None
        return min(candidates, key=lambda d: (d.distance, -d.confidence))

    def _refresh_active_detection(self, detections: list[Detection]) -> Detection | None:
        if self.active_detection is None:
            return None
        for det in detections:
            if det.track_id == self.active_detection.track_id:
                return det
        if self.phase_steps < 30:
            return self.active_detection
        return None

    def _search_output(self, pose: Pose2D) -> PlannerOutput:
        wp = self.SEARCH_WAYPOINTS[self.waypoint_idx]
        if pose.distance_to((wp[0], wp[1])) < 0.45:
            self.waypoint_idx = (self.waypoint_idx + 1) % len(self.SEARCH_WAYPOINTS)
            wp = self.SEARCH_WAYPOINTS[self.waypoint_idx]
        return PlannerOutput("search", self._drive_to(pose, wp[0], wp[1], wp[2], 0.35), "stow", (wp[0], wp[1]))

    def _approach_output(self, pose: Pose2D, det: Detection) -> PlannerOutput:
        bearing = pose.bearing_to((det.world_x, det.world_y))
        standoff = 0.42
        tx = det.world_x - standoff * math.cos(bearing)
        ty = det.world_y - standoff * math.sin(bearing)
        return PlannerOutput("approach_object", self._drive_to(pose, tx, ty, bearing, 0.28), "stow", (det.world_x, det.world_y))

    def _push_output(self, pose: Pose2D, det: Detection) -> PlannerOutput:
        desired_yaw = math.atan2(self.TARGET_CENTER[1] - det.world_y, self.TARGET_CENTER[0] - det.world_x)
        yaw_err = _wrap_to_pi(desired_yaw - pose.yaw)
        vx = 0.18 if abs(yaw_err) < 0.45 else 0.0
        wz = _clamp(1.8 * yaw_err, -0.7, 0.7)
        return PlannerOutput("push_to_goal", (vx, 0.0, wz), "left_push", (det.world_x, det.world_y))

    @staticmethod
    def _near_target(xy: tuple[float, float], max_distance: float) -> bool:
        return math.hypot(xy[0] - TaskBPlanner.TARGET_CENTER[0], xy[1] - TaskBPlanner.TARGET_CENTER[1]) <= max_distance

    @staticmethod
    def _face_and_creep(det: Detection) -> tuple[float, float, float]:
        yaw_err = math.atan2(det.rel_y, max(det.rel_x, 1e-6))
        vx = 0.10 if abs(yaw_err) < 0.35 else 0.0
        return vx, 0.0, _clamp(2.0 * yaw_err, -0.5, 0.5)

    @staticmethod
    def _drive_to(pose: Pose2D, tx: float, ty: float, tyaw: float, max_vx: float) -> tuple[float, float, float]:
        ex = tx - pose.x
        ey = ty - pose.y
        c = math.cos(pose.yaw)
        s = math.sin(pose.yaw)
        body_x = c * ex + s * ey
        body_y = -s * ex + c * ey
        yaw_err = _wrap_to_pi(tyaw - pose.yaw)
        vx = _clamp(0.9 * body_x, -0.18, max_vx)
        vy = _clamp(0.8 * body_y, -0.22, 0.22)
        wz = _clamp(1.8 * yaw_err, -0.7, 0.7)
        if abs(yaw_err) > 0.9:
            vx = min(vx, 0.05)
            vy = _clamp(vy, -0.08, 0.08)
        return vx, vy, wz
```

- [ ] **Step 4: Run planner tests to verify they pass**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py::TaskBPlannerTest -q
```

Expected: PASS with 4 tests.

- [ ] **Step 5: Run all current Task B unit tests**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py -q
```

Expected: PASS with 13 tests.

- [ ] **Step 6: Commit**

```bash
git add demo/solution_task_b_g1.py tests/task_b/test_solution_task_b_g1.py
git commit -m "feat: add Task B search and contact planner"
```

---

### Task 5: Implement local G1 hand/arm override

**Files:**
- Modify: `tests/task_b/test_solution_task_b_g1.py`
- Modify: `demo/solution_task_b_g1.py`

- [ ] **Step 1: Add failing local interaction tests**

Append these tests before the `if __name__ == "__main__"` block:

```python
class LocalObjectInteractionTest(unittest.TestCase):
    def test_stow_leaves_action_unchanged(self):
        interaction = sol.LocalObjectInteraction()
        action = [0.1] * 33
        out = interaction.apply_arm_override(action, "stow")
        self.assertEqual(out, action)

    def test_left_touch_overrides_left_arm_and_hands_only(self):
        interaction = sol.LocalObjectInteraction()
        action = [0.0] * 33
        out = interaction.apply_arm_override(action, "left_touch")
        changed = {i for i, (a, b) in enumerate(zip(action, out)) if a != b}
        self.assertTrue({15, 16, 17, 18, 19, 20, 21}.issubset(changed))
        self.assertTrue({29, 30}.issubset(changed))
        self.assertNotIn(0, changed)
        self.assertNotIn(6, changed)

    def test_left_push_uses_more_forward_pose_than_left_touch(self):
        interaction = sol.LocalObjectInteraction()
        touch = interaction.apply_arm_override([0.0] * 33, "left_touch")
        push = interaction.apply_arm_override([0.0] * 33, "left_push")
        self.assertGreaterEqual(push[15], touch[15])
        self.assertGreaterEqual(push[18], touch[18])
```

- [ ] **Step 2: Run local interaction tests to verify they fail**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py::LocalObjectInteractionTest -q
```

Expected: FAIL with `AttributeError` because `LocalObjectInteraction` does not exist.

- [ ] **Step 3: Add local interaction implementation**

Insert this class after `TaskBPlanner` in `demo/solution_task_b_g1.py`:

```python
class LocalObjectInteraction:
    """Conservative G1 upper-body action override.

    Action indices follow UNITREE_G1_29DOF_DEX1_CFG.joint_names:
    left arm 15..21, right arm 22..28, hands 29..32. The action space uses
    default-offset joint position targets scaled by 0.5, so these are small
    normalized offsets rather than absolute joint angles.
    """

    LEFT_TOUCH = {
        15: 0.28,   # left_shoulder_pitch_joint
        16: 0.18,   # left_shoulder_roll_joint
        17: 0.00,   # left_shoulder_yaw_joint
        18: 0.22,   # left_elbow_joint
        19: 0.00,
        20: -0.08,
        21: 0.00,
        29: 0.20,
        30: 0.20,
    }
    LEFT_PUSH = {
        15: 0.38,
        16: 0.20,
        17: 0.00,
        18: 0.32,
        19: 0.00,
        20: -0.10,
        21: 0.00,
        29: 0.25,
        30: 0.25,
    }

    def apply_arm_override(self, action: Sequence[float], arm_mode: str) -> list[float]:
        out = [float(v) for v in action]
        if arm_mode == "left_touch":
            self._apply(out, self.LEFT_TOUCH)
        elif arm_mode == "left_push":
            self._apply(out, self.LEFT_PUSH)
        return out

    @staticmethod
    def _apply(action: list[float], values: dict[int, float]) -> None:
        for idx, value in values.items():
            if idx < len(action):
                action[idx] = float(value)
```

- [ ] **Step 4: Run local interaction tests to verify they pass**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py::LocalObjectInteractionTest -q
```

Expected: PASS with 3 tests.

- [ ] **Step 5: Run all current Task B unit tests**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py -q
```

Expected: PASS with 16 tests.

- [ ] **Step 6: Commit**

```bash
git add demo/solution_task_b_g1.py tests/task_b/test_solution_task_b_g1.py
git commit -m "feat: add conservative G1 hand override"
```

---

### Task 6: Implement submission-style RGB-D perception baseline

**Files:**
- Modify: `tests/task_b/test_solution_task_b_g1.py`
- Modify: `demo/solution_task_b_g1.py`

- [ ] **Step 1: Add failing perception tests**

Append these tests before the `if __name__ == "__main__"` block:

```python
@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class TaskBRgbdPerceptionTest(unittest.TestCase):
    def make_image_obs(self):
        rgb = torch.zeros((1, 64, 96, 3), dtype=torch.uint8)
        depth = torch.full((1, 64, 96, 1), 4.0, dtype=torch.float32)
        rgb[0, 28:38, 44:54, 0] = 230
        rgb[0, 28:38, 44:54, 1] = 190
        rgb[0, 28:38, 44:54, 2] = 30
        depth[0, 28:38, 44:54, 0] = 2.0
        return {"head_rgb": rgb, "head_depth": depth}

    def test_detects_synthetic_colored_blob(self):
        perception = sol.TaskBRgbdPerception(min_pixels=20)
        detections = perception.update(self.make_image_obs(), sol.Pose2D(-10.0, -10.0, 0.0))
        self.assertEqual(len(detections), 1)
        det = detections[0]
        self.assertEqual(det.label, "colored_object")
        self.assertGreater(det.confidence, 0.2)
        self.assertAlmostEqual(det.distance, 2.0, delta=0.1)
        self.assertGreater(det.world_x, -10.0)

    def test_returns_empty_when_no_image_keys_exist(self):
        perception = sol.TaskBRgbdPerception(min_pixels=20)
        detections = perception.update({}, sol.Pose2D(-10.0, -10.0, 0.0))
        self.assertEqual(detections, [])

    def test_tracks_same_blob_with_stable_id(self):
        perception = sol.TaskBRgbdPerception(min_pixels=20)
        first = perception.update(self.make_image_obs(), sol.Pose2D(-10.0, -10.0, 0.0))[0]
        second = perception.update(self.make_image_obs(), sol.Pose2D(-10.0, -10.0, 0.0))[0]
        self.assertEqual(first.track_id, second.track_id)
```

- [ ] **Step 2: Run perception tests to verify they fail**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py::TaskBRgbdPerceptionTest -q
```

Expected: FAIL with `AttributeError` because `TaskBRgbdPerception` does not exist.

- [ ] **Step 3: Add RGB-D perception implementation**

Insert this class after `LocalObjectInteraction` in `demo/solution_task_b_g1.py`:

```python
class TaskBRgbdPerception:
    def __init__(
        self,
        hfov_deg: float = 70.0,
        min_pixels: int = 35,
        max_depth: float = 8.0,
        track_match_dist: float = 0.75,
    ):
        self.hfov = math.radians(float(hfov_deg))
        self.min_pixels = int(min_pixels)
        self.max_depth = float(max_depth)
        self.track_match_dist = float(track_match_dist)
        self.next_track_id = 1
        self.tracks: dict[int, tuple[float, float]] = {}

    def reset(self) -> None:
        self.next_track_id = 1
        self.tracks.clear()

    def update(self, image_obs: dict, pose: Pose2D) -> list[Detection]:
        if torch is None:
            return []
        rgb, depth = self._extract_head_rgbd(image_obs)
        if rgb is None or depth is None:
            return []
        rgb = rgb.detach().float().cpu()
        depth = depth.detach().float().cpu()
        if rgb.ndim == 4:
            rgb = rgb[0]
        if depth.ndim == 4:
            depth = depth[0]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if rgb.ndim != 3 or rgb.shape[-1] < 3 or depth.ndim != 2:
            return []

        mask = self._colored_object_mask(rgb[..., :3], depth)
        components = self._components(mask)
        detections: list[Detection] = []
        for pixels in components:
            if len(pixels) < self.min_pixels:
                continue
            ys = [p[0] for p in pixels]
            xs = [p[1] for p in pixels]
            y0, y1 = min(ys), max(ys)
            x0, x1 = min(xs), max(xs)
            cy = int(round(sum(ys) / len(ys)))
            cx = int(round(sum(xs) / len(xs)))
            local_depth = depth[y0:y1 + 1, x0:x1 + 1]
            finite = torch.isfinite(local_depth) & (local_depth > 0.05) & (local_depth < self.max_depth)
            if not bool(finite.any()):
                continue
            dist = float(local_depth[finite].median().item())
            rel_x, rel_y = self._pixel_to_robot_xy(cx, rgb.shape[1], dist)
            world_x = pose.x + math.cos(pose.yaw) * rel_x - math.sin(pose.yaw) * rel_y
            world_y = pose.y + math.sin(pose.yaw) * rel_x + math.cos(pose.yaw) * rel_y
            track_id = self._assign_track(world_x, world_y)
            confidence = _clamp(len(pixels) / 250.0, 0.05, 1.0)
            detections.append(
                Detection(
                    track_id=track_id,
                    label="colored_object",
                    rel_x=rel_x,
                    rel_y=rel_y,
                    distance=dist,
                    confidence=confidence,
                    world_x=world_x,
                    world_y=world_y,
                    bbox=(x0, y0, x1, y1),
                )
            )
        detections.sort(key=lambda d: (d.distance, -d.confidence))
        return detections[:5]

    @staticmethod
    def _extract_head_rgbd(image_obs: dict):
        if not isinstance(image_obs, dict):
            return None, None
        rgb = image_obs.get("head_rgb")
        depth = image_obs.get("head_depth")
        return rgb, depth

    def _colored_object_mask(self, rgb, depth):
        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]
        maxc = torch.maximum(torch.maximum(r, g), b)
        minc = torch.minimum(torch.minimum(r, g), b)
        saturation = maxc - minc
        yellow = (r > 120.0) & (g > 90.0) & (b < 130.0)
        red_or_orange = (r > 130.0) & (g > 45.0) & (b < 150.0) & (r > b + 35.0)
        bright_colored = (maxc > 110.0) & (saturation > 45.0)
        depth_ok = torch.isfinite(depth) & (depth > 0.15) & (depth < self.max_depth)
        return (yellow | red_or_orange | bright_colored) & depth_ok

    @staticmethod
    def _components(mask) -> list[list[tuple[int, int]]]:
        h, w = int(mask.shape[0]), int(mask.shape[1])
        visited = torch.zeros((h, w), dtype=torch.bool)
        components: list[list[tuple[int, int]]] = []
        for y in range(h):
            for x in range(w):
                if visited[y, x] or not bool(mask[y, x]):
                    continue
                stack = [(y, x)]
                visited[y, x] = True
                pixels: list[tuple[int, int]] = []
                while stack:
                    cy, cx = stack.pop()
                    pixels.append((cy, cx))
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and bool(mask[ny, nx]):
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                components.append(pixels)
        return components

    def _pixel_to_robot_xy(self, cx: int, width: int, depth: float) -> tuple[float, float]:
        x_norm = (float(cx) + 0.5) / max(float(width), 1.0) - 0.5
        lateral_angle = x_norm * self.hfov
        rel_x = float(depth)
        rel_y = math.tan(lateral_angle) * float(depth)
        return rel_x, rel_y

    def _assign_track(self, world_x: float, world_y: float) -> int:
        best_id = None
        best_dist = self.track_match_dist
        for track_id, (tx, ty) in self.tracks.items():
            d = math.hypot(world_x - tx, world_y - ty)
            if d < best_dist:
                best_dist = d
                best_id = track_id
        if best_id is None:
            best_id = self.next_track_id
            self.next_track_id += 1
        self.tracks[best_id] = (world_x, world_y)
        return best_id
```

- [ ] **Step 4: Run perception tests to verify they pass**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py::TaskBRgbdPerceptionTest -q
```

Expected: PASS with 3 tests.

- [ ] **Step 5: Run all current Task B unit tests**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py -q
```

Expected: PASS with 19 tests.

- [ ] **Step 6: Commit**

```bash
git add demo/solution_task_b_g1.py tests/task_b/test_solution_task_b_g1.py
git commit -m "feat: add Task B RGB-D perception baseline"
```

---

### Task 7: Wire full `AlgSolution` controller

**Files:**
- Modify: `tests/task_b/test_solution_task_b_g1.py`
- Modify: `demo/solution_task_b_g1.py`

- [ ] **Step 1: Add failing `AlgSolution` integration tests**

Append these tests before the `if __name__ == "__main__"` block:

```python
@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class AlgSolutionGlueTest(unittest.TestCase):
    class FakeBridge:
        def __init__(self):
            self.reset_calls = 0
            self.commands = []

        def reset(self):
            self.reset_calls += 1

        def act(self, proprio, command):
            self.commands.append(tuple(command))
            return [0.0] * 33

    class FakePerception:
        def __init__(self, detections):
            self.detections = detections
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

        def update(self, image_obs, pose):
            return self.detections

    def make_solution_with_fakes(self, detections):
        instance = sol.AlgSolution.__new__(sol.AlgSolution)
        instance.bridge = self.FakeBridge()
        instance.odom = sol.DeadReckoningOdometry()
        instance.perception = self.FakePerception(detections)
        instance.planner = sol.TaskBPlanner()
        instance.interaction = sol.LocalObjectInteraction()
        return instance

    def proprio(self):
        row = torch.zeros((1, 12 + 3 * 33), dtype=torch.float32)
        row[0, 9:12] = torch.tensor([0.0, 0.0, -1.0])
        return row

    def test_predicts_returns_33_dim_action_and_no_giveup(self):
        det = sol.Detection(1, "object", 1.0, 0.0, 1.0, 0.9, -9.0, -10.0, (0, 0, 3, 3))
        solution = self.make_solution_with_fakes([det])
        out = solution.predicts({"proprio": self.proprio(), "image": {}}, current_score=0.0)
        self.assertFalse(out["giveup"])
        self.assertEqual(len(out["action"]), 33)
        self.assertEqual(len(solution.bridge.commands), 1)

    def test_reset_resets_all_components(self):
        solution = self.make_solution_with_fakes([])
        solution.reset()
        self.assertEqual(solution.bridge.reset_calls, 1)
        self.assertEqual(solution.perception.reset_calls, 1)
        self.assertEqual(solution.planner.phase, "search")
        self.assertEqual(solution.odom.pose, sol.Pose2D(-10.0, -10.0, 0.0))
```

- [ ] **Step 2: Run glue tests to verify they fail**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py::AlgSolutionGlueTest -q
```

Expected: FAIL because the current temporary `AlgSolution` does not call bridge, odometry, perception, planner, and interaction components.

- [ ] **Step 3: Replace `AlgSolution` with full controller**

Replace the temporary `AlgSolution` class at the bottom of `demo/solution_task_b_g1.py` with:

```python
class AlgSolution:
    def __init__(self):
        self.bridge = G1VelocityPolicyBridge(policy_path=_POLICY_PATH)
        self.odom = DeadReckoningOdometry(dt=0.02, x0=-10.0, y0=-10.0)
        self.perception = TaskBRgbdPerception()
        self.planner = TaskBPlanner()
        self.interaction = LocalObjectInteraction()

    def reset(self, **kwargs) -> None:
        self.bridge.reset()
        self.odom.reset()
        self.perception.reset()
        self.planner.reset()

    def predicts(self, obs: dict, current_score: float):
        proprio = obs["proprio"]
        row = proprio[0] if hasattr(proprio, "shape") and len(proprio.shape) >= 2 else proprio
        pose = self.odom.update(row)
        image_obs = obs.get("image", {})
        detections = self.perception.update(image_obs, pose)
        plan = self.planner.step(pose, detections, current_score)
        action = self.bridge.act(proprio, plan.command)
        action = self.interaction.apply_arm_override(action, plan.arm_mode)
        return {"action": action, "giveup": False}
```

- [ ] **Step 4: Run glue tests to verify they pass**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py::AlgSolutionGlueTest -q
```

Expected: PASS with 2 tests.

- [ ] **Step 5: Run all Task B unit tests**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py -q
```

Expected: PASS with 21 tests.

- [ ] **Step 6: Commit**

```bash
git add demo/solution_task_b_g1.py tests/task_b/test_solution_task_b_g1.py
git commit -m "feat: wire Task B G1 solution controller"
```

---

### Task 8: Add Task B G1 camera probe script

**Files:**
- Create: `scripts/probe_task_b_g1_camera.py`

- [ ] **Step 1: Create the camera probe script**

Create `scripts/probe_task_b_g1_camera.py` with:

```python
"""Capture Task B G1 head RGB-D frames and print observation/sensor metadata.

Usage:
  PYTHONPATH=. python scripts/probe_task_b_g1_camera.py \
    --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --out outputs/task_b_g1_camera
"""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe Task B G1 camera observations.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-G1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=80)
parser.add_argument("--capture_every", type=int, default=20)
parser.add_argument("--out", type=str, default="outputs/task_b_g1_camera")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

try:
    from PIL import Image  # noqa: E402
except Exception:  # pragma: no cover - optional debug dependency
    Image = None


def _tensor_summary(x):
    t = x.detach().float().cpu()
    finite = torch.isfinite(t)
    if bool(finite.any()):
        vals = t[finite]
        return {
            "shape": list(t.shape),
            "dtype": str(x.dtype),
            "min": float(vals.min().item()),
            "max": float(vals.max().item()),
            "mean": float(vals.mean().item()),
        }
    return {"shape": list(t.shape), "dtype": str(x.dtype), "min": None, "max": None, "mean": None}


def _save_rgb_png(rgb, path):
    if Image is None:
        return False
    arr = rgb.detach().cpu()
    if arr.ndim == 4:
        arr = arr[0]
    arr = arr[..., :3].clamp(0, 255).to(torch.uint8).numpy()
    Image.fromarray(arr).save(path)
    return True


def _save_depth_png(depth, path):
    if Image is None:
        return False
    d = depth.detach().float().cpu()
    if d.ndim == 4:
        d = d[0]
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    finite = torch.isfinite(d) & (d > 0)
    out = torch.zeros_like(d)
    if bool(finite.any()):
        vals = d[finite]
        out[finite] = ((d[finite] - vals.min()) / (vals.max() - vals.min() + 1e-6) * 255.0)
    Image.fromarray(out.clamp(0, 255).to(torch.uint8).numpy()).save(path)
    return True


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    summaries = []
    print(f"[probe] obs groups: {list(obs.keys())}")
    print(f"[probe] image keys: {list(obs.get('image', {}).keys())}")

    for step in range(args_cli.num_steps):
        action_dim = int(env.unwrapped.action_space.shape[-1])
        actions = torch.zeros((args_cli.num_envs, action_dim), dtype=torch.float32, device=args_cli.device)
        obs, reward, terminated, truncated, info = env.step(actions)
        if step % args_cli.capture_every != 0:
            continue
        image_obs = obs.get("image", {})
        summary = {"step": step, "keys": list(image_obs.keys()), "terms": {}}
        for key, value in image_obs.items():
            if hasattr(value, "shape"):
                summary["terms"][key] = _tensor_summary(value)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))
        if "head_rgb" in image_obs:
            _save_rgb_png(image_obs["head_rgb"], os.path.join(args_cli.out, f"head_rgb_{step:04d}.png"))
            torch.save(image_obs["head_rgb"].detach().cpu(), os.path.join(args_cli.out, f"head_rgb_{step:04d}.pt"))
        if "head_depth" in image_obs:
            _save_depth_png(image_obs["head_depth"], os.path.join(args_cli.out, f"head_depth_{step:04d}.png"))
            torch.save(image_obs["head_depth"].detach().cpu(), os.path.join(args_cli.out, f"head_depth_{step:04d}.pt"))
        if bool(terminated.any()) or bool(truncated.any()):
            break

    with open(os.path.join(args_cli.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"[probe] wrote {len(summaries)} captures to {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Run a syntax check**

Run:

```bash
python -m py_compile scripts/probe_task_b_g1_camera.py
```

Expected: command exits with status 0 and prints no output.

- [ ] **Step 3: Run the probe in Isaac**

Run:

```bash
PYTHONPATH=. python scripts/probe_task_b_g1_camera.py \
  --task ATEC-TaskB-G1 \
  --num_envs 1 \
  --enable_cameras \
  --headless \
  --num_steps 40 \
  --capture_every 20 \
  --out outputs/task_b_g1_camera
```

Expected:

- Console prints `image keys` containing `head_rgb` and `head_depth`.
- `outputs/task_b_g1_camera/summary.json` exists.
- At least one `head_rgb_*.pt` and one `head_depth_*.pt` file exists.

- [ ] **Step 4: Commit**

```bash
git add scripts/probe_task_b_g1_camera.py
git commit -m "feat: add Task B G1 camera probe"
```

---

### Task 9: Add Task B G1 object calibration probe

**Files:**
- Create: `scripts/probe_task_b_g1_objects.py`

- [ ] **Step 1: Create the object calibration probe**

Create `scripts/probe_task_b_g1_objects.py` with:

```python
"""Compare Task B G1 submission-style detections against env.scene object truth.

This script is for local calibration only. It reads env.scene object positions, so
none of its privileged access belongs in demo/solution_task_b_g1.py.

Usage:
  PYTHONPATH=. python scripts/probe_task_b_g1_objects.py \
    --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --out outputs/task_b_g1_objects
"""

from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe Task B object truth vs RGB-D detections.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-G1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=120)
parser.add_argument("--out", type=str, default="outputs/task_b_g1_objects")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from demo.solution_task_b_g1 import Pose2D, TaskBRgbdPerception, _wrap_to_pi  # noqa: E402


def yaw_from_quat_wxyz(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def nearest_error(det, objects):
    best = None
    for obj in objects:
        d = math.hypot(det.world_x - obj["x"], det.world_y - obj["y"])
        if best is None or d < best["error_xy"]:
            best = {"object": obj["name"], "error_xy": d}
    return best


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    scene = env.unwrapped.scene
    robot = scene["robot"]
    perception = TaskBRgbdPerception()
    obs, _ = env.reset()

    rows = []
    for step in range(args_cli.num_steps):
        rpos = robot.data.root_pos_w[0].detach().cpu().tolist()
        rquat = robot.data.root_quat_w[0].detach().cpu().tolist()
        pose = Pose2D(float(rpos[0]), float(rpos[1]), yaw_from_quat_wxyz(rquat))
        detections = perception.update(obs.get("image", {}), pose)
        objects = []
        for idx in range(1, 19):
            obj = scene[f"object_{idx}"]
            pos = obj.data.root_pos_w[0].detach().cpu().tolist()
            objects.append({"name": f"object_{idx}", "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])})
        det_rows = []
        for det in detections:
            match = nearest_error(det, objects)
            det_rows.append(
                {
                    "track_id": det.track_id,
                    "bbox": list(det.bbox),
                    "distance": det.distance,
                    "world_x": det.world_x,
                    "world_y": det.world_y,
                    "nearest": match,
                }
            )
        row = {
            "step": step,
            "robot": {"x": pose.x, "y": pose.y, "yaw": pose.yaw},
            "objects": objects,
            "detections": det_rows,
        }
        if step % 20 == 0:
            print(f"[objects] step={step} detections={len(det_rows)} robot=({pose.x:.2f},{pose.y:.2f},{pose.yaw:.2f})")
            for det in det_rows[:3]:
                nearest = det["nearest"]
                err = nearest["error_xy"] if nearest else None
                print(f"  det track={det['track_id']} xy=({det['world_x']:.2f},{det['world_y']:.2f}) nearest={nearest} err={err}")
        rows.append(row)

        action_dim = int(env.unwrapped.action_space.shape[-1])
        actions = torch.zeros((args_cli.num_envs, action_dim), dtype=torch.float32, device=args_cli.device)
        obs, reward, terminated, truncated, info = env.step(actions)
        if bool(terminated.any()) or bool(truncated.any()):
            break

    with open(os.path.join(args_cli.out, "detections_vs_truth.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"[objects] wrote {len(rows)} rows to {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Run a syntax check**

Run:

```bash
python -m py_compile scripts/probe_task_b_g1_objects.py
```

Expected: command exits with status 0 and prints no output.

- [ ] **Step 3: Run the calibration probe in Isaac**

Run:

```bash
PYTHONPATH=. python scripts/probe_task_b_g1_objects.py \
  --task ATEC-TaskB-G1 \
  --num_envs 1 \
  --enable_cameras \
  --headless \
  --num_steps 60 \
  --out outputs/task_b_g1_objects
```

Expected:

- Console prints at least three `[objects] step=...` lines.
- `outputs/task_b_g1_objects/detections_vs_truth.json` exists.
- The JSON contains 18 object truth entries per row.

- [ ] **Step 4: Commit**

```bash
git add scripts/probe_task_b_g1_objects.py
git commit -m "feat: add Task B object calibration probe"
```

---

### Task 10: Add direct Task B G1 evaluator

**Files:**
- Create: `scripts/eval_task_b_g1.py`

- [ ] **Step 1: Create the evaluator script**

Create `scripts/eval_task_b_g1.py` with:

```python
"""Run demo.solution_task_b_g1.AlgSolution on ATEC-TaskB-G1 without replacing demo/solution.py.

Usage:
  PYTHONPATH=. python scripts/eval_task_b_g1.py \
    --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless --debug
"""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Task B G1 solution_task_b_g1 locally.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-G1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=6000)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401,E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from demo.solution_task_b_g1 import AlgSolution  # noqa: E402


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    solution = AlgSolution()
    obs, _ = env.reset()
    solution.reset()
    total_score = 0.0
    elapsed = 0.0
    step_dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else 0.02
    last_phase = None

    for step in range(args_cli.max_steps):
        if not simulation_app.is_running():
            break
        start = time.time()
        with torch.inference_mode():
            resp = solution.predicts(obs, total_score)
            if resp.get("giveup", False):
                print(f"[eval-b] giveup at step={step}")
                break
            actions = torch.tensor(resp["action"], dtype=torch.float32, device=args_cli.device).view(1, -1)
            obs, reward, terminated, truncated, info = env.step(actions)
        sim_dt = info.get("Step_dt", step_dt) if isinstance(info, dict) else step_dt
        sim_dt_value = sim_dt.item() if hasattr(sim_dt, "item") else float(sim_dt)
        reward_value = reward.mean().item() if isinstance(reward, torch.Tensor) else float(reward)
        total_score += reward_value / sim_dt_value
        if isinstance(info, dict) and "Elapsed_Time" in info:
            raw_elapsed = info["Elapsed_Time"]
            elapsed = raw_elapsed.item() if hasattr(raw_elapsed, "item") else float(raw_elapsed)
        else:
            elapsed += step_dt

        phase = getattr(solution.planner, "phase", "unknown")
        if args_cli.debug and (step % 50 == 0 or phase != last_phase):
            det_count = len(getattr(solution.perception, "tracks", {}))
            print(f"[eval-b] step={step} score={total_score:.2f} elapsed={elapsed:.2f} phase={phase} tracks={det_count}")
        last_phase = phase

        done = bool(terminated.item() if hasattr(terminated, "item") else terminated) or bool(
            truncated.item() if hasattr(truncated, "item") else truncated
        )
        if done:
            print(f"[eval-b] done at step={step}")
            break
        if args_cli.real_time:
            sleep_time = step_dt - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()
    print(f"score: {total_score:.2f}, elapsed_time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Run a syntax check**

Run:

```bash
python -m py_compile scripts/eval_task_b_g1.py
```

Expected: command exits with status 0 and prints no output.

- [ ] **Step 3: Run the evaluator for a short smoke test**

Run:

```bash
PYTHONPATH=. python scripts/eval_task_b_g1.py \
  --task ATEC-TaskB-G1 \
  --num_envs 1 \
  --enable_cameras \
  --headless \
  --max_steps 200 \
  --debug
```

Expected:

- The script starts Isaac and creates `ATEC-TaskB-G1`.
- It prints `[eval-b]` lines with phase and score.
- It ends by printing `score: ... elapsed_time: ... seconds`.
- If it fails because `demo/policy_a.pt` is missing, copy or symlink the existing G1 locomotion policy into `demo/policy_a.pt`, then rerun this exact command.

- [ ] **Step 4: Commit**

```bash
git add scripts/eval_task_b_g1.py
git commit -m "feat: add direct Task B G1 evaluator"
```

---

### Task 11: Run non-Isaac verification and initial Isaac probes

**Files:**
- No source changes expected unless a verification command reveals a concrete defect.

- [ ] **Step 1: Run all non-Isaac Task B tests**

Run:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py -q
```

Expected: PASS with 21 tests.

- [ ] **Step 2: Run syntax checks for all new scripts and solution**

Run:

```bash
python -m py_compile \
  demo/solution_task_b_g1.py \
  scripts/probe_task_b_g1_camera.py \
  scripts/probe_task_b_g1_objects.py \
  scripts/eval_task_b_g1.py
```

Expected: command exits with status 0 and prints no output.

- [ ] **Step 3: Run camera probe**

Run:

```bash
PYTHONPATH=. python scripts/probe_task_b_g1_camera.py \
  --task ATEC-TaskB-G1 \
  --num_envs 1 \
  --enable_cameras \
  --headless \
  --num_steps 40 \
  --capture_every 20 \
  --out outputs/task_b_g1_camera
```

Expected: `outputs/task_b_g1_camera/summary.json` exists and lists `head_rgb` plus `head_depth`.

- [ ] **Step 4: Run object calibration probe**

Run:

```bash
PYTHONPATH=. python scripts/probe_task_b_g1_objects.py \
  --task ATEC-TaskB-G1 \
  --num_envs 1 \
  --enable_cameras \
  --headless \
  --num_steps 60 \
  --out outputs/task_b_g1_objects
```

Expected: `outputs/task_b_g1_objects/detections_vs_truth.json` exists and includes robot pose, 18 object positions, and any RGB-D detections.

- [ ] **Step 5: Run short evaluator smoke test**

Run:

```bash
PYTHONPATH=. python scripts/eval_task_b_g1.py \
  --task ATEC-TaskB-G1 \
  --num_envs 1 \
  --enable_cameras \
  --headless \
  --max_steps 200 \
  --debug
```

Expected: evaluator runs for up to 200 steps and prints phase transitions. A score of 0.00 is acceptable for this smoke test because object contact is not expected within 200 steps.

- [ ] **Step 6: Commit verification fixes if needed**

If a verification command reveals a defect and you edit code, run the exact failing command again until it passes, then commit:

```bash
git add demo/solution_task_b_g1.py scripts/probe_task_b_g1_camera.py scripts/probe_task_b_g1_objects.py scripts/eval_task_b_g1.py tests/task_b/test_solution_task_b_g1.py
git commit -m "fix: stabilize Task B G1 baseline smoke tests"
```

If no files changed, skip this commit step.

---

### Task 12: Run longer baseline evaluation and record tuning notes

**Files:**
- Modify if needed: `demo/solution_task_b_g1.py`
- Optional create: `docs/task_b_g1_baseline_notes.md`

- [ ] **Step 1: Run a longer baseline evaluation**

Run:

```bash
PYTHONPATH=. python scripts/eval_task_b_g1.py \
  --task ATEC-TaskB-G1 \
  --num_envs 1 \
  --enable_cameras \
  --headless \
  --max_steps 6000 \
  --debug
```

Expected:

- The script runs until done or 6000 steps.
- Console output shows planner phases.
- If a nonzero score appears, the log makes clear which phase was active near the score increase.

- [ ] **Step 2: Record the first baseline result**

Create `docs/task_b_g1_baseline_notes.md` with this structure, replacing the numeric fields with the observed values from the command above:

```markdown
# Task B G1 Baseline Notes

## Run 1

Command:

```bash
PYTHONPATH=. python scripts/eval_task_b_g1.py \
  --task ATEC-TaskB-G1 \
  --num_envs 1 \
  --enable_cameras \
  --headless \
  --max_steps 6000 \
  --debug
```

Observed result:

- Score: 0.00
- Elapsed simulation time: 0.00 seconds
- Last phase: search
- Tracks detected: 0
- Failure mode: camera did not detect objects during the short baseline route

Next tuning action:

- Inspect `outputs/task_b_g1_camera/summary.json` and `outputs/task_b_g1_objects/detections_vs_truth.json`.
- If images show visible objects but no detections, loosen the RGB thresholds in `TaskBRgbdPerception._colored_object_mask`.
- If images do not show objects, adjust `TaskBPlanner.SEARCH_WAYPOINTS` or head-camera scanning behavior.
```

Use the actual observed values in the bullet list. Keep the `Next tuning action` bullet that matches the run.

- [ ] **Step 3: Commit notes and any tuning edits**

If only notes were added:

```bash
git add docs/task_b_g1_baseline_notes.md
git commit -m "docs: record initial Task B G1 baseline run"
```

If code was tuned, include the changed code and rerun the relevant unit tests plus the evaluator command before committing:

```bash
python -m pytest tests/task_b/test_solution_task_b_g1.py -q
PYTHONPATH=. python scripts/eval_task_b_g1.py \
  --task ATEC-TaskB-G1 \
  --num_envs 1 \
  --enable_cameras \
  --headless \
  --max_steps 6000 \
  --debug

git add demo/solution_task_b_g1.py docs/task_b_g1_baseline_notes.md
git commit -m "tune: improve Task B G1 baseline detection"
```

Expected: tests pass, evaluator runs, and notes capture the new score/failure mode.

---

## Self-review checklist

Spec coverage:

- RGB-D camera observation probe: Task 8.
- Non-fixed-seed visual detection baseline: Task 6 plus Task 9 calibration.
- G1 locomotion bridge reuse with 960-dim term-major history: Task 2.
- Dead-reckoning odometry from proprio: Task 3.
- Search, approach, touch, push, verify state machine: Task 4.
- Conservative hand/arm override for contact scoring: Task 5.
- Self-contained `AlgSolution.predicts()` data flow: Task 7.
- Fast non-Isaac tests: Tasks 1 through 7 and Task 11.
- Local evaluation command for `ATEC-TaskB-G1`: Task 10 and Task 12.
- Avoid disturbing current Task D `demo/solution.py`: file map and evaluator design keep Task B in `solution_task_b_g1.py`.

Placeholder scan:

- The plan contains exact file paths, commands, expected outcomes, and code snippets for each source change.
- The plan does not rely on fixed object seed or env.scene truth in the submission solution.

Type/signature consistency:

- `Detection`, `Pose2D`, and `PlannerOutput` fields are defined before planner/perception tests use them.
- `TaskBPlanner.step(pose, detections, current_score)` returns `PlannerOutput` consumed by `AlgSolution.predicts()`.
- `LocalObjectInteraction.apply_arm_override(action, arm_mode)` matches `PlannerOutput.arm_mode`.
- `G1VelocityPolicyBridge.act(proprio, command)` returns a Python `list[float]` consumed by `LocalObjectInteraction` and evaluator scripts.
