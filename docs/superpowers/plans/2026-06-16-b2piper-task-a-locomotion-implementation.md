# B2Piper Task-A-style Locomotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a B2Piper Task-A-style locomotion training environment that uses the B2Piper robot asset, exposes a 12-dimensional leg-only policy action space, keeps Piper out of the policy action space, and refactors the B2 training reward/terrain design using transferable G1 Task A reward ideas.

**Architecture:** Add a focused B2Piper Task A config beside the existing Unitree B2 quadruped configs. The new config subclasses the current B2 rough config, swaps in `UNITREE_B2_PIPER_CFG`, keeps the action and actor observation joints restricted to the 12 B2 leg joints, adds Task-A-like terrain, and adds robot-agnostic reward terms from the G1 Task A design. Existing B2 flat/rough configs remain unchanged.

**Tech Stack:** Python, IsaacLab manager-based RL configs, Gymnasium registration, RSL-RL PPO, pytest static regression tests.

---

## File Structure

- Create: `tests/test_b2piper_task_a_training_cfg.py`
  - Static regression tests for the new B2Piper Task A training config, environment registration, 12-leg-only action list, and runner config.
  - Avoids importing Isaac Sim/IsaacLab so it can run quickly in a normal pytest process.

- Create: `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/task_a_b2piper_env_cfg.py`
  - Defines `UnitreeB2PiperTaskAEnvCfg`.
  - Uses `UNITREE_B2_PIPER_CFG` as the robot.
  - Keeps `joint_names` to 12 B2 leg joints.
  - Adds Task-A-like terrain and reward changes.

- Modify: `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/__init__.py`
  - Registers `ATEC-Isaac-Velocity-TaskA-Unitree-B2Piper-v0`.

- Modify: `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_cfg.py`
  - Adds `UnitreeB2PiperTaskAPPORunnerCfg` with experiment name `unitree_b2piper_task_a` and `max_iterations=10000`.

- Existing files to keep verified:
  - `scripts/rsl_rl/train.py`
  - `scripts/rsl_rl/play.py`
  - `tests/test_rsl_rl_script_compat.py`

---

## Task 1: Add Static Regression Tests

**Files:**
- Create: `tests/test_b2piper_task_a_training_cfg.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_b2piper_task_a_training_cfg.py` with this full content:

```python
"""Static tests for the B2Piper Task-A-style locomotion training config.

These tests intentionally avoid importing IsaacLab/Isaac Sim. They verify the
source-level contract of the new config: B2Piper robot asset, 12 leg-only
policy actions, Task-A-style terrain/reward terms, and Gym registration.
"""

from __future__ import annotations

import ast
from pathlib import Path


_REPO = Path(__file__).resolve().parents[1]
_ENV_CFG = _REPO / "source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/task_a_b2piper_env_cfg.py"
_INIT = _REPO / "source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/__init__.py"
_RUNNER_CFG = _REPO / "source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_cfg.py"

_ENV_ID = "ATEC-Isaac-Velocity-TaskA-Unitree-B2Piper-v0"
_EXPECTED_LEG_JOINTS = {
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
}


def _read_existing(path: Path) -> str:
    assert path.exists(), f"missing expected file: {path}"
    return path.read_text(encoding="utf-8")


def _class_assignment_literal(source: str, class_name: str, assignment_name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == assignment_name:
                            return ast.literal_eval(stmt.value)
    raise AssertionError(f"{class_name}.{assignment_name} assignment not found")


def test_b2piper_task_a_config_uses_b2piper_asset_and_leg_only_actions():
    src = _read_existing(_ENV_CFG)
    joint_names = _class_assignment_literal(src, "UnitreeB2PiperTaskAEnvCfg", "joint_names")

    assert "class UnitreeB2PiperTaskAEnvCfg(UnitreeB2RoughEnvCfg)" in src
    assert "UNITREE_B2_PIPER_CFG" in src
    assert "self.scene.robot = UNITREE_B2_PIPER_CFG.replace" in src
    assert "self.actions.joint_pos.joint_names = self.joint_names" in src
    assert len(joint_names) == 12
    assert set(joint_names) == _EXPECTED_LEG_JOINTS
    assert all("arm_joint" not in name for name in joint_names)


def test_b2piper_task_a_config_adds_task_a_terrain_and_reward_terms():
    src = _read_existing(_ENV_CFG)

    assert "BetterTerrainGeneratorCfg" in src
    assert "BetterTerrainImporter" in src
    assert "MeshPlaneTerrainCfg(proportion=0.45)" in src
    assert "HfRandomUniformTerrainCfg" in src
    assert "proportion=0.20" in src
    assert "MeshInvertedPyramidStairsTerrainCfg" in src
    assert "proportion=0.35" in src
    assert "self.rewards.track_lin_vel_xy_exp.weight = 4.0" in src
    assert "self.rewards.track_ang_vel_z_exp.weight = 3.0" in src
    assert "self.rewards.lin_vel_y_l2 = RewTerm(func=mdp.lin_vel_y_l2, weight=-0.35)" in src
    assert "self.rewards.action_smoothness = RewTerm(func=mdp.action_smoothness, weight=-0.003)" in src
    assert "self.rewards.idle_when_commanded = RewTerm(" in src
    assert "func=mdp.idle_when_commanded" in src
    assert "self.rewards.feet_slide.weight = -0.15" in src
    assert "self.rewards.feet_stumble.weight = -1.0" in src


def test_b2piper_task_a_env_is_registered_with_runner_config():
    init_src = _read_existing(_INIT)
    runner_src = _read_existing(_RUNNER_CFG)

    assert _ENV_ID in init_src
    assert "task_a_b2piper_env_cfg:UnitreeB2PiperTaskAEnvCfg" in init_src
    assert "rsl_rl_ppo_cfg:UnitreeB2PiperTaskAPPORunnerCfg" in init_src
    assert "class UnitreeB2PiperTaskAPPORunnerCfg(UnitreeB2RoughPPORunnerCfg)" in runner_src
    assert 'self.experiment_name = "unitree_b2piper_task_a"' in runner_src
    assert "self.max_iterations = 10000" in runner_src
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_b2piper_task_a_training_cfg.py -q
```

Expected: FAIL. The first failure should mention the missing file:

```text
missing expected file: .../task_a_b2piper_env_cfg.py
```

If the test errors because of a syntax mistake in the test itself, fix the test and rerun until it fails because the implementation is missing.

- [ ] **Step 3: Optional checkpoint if commits are authorized**

Only run this if the user has explicitly authorized commits in this session:

```bash
git add tests/test_b2piper_task_a_training_cfg.py
git commit -m "test: add b2piper task a training config contract"
```

Expected: one commit containing only the new test file.

---

## Task 2: Add B2Piper Task A Environment Config

**Files:**
- Create: `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/task_a_b2piper_env_cfg.py`
- Test: `tests/test_b2piper_task_a_training_cfg.py`

- [ ] **Step 1: Create the environment config**

Create `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/task_a_b2piper_env_cfg.py` with this full content:

```python
"""Task-A-style B2Piper locomotion training configuration.

The policy controls only the 12 B2 leg joints while the Piper arm is kept out of
the policy action space. The robot asset still includes the Piper arm so the
locomotion policy trains with the correct mounted mass and inertia.
"""

from __future__ import annotations

import copy

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.velocity.mdp as mdp
from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG
from atec_rl_lab.tasks.task_base import BetterTerrainGenerator, BetterTerrainGeneratorCfg, BetterTerrainImporter
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.rough_env_cfg import UnitreeB2RoughEnvCfg


@configclass
class UnitreeB2PiperTaskAEnvCfg(UnitreeB2RoughEnvCfg):
    """B2Piper velocity-tracking env with Task-A-style terrain and rewards."""

    task_a_num_rows: int = 10
    task_a_num_cols: int = 20

    # Keep the policy action space leg-only. The B2Piper asset has 8 additional
    # arm joints, but they are intentionally not exposed to this locomotion policy.
    # fmt: off
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    # fmt: on

    def __post_init__(self) -> None:
        super().__post_init__()

        physics_material = copy.deepcopy(self.scene.terrain.physics_material)
        visual_material = copy.deepcopy(self.scene.terrain.visual_material)

        # ------------------------------ Scene ------------------------------
        self.scene.robot = UNITREE_B2_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        if self.scene.height_scanner_base is not None:
            self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------ Task-A terrain ------------------------------
        self.scene.terrain = TerrainImporterCfg(
            class_type=BetterTerrainImporter,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=BetterTerrainGeneratorCfg(
                class_type=BetterTerrainGenerator,
                seed=0,
                size=(20.0, 20.0),
                border_width=0.0,
                num_rows=self.task_a_num_rows,
                num_cols=self.task_a_num_cols,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                curriculum=True,
                terrain_sequence=None,
                sub_terrains={
                    "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.45),
                    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                        proportion=0.20,
                        noise_range=(0.02, 0.10),
                        noise_step=0.02,
                        border_width=0.25,
                    ),
                    "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                        proportion=0.35,
                        step_height_range=(0.20, 0.40),
                        step_width=0.5,
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

        # ------------------------------ Observations/actions ------------------------------
        # Parent B2 config already binds policy joint_pos/joint_vel and action joints
        # to self.joint_names. Repeat the action binding here to make the contract clear.
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------ Rewards ------------------------------
        self._restrict_joint_reward_assets_to_legs()

        # Task-A-inspired tracking and anti-laziness terms.
        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.lin_vel_y_l2 = RewTerm(func=mdp.lin_vel_y_l2, weight=-0.35)
        self.rewards.action_smoothness = RewTerm(func=mdp.action_smoothness, weight=-0.003)
        self.rewards.idle_when_commanded = RewTerm(
            func=mdp.idle_when_commanded,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "cmd_threshold": 0.2,
                "vel_threshold": 0.1,
            },
        )

        # Conservative quadruped foot-contact shaping. Keep these mild so early
        # learning is not dominated by contact penalties.
        self.rewards.feet_slide.weight = -0.15
        self.rewards.feet_stumble.weight = -1.0

        if self.__class__.__name__ == "UnitreeB2PiperTaskAEnvCfg":
            self.disable_zero_weight_rewards()

    def _restrict_joint_reward_assets_to_legs(self) -> None:
        """Avoid penalizing passive Piper arm joints in leg-locomotion rewards."""

        reward_names = (
            "joint_torques_l2",
            "joint_vel_l2",
            "joint_acc_l2",
            "joint_pos_limits",
            "joint_vel_limits",
            "joint_power",
            "stand_still",
            "joint_pos_penalty",
        )
        for reward_name in reward_names:
            reward = getattr(self.rewards, reward_name, None)
            if reward is None:
                continue
            asset_cfg = reward.params.get("asset_cfg") if reward.params is not None else None
            if asset_cfg is not None:
                asset_cfg.joint_names = self.joint_names
```

- [ ] **Step 2: Run the focused static test**

Run:

```bash
python -m pytest tests/test_b2piper_task_a_training_cfg.py::test_b2piper_task_a_config_uses_b2piper_asset_and_leg_only_actions -q
```

Expected: PASS for this test. Other tests in the file may still fail because env registration and runner config are not implemented yet.

- [ ] **Step 3: Compile the new config**

Run:

```bash
python -m py_compile source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/task_a_b2piper_env_cfg.py
```

Expected: exit code 0 and no Python syntax errors.

- [ ] **Step 4: Optional checkpoint if commits are authorized**

Only run this if the user has explicitly authorized commits in this session:

```bash
git add source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/task_a_b2piper_env_cfg.py tests/test_b2piper_task_a_training_cfg.py
git commit -m "feat: add b2piper task a locomotion config"
```

Expected: one commit containing the new env config and the still-partial static test file.

---

## Task 3: Register the Environment and Runner Config

**Files:**
- Modify: `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/__init__.py`
- Modify: `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_cfg.py`
- Test: `tests/test_b2piper_task_a_training_cfg.py`

- [ ] **Step 1: Add the PPO runner config**

Append this class to `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_cfg.py` after `UnitreeB2FlatPPORunnerCfg`:

```python

@configclass
class UnitreeB2PiperTaskAPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "unitree_b2piper_task_a"
```

- [ ] **Step 2: Register the new Gym environment**

Append this block to `source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/__init__.py` after the existing rough B2 registration:

```python

gym.register(
    id="ATEC-Isaac-Velocity-TaskA-Unitree-B2Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_a_b2piper_env_cfg:UnitreeB2PiperTaskAEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2PiperTaskAPPORunnerCfg",
    },
)
```

- [ ] **Step 3: Run all static contract tests**

Run:

```bash
python -m pytest tests/test_b2piper_task_a_training_cfg.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 4: Compile modified config files**

Run:

```bash
python -m py_compile \
  source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/task_a_b2piper_env_cfg.py \
  source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/__init__.py \
  source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_cfg.py
```

Expected: exit code 0 and no syntax errors.

- [ ] **Step 5: Verify environment listing**

Run:

```bash
python scripts/list_envs.py | grep ATEC-Isaac-Velocity-TaskA-Unitree-B2Piper-v0
```

Expected output contains exactly the new environment ID:

```text
ATEC-Isaac-Velocity-TaskA-Unitree-B2Piper-v0
```

- [ ] **Step 6: Optional checkpoint if commits are authorized**

Only run this if the user has explicitly authorized commits in this session:

```bash
git add \
  source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/__init__.py \
  source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_cfg.py \
  tests/test_b2piper_task_a_training_cfg.py
git commit -m "feat: register b2piper task a locomotion env"
```

Expected: one commit containing the registration and runner config changes.

---

## Task 4: Run Full Lightweight Verification

**Files:**
- Verify: `tests/test_b2piper_task_a_training_cfg.py`
- Verify: `tests/test_rsl_rl_script_compat.py`
- Verify: `scripts/rsl_rl/train.py`
- Verify: `scripts/rsl_rl/play.py`

- [ ] **Step 1: Run new B2Piper static tests**

Run:

```bash
python -m pytest tests/test_b2piper_task_a_training_cfg.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 2: Run existing rsl-rl compatibility tests**

Run:

```bash
python -m pytest tests/test_rsl_rl_script_compat.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 3: Compile all touched Python files**

Run:

```bash
python -m py_compile \
  scripts/rsl_rl/play.py \
  scripts/rsl_rl/train.py \
  tests/test_rsl_rl_script_compat.py \
  tests/test_b2piper_task_a_training_cfg.py \
  source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/task_a_b2piper_env_cfg.py \
  source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/__init__.py \
  source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_cfg.py
```

Expected: exit code 0 and no syntax errors.

- [ ] **Step 4: Check git diff for scope**

Run:

```bash
git diff -- \
  scripts/rsl_rl/play.py \
  scripts/rsl_rl/train.py \
  tests/test_rsl_rl_script_compat.py \
  tests/test_b2piper_task_a_training_cfg.py \
  docs/superpowers/specs/2026-06-16-b2piper-task-a-locomotion-design.md \
  docs/superpowers/plans/2026-06-16-b2piper-task-a-locomotion-implementation.md \
  source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/task_a_b2piper_env_cfg.py \
  source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/__init__.py \
  source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_cfg.py
```

Expected: diff only covers the approved rsl-rl compatibility fix, design/plan docs, new tests, new B2Piper env config, env registration, and runner config.

- [ ] **Step 5: Optional checkpoint if commits are authorized**

Only run this if the user has explicitly authorized commits in this session:

```bash
git add \
  scripts/rsl_rl/play.py \
  scripts/rsl_rl/train.py \
  tests/test_rsl_rl_script_compat.py \
  tests/test_b2piper_task_a_training_cfg.py \
  docs/superpowers/specs/2026-06-16-b2piper-task-a-locomotion-design.md \
  docs/superpowers/plans/2026-06-16-b2piper-task-a-locomotion-implementation.md \
  source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/task_a_b2piper_env_cfg.py \
  source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/__init__.py \
  source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/quadruped/unitree_b2/agents/rsl_rl_ppo_cfg.py
git commit -m "feat: add b2piper task a locomotion training env"
```

Expected: one commit containing the full implementation only if commits were explicitly authorized.

---

## Task 5: Run IsaacLab Smoke Training

**Files:**
- Runtime verification of `ATEC-Isaac-Velocity-TaskA-Unitree-B2Piper-v0`

- [ ] **Step 1: Run a two-iteration smoke train**

Run from repo root with the `atec` environment active:

```bash
python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-TaskA-Unitree-B2Piper-v0 \
  --num_envs 16 \
  --max_iterations 2 \
  --headless
```

Expected output includes:

```text
[INFO]: Parsing configuration from: atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.task_a_b2piper_env_cfg:UnitreeB2PiperTaskAEnvCfg
[INFO] Action Manager:  <ActionManager> contains 1 active terms.
|  Active Action Terms (shape: 12)   |
```

Expected outcome: command exits without a Python traceback. Isaac/PhysX warnings are acceptable if the training loop starts and completes two iterations.

- [ ] **Step 2: If smoke fails, classify the failure before changing code**

Use the first Python traceback and classify it into one of these categories:

```text
registration/import error: env id, module path, or class name mismatch
config schema error: invalid IsaacLab config field or reward term
robot/action error: B2Piper asset cannot run with 12-leg-only action term
terrain error: BetterTerrain config not accepted for B2Piper env
runtime instability: training starts, then NaN/fall/arm instability dominates
```

For registration/import/config/terrain errors, fix the exact failing field and rerun Step 1.

For robot/action or runtime instability, stop and report the exact traceback/output. Do not convert the policy to 20 actions without explicit approval; the design requires keeping the policy interface at 12 actions.

- [ ] **Step 3: Record the smoke result**

Append a short note to the final implementation summary, not necessarily to a file:

```text
Smoke command: python scripts/rsl_rl/train.py --task ATEC-Isaac-Velocity-TaskA-Unitree-B2Piper-v0 --num_envs 16 --max_iterations 2 --headless
Result: PASS or FAIL
Evidence: key output line or traceback category
```

- [ ] **Step 4: Optional checkpoint if commits are authorized**

Only run this if the user has explicitly authorized commits and Task 5 passed or the failure was documented in a deliberate follow-up commit:

```bash
git status --short
git commit --allow-empty -m "test: smoke b2piper task a locomotion env"
```

Expected: only use this as a marker commit if the user wants commit-level checkpoints. Otherwise skip it.

---

## Task 6: Document Training Commands in the Final Response

**Files:**
- No file edits required unless the user asks for a persistent runbook.

- [ ] **Step 1: Report the short training command**

Include this command in the final response after implementation verification:

```bash
python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-TaskA-Unitree-B2Piper-v0 \
  --num_envs 512 \
  --max_iterations 500 \
  --headless
```

- [ ] **Step 2: Report the full training command**

Include this command in the final response after implementation verification:

```bash
python scripts/rsl_rl/train.py \
  --task ATEC-Isaac-Velocity-TaskA-Unitree-B2Piper-v0 \
  --num_envs 2048 \
  --max_iterations 10000 \
  --headless
```

Mention that `--num_envs 4096` can be tried if the RTX 3090 has enough free memory.

- [ ] **Step 3: Report the play/export command without placeholders**

Use a shell variable to select the newest model checkpoint without writing a placeholder path:

```bash
ckpt=$(find logs/rsl_rl/unitree_b2piper_task_a -name 'model_*.pt' | sort -V | tail -1)
python scripts/rsl_rl/play.py \
  --task ATEC-Isaac-Velocity-TaskA-Unitree-B2Piper-v0 \
  --checkpoint "$ckpt"
```

Explain that this command expects an RSL-RL training checkpoint and that exported TorchScript `policy.pt` should be loaded directly with `torch.jit.load()` in deployment code.

---

## Self-Review Notes

- Spec coverage:
  - New env ID: Task 3.
  - B2Piper asset: Task 2.
  - 12-dimensional leg-only action: Task 1 static test and Task 2 config.
  - Piper excluded from policy action: Task 2 config and Task 5 failure guard.
  - Task-A terrain: Task 2 config.
  - G1 Task A reward transfer: Task 2 reward edits and Task 1 test.
  - Existing B2 envs preserved: Task 2 subclasses instead of modifying rough/flat behavior directly.
  - Static and smoke verification: Tasks 4 and 5.

- Placeholder scan:
  - Commands avoid unresolved `RUN_DIR`/`MODEL_FILE` placeholders by using either concrete smoke commands or a shell variable for newest checkpoint.
  - Optional commits are explicitly gated on user authorization because this session's higher-priority rules prohibit committing without explicit user request.

- Type consistency:
  - New env class: `UnitreeB2PiperTaskAEnvCfg`.
  - New runner class: `UnitreeB2PiperTaskAPPORunnerCfg`.
  - New env ID: `ATEC-Isaac-Velocity-TaskA-Unitree-B2Piper-v0`.
  - New experiment name: `unitree_b2piper_task_a`.
