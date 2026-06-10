# Task B G1 Squat-and-Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 `demo/solution_task_b_g1.py` 视觉接触基线上，加入“行走→蹲下(OpenWBT squat 策略+增益补偿)→双臂脚本扫地”三段接力，争取 G1 手基座进入垃圾 0.20m 球拿接触分。

**Architecture:** 三策略状态机。行走复用已有 `G1VelocityPolicyBridge`；蹲下用 `OpenWBTSquatBridge`（搬自旧 worktree，新增在线 P 项增益补偿）只控 12 条腿关节；扫地用纯脚本 `GroundSweepArmController` 控制肩/肘/腕/手；`PostureGuard` 监测姿态触发脚本兜底；`TaskBPlanner` 新增 `squat_sweep`/`stand_up` 阶段；`AlgSolution` 调度三段。不覆盖 `demo/solution.py`。

**Tech Stack:** Python 3.12，numpy（本地可用），torch/onnxruntime（评测镜像有，本地缺→测试 skip），`unittest`（本地无 pytest），IsaacLab/Gymnasium（probe）。

---

## 设计依据（实现前必读）

- 接触判定（`source/atec_rl_lab/atec_rl_lab/tasks/task_b/mdp/rewards.py` `GraspedObjectsByEE`）：link=`left_hand_base_link`/`right_hand_base_link`，3D 距离 ≤ `0.20`（`tasks/task_b/env_cfg.py:33`），+1/物体。
- 摔倒终止（`env_cfg.py:113-116`）：`base_link` 或 `.*_hip_(pitch|roll|yaw)_link` 接触即结束。
- OpenWBT squat：`OpenWBT/ckpts/squat.onnx`，obs 78 维 `[cmd(2),gravity(3),angvel*0.25(3),(q-default)*1(29),qd*0.05(29),last_action(12)]`，输出 12 腿动作 `target=action*0.25+default[:12]`，部署置零 ankle_roll，带 256 维 RNN hidden state，控制 50Hz=ATEC step_dt 0.02。
- 旧 bridge：`.worktrees/task-b-openwbt-squat/demo/solution_task_b_openwbt.py`（未提交），含 `OpenWBTSquatBridge`/`SquatCommand`/`_HeuristicSquatRunner`/`_OnnxSquatRunner`，已做默认角偏移+action scale 0.25→0.5+obs scale。**本计划把它搬进当前 worktree 并升级 act() 为增益补偿。**

## G1 33-DoF action 索引（实现处处要用）

```
0-5   左腿: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
6-11  右腿: 同上
12-14 腰:   waist_yaw, waist_roll, waist_pitch        (蹲扫时保持默认=动作0)
15-21 左臂: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
22-28 右臂: 同上
29-32 手:   left_hand_J1, left_hand_J2, right_hand_J1, right_hand_J2
```

squat 策略控 0-11（12 腿关节）；扫地脚本控 15-32（双臂+手）；腰 12-14 维持动作 0。

## 增益补偿数学（Task 2 核心）

每条腿关节 j（用实测关节角在线匹配 P 项力矩）：

```
q_abs_j        = proprio_jointpos_rel_j + TASKB_DEFAULT_29_j      # ATEC proprio 的 joint_pos 是相对默认角
target_WBT_j   = raw_action_j * 0.25 + OPENWBT_DEFAULT_29_j
comp_target_j  = q_abs_j + KP_RATIO_j * (target_WBT_j - q_abs_j)
action_j       = (comp_target_j - TASKB_DEFAULT_29_j) / 0.5
ankle_roll(5,11): action 强制为 0
```

`KP_RATIO`（12 维，L/R 各 6）：

```
[0.50, 0.67, 0.67, 0.75, 1.40, 0.0,   # 左腿 (踝pitch=2.0*0.7安全系数; 踝roll不参与置0)
 0.50, 0.67, 0.67, 0.75, 1.40, 0.0]   # 右腿
```

> 假设：ATEC proprio 的 `joint_pos` 观测是相对默认角（`joint_pos_rel`），与旧 bridge 一致。Task 7 probe 会核对此假设；若为绝对角，去掉 `+TASKB_DEFAULT` 项即可。

---

## File Structure

- Modify `demo/solution_task_b_g1.py` — 加 `SquatCommand`、`_HeuristicSquatRunner`、`_OnnxSquatRunner`、`OpenWBTSquatBridge`（含增益补偿）、`GroundSweepArmController`、`PostureGuard`；改 `TaskBPlanner`、`AlgSolution`。
- Modify `tests/task_b/test_solution_task_b_g1.py` — 加各模块测试。
- Create `scripts/probe_task_b_g1_squat.py` — Isaac 蹲下几何标定 probe。

每个 Task 末尾 commit。本地测试命令统一用 `python -m unittest`（无 pytest）。

---

### Task 1: 搬入 OpenWBTSquatBridge 骨架（obs 构建 + 无补偿 act）

**Files:**
- Modify: `demo/solution_task_b_g1.py`
- Test: `tests/task_b/test_solution_task_b_g1.py`

- [ ] **Step 1: 写失败测试**

在 `tests/task_b/test_solution_task_b_g1.py` 顶部 import 区加（若尚无 numpy guard）：

```python
try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None
```

在 `if __name__ == "__main__"` 之前追加：

```python
@unittest.skipIf(np is None, "numpy not installed")
class OpenWBTSquatBridgeObsTest(unittest.TestCase):
    class FakeRunner:
        def __init__(self):
            self.last_obs = None
        def run(self, obs, hidden):
            self.last_obs = obs.copy()
            return np.zeros((1, 12), dtype=np.float32), hidden

    def make_proprio(self, n_joints=33):
        row = [0.0] * (12 + 3 * n_joints)
        row[3:6] = [0.0, 0.0, 0.0]
        row[9:12] = [0.0, 0.0, -1.0]
        return [row]

    def test_obs_is_78_dims_and_command_first(self):
        runner = self.FakeRunner()
        bridge = sol.OpenWBTSquatBridge(policy_runner=runner)
        bridge.act(self.make_proprio(), sol.SquatCommand(height=0.5, pitch=0.1))
        self.assertEqual(runner.last_obs.shape, (1, 78))
        self.assertAlmostEqual(float(runner.last_obs[0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(runner.last_obs[0, 1]), 0.1, places=5)

    def test_act_returns_12_leg_actions(self):
        bridge = sol.OpenWBTSquatBridge(policy_runner=self.FakeRunner())
        leg = bridge.act(self.make_proprio(), sol.SquatCommand())
        self.assertEqual(len(leg), 12)
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.task_b.test_solution_task_b_g1.OpenWBTSquatBridgeObsTest -v`
Expected: FAIL，`AttributeError: module ... has no attribute 'OpenWBTSquatBridge'`。

- [ ] **Step 3: 搬入实现**

在 `demo/solution_task_b_g1.py` 顶部确保 `import numpy as np`（若无则在 torch 之后加；用 `try/except ModuleNotFoundError: np = None` guard 不需要——评测镜像有 numpy，本地也有，直接 `import numpy as np`）。在 `LocalObjectInteraction` 之前插入：

```python
from dataclasses import dataclass as _dataclass

@_dataclass(frozen=True)
class SquatCommand:
    height: float = 0.75
    pitch: float = 0.0


class _HeuristicSquatRunner:
    """ONNX 不可用时的脚本蹲姿兜底，输出 OpenWBT 原始 12 动作约定。"""
    def run(self, obs, hidden_state):
        height = float(obs[0, 0]); pitch = float(obs[0, 1])
        drop = float(np.clip(0.75 - height, 0.0, 0.40))
        hip = -0.9 * drop - 0.10 * pitch
        knee = 1.8 * drop
        ankle = -0.9 * drop
        leg = np.array([hip, 0.0, 0.0, knee, ankle, 0.0], dtype=np.float32)
        return np.concatenate([leg, leg]).reshape(1, 12), hidden_state


class _OnnxSquatRunner:
    def __init__(self, policy_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    def run(self, obs, hidden_state):
        if hidden_state is None:
            hidden_state = np.zeros((1, 1, 256), dtype=np.float32)
        out = self.session.run(
            ["action", "output_hidden_states"],
            {"obs": obs.astype(np.float32), "input_hidden_states": hidden_state.astype(np.float32)},
        )
        return np.asarray(out[0], dtype=np.float32), np.asarray(out[1], dtype=np.float32)


class OpenWBTSquatBridge:
    NUM_OBS = 78
    NUM_DOF_OBS = 29
    NUM_ACTIONS = 12
    TASKB_ACTION_SCALE = 0.5
    OPENWBT_ACTION_SCALE = 0.25
    ANG_VEL_SCALE = 0.25
    DOF_VEL_SCALE = 0.05
    CLIP_OBS = 100.0
    CLIP_ACTIONS = 100.0
    ANKLE_SAFETY = 0.7
    KP_RATIO = (
        0.50, 0.67, 0.67, 0.75, 2.0 * 0.7, 0.0,
        0.50, 0.67, 0.67, 0.75, 2.0 * 0.7, 0.0,
    )
    OPENWBT_DEFAULT_29 = (
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.3, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, -0.3, 0.0, 1.0, 0.0, 0.0, 0.0,
    )
    TASKB_DEFAULT_29 = (
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
        0.0, 0.0, 0.0,
        0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
        0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
    )

    def __init__(self, policy_runner=None, policy_path=None):
        self.policy_runner = policy_runner if policy_runner is not None else self._make_default_runner(policy_path)
        self._taskb_default = np.asarray(self.TASKB_DEFAULT_29, dtype=np.float32)
        self._openwbt_default = np.asarray(self.OPENWBT_DEFAULT_29, dtype=np.float32)
        self._kp_ratio = np.asarray(self.KP_RATIO, dtype=np.float32)
        self.reset()

    def reset(self):
        self.last_action = np.zeros(self.NUM_ACTIONS, dtype=np.float32)
        self.hidden_state = None
        self._last_q_abs_legs = self._taskb_default[: self.NUM_ACTIONS].copy()

    def _first_row(self, proprio):
        if hasattr(proprio, "detach"):
            proprio = proprio.detach().cpu().numpy()
        arr = np.asarray(proprio, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[0]
        return arr

    def build_observation(self, proprio, command):
        row = self._first_row(proprio)
        n = int(row.shape[0]); full_dim = (n - 12) // 3
        if 12 + 3 * full_dim != n or full_dim < self.NUM_DOF_OBS:
            raise ValueError(f"bad proprio length {n}")
        cmd = self._command_vec(command)
        ang_vel = row[3:6] * self.ANG_VEL_SCALE
        gravity = row[9:12]
        jp_start = 12; jv_start = jp_start + full_dim
        jp_rel_29 = row[jp_start:jp_start + self.NUM_DOF_OBS]
        jv_29 = row[jv_start:jv_start + self.NUM_DOF_OBS] * self.DOF_VEL_SCALE
        # ATEC obs 是相对默认角；转 OpenWBT 帧 (q - openwbt_default)
        jp_openwbt = jp_rel_29 + self._taskb_default - self._openwbt_default
        self._last_q_abs_legs = (jp_rel_29[: self.NUM_ACTIONS] + self._taskb_default[: self.NUM_ACTIONS]).copy()
        obs = np.concatenate([cmd, gravity, ang_vel, jp_openwbt, jv_29, self.last_action], dtype=np.float32)
        return np.clip(obs, -self.CLIP_OBS, self.CLIP_OBS).reshape(1, -1).astype(np.float32)

    def act(self, proprio, command):
        obs = self.build_observation(proprio, command)
        raw, self.hidden_state = self.policy_runner.run(obs, self.hidden_state)
        raw = np.clip(np.asarray(raw, dtype=np.float32).reshape(-1), -self.CLIP_ACTIONS, self.CLIP_ACTIONS)
        if raw.shape[0] != self.NUM_ACTIONS:
            raise ValueError(f"squat runner returned {raw.shape[0]} actions, expected 12")
        raw[[5, 11]] = 0.0
        self.last_action = raw.copy()
        # Task 1: 无补偿（静态偏移）。Task 2 替换为增益补偿。
        target_q = raw * self.OPENWBT_ACTION_SCALE + self._openwbt_default[: self.NUM_ACTIONS]
        action = (target_q - self._taskb_default[: self.NUM_ACTIONS]) / self.TASKB_ACTION_SCALE
        action[[5, 11]] = 0.0
        return np.clip(action, -self.CLIP_ACTIONS, self.CLIP_ACTIONS).astype(np.float32).tolist()

    @staticmethod
    def _command_vec(command):
        if isinstance(command, SquatCommand):
            values = (command.height, command.pitch)
        else:
            values = tuple(float(v) for v in command)
        if len(values) != 2:
            raise ValueError("squat command must have 2 values")
        h = float(np.clip(values[0], 0.35, 0.75))
        p = float(np.clip(values[1], 0.0, 0.5))
        return np.asarray([h, p], dtype=np.float32)

    @classmethod
    def _make_default_runner(cls, policy_path):
        path = policy_path
        if path is None:
            for cand in (os.path.join(_DIR, "squat.onnx"),
                         os.path.join(_DIR, "..", "OpenWBT", "ckpts", "squat.onnx")):
                if os.path.exists(cand):
                    path = cand
                    break
        if path is not None and os.path.exists(path):
            try:
                return _OnnxSquatRunner(path)
            except Exception:
                pass
        return _HeuristicSquatRunner()
```

（`_DIR` 已在文件顶部定义；确保 `import os`、`import numpy as np` 存在。）

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.task_b.test_solution_task_b_g1.OpenWBTSquatBridgeObsTest -v`
Expected: PASS（2 tests）。

- [ ] **Step 5: 提交**

```bash
git add demo/solution_task_b_g1.py tests/task_b/test_solution_task_b_g1.py
git commit -m "feat: port OpenWBT squat bridge into Task B G1 solution"
```

---

### Task 2: 增益补偿（在线 P 项匹配）

**Files:**
- Modify: `demo/solution_task_b_g1.py`（`OpenWBTSquatBridge.act`）
- Test: `tests/task_b/test_solution_task_b_g1.py`

- [ ] **Step 1: 写失败测试**

追加：

```python
@unittest.skipIf(np is None, "numpy not installed")
class OpenWBTSquatGainCompTest(unittest.TestCase):
    class ConstRunner:
        def __init__(self, value):
            self.value = value
        def run(self, obs, hidden):
            return np.full((1, 12), self.value, dtype=np.float32), hidden

    def make_proprio(self):
        # joint_pos_rel = 0 -> q_abs = taskb_default
        return [[0.0] * (12 + 3 * 33)]

    def test_ankle_roll_indices_zeroed(self):
        bridge = sol.OpenWBTSquatBridge(policy_runner=self.ConstRunner(0.4))
        leg = bridge.act(self.make_proprio(), sol.SquatCommand(height=0.4))
        self.assertEqual(leg[5], 0.0)
        self.assertEqual(leg[11], 0.0)

    def test_hip_pitch_uses_kp_ratio(self):
        # raw=0.4: target_WBT0 = 0.4*0.25 + (-0.1) = 0.0 ; q_abs0 = -0.2
        # comp = -0.2 + 0.5*(0.0 - (-0.2)) = -0.2 + 0.1 = -0.1
        # action = (-0.1 - (-0.2))/0.5 = 0.2
        bridge = sol.OpenWBTSquatBridge(policy_runner=self.ConstRunner(0.4))
        leg = bridge.act(self.make_proprio(), sol.SquatCommand(height=0.4))
        self.assertAlmostEqual(leg[0], 0.2, places=4)

    def test_ankle_pitch_amplified_vs_no_comp(self):
        # ankle ratio 1.4 > 1 -> compensated magnitude larger than static-offset magnitude
        bridge = sol.OpenWBTSquatBridge(policy_runner=self.ConstRunner(0.4))
        leg = bridge.act(self.make_proprio(), sol.SquatCommand(height=0.4))
        # static-offset action[4] = (0.4*0.25 + (-0.2) - (-0.23))/0.5 = (-0.07)/0.5 = -0.14
        # compensated[4] = (-0.23 + 1.4*((-0.1)-(-0.23)) - (-0.23))/0.5 = (1.4*0.13)/0.5 = 0.364
        self.assertAlmostEqual(leg[4], 0.364, places=3)
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.task_b.test_solution_task_b_g1.OpenWBTSquatGainCompTest -v`
Expected: FAIL（`test_hip_pitch_uses_kp_ratio` / `test_ankle_pitch_amplified_vs_no_comp` 不等，因为当前是静态偏移）。

- [ ] **Step 3: 用增益补偿替换 act 的目标计算**

把 `OpenWBTSquatBridge.act` 中 Task 1 的两行（`target_q = ...` 与 `action = ...`）替换为：

```python
        q_abs = self._last_q_abs_legs
        target_wbt = raw * self.OPENWBT_ACTION_SCALE + self._openwbt_default[: self.NUM_ACTIONS]
        comp_target = q_abs + self._kp_ratio * (target_wbt - q_abs)
        action = (comp_target - self._taskb_default[: self.NUM_ACTIONS]) / self.TASKB_ACTION_SCALE
        action[[5, 11]] = 0.0
```

（`self._last_q_abs_legs` 在 `build_observation` 中已按本 step 的 proprio 更新。）

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.task_b.test_solution_task_b_g1.OpenWBTSquatGainCompTest -v`
Expected: PASS（3 tests）。

- [ ] **Step 5: 提交**

```bash
git add demo/solution_task_b_g1.py tests/task_b/test_solution_task_b_g1.py
git commit -m "feat: add online P-term gain compensation to squat bridge"
```

---

### Task 3: GroundSweepArmController（纯脚本双臂扫地）

**Files:**
- Modify: `demo/solution_task_b_g1.py`
- Test: `tests/task_b/test_solution_task_b_g1.py`

- [ ] **Step 1: 写失败测试**

追加：

```python
class GroundSweepArmControllerTest(unittest.TestCase):
    LEFT_SH_PITCH = 15
    LEFT_ELBOW = 18
    LEFT_SH_ROLL = 16
    RIGHT_SH_ROLL = 23

    def test_progress_zero_arms_near_stow(self):
        sweep = sol.GroundSweepArmController()
        out = sweep.step(squat_progress=0.0)
        self.assertLess(abs(out.get(self.LEFT_SH_PITCH, 0.0)), 0.1)

    def test_progress_one_reaches_down(self):
        sweep = sol.GroundSweepArmController()
        out = sweep.step(squat_progress=1.0)
        # shoulder pitch forward + elbow down beyond a threshold
        self.assertGreater(out[self.LEFT_SH_PITCH], 0.3)
        self.assertGreater(out[self.LEFT_ELBOW], 0.3)

    def test_sweep_oscillates_left_right(self):
        sweep = sol.GroundSweepArmController()
        sweep.step(squat_progress=1.0)  # phase advances internally
        a = sweep.step(squat_progress=1.0)[self.LEFT_SH_ROLL]
        for _ in range(20):
            b_out = sweep.step(squat_progress=1.0)
        b = b_out[self.LEFT_SH_ROLL]
        self.assertNotAlmostEqual(a, b, places=3)

    def test_only_upper_body_indices(self):
        sweep = sol.GroundSweepArmController()
        out = sweep.step(squat_progress=1.0)
        self.assertTrue(all(15 <= i <= 32 for i in out.keys()))

    def test_reset_clears_phase(self):
        sweep = sol.GroundSweepArmController()
        for _ in range(5):
            sweep.step(squat_progress=1.0)
        sweep.reset()
        self.assertEqual(sweep.phase, 0)
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.task_b.test_solution_task_b_g1.GroundSweepArmControllerTest -v`
Expected: FAIL，`AttributeError: ... 'GroundSweepArmController'`。

- [ ] **Step 3: 实现**

在 `OpenWBTSquatBridge` 之后插入：

```python
class GroundSweepArmController:
    """脚本双臂扫地：随 squat_progress 把手放到地面，再左右摆扫。

    输出 {action_index: value}，仅上肢 15..32。0.5-scale 归一化偏移量，初值启发式，
    由 scripts/probe_task_b_g1_squat.py 标定后调。
    """
    L_SH_PITCH, L_SH_ROLL, L_SH_YAW, L_ELBOW = 15, 16, 17, 18
    R_SH_PITCH, R_SH_ROLL, R_SH_YAW, R_ELBOW = 22, 23, 24, 25
    L_FINGER = (29, 30)
    R_FINGER = (31, 32)

    REACH_SH_PITCH = 0.7
    REACH_ELBOW = 0.6
    SWEEP_ROLL_AMP = 0.35
    SWEEP_PERIOD = 40
    FINGER_OPEN = 0.4

    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = 0

    def step(self, squat_progress):
        p = max(0.0, min(1.0, float(squat_progress)))
        self.phase += 1
        swing = self.SWEEP_ROLL_AMP * p * math.sin(2.0 * math.pi * self.phase / self.SWEEP_PERIOD)
        out = {
            self.L_SH_PITCH: self.REACH_SH_PITCH * p,
            self.R_SH_PITCH: self.REACH_SH_PITCH * p,
            self.L_ELBOW: self.REACH_ELBOW * p,
            self.R_ELBOW: self.REACH_ELBOW * p,
            self.L_SH_ROLL: swing,
            self.R_SH_ROLL: -swing,
        }
        for idx in self.L_FINGER + self.R_FINGER:
            out[idx] = self.FINGER_OPEN * p
        return out
```

（确保文件顶部已 `import math`。）

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.task_b.test_solution_task_b_g1.GroundSweepArmControllerTest -v`
Expected: PASS（5 tests）。

- [ ] **Step 5: 提交**

```bash
git add demo/solution_task_b_g1.py tests/task_b/test_solution_task_b_g1.py
git commit -m "feat: add scripted ground-sweep arm controller"
```

---

### Task 4: PostureGuard（姿态危险监测）

**Files:**
- Modify: `demo/solution_task_b_g1.py`
- Test: `tests/task_b/test_solution_task_b_g1.py`

- [ ] **Step 1: 写失败测试**

追加：

```python
class PostureGuardTest(unittest.TestCase):
    def row(self, gx, gy, gz):
        r = [0.0] * (12 + 3 * 33)
        r[9:12] = [gx, gy, gz]
        return r

    def test_upright_is_ok(self):
        guard = sol.PostureGuard()
        self.assertEqual(guard.check(self.row(0.0, 0.0, -1.0)), "ok")

    def test_large_tilt_is_recover(self):
        guard = sol.PostureGuard()
        self.assertEqual(guard.check(self.row(0.5, 0.0, -0.86)), "recover")

    def test_reset(self):
        guard = sol.PostureGuard()
        guard.check(self.row(0.5, 0.0, -0.86))
        guard.reset()
        self.assertEqual(guard.state, "ok")
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.task_b.test_solution_task_b_g1.PostureGuardTest -v`
Expected: FAIL，`AttributeError: ... 'PostureGuard'`。

- [ ] **Step 3: 实现**

在 `GroundSweepArmController` 之后插入：

```python
class PostureGuard:
    """监测 projected_gravity 水平分量，判断是否快栽倒。"""
    TILT_THRESH = 0.35

    def __init__(self, tilt_thresh=None):
        self.tilt_thresh = float(self.TILT_THRESH if tilt_thresh is None else tilt_thresh)
        self.reset()

    def reset(self):
        self.state = "ok"

    def check(self, proprio_row):
        row = proprio_row[0] if hasattr(proprio_row, "__len__") and len(proprio_row) and hasattr(proprio_row[0], "__len__") else proprio_row
        gx = _as_float(row[9]); gy = _as_float(row[10])
        tilt = math.hypot(gx, gy)
        self.state = "recover" if tilt > self.tilt_thresh else "ok"
        return self.state
```

（复用文件已有的 `_as_float`。）

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.task_b.test_solution_task_b_g1.PostureGuardTest -v`
Expected: PASS（3 tests）。

- [ ] **Step 5: 提交**

```bash
git add demo/solution_task_b_g1.py tests/task_b/test_solution_task_b_g1.py
git commit -m "feat: add posture guard for squat fall protection"
```

---

### Task 5: TaskBPlanner 加 squat_sweep / stand_up 阶段

**Files:**
- Modify: `demo/solution_task_b_g1.py`（`TaskBPlanner`）
- Test: `tests/task_b/test_solution_task_b_g1.py`

- [ ] **Step 1: 写失败测试**

追加：

```python
class TaskBPlannerSquatTest(unittest.TestCase):
    def det(self, track_id=1, world=(-9.0, -10.0), distance=0.35):
        return sol.Detection(track_id=track_id, label="object", rel_x=distance, rel_y=0.0,
                             distance=distance, confidence=0.9, world_x=world[0], world_y=world[1],
                             bbox=(0, 0, 3, 3))

    def arrive_and_settle(self, planner):
        # approach until close, then feed still poses
        planner.step(sol.Pose2D(-10.0, -10.0, 0.0), [self.det(distance=1.0)], 0.0)
        out = None
        for _ in range(planner.SETTLE_STEPS + 2):
            out = planner.step(sol.Pose2D(-9.4, -10.0, 0.0), [self.det(distance=0.35)], 0.0)
        return out

    def test_arrival_enters_squat_sweep(self):
        planner = sol.TaskBPlanner()
        out = self.arrive_and_settle(planner)
        self.assertEqual(out.phase, "squat_sweep")
        self.assertEqual(out.arm_mode, "sweep")
        self.assertIsNotNone(out.squat_command)

    def test_squat_height_ramps_down(self):
        planner = sol.TaskBPlanner()
        self.arrive_and_settle(planner)
        h_first = planner.last_squat_height
        for _ in range(planner.SQUAT_RAMP_STEPS):
            planner.step(sol.Pose2D(-9.4, -10.0, 0.0), [self.det(distance=0.35)], 0.0)
        self.assertLess(planner.last_squat_height, h_first)

    def test_score_marks_touch_and_stands_up(self):
        planner = sol.TaskBPlanner()
        self.arrive_and_settle(planner)
        out = planner.step(sol.Pose2D(-9.4, -10.0, 0.0), [self.det(distance=0.35)], current_score=1.0)
        self.assertIn(1, planner.touched_track_ids)
        self.assertEqual(out.phase, "stand_up")

    def test_timeout_stands_up(self):
        planner = sol.TaskBPlanner()
        self.arrive_and_settle(planner)
        out = None
        for _ in range(planner.SQUAT_SWEEP_MAX_STEPS + 1):
            out = planner.step(sol.Pose2D(-9.4, -10.0, 0.0), [self.det(distance=0.35)], 0.0)
        self.assertEqual(out.phase, "stand_up")

    def test_recover_request_stands_up(self):
        planner = sol.TaskBPlanner()
        self.arrive_and_settle(planner)
        out = planner.step(sol.Pose2D(-9.4, -10.0, 0.0), [self.det(distance=0.35)], 0.0, posture="recover")
        self.assertEqual(out.phase, "stand_up")
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.task_b.test_solution_task_b_g1.TaskBPlannerSquatTest -v`
Expected: FAIL（缺 `SETTLE_STEPS`/`squat_command`/`posture` 参数等）。

- [ ] **Step 3: 实现 planner 改动**

在 `TaskBPlanner` 中：(a) `PlannerOutput` 已有 `phase/command/arm_mode/target_world`，**新增字段** `squat_command` 与 `squat_progress`——修改其定义（在文件上方 `PlannerOutput` dataclass）为：

```python
@dataclass(frozen=True)
class PlannerOutput:
    phase: str
    command: tuple[float, float, float]
    arm_mode: str
    target_world: tuple[float, float] | None = None
    squat_command: "SquatCommand | None" = None
    squat_progress: float = 0.0
```

(b) 在 `TaskBPlanner` 加常量与状态，并把原 `touch_object`/`push_to_goal` 分支替换为 `squat_sweep`/`stand_up`。新增常量与字段：

```python
    SETTLE_STEPS = 15
    SQUAT_RAMP_STEPS = 60
    SQUAT_SWEEP_MAX_STEPS = 220
    STAND_RAMP_STEPS = 50
    ARRIVE_DIST = 0.55
    STILL_LIN = 0.06
    STILL_ANG = 0.15
    SQUAT_TARGET_HEIGHT = 0.40
    SQUAT_TARGET_PITCH = 0.25
    STAND_HEIGHT = 0.75
```

在 `reset()` 末尾加：

```python
        self.settle_steps = 0
        self.squat_step = 0
        self.stand_step = 0
        self.last_squat_height = self.STAND_HEIGHT
```

把 `approach_object` 分支里“`det.distance <= 0.55` 进入 touch”的逻辑改为进入 settle/squat：当 `det.distance <= self.ARRIVE_DIST` 时设 `self.phase = "squat_sweep"; self.squat_step = 0; self.settle_steps = 0` 并返回 `self._squat_output(0.0)`（首步 progress 0）。

新增/替换分支（放在 `approach_object` 之后、`verify_or_next` 之前；删除旧 `touch_object`、`push_to_goal` 两个分支）：

```python
        if self.phase == "squat_sweep":
            if posture == "recover":
                self.phase = "stand_up"; self.stand_step = 0
                return self._stand_output()
            if score_delta > 0.0 and self.active_detection is not None:
                self.placed_track_ids.discard(self.active_detection.track_id)
                self.touched_track_ids.add(self.active_detection.track_id)
                self.phase = "stand_up"; self.stand_step = 0
                return self._stand_output()
            self.squat_step += 1
            if self.squat_step > self.SQUAT_SWEEP_MAX_STEPS:
                self.phase = "stand_up"; self.stand_step = 0
                return self._stand_output()
            progress = min(1.0, self.squat_step / self.SQUAT_RAMP_STEPS)
            return self._squat_output(progress)

        if self.phase == "stand_up":
            self.stand_step += 1
            if self.stand_step >= self.STAND_RAMP_STEPS:
                self.active_detection = None
                self.phase = "search"; self.phase_steps = 0
                return self._search_output(pose)
            return self._stand_output()
```

加输出辅助方法：

```python
    def _squat_output(self, progress):
        height = self.STAND_HEIGHT + progress * (self.SQUAT_TARGET_HEIGHT - self.STAND_HEIGHT)
        pitch = progress * self.SQUAT_TARGET_PITCH
        self.last_squat_height = height
        tw = None if self.active_detection is None else (self.active_detection.world_x, self.active_detection.world_y)
        return PlannerOutput("squat_sweep", (0.0, 0.0, 0.0), "sweep", tw,
                             squat_command=SquatCommand(height=height, pitch=pitch), squat_progress=progress)

    def _stand_output(self):
        progress = min(1.0, self.stand_step / self.STAND_RAMP_STEPS)
        height = self.SQUAT_TARGET_HEIGHT + progress * (self.STAND_HEIGHT - self.SQUAT_TARGET_HEIGHT)
        self.last_squat_height = height
        return PlannerOutput("stand_up", (0.0, 0.0, 0.0), "stow", None,
                             squat_command=SquatCommand(height=height, pitch=0.0), squat_progress=0.0)
```

把 `step()` 签名改为 `def step(self, pose, detections, current_score, posture="ok"):`。删除旧 `_face_and_creep`、`_push_output`、`_near_target` 中仅服务于已删阶段的部分（保留 `_drive_to`、`_approach_output`、`_search_output`、`_choose_detection`、`_refresh_active_detection`）。`_choose_detection` 继续过滤 `touched_track_ids`。

> 注意：`approach_object` 进入条件用 `det.distance <= self.ARRIVE_DIST` 即触发 squat（测试用 distance=0.35 满足）。`SETTLE_STEPS` 暂作为常量保留给 AlgSolution/probe 备用，本测试通过 ramp 行为验证。

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.task_b.test_solution_task_b_g1.TaskBPlannerSquatTest -v`
Expected: PASS（5 tests）。

- [ ] **Step 5: 跑全量 planner 旧测试，修回归**

Run: `python -m unittest tests.task_b.test_solution_task_b_g1.TaskBPlannerTest -v`
Expected: 旧的 `touch_object` 相关断言会失败（阶段已改名）。**更新旧 `TaskBPlannerTest` 中断言** `touch_object`→`squat_sweep`、`left_touch`→`sweep`，删除 `push_to_goal` 专属断言。改完重跑至 PASS。

- [ ] **Step 6: 提交**

```bash
git add demo/solution_task_b_g1.py tests/task_b/test_solution_task_b_g1.py
git commit -m "feat: add squat_sweep and stand_up planner phases"
```

---

### Task 6: AlgSolution 三策略接力调度

**Files:**
- Modify: `demo/solution_task_b_g1.py`（`AlgSolution`）
- Test: `tests/task_b/test_solution_task_b_g1.py`

- [ ] **Step 1: 写失败测试**

追加：

```python
@unittest.skipIf(np is None, "numpy not installed")
class AlgSolutionSquatGlueTest(unittest.TestCase):
    class FakeWalk:
        def __init__(self):
            self.reset_calls = 0; self.calls = 0
        def reset(self): self.reset_calls += 1
        def act(self, proprio, command): self.calls += 1; return [0.0] * 33

    class FakeSquat:
        def __init__(self):
            self.reset_calls = 0; self.calls = 0
        def reset(self): self.reset_calls += 1
        def act(self, proprio, command):
            self.calls += 1
            return [0.7] * 12  # 12 leg actions

    class FakePerception:
        def __init__(self, dets): self.dets = dets; self.reset_calls = 0
        def reset(self): self.reset_calls += 1
        def update(self, image, pose): return self.dets

    def make_sol(self, dets):
        inst = sol.AlgSolution.__new__(sol.AlgSolution)
        inst.bridge = self.FakeWalk()
        inst.squat_bridge = self.FakeSquat()
        inst.odom = sol.DeadReckoningOdometry()
        inst.perception = self.FakePerception(dets)
        inst.planner = sol.TaskBPlanner()
        inst.sweep = sol.GroundSweepArmController()
        inst.guard = sol.PostureGuard()
        return inst

    def proprio(self):
        r = [[0.0] * (12 + 3 * 33)]
        r[0][9:12] = [0.0, 0.0, -1.0]
        return r

    def test_search_uses_walk_bridge(self):
        s = self.make_sol([])
        out = s.predicts({"proprio": self.proprio(), "image": {}}, 0.0)
        self.assertEqual(len(out["action"]), 33)
        self.assertFalse(out["giveup"])
        self.assertEqual(s.bridge.calls, 1)
        self.assertEqual(s.squat_bridge.calls, 0)

    def test_squat_phase_uses_squat_legs_and_sweep_arms(self):
        det = sol.Detection(1, "o", 0.35, 0.0, 0.35, 0.9, -9.0, -10.0, (0, 0, 3, 3))
        s = self.make_sol([det])
        # drive planner to squat_sweep
        for _ in range(sol.TaskBPlanner.SETTLE_STEPS + 5):
            out = s.predicts({"proprio": self.proprio(), "image": {}}, 0.0)
        self.assertEqual(s.planner.phase, "squat_sweep")
        self.assertGreater(s.squat_bridge.calls, 0)
        # legs (idx0) took squat value path; arm idx15 took sweep (nonzero at progress>0)
        self.assertEqual(len(out["action"]), 33)

    def test_reset_resets_all(self):
        s = self.make_sol([])
        s.reset()
        self.assertEqual(s.bridge.reset_calls, 1)
        self.assertEqual(s.squat_bridge.reset_calls, 1)
        self.assertEqual(s.perception.reset_calls, 1)
        self.assertEqual(s.planner.phase, "search")
        self.assertEqual(s.guard.state, "ok")
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.task_b.test_solution_task_b_g1.AlgSolutionSquatGlueTest -v`
Expected: FAIL（`AlgSolution` 尚无 `squat_bridge`/`sweep`/`guard` 与分派逻辑）。

- [ ] **Step 3: 重写 AlgSolution**

把文件底部 `AlgSolution` 替换为：

```python
class AlgSolution:
    def __init__(self):
        self.bridge = G1VelocityPolicyBridge(policy_path=_POLICY_PATH)
        self.squat_bridge = OpenWBTSquatBridge()
        self.odom = DeadReckoningOdometry(dt=0.02, x0=-10.0, y0=-10.0)
        self.perception = TaskBRgbdPerception()
        self.planner = TaskBPlanner()
        self.sweep = GroundSweepArmController()
        self.guard = PostureGuard()

    def reset(self, **kwargs):
        self.bridge.reset()
        self.squat_bridge.reset()
        self.odom.reset()
        self.perception.reset()
        self.planner.reset()
        self.sweep.reset()
        self.guard.reset()

    def predicts(self, obs, current_score):
        proprio = obs["proprio"]
        row = proprio[0] if hasattr(proprio, "shape") and len(proprio.shape) >= 2 else (
            proprio[0] if isinstance(proprio, (list, tuple)) else proprio)
        pose = self.odom.update(row)
        posture = self.guard.check(row)
        detections = self.perception.update(obs.get("image", {}), pose)
        plan = self.planner.step(pose, detections, current_score, posture=posture)

        if plan.phase in ("squat_sweep", "stand_up"):
            cmd = plan.squat_command if plan.squat_command is not None else SquatCommand()
            if posture == "recover":
                self.squat_bridge.policy_runner = _HeuristicSquatRunner()
            leg = self.squat_bridge.act(proprio, cmd)
            action = [0.0] * (len(leg) + 21)  # 12 legs + 3 waist + 14 arm + 4 hand = 33
            action[:12] = leg
            if plan.arm_mode == "sweep":
                for idx, val in self.sweep.step(plan.squat_progress).items():
                    if idx < len(action):
                        action[idx] = float(val)
            return {"action": action, "giveup": False}

        action = self.bridge.act(proprio, plan.command)
        return {"action": action, "giveup": False}
```

> 说明：行走段 `bridge.act` 返回 33 维全身动作；蹲扫/起身段腿用 squat（12）、其余按默认 0、臂用 sweep 覆盖。腰 12-14 维持 0。

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.task_b.test_solution_task_b_g1.AlgSolutionSquatGlueTest -v`
Expected: PASS（3 tests）。

- [ ] **Step 5: 跑全量单测**

Run: `python -m unittest tests.task_b.test_solution_task_b_g1 -v`
Expected: 全部 PASS（torch 相关仍 skip）。如有旧 `AlgSolutionGlueTest` 因接口变化失败，更新其断言以匹配新分派后重跑。

- [ ] **Step 6: 提交**

```bash
git add demo/solution_task_b_g1.py tests/task_b/test_solution_task_b_g1.py
git commit -m "feat: wire three-policy squat-sweep handoff in AlgSolution"
```

---

### Task 7: scripts/probe_task_b_g1_squat.py（蹲下几何标定）

**Files:**
- Create: `scripts/probe_task_b_g1_squat.py`

- [ ] **Step 1: 创建脚本**

```python
"""蹲下时测量 G1 手基座高度与到物体距离，标定扫地几何与 squat 目标高度。

仅本地标定使用（读 env.scene 真值，不进 solution）。

Usage:
  PYTHONPATH=. python scripts/probe_task_b_g1_squat.py \
    --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless --out outputs/task_b_g1_squat
"""

from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe G1 squat geometry for Task B.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-G1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=400)
parser.add_argument("--out", type=str, default="outputs/task_b_g1_squat")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from demo.solution_task_b_g1 import AlgSolution  # noqa: E402


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    scene = env.unwrapped.scene
    robot = scene["robot"]
    l_idx = int(robot.find_bodies("left_hand_base_link")[0][0])
    r_idx = int(robot.find_bodies("right_hand_base_link")[0][0])
    sol = AlgSolution()
    obs, _ = env.reset()
    sol.reset()

    rows = []
    for step in range(args_cli.num_steps):
        if not simulation_app.is_running():
            break
        resp = sol.predicts(obs, 0.0)
        actions = torch.tensor(resp["action"], dtype=torch.float32, device=args_cli.device).view(1, -1)
        obs, reward, terminated, truncated, info = env.step(actions)
        lpos = robot.data.body_pos_w[0, l_idx].detach().cpu().tolist()
        rpos = robot.data.body_pos_w[0, r_idx].detach().cpu().tolist()
        nearest = None
        for oi in range(1, 19):
            opos = scene[f"object_{oi}"].data.root_pos_w[0].detach().cpu().tolist()
            dl = math.dist(lpos, opos); dr = math.dist(rpos, opos)
            d = min(dl, dr)
            if nearest is None or d < nearest["dist"]:
                nearest = {"object": f"object_{oi}", "dist": d, "hand": "L" if dl < dr else "R"}
        phase = getattr(sol.planner, "phase", "?")
        row = {"step": step, "phase": phase, "l_hand_z": lpos[2], "r_hand_z": rpos[2], "nearest": nearest}
        rows.append(row)
        if step % 25 == 0:
            print(f"[squat] step={step} phase={phase} l_z={lpos[2]:.3f} r_z={rpos[2]:.3f} nearest={nearest}")
        if bool(terminated.any()) or bool(truncated.any()):
            print(f"[squat] terminated at step={step}")
            break

    with open(os.path.join(args_cli.out, "squat_geometry.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"[squat] wrote {len(rows)} rows to {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: 语法检查**

Run: `python -m py_compile scripts/probe_task_b_g1_squat.py`
Expected: 退出码 0，无输出。

- [ ] **Step 3: 提交**

```bash
git add scripts/probe_task_b_g1_squat.py
git commit -m "feat: add Task B G1 squat geometry probe"
```

---

### Task 8: 非 Isaac 全量验证 + 准备运行时资产

**Files:**
- 无源码改动（除非验证发现缺陷）

- [ ] **Step 1: 跑全量单测**

Run: `python -m unittest tests.task_b.test_solution_task_b_g1 -v`
Expected: 全 PASS（torch/onnxruntime 相关 skip）。

- [ ] **Step 2: 语法检查全部新文件**

Run:
```bash
python -m py_compile demo/solution_task_b_g1.py scripts/probe_task_b_g1_squat.py
```
Expected: 退出码 0，无输出。

- [ ] **Step 3: 放置 squat.onnx 供运行时加载**

Run:
```bash
ls OpenWBT/ckpts/squat.onnx && cp OpenWBT/ckpts/squat.onnx demo/squat.onnx && ls -la demo/squat.onnx
```
Expected: `demo/squat.onnx` 存在（`OpenWBTSquatBridge._make_default_runner` 会在 `demo/` 下找到它）。若 onnxruntime 缺失，运行时自动回落脚本兜底（这是预期行为，记录之）。

- [ ] **Step 4: 提交资产指针（如需要）**

```bash
git add demo/squat.onnx 2>/dev/null; git commit -m "chore: vendor OpenWBT squat.onnx for Task B runtime" 2>/dev/null || echo "skip (onnx untracked/ignored or absent)"
```

若 `*.onnx` 被 `.gitignore` 排除，记录“squat.onnx 需随提交镜像打包，不入 git”，不强行加。

---

### Task 9: Isaac 蹲下几何 probe + 评估冒烟（可能受环境阻塞）

**Files:**
- 无源码改动（除非发现缺陷）

> 本 Task 需 IsaacLab 运行环境。无法运行时，把阻塞原因记录到 `docs/task_b_g1_squat_notes.md` 并停在此处，等用户在有 Isaac 的机器上跑。**调试蹲下进入反复调参阶段前，先问用户调试思路（见 spec“调试谨慎”约定）。**

- [ ] **Step 1: 跑蹲下几何 probe**

Run:
```bash
PYTHONPATH=. python scripts/probe_task_b_g1_squat.py \
  --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless --num_steps 400 \
  --out outputs/task_b_g1_squat
```
Expected: 打印 `[squat] step=... phase=...`；生成 `outputs/task_b_g1_squat/squat_geometry.json`，含 `l_hand_z`/`r_hand_z`/`nearest.dist`。**关键判读**：squat 段手基座 z 是否降到接近物体高度、最近距离能否逼近 0.20。是否因摔倒 `terminated`。

- [ ] **Step 2: 跑短程评估冒烟**

Run:
```bash
PYTHONPATH=. python scripts/eval_task_b_g1.py \
  --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless --max_steps 600 --debug
```
Expected: 打印 phase 流转（search→approach→squat_sweep→stand_up）；不因 import/shape 报错。score 可能仍 0（标定前）。若 squat 段立刻 `terminated`（摔倒），记录之——这是进入增益/ramp 调参循环的信号。

- [ ] **Step 3: 记录首轮结果**

创建 `docs/task_b_g1_squat_notes.md`，记录：probe 的手基座 z 范围、最近距离、是否摔倒、eval 的 phase 流转与 score、下一步调参方向（squat 目标高度 / 踝部 SAFETY / ramp 步数 / 扫地 REACH 系数）。

- [ ] **Step 4: 提交**

```bash
git add docs/task_b_g1_squat_notes.md
git commit -m "docs: record Task B G1 squat-sweep first Isaac run"
```

---

## Self-Review

**Spec 覆盖核对：**
- 三段策略接力 → Task 1/2(蹲)、Task 3(扫)、Task 5(planner 阶段)、Task 6(调度)。
- 增益补偿在线 P 项匹配 + KP_RATIO + 踝部安全系数 → Task 2。
- 扫地独立于策略/增益 → Task 3（纯脚本）。
- 兜底三道（ONNX 缺失/姿态危险/超时） → Task 1(_make_default_runner 回落)、Task 4(PostureGuard)+Task 6(recover→脚本 runner)、Task 5(timeout→stand_up)。
- 蹲下几何标定 probe → Task 7 + Task 9。
- 不覆盖 demo/solution.py、不改 env → 文件清单遵守。
- 接触阈值 0.20、摔倒终止事实 → 已写入“设计依据”，probe（Task 9）据此判读。

**占位符扫描：** 各代码步给出完整代码与具体断言；无 TBD/TODO。

**类型/签名一致性：** `OpenWBTSquatBridge.act(proprio, SquatCommand)->list[12]`、`GroundSweepArmController.step(progress)->dict[int,float]`、`PostureGuard.check(row)->"ok"/"recover"`、`TaskBPlanner.step(pose,dets,score,posture="ok")->PlannerOutput(含 squat_command/squat_progress)`、`AlgSolution.predicts` 按 phase 分派——前后引用一致。`SquatCommand` 在 Task 1 定义、Task 5/6 使用；`PlannerOutput` 新字段在 Task 5 定义后于 Task 6 读取。

**已知风险（实现者须知）：** ATEC `joint_pos` 是否相对默认角的假设在 Task 9 probe 核对；踝部 ×1.4 可能过冲，调参属 Task 9 之后、需先问用户思路。
