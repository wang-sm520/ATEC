# Task B G1 拿分路线图：感知评估 + 流程优化 + 夹取进阶 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Task B 的 touch 分（手碰物体 0.20m，+1/件）打稳打满，然后在此基础上增加"夹取→搬运→放入目标圈"（再 +1/件），单局上限 36 分。

**Architecture:** 现有 MiniWBC 全身控制器（policy18.onnx）+ 颜色/深度地面投影感知 + TaskBPlanner FSM 不动骨架。Phase 1 修导航层（盲走到达判定 + 多目标记忆），Phase 2 在 WBC 手位姿命令之上加脚本化抓取-搬运，ACT 仅作为脚本抓取失败时的条件分支。

**Tech Stack:** Python（demo/ 提交件纯 numpy+onnxruntime，不依赖 atec_rl_lab）；Isaac Lab 探针/评估脚本在 `scripts/`；单测用 `python -m unittest`（无 pytest）。

**工作分支：** 全部在 worktree `worktree-task-b-g1-agent-team`（`.claude/worktrees/task-b-g1-agent-team/`）进行，main 上没有这套代码。

---

## 0. 执行者必读的背景结论（已验证事实，不要重新发明）

计分与终止（`source/atec_rl_lab/atec_rl_lab/tasks/task_b/env_cfg.py`）：
- **touch 分**：`left/right_hand_base_link` 到物体 root 3D 距离 ≤ **0.20m**，每物体一次性 +1。
- **入圈分**：物体进入圆心 `(-3,-10)`、半径 **1.0m**、z∈[0,0.5] 的圈，每物体一次性 +1。
- 18 件物体（Sugar/Mustard/Banana/Cracker，全是高饱和度黄/红色），满分 36。
- **单局 1200s（20 分钟）**，控制 50Hz。**摔倒（base/hip 链接触地）→ 整局终止**，这是最大的风险项：任何新行为都不能牺牲稳定性。

感知现状（来自 `docs/task_b_g1_squat_notes.md` Run 3/6 与 `scripts/confirm_detection.py` 实测）：
- 颜色 blob + 深度针孔反投影（47.6° 俯仰修正）后，**检测本身的定位误差 0.03–0.05m**（以真值机器人位姿计）。
- 用 dead-reckoning 里程计换算世界坐标后误差 median 0.24m / p90 1.27m —— **误差大头是里程计漂移，不是检测**。
- 头相机几何约束：俯视 47.6°、高 1.37m，**只能看到前方 ~0.62–2.5m 的地面带**；物体一旦近于 ~0.7m 就出视野（机器人最需要精确贴近时"瞎"）。
- 手相机（`ee_dual_*` 右手）近场可见但手臂关节几乎改变不了深度（Jacobian 实测 ±0.005m/unit），**手眼伺服已被证伪为主力方案**，不要再投入。

当前不稳定的根因（本计划 Task 1 修的就是它）：
`TaskBPlanner` 的到达判定是 `det.distance <= ARRIVE_DIST(0.55)`，依赖**新鲜检测**。但相机在前方 <0.62m 处是盲区，物体只在 0.55–0.62m 的一条极窄环带内既可见又满足阈值——大多数接近过程要么永远触发不了蹲下，要么在丢失目标 30 步后回退 search。**修法：检测丢失后用记住的世界坐标 + 里程计盲走最后 0.7m，到达判定改用里程计地面距离。** 短程盲走（<1m、几秒）内里程计漂移很小，且目标世界坐标本来就是用同一套里程计算的，系统性漂移在相减时大部分抵消。

**约定（重要）**：蹲扫的平衡参数（SQUAT_HEIGHT=0.30、HAND_Z=-0.20、sweep 幅度/周期、MiniWBC 命令范围）目前稳定不摔，**本计划所有任务都不改它们**。WBC 命令有效范围：base height [0.3,0.9]、hand x [-0.2,0.6]、hand z [-0.2,0.65]、left y [-0.1,0.6]、right y [-0.6,0.1]，越界会失稳。

---

## 决策记录：现在不上 YOLO

**结论：不引入 YOLO/学习型检测器。** 理由：
1. 瓶颈不在识别。颜色+深度方案的检测误差 0.04m，已远小于 0.20m 计分球和 ±0.14m 平扫覆盖；丢分来自相机视野几何与里程计漂移，YOLO 一个都解决不了。
2. 目标物全部是高饱和黄/红刚体，背景是灰色地面——颜色阈值正是这种场景的强方案；YOLO 需要采集渲染数据、训练、打包权重进无网络容器，成本高收益≈0。
3. 真正值得做的"感知升级"在别处：目标圈橙色标记的视觉确认（Phase 2 搬运用，已有 `_is_large_target_background` 的颜色逻辑可复用）和手相机近场抓取确认（Phase 2 验证抓住与否）。

**何时重新考虑**：若官方线上环境出现本地没有的干扰物/光照（表现为线上得分显著低于本地且日志显示大量假阳性逼近），再训一个小检测器替换 `_colored_object_mask`，接口（输入 RGB-D、输出 mask）保持不变。

---

## Phase 1 — 稳定拿满 touch 分

目标验收：3 个不同布局/seed 的 300s 局，平均 touch ≥ 4 件且无摔倒（20 分钟整局外推 ≥12 件）。

### Task 1: 里程计盲走到达判定（修"够不着"的根因）

**Files:**
- Modify: `demo/solution_task_b_g1.py`（`TaskBPlanner`，约 256–420 行）
- Test: `tests/task_b/test_planner_blind_walk.py`（新建）

- [ ] **Step 1: 写失败的测试**

```python
"""Blind-walk arrival: planner must keep approaching a remembered target after
the camera loses it (<0.7m is a blind zone), and trigger squat on odometry
ground distance instead of fresh-detection distance."""
import math
import unittest

from demo.solution_task_b_g1 import Detection, Pose2D, TaskBPlanner


def det(track_id, wx, wy, pose, conf=0.8):
    dist = math.hypot(wx - pose.x, wy - pose.y)
    return Detection(track_id=track_id, label="colored_object",
                     rel_x=wx - pose.x, rel_y=wy - pose.y, distance=dist,
                     confidence=conf, world_x=wx, world_y=wy, bbox=(0, 0, 1, 1))


class BlindWalkArrivalTest(unittest.TestCase):
    def setUp(self):
        self.p = TaskBPlanner()

    def test_blind_walk_survives_long_detection_loss(self):
        pose = Pose2D(0.0, 0.0, 0.0)
        out = self.p.step(pose, [det(1, 1.6, 0.0, pose)], 0.0)
        self.assertEqual(out.phase, "approach_object")
        # 60 steps with NO detections (old code fell back to search at 30)
        for i in range(60):
            pose = Pose2D(0.01 * i, 0.0, 0.0)  # creeping forward
            out = self.p.step(pose, [], 0.0)
        self.assertEqual(out.phase, "approach_object")

    def test_arrival_gates_on_odometry_ground_distance(self):
        pose = Pose2D(0.0, 0.0, 0.0)
        self.p.step(pose, [det(1, 1.0, 0.0, pose)], 0.0)
        # walk to within ARRIVE_DIST of the remembered world coords, blind
        pose = Pose2D(1.0 - self.p.ARRIVE_DIST + 0.01, 0.0, 0.0)
        out = self.p.step(pose, [], 0.0)
        self.assertEqual(out.phase, "squat_sweep")

    def test_blind_walk_eventually_gives_up(self):
        pose = Pose2D(0.0, 0.0, 0.0)
        self.p.step(pose, [det(1, 1.6, 0.0, pose)], 0.0)
        out = None
        for _ in range(self.p.BLIND_WALK_STEPS + 2):
            out = self.p.step(pose, [], 0.0)  # not moving, never arrives
        self.assertEqual(out.phase, "search")

    def test_fresh_detection_refreshes_target(self):
        pose = Pose2D(0.0, 0.0, 0.0)
        self.p.step(pose, [det(1, 1.6, 0.0, pose)], 0.0)
        out = self.p.step(pose, [det(1, 1.5, 0.2, pose)], 0.0)
        self.assertEqual(out.target_world, (1.5, 0.2))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd .claude/worktrees/task-b-g1-agent-team
python -m unittest tests.task_b.test_planner_blind_walk -v
```
预期：`test_blind_walk_survives_long_detection_loss` FAIL（30 步即回 search）；`AttributeError: BLIND_WALK_STEPS`。

- [ ] **Step 3: 实现**

`TaskBPlanner` 改动（保持其余不变）：

```python
class TaskBPlanner:
    # ... TARGET_CENTER / SEARCH_WAYPOINTS 等不动 ...
    ARRIVE_DIST = 0.35          # 改: 里程计地面距离到达阈值 (原 0.55, 语义是检测距离)
    BLIND_WALK_STEPS = 250      # 新: 检测丢失后仍朝记忆坐标走的最大步数 (5s@50Hz)
    APPROACH_STANDOFF = 0.25    # 新: 站位点离物体的距离 (原 _approach_output 内写死 0.42)

    def reset(self) -> None:
        # ... 原有字段 ...
        self.steps_since_seen = 0   # 新

    def step(self, pose, detections, current_score, posture="ok"):
        # ... search 分支不动 ...
        if self.phase == "approach_object":
            self.phase_steps += 1
            det = self._refresh_active_detection(detections)
            if det is None:
                self.phase = "search"
                self.active_detection = None
                self.phase_steps = 0
                return self._search_output(pose)
            self.active_detection = det
            # 改: 到达判定用里程计地面距离, 不再用 det.distance
            if pose.distance_to((det.world_x, det.world_y)) <= self.ARRIVE_DIST:
                self.phase = "squat_sweep"
                self.squat_step = 0
                return self._squat_output(0.0)
            return self._approach_output(pose, det)
        # ... 其余分支不动 ...

    def _refresh_active_detection(self, detections):
        if self.active_detection is None:
            return None
        for det in detections:
            if det.track_id == self.active_detection.track_id:
                self.steps_since_seen = 0      # 新: 看到了, 重置盲走计数
                return det
        self.steps_since_seen += 1             # 改: 原来用 phase_steps<30
        if self.steps_since_seen <= self.BLIND_WALK_STEPS:
            return self.active_detection       # 盲走: 沿用记忆中的世界坐标
        return None

    def _approach_output(self, pose, det):
        bearing = pose.bearing_to((det.world_x, det.world_y))
        standoff = self.APPROACH_STANDOFF      # 改: 0.42 -> 类常量 0.25
        tx = det.world_x - standoff * math.cos(bearing)
        ty = det.world_y - standoff * math.sin(bearing)
        return PlannerOutput("approach_object",
                             self._drive_to(pose, tx, ty, bearing, 0.28),
                             "stow", (det.world_x, det.world_y))
```

注意：进入 approach（search 分支选中目标）时也要 `self.steps_since_seen = 0`。

- [ ] **Step 4: 跑测试确认通过 + 老测试不回归**

```bash
python -m unittest tests.task_b.test_planner_blind_walk tests.task_b.test_solution_task_b_g1 -v
```
预期：新测试 4 个 PASS；老测试若有依赖 `ARRIVE_DIST=0.55`/30 步回退行为的断言，按新语义更新断言（行为变化是本任务目的，不是回归）。

- [ ] **Step 5: Commit**

```bash
git add demo/solution_task_b_g1.py tests/task_b/test_planner_blind_walk.py tests/task_b/test_solution_task_b_g1.py
git commit -m "feat(task-b): odometry blind-walk arrival gate, fixes unreachable squat trigger"
```

### Task 2: 多目标记忆（ObjectMemory）——少转圈、按近邻顺序连续清场

现状：每碰完一件就回 `search` 原地转圈重新找。感知一帧能看到 1–2 件、转一圈 ~3 件，记下来不用白不用。

**Files:**
- Modify: `demo/solution_task_b_g1.py`（`TaskBPlanner`）
- Test: `tests/task_b/test_planner_memory.py`（新建）

- [ ] **Step 1: 写失败的测试**

```python
import unittest

from demo.solution_task_b_g1 import Pose2D, TaskBPlanner
from tests.task_b.test_planner_blind_walk import det


class ObjectMemoryTest(unittest.TestCase):
    def setUp(self):
        self.p = TaskBPlanner()

    def test_remembers_all_detections_not_just_active(self):
        pose = Pose2D(0.0, 0.0, 0.0)
        self.p.step(pose, [det(1, 1.0, 0.0, pose), det(2, 2.5, 1.0, pose)], 0.0)
        self.assertIn(2, self.p.memory)

    def test_goes_to_remembered_object_after_standup_without_search(self):
        pose = Pose2D(0.0, 0.0, 0.0)
        self.p.step(pose, [det(1, 0.5, 0.0, pose), det(2, 2.5, 1.0, pose)], 0.0)
        # arrive -> squat -> score -> stand_up 完整走完
        self.p.step(Pose2D(0.2, 0.0, 0.0), [], 0.0)          # squat (arrived)
        self.p.step(Pose2D(0.2, 0.0, 0.0), [], 1.0)          # score_delta>0 -> stand_up
        out = None
        for _ in range(TaskBPlanner.STAND_RAMP_STEPS + 1):
            out = self.p.step(Pose2D(0.2, 0.0, 0.0), [], 1.0)
        # 起身后应直奔记忆中的 object 2, 而不是 search
        self.assertEqual(out.phase, "approach_object")
        self.assertEqual(out.target_world, (2.5, 1.0))

    def test_gives_up_object_after_max_attempts(self):
        pose = Pose2D(0.0, 0.0, 0.0)
        for _ in range(TaskBPlanner.MAX_ATTEMPTS):
            self.p.step(pose, [det(1, 0.5, 0.0, pose)], 0.0)   # select
            self.p.step(Pose2D(0.2, 0.0, 0.0), [], 0.0)        # arrive -> squat
            for _ in range(TaskBPlanner.SQUAT_SWEEP_MAX_STEPS + 1):
                self.p.step(Pose2D(0.2, 0.0, 0.0), [], 0.0)    # timeout, no score
            for _ in range(TaskBPlanner.STAND_RAMP_STEPS + 1):
                self.p.step(Pose2D(0.2, 0.0, 0.0), [], 0.0)    # stand_up done
        out = self.p.step(pose, [det(1, 0.5, 0.0, pose)], 0.0)
        self.assertEqual(out.phase, "search")  # 两次没碰到就放弃, 不死磕


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试确认失败**

```bash
python -m unittest tests.task_b.test_planner_memory -v
```
预期：`AttributeError: memory` / `MAX_ATTEMPTS`。

- [ ] **Step 3: 实现**

```python
class TaskBPlanner:
    MAX_ATTEMPTS = 2   # 新: 每件物体最多蹲扫尝试次数

    def reset(self) -> None:
        # ... 原有字段 ...
        self.memory: dict[int, tuple[float, float]] = {}   # track_id -> world xy
        self.attempts: dict[int, int] = {}                 # track_id -> 蹲扫次数

    def step(self, pose, detections, current_score, posture="ok"):
        for d in detections:                               # 新: 任意阶段都记忆
            self.memory[d.track_id] = (d.world_x, d.world_y)
        # ... 原有逻辑, 两处改动:
        # (1) squat_sweep 进入时(从 approach 切换处)记一次尝试:
        #     self.attempts[det.track_id] = self.attempts.get(det.track_id, 0) + 1
        # (2) search 分支: 先查记忆再转圈:
        if self.phase == "search":
            fresh = self._choose_detection(detections)
            if fresh is not None:
                ...  # 原逻辑
            remembered = self._choose_from_memory(pose)    # 新
            if remembered is not None:
                self.active_detection = remembered
                self.steps_since_seen = 0
                self.phase = "approach_object"
                self.phase_steps = 0
                return self._approach_output(pose, remembered)
            return self._search_output(pose)

    def _exhausted(self, track_id: int) -> bool:
        return (track_id in self.touched_track_ids
                or track_id in self.placed_track_ids
                or self.attempts.get(track_id, 0) >= self.MAX_ATTEMPTS)

    def _choose_from_memory(self, pose) -> Detection | None:
        candidates = [(tid, xy) for tid, xy in self.memory.items() if not self._exhausted(tid)]
        if not candidates:
            return None
        tid, (wx, wy) = min(candidates, key=lambda kv: pose.distance_to(kv[1]))
        return Detection(track_id=tid, label="memory", rel_x=wx - pose.x, rel_y=wy - pose.y,
                         distance=pose.distance_to((wx, wy)), confidence=0.5,
                         world_x=wx, world_y=wy, bbox=(0, 0, 0, 0))
```

同时把 `_choose_detection` 的过滤改为 `not self._exhausted(d.track_id)`（原来只查 touched|placed）。

- [ ] **Step 4: 跑全部 task_b 单测确认通过**

```bash
python -m unittest discover -s tests/task_b -p 'test_*.py' -v
```

- [ ] **Step 5: Commit**

```bash
git add demo/solution_task_b_g1.py tests/task_b/test_planner_memory.py
git commit -m "feat(task-b): object memory map, nearest-first chaining, per-object attempt cap"
```

### Task 3: Isaac 实测验证 + 调参收敛（Phase 1 验收）

**Files:** 只跑命令、记录 `docs/task_b_g1_squat_notes.md`；如需调参只动本任务列出的旋钮。

- [ ] **Step 1: 蹲扫几何 probe（确认手真能进 0.20m 球）**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate atec
cd .claude/worktrees/task-b-g1-agent-team
PYTHONPATH=. python scripts/probe_task_b_g1_squat.py \
  --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless --num_steps 600 \
  --out outputs/task_b_g1_squat_blindwalk
```
看 `squat_geometry.json`：squat 段 `nearest.dist` min 应 ≤ 0.20（原基线 0.56）。Isaac 启动需数分钟，放后台跑、轮询日志。

- [ ] **Step 2: 多 seed 评分**

```bash
for i in 1 2 3; do
  PYTHONPATH=. python scripts/eval_task_b_g1_wbc.py \
    --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless \
    --seconds 300 --out logs/videos/task_b_g1_wbc/roadmap_run$i.mp4
done
```
记录每局 score / 是否摔倒 / 各 phase 步数占比。**验收：平均 touch ≥ 4 件/300s，0 摔倒。**（eval 脚本如不支持 --seed，三次运行的随机布局差异即来自环境默认随机性；如布局固定，给脚本加 `--seed` 透传。）

- [ ] **Step 3: 仅当未达标时按序调参（一次只动一个，每次重跑 Step 1）**

| 症状 | 旋钮 | 方向 |
|---|---|---|
| 蹲下时物体仍在脚前 >0.4m | `APPROACH_STANDOFF` 0.25→0.15；solution.py `CREEP_STEPS` 22→35 | 停更近/蹭更多 |
| 蹲点准但扫不到 | `SWEEP_LAT_AMP` 0.14→0.18（勿动 HAND_Z/SQUAT_HEIGHT） | 扫更宽 |
| 盲走走过头/绕圈 | `ARRIVE_DIST` 0.35→0.45 | 放宽到达 |
| 蹲下瞬间晃动 | 在 solution.py creep 与蹲下之间加 25 步零速 settle gate | 先停稳再蹲 |

- [ ] **Step 4: 把结果追加到 `docs/task_b_g1_squat_notes.md` 并 commit**

```bash
git add docs/task_b_g1_squat_notes.md
git commit -m "docs(task-b): blind-walk + memory eval results"
```

- [ ] **Step 5: 同步提交包**：把改动后的 `solution_task_b_g1.py` 复制到 `demo/task_b/` 暂存目录，跑一遍 `python -m unittest discover -s tests/task_b -v` 后 commit。

---

## Phase 2 — 夹取搬运进圈（每件再 +1）

### 总体策略与风险控制（执行前先读）

- **两段式清场（EV 最优）**：第一遍先用 Phase 1 流程把能碰的物体全碰一遍（把 touch 分**落袋**），第二遍再对记忆中的物体做"抓取-搬运-入圈"。原因：摔倒整局终止——若先搬运、中途摔倒，连没碰过的 touch 分也全没了；20 分钟时长足够两遍（Phase 1 验收会给出每件耗时数据）。
- 抓取尝试本身手必然进 0.20m 球 → 第二遍对"第一遍没碰到的"物体也是补 touch 分的机会。
- 搬运导航精度要求低：圈半径 1.0m，里程计漂移 median 0.24m，纯盲走可达；目标圈是地面橙红色大圆盘，感知里 `_is_large_target_background` 的颜色判据可直接复用为"圈检测器"做视觉确认。
- **先脚本抓取，ACT 是条件分支**：dex1 平行夹爪 + 已知物体 base 系坐标（误差 0.04m）+ WBC 手位姿 IK，脚本化抓取很可能够用。Task 4 的 probe 给出量化数据后再决定是否上 ACT（决策门在 Task 4 Step 4）。

### Task 4: MiniWBC 手指命令 + 抓取可行性 probe（决策门）

**Files:**
- Modify: `demo/mini_wbc.py:80-116`（`MiniWBC.act`）
- Create: `scripts/probe_task_b_g1_grasp.py`
- Test: `tests/task_b/test_mini_wbc_fingers.py`（新建）

- [ ] **Step 1: 查清 dex1 手指**：读 `source/atec_rl_lab/atec_rl_lab/assets/robots/g1/g1_29dof_dex1.py`，记下动作位 29–32 对应的 4 个手指关节名、默认角、上下限、开/合方向。动作语义是 `target = default + 0.5 * action`。把结论写进本计划文件此处再继续。

- [ ] **Step 2: 写失败的测试**

```python
import unittest
import numpy as np

try:
    from demo.mini_wbc import MiniWBC, DEFAULT_LEFT_HAND, DEFAULT_RIGHT_HAND
    HAVE_ORT = True
except (RuntimeError, ModuleNotFoundError):
    HAVE_ORT = False


@unittest.skipIf(not HAVE_ORT, "onnxruntime not available")
class FingerCommandTest(unittest.TestCase):
    def test_fingers_passthrough_to_action_29_32(self):
        wbc = MiniWBC()
        row = np.zeros(12 + 3 * 33, dtype=np.float32)
        row[11] = -1.0  # projected gravity z
        out = wbc.act(row, [0, 0, 0], 0.75, [0, 0, 0],
                      list(DEFAULT_LEFT_HAND), list(DEFAULT_RIGHT_HAND),
                      fingers=[0.3, 0.3, -0.4, -0.4])
        self.assertEqual(out[29:33], [0.3, 0.3, -0.4, -0.4])

    def test_fingers_default_zero(self):
        wbc = MiniWBC()
        row = np.zeros(12 + 3 * 33, dtype=np.float32)
        row[11] = -1.0
        out = wbc.act(row, [0, 0, 0], 0.75, [0, 0, 0],
                      list(DEFAULT_LEFT_HAND), list(DEFAULT_RIGHT_HAND))
        self.assertEqual(out[29:33], [0.0, 0.0, 0.0, 0.0])
```

跑：`python -m unittest tests.task_b.test_mini_wbc_fingers -v`，预期 `TypeError: unexpected keyword 'fingers'`。

- [ ] **Step 3: 实现（`MiniWBC.act` 末尾）**

```python
def act(self, proprio_row, vel_cmd, base_height, waist_rpy, left_hand, right_hand,
        waist_weight=1.0, fingers=None):
    # ... 原实现不动 ...
    action_33 = np.zeros(33, dtype=np.float32)
    action_33[:29] = atec_action_body
    if fingers is not None:
        action_33[29:33] = np.asarray(fingers, dtype=np.float32)
    return action_33.tolist()
```

跑测试通过后 commit：`git commit -m "feat(task-b): finger command passthrough in MiniWBC"`。

- [ ] **Step 4: 抓取 probe（特权脚本，仿照 `scripts/probe_task_b_g1_squat.py` 的 AppLauncher/场景读取骨架）**

`scripts/probe_task_b_g1_grasp.py` 逻辑：对最近的每种物体类型各跑一次——用**真值**物体位姿（`env.scene[f"object_{i}"].data.root_pos_w`）驱动 MiniWBC：走到物体前 0.25m → 蹲 0.30 → 右手位姿命令到物体正上方 0.05m（base 系，钳到 WBC 有效范围）→ 手指闭合 100 步 → base 高度升到 0.6 → 保持 150 步。记录 JSON：物体 z 是否升离地面 >0.15m、150 步末是否仍 >0.15m（hold 成功）、是否摔倒。每类型重复 3 次（不同局）。

```bash
PYTHONPATH=. python scripts/probe_task_b_g1_grasp.py \
  --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless \
  --out outputs/task_b_g1_grasp
```

**决策门：** 任意 ≥2 种物体类型 hold 成功率 ≥ 50% → 走 Task 5（脚本抓取），**跳过 Task 6**。全部 <50% → 先只对成功率最高的类型走 Task 5，同时启动 Task 6（ACT）。0 成功且摔倒频发 → 停下来与用户讨论（参见 squat-debugging-caution 约定），不要自行加大动作幅度硬调。

### Task 5: 脚本化 grasp→carry→drop 全流程

**Files:**
- Modify: `demo/solution_task_b_g1.py`（TaskBPlanner 加阶段）、`demo/solution.py`（新阶段的 WBC 命令映射）
- Test: `tests/task_b/test_planner_carry.py`（新建）

FSM 扩展（在 Phase 1 的 FSM 上加，蹲扫路径保留为 fallback）：

```
squat_sweep 之后插入分支:
  grasp(右手到物体上方→下压→合指, 120步) → lift(升高到0.6, 验证持有) →
    持有成功: carry(导航到圈边 (-3,-10) 外 0.6m) → drop(蹲0.45, 开指, 后退) → search
    持有失败(物体没升起/中途掉): 回 squat_sweep 平扫保底 touch → search
```

实现要点（执行者按 Task 1/2 的代码风格写,逐步 TDD）：
1. 持有验证不依赖视觉：lift 后连续 30 步检查手指关节角（proprio joint_pos 29–32 位）卡在"半闭合"区间（完全闭合=夹空了，张开=没夹）。阈值由 Task 4 probe 数据给出，写成类常量。
2. carry 阶段速度钳到 0.3 m/s、wz 0.5，手位姿固定在胸前 `[0.25, ∓0.2, 0.35]`（有效范围内）；`PostureGuard` 返回 `recover` 时立即 `drop`（开指+站直），宁可丢物体不可摔倒。
3. drop 触发条件：`pose.distance_to(TARGET_CENTER) <= 1.4`（圈半径 1.0 + 余量，确保物体落点入圈）；可选视觉确认：头相机看到大面积橙红色 blob（复用 `_is_large_target_background` 判据）时对圈心做一次世界坐标修正。
4. score_delta 在 carry/drop 期间 +1 视为入圈成功（注意与 touch 分在 grasp 时刻的 +1 区分开：grasp 段的 delta 记 touch，drop 段的 delta 记入圈）。
5. 两段式调度在 `AlgSolution` 层做：`self.stage = "touch_all" | "carry"`；`touch_all` 跑 Phase 1 行为，当 planner 记忆中无未尝试目标或局时过半（30000 步）时切 `carry`，carry 阶段对记忆中物体逐个执行 grasp 流程。
6. Isaac 验收：300s 局至少完成 1 次完整 grasp→carry→drop 且 score 含入圈分；整局无摔倒。录像确认动作（`--out logs/videos/task_b_g1_wbc/carry_run.mp4`）。

每个子步骤先写 planner 单测（纯 Python，参照 Task 1/2 的测试写法），再 Isaac 验证，再 commit。

### Task 6（条件触发）: ACT 抓取技能 —— 仅当 Task 4 决策门判定脚本抓取不够

**设计要点（按此开题，先写 spec 再动代码）：**
- **动作空间用 WBC 命令而不是关节**：`[右手位姿7 + 右手指2 + base高度1] = 10 维`。WBC 负责全身平衡与 IK，ACT 只学"手往哪放、何时合指"，比 29 维关节空间易学一个量级，也天然不破坏平衡（命令始终钳在有效范围）。
- **观测**：`ee_dual_rgb`（右手相机，降采样到 224×224）+ `head_rgb` 同尺寸 + 本体（右臂 7 关节角 + 手指 2 + base 高度 + 上一步命令）。
- **数据采集**：脚本专家 + 特权真值。用 Task 4 的 probe 脚本作为专家（真值物体位姿驱动），随机化物体类型/位置/朝向与机器人停位误差（±0.1m，模拟感知误差），采 300–500 条成功轨迹。仿照 `scripts/act/collect_demos_task_e.py` 写 `scripts/act/collect_demos_task_b.py`，复用 `scripts/act/filter_demos.py` 过滤失败条目。
- **训练**：复用 `scripts/act/train_task_e.py`/`baseline.sh` 的 ACT 实现，chunk size 30（0.6s@50Hz），训练后导出 ONNX（提交容器无训练栈）。
- **部署**：作为 `grasp` 阶段的策略替换脚本版（FSM 不变），超时 200 步回退平扫。打包新增 ACT onnx 进 `demo/task_b/`。
- **不做**：不用 ACT 学走路/搬运/全流程——那些 WBC+脚本已覆盖，ACT 只补"最后 30cm 的手部精细动作"。

---

## 提交打包检查单（两个 Phase 各自收尾时跑）

1. `demo/task_b/` 暂存目录五件套同步：`solution.py`、`mini_wbc.py`、`solution_task_b_g1.py`、`policy18.onnx`、`requirements.txt`（Phase 2 如加 ACT 权重则 +1 个 onnx）。
2. `python -m unittest discover -s tests/task_b -v` 全绿。
3. 启动日志确认 WBC 加载的是 ONNX 路径（无静默 fallback）。
4. 别上传 `run.sh`/`server.py`（平台注入）；容器无网络，依赖全部打包。

## 风险表

| 风险 | 影响 | 缓解 |
|---|---|---|
| 摔倒终止整局 | 全部已得分外的机会清零 | 不动平衡参数；两段式清场先落袋 touch 分；PostureGuard→立即弃物回稳 |
| 盲走撞到物体把它踢走 | 物体被踢离原位扫不到 | ARRIVE_DIST 0.35 留了脚前余量；attempts 上限 2 防死磕；被踢走的物体会被重新检测成新位置 |
| 线上环境与本地有差异 | 颜色阈值/外参失配 | 决策记录中的 YOLO 重新评估条件；外参来自实测 probe，提交前用官方 play_atec_task.py 入口再验一遍 |
| dex1 夹不住某些物体（如 Banana 太扁） | 该类型只能拿 touch 分 | Task 4 按类型量化，carry 阶段只对可抓类型执行 |
