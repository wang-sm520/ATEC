# Task B G1 Squat-and-Sweep Design

日期：2026-06-10

## Goal

为 ATEC Task B (G1) 增加一个 **蹲下扫地拿接触分** 的能力：视觉检测垃圾 → 行走到物体前 → 切换到 OpenWBT 蹲下策略蹲下 → 双臂脚本扫地，力求让 `left_hand_base_link` / `right_hand_base_link` 进入物体的接触判定球内拿分。

本设计是对现有 `demo/solution_task_b_g1.py` 视觉接触基线的扩展，不从零搭建，也不覆盖 `demo/solution.py`（保护 Task D 现有方案）。

## 评分事实（决定本设计可行性）

来自 `source/atec_rl_lab/atec_rl_lab/tasks/task_b/`：

- **接触分** `GraspedObjectsByEE`（`mdp/rewards.py`）：判定 link 为 `left_hand_base_link` / `right_hand_base_link`（Task B G1 在 `env_cfg.py:109-112` 覆盖），**3D 距离 ≤ `grasp_dist_thresh = 0.20` m**（`env_cfg.py:33` 把默认 0.12 放宽到 0.20），每个物体首次接触 +1，共 18 个。
- **投放分** `ObjectsInCircle`：物体 root 落入圆心 `(-3.0, -10.0)`、半径 1.0、z∈[0,0.5] +1/个。**本设计不追求投放分**。
- **致命约束** `illegal_contact` 终止（`env_cfg.py:113-116`）：`base_link` 或 `.*_hip_(pitch|roll|yaw)_link` 接触即整局结束。**蹲下失稳摔倒 = 直接结束本局**。
- 机器人初始位姿 `pos=(-10, -10, 0.9)`（`env_cfg.py:104`）。

结论：0.20m 容差宽松，蹲下把手放到地面附近左右扫，几何上有现实的得分机会；但蹲姿必须稳，摔倒代价是整局归零。

## 核心约束：kp/kd 不可从 solution 端设置

ATEC 评测环境的 PD 增益写死在 actuator 配置里（`assets/robots/g1/g1_29dof_dex1.py`），solution 只能输出关节**位置目标**，env 内部用它自己的增益算力矩。两边增益差异：

| 关节 | OpenWBT 训练 kp / kd | ATEC env 固定 kp / kd |
|---|---|---|
| 髋 pitch | 100 / 2 | 200 / 5 |
| 髋 roll/yaw | 100 / 2 | 150 / 5 |
| 膝 | 150 / 4 | 200 / 5 |
| 踝 pitch/roll | 40 / 2 | 20 / 2 |
| 肩 pitch/roll | 100 / 2 | 100 / 2 |

因此“字面匹配 kp/kd”不可能。**可做的等价替代**是在动作端在线匹配 P 项力矩（见下文增益补偿）。注意阻尼失配方向是安全的：ATEC 腿部 kd=5 > WBT 2-4（更阻尼=更稳），踝部 kd 两边都是 2。唯一风险点是踝部刚度只有一半。

## OpenWBT 蹲下策略接口事实

来自 `OpenWBT/`（`deploy/configs/g1_squat.yaml`、`deploy/helpers/policy_unified.py`、`deploy/controllers/`）：

- 策略文件 `OpenWBT/ckpts/squat.onnx`，用 `onnxruntime` 跑，**带 256 维 RNN hidden state**（`[1,1,256]`，跨 step 持续，reset 清零）。
- **obs 78 维**，顺序：`[squat_cmd(2), gravity_orientation(3), ang_vel*0.25 (3), (q-default)*1.0 (29), qd*0.05 (29), last_action(12)]`，clip 到 ±100。
- **输出 12 维**（仅 12 条腿关节，action_idx [0..11]）；后处理 `target_q = action*0.25 + default[0:12]`；部署时把 ankle_roll 目标 [5,11] 置零。
- **squat_cmd = [height, pitch]**：height 默认 0.75、范围 [0.35, 0.75]（越小越蹲）；pitch 默认 0、范围 [0, 0.5]。
- 控制频率 50Hz（decimation 4、sim dt 0.005），**与 ATEC env 的 step_dt=0.02 完全一致**——无需重采样。
- OpenWBT 29-DoF 关节顺序与 ATEC 前 29 个 body joint 顺序一致（L/R 腿×6、腰×3、L/R 臂×7）。

关键含义：**squat 策略只控腿；手臂是它的 `upper_action` 槽位，由本设计脚本控制。** 增益问题只影响腿能否稳住蹲姿，扫地手臂动作独立可控。

## 现有可复用资产

`demo/solution_task_b_g1.py`（当前 worktree）：

- `TaskBRgbdPerception` — head RGB-D 颜色+深度检测，输出世界坐标 `Detection`。复用。
- `G1VelocityPolicyBridge` — 960 维 term-major history 行走桥接（控全 29 身体关节，补齐 33）。复用作导航段。
- `DeadReckoningOdometry` — 从 proprio 积分平面位姿。复用。
- `TaskBPlanner` — search/approach/touch/push/verify 状态机。**改**：用蹲扫阶段替换 left_touch/push。
- `LocalObjectInteraction` — 保守左手 override。**被新的扫地控制器取代**。

`.worktrees/task-b-openwbt-squat/demo/solution_task_b_openwbt.py`（旧 worktree，未提交）：

- `OpenWBTSquatBridge` — 已实现 Task B proprio→78 维 obs、12 维输出→ATEC action 的转换，含默认角偏移、action scale 0.25→0.5、obs scale、ankle-roll 置零、ONNX/脚本兜底 runner。**搬入当前 worktree 并加增益补偿。**
- `_HeuristicSquatRunner` — 脚本蹲姿兜底。搬入。

## 架构：三段策略接力状态机

三段各管不同关节，接力点是“停稳”和“起身”：

```
[感知] head RGB-D ──► Detection(world_x, world_y) ──► planner 选目标
                                                          │
  ① 行走 G1VelocityPolicyBridge  ──停稳切换──►  ② 蹲下 OpenWBTSquatBridge(腿,+增益补偿)
     (控全身, 走到 standoff)                          ③ 扫地 GroundSweepArmController(脚本控臂)
                                                          │
                                          得分 / 超时 / 危险 ──► 起身 ──► 切回 ①, 下一个目标
```

### 数据流（每 step，`AlgSolution.predicts`）

1. `row = proprio[0]`；`pose = odom.update(row)`。
2. `detections = perception.update(obs["image"], pose)`。
3. `plan = planner.step(pose, detections, current_score)` → 返回 `phase` 与该段所需指令。
4. `guard = posture_guard.check(row)` → 是否危险（projected_gravity 偏离阈值）。
5. 按 `plan.phase` 分派：
   - `search` / `approach_object`：`action = bridge.act(proprio, plan.command)`（行走，手臂收拢）。
   - `squat_sweep`：
     - `leg_action = squat_bridge.act(proprio, SquatCommand(height, pitch))`（含增益补偿；危险时 `guard` 强制脚本兜底蹲姿）。
     - `arm_action = sweep.step(squat_progress)`（脚本双臂轨迹）。
     - 合并：腿 12 关节用 `leg_action`，臂+手用 `arm_action`，其余默认。
   - `stand_up`：squat_bridge 把 height 拉回 0.75，臂收拢。
6. 返回 `{"action": action_33, "giveup": False}`。

## 模块设计

### OpenWBTSquatBridge（搬入 + 增益补偿）

在旧实现基础上，把 12 维腿动作的转换从“静态默认角偏移”升级为“在线 P 项匹配”：

```
对每条腿关节 j:
  target_WBT_j = action_WBT_j * 0.25 + default_WBT_j        # OpenWBT 原始目标
  q_j          = 实测关节角 (来自 proprio, 已加回 ATEC 默认角)
  ratio_j      = KP_RATIO[j]                                 # = kp_WBT_j / kp_ATEC_j, 踝部再乘安全系数
  target_ATEC_j = q_j + ratio_j * (target_WBT_j - q_j)       # 在线匹配 P 项力矩
  action_ATEC_j = (target_ATEC_j - default_ATEC_j) / 0.5     # 转 ATEC 动作
```

`KP_RATIO`（12 条腿，按 L/R 各 6：hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll）：

```
hip_pitch  = 100/200 = 0.50
hip_roll   = 100/150 = 0.67
hip_yaw    = 100/150 = 0.67
knee       = 150/200 = 0.75
ankle_pitch= 40/20 * SAFETY = 2.0 * 0.7 = 1.40
ankle_roll = (置零, 不参与)
```

`SAFETY` 默认 0.7（踝部专用），Isaac 实测后可调。obs 端、hidden state、ankle-roll 置零、ONNX/脚本兜底沿用旧实现。RNN hidden state 在 `reset()` 与每次进入 `squat_sweep` 前清零（避免跨目标污染）。

接口：`act(proprio, SquatCommand) -> leg_action_12: list[float]`（只返回腿，臂由调用方填）。

### GroundSweepArmController（新建，脚本）

纯脚本，输出 G1 上肢关节（肩 15-21 / 22-28、手 29-32）的位置偏移动作，不依赖策略、不依赖增益。

- 输入：`squat_progress ∈ [0,1]`（蹲下进度，来自 planner）与一个 `sweep_phase`（内部步数）。
- 轨迹：随 `squat_progress` 增大，肩 pitch 前展、肘下压，把 `*_hand_base_link` 降到地面高度；到位后双臂随 `sweep_phase` 做正弦左右摆扫（肩 roll/yaw 摆动），覆盖身体前方一个扇区。
- 手指：扫地时半开（增大手基座到地面物体的接触面），不强求闭合。
- 所有数值是 0.5-scale 下的归一化偏移量，初值用启发式，由 `scripts/probe_task_b_g1_squat.py` 量出的几何标定后调。

接口：`step(squat_progress: float) -> dict[int, float]`（关节 index → 动作值），由 `AlgSolution` 写入对应 action 槽位。

### PostureGuard（新建）

监测姿态危险，触发脚本兜底或放弃。

- 输入：proprio row 的 `projected_gravity = row[9:12]`。
- 危险判据：`projected_gravity` 的水平分量模长 `hypot(g_x, g_y)` 超阈值（接近栽倒），或 z 分量绝对值偏离 1 过大。
- 输出：`PostureState`（`ok` / `recover`）。`recover` 时 `OpenWBTSquatBridge` 改用 `_HeuristicSquatRunner` 的保守浅蹲，且 planner 进入 `stand_up`。

阈值用保守初值（例如水平分量 > 0.35 视为危险），Isaac 实测调。

### TaskBPlanner（改）

新增/替换阶段：

- 保留 `search`、`approach_object`（复用现有导航逻辑，手臂收拢）。
- **新增 `squat_sweep`**：进入条件——到达目标 standoff（distance ≤ 阈值）且停稳（lin/ang vel 低于阈值若干步）。该阶段：
  - 输出 `SquatCommand`：height 从 0.75 按 ramp 缓降到 ~0.40，pitch 升到 ~0.25；
  - 输出 `squat_progress` 给扫地控制器；
  - 持续到拿分（score_delta>0）、或超时 `SQUAT_SWEEP_MAX_STEPS`、或 `PostureGuard=recover`。
- **新增 `stand_up`**：height ramp 回 0.75，臂收拢；姿态恢复且站稳后切回 `search`/`approach`。
- 选目标时过滤已接触 track（沿用 `touched_track_ids`）。

### AlgSolution（改）

三策略接力调度（见数据流）。`reset()` 复位 perception/odom/planner/squat_bridge(含 hidden state)/sweep/guard。`__init__` 加载行走 policy（`policy_a.pt`/`policy.pt`）与 squat ONNX（缺失则脚本兜底）。

### scripts/probe_task_b_g1_squat.py（新建，仅本地标定）

蹲下时读取 `env.scene` 真值：记录 `*_hand_base_link` 的世界 z 高度、到最近物体 root 的 3D 距离、躯干高度与姿态。用于回答“蹲到 0.40 + 双臂下压，手基座能否进入物体 0.20m 球”，并标定扫地轨迹与 squat 目标 height。privileged，不进 solution。

## Error handling / 兜底

1. **ONNX 缺失**：`OpenWBTSquatBridge` 自动回落 `_HeuristicSquatRunner`（脚本蹲姿），solution 仍可运行。
2. **姿态危险**：`PostureGuard=recover` → 脚本兜底浅蹲 + planner 转 `stand_up`，宁可不得分不摔倒。
3. **蹲扫超时无分**：`stand_up` → 切回行走找下一个，不恋战。
4. **踝部过冲**：`SAFETY` 系数与 `SQUAT` height ramp 步数兜住；实测振荡则减小 `SAFETY` / 放慢 ramp。

## Testing

非 Isaac 快速测试（`unittest`，torch/numpy 可缺省时 skip）：

- `OpenWBTSquatBridge` 增益补偿：给定已知 `action_WBT`、`q`、`KP_RATIO`，断言 `target_ATEC = q + ratio*(target_WBT-q)`，并验证 ankle_roll 置零、输出 12 维。
- `GroundSweepArmController`：`squat_progress=0` 时手臂接近收拢；`=1` 时肩 pitch 前展、肘下压超过阈值；`sweep_phase` 变化时肩 roll 左右摆动符号翻转；只改上肢 index、不动腿。
- `PostureGuard`：竖直 gravity 判 `ok`；大水平分量判 `recover`。
- `TaskBPlanner`：到达+停稳进入 `squat_sweep`；score_delta>0 标记 touched 并转 `stand_up`/下一个；超时转 `stand_up`；`recover` 转 `stand_up`。
- `AlgSolution`（fake bridges）：`squat_sweep` 段腿来自 squat_bridge、臂来自 sweep、返回 33 维、no giveup；`reset` 复位全部组件含 hidden state。

Isaac 验证：

- `scripts/probe_task_b_g1_squat.py` 量手基座 z 与到物体距离，确认几何可达。
- `scripts/eval_task_b_g1.py` 跑长程评估，记录是否拿到接触分、是否摔倒终止、踝部是否振荡。

## Scope / 文件清单

修改 `demo/solution_task_b_g1.py`：搬入 `OpenWBTSquatBridge`（+增益补偿）、`SquatCommand`、`_HeuristicSquatRunner`/`_OnnxSquatRunner`；新增 `GroundSweepArmController`、`PostureGuard`；改 `TaskBPlanner`、`AlgSolution`；移除/取代 `LocalObjectInteraction`。

修改 `tests/task_b/test_solution_task_b_g1.py`：加上述测试。

新增 `scripts/probe_task_b_g1_squat.py`。

不改 `demo/solution.py`。不改 env / 资产 / 训练配置。

## 非目标（YAGNI）

- 不做投放分（push 进圈）。
- 不重训 squat 策略。
- 不做精确抓取/夹取，只做接触级蹲扫。
- 不在 solution 内访问 env 真值（仅 probe 脚本可）。
