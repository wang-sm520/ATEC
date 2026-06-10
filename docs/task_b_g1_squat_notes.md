# Task B G1 Squat-and-Sweep — Build Notes & Isaac Run Guide

状态：Task 1–8（非 Isaac 构建 + 验证）完成。Task 9（Isaac probe + 评估）待在有 IsaacLab 的机器上运行。

设计：`docs/superpowers/specs/2026-06-10-task-b-g1-squat-sweep-design.md`
计划：`docs/superpowers/plans/2026-06-10-task-b-g1-squat-sweep.md`

## 已完成（commits 50b8db6 → 98f559b）

全部在 `demo/solution_task_b_g1.py`（不覆盖 `demo/solution.py`）：

- `OpenWBTSquatBridge` — Task B proprio → OpenWBT 78 维 obs → 12 腿动作，含**在线 P 项增益补偿**（`KP_RATIO`，踝部 ×1.4=2.0×ANKLE_SAFETY 0.7）、ankle_roll 置零、RNN hidden state、ONNX/脚本兜底 runner（reset 恢复主 runner）。
- `GroundSweepArmController` — 纯脚本双臂扫地（肩 pitch 前展 + 肘下压 + 肩 roll 正弦摆扫 + 手指半开），仅写上肢 15–32。
- `PostureGuard` — projected_gravity 水平分量 > 0.35（≈20°）判 `recover`，ndim-aware 兼容 torch/numpy/list 输入。
- `TaskBPlanner` — 阶段 `search → approach_object → squat_sweep → stand_up`；squat_sweep 在 score_delta>0 / 超时 / posture=recover 时起身。
- `AlgSolution` — 三策略接力调度，保留感知节流（PERCEPTION_INTERVAL=5）；squat/stand 段腿用 squat_bridge、臂用 sweep、腰默认；recover 段切脚本兜底。
- `scripts/probe_task_b_g1_squat.py` — 蹲下几何标定 probe（量 `*_hand_base_link` 的 z 与到物体 3D 距离）。
- `demo/squat.onnx` — 已 vendored（1.4MB，来自 OpenWBT/ckpts）。

单测：`python -m unittest tests.task_b.test_solution_task_b_g1 -v` → **56 tests OK（skipped=14，torch/onnx 缺失时跳过）**。

## 评分依据（决定能否拿分）

- 接触分：`*_hand_base_link` 到物体 root **3D 距离 ≤ 0.20m**，每物体首次 +1。
- 摔倒终止：`base_link` / `.*_hip_(pitch|roll|yaw)_link` 接触即整局结束 → 蹲姿失稳代价极大。

## Task 9：Isaac 运行步骤（在有 IsaacLab 的机器）

1. 蹲下几何 probe（关键：手基座能否降到物体附近、能否逼近 0.20m、是否摔倒）：
```bash
PYTHONPATH=. python scripts/probe_task_b_g1_squat.py \
  --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless --num_steps 400 \
  --out outputs/task_b_g1_squat
```
看 `outputs/task_b_g1_squat/squat_geometry.json`：squat 段 `l_hand_z`/`r_hand_z` 是否降到接近物体高度、`nearest.dist` 能否接近 0.20、是否 `terminated`（摔倒）。

2. 短程评估冒烟（看 phase 流转 search→approach→squat_sweep→stand_up，是否 import/shape 报错，是否立刻摔倒）：
```bash
PYTHONPATH=. python scripts/eval_task_b_g1.py \
  --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless --max_steps 600 --debug
```
启动时会打印 `[task-b-g1] squat runner: ONNX (...)` 或 `heuristic` —— **务必确认是 ONNX**（否则没在测真正的蹲下策略，见下方 I1）。

## 调参旋钮（进入蹲下反复调参前先和用户对思路）

- 增益：`OpenWBTSquatBridge.ANKLE_SAFETY`（踝部过冲就调小）、`KP_RATIO`。
- 蹲姿：`TaskBPlanner.SQUAT_TARGET_HEIGHT`（0.40，蹲多深）、`SQUAT_TARGET_PITCH`（0.25）、`SQUAT_RAMP_STEPS`（60，蹲多快）、`SQUAT_SWEEP_MAX_STEPS`（220）。
- 扫地：`GroundSweepArmController.REACH_SH_PITCH/REACH_ELBOW`（手放多低）、`SWEEP_ROLL_AMP/SWEEP_PERIOD`（扫多宽多快）、`FINGER_OPEN`。
- 安全：`PostureGuard.TILT_THRESH`（0.35，多敏感触发兜底）。
- 是否需要"到达后先停稳再蹲"的 settle gate（当前到 ARRIVE_DIST=0.55 立即蹲；若蹲下时还在动导致失稳，需加 settle gate，要用到速度信息）。

## 已知操作性问题（Isaac 运行/提交前处理）

- **I1（重要）**：`onnxruntime` 不在 `demo/requirements.txt`，基础镜像是 PyTorch 镜像，不保证预装。若缺失，`OpenWBTSquatBridge` 会**静默回落脚本兜底**（启动日志会显示 `heuristic`）。要真正跑 squat.onnx，需把 `onnxruntime` 加入提交镜像依赖，并在首跑确认日志是 ONNX。
- **I2（打包，预期内）**：`demo/Dockerfile` 当前只 COPY `solution.py` + `policy.pt`。真正提交本方案时需打包 `solution_task_b_g1.py`（改名为 `solution.py`）、`squat.onnx`、走路策略 `policy_a.pt`/`policy.pt`。设计刻意把打包留到后续单独分支，未动 `demo/solution.py`/Dockerfile。
- 走路策略权重：`AlgSolution` 的 `G1VelocityPolicyBridge` 需要 `demo/policy_a.pt` 或 `demo/policy.pt`（Task A 训练的 G1 locomotion TorchScript）。Isaac 运行前确认该文件在 `demo/` 下。

## Task 9 运行结果（待填）

- squat 段手基座 z 范围：TODO
- 到最近物体最小距离：TODO（能否 ≤0.20）
- 是否摔倒终止：TODO
- 评估 phase 流转 / score：TODO
- 首轮调参方向：TODO
