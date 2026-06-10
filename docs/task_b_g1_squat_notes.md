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

## Task 9 运行结果（Run 1, geometry probe, 400 steps）

命令见上。env：conda `atec`（torch 2.7.0+cu128, onnxruntime 1.26.0, IsaacLab）。

- **是否摔倒：否**（400 步无 `terminated`）—— 蹲姿稳定，增益补偿没把机器人弄翻。这是最难的一关，过了。
- phase 流转：approach_object(127) → squat_sweep(221, 打满 220 超时) → stand_up(50) → search(2)。400 步只完成了**一次**蹲扫尝试（对 object_6）。
- 手基座高度：squat 段 `l_hand_z` 最低 ~0.40m、`r_hand_z` ~0.40m；stand 段一度到 0.36m。
- **到最近物体 3D 距离：始终 ≥ 0.56m**（squat 段最近 0.73m，全程最近 0.56m）。**远高于 0.20m 接触阈值 → 0 分。**
- 几何拆解（object 在地面 z≈0.05）：最近时手 z≈0.36、3D 距离 0.56m → 水平差 ≈0.47m。**机器人停得太远 + 手够不到。**

### 核心结论
蹲下稳了，但**手离物体差 ~0.5m**，问题不在"蹲"而在"够":
1. 手最低只到 ~0.40m（扫地姿态前展/下压不够，或蹲得不够低）；
2. 机器人停在离物体 ~0.5m 处（approach standoff 0.42 + 感知/里程计定位残差），手臂前伸覆盖不了这个水平距离。

### 首轮调参方向（待和用户对思路后再动）
- 让手够到地面：加大 `GroundSweepArmController.REACH_SH_PITCH/REACH_ELBOW`（手更前伸下压），和/或蹲更低 `SQUAT_TARGET_HEIGHT` 0.40→更小。
- 让机器人停更近：减小 approach standoff（`_approach_output` standoff=0.42）与 `ARRIVE_DIST`，但要先确认感知/里程计把 object 世界坐标定得准（用 `probe_task_b_g1_objects.py` 的 detections-vs-truth 校准残差）。
- 定位残差是关键未知：若感知把 object 位置定偏 ~0.5m，再怎么调 standoff 也停不准 → 先量定位误差。

## Task 9 运行结果（Run 2, eval smoke, 600 steps）

- **squat runner: ONNX**（确认加载真实蹲下策略，非脚本兜底）。
- ee bodies：`left_hand_base_link`/`right_hand_base_link`（idx 37/38）。
- **score: 0.00**，elapsed 11.98s，无摔倒。
- 关键现象：**step 1 approach → step 2 就进入 squat_sweep 并一直蹲着**。即机器人几乎在出生点就判定"到达"并下蹲，没有真正走到物体边。`seen=[1,2,3,4]` 有检测，但 active 目标的 distance 一进场就 ≤ ARRIVE_DIST=0.55 → 过早下蹲。
- 对比 probe(Run1) 走了 127 步才蹲：两次随机物体布局不同，但**共同点是手到物体始终 ≥0.5m、0 分**。

## Task 9 运行结果（Run 3, objects detection-vs-truth, 80 steps）

- 检测定位误差（detection → 最近真值 object 的 XY 误差）：**~0.16–0.19m**（step0: 2检测 err 0.194；step20: 1检测 err 0.159）。**感知其实够准（≤0.2m），定位不是 0.5m 偏差。**
- **但检测极稀疏**：18 个物体每帧只检出 1–2 个，且 step 40/60 起降到 0（机器人移动/转向后丢失）。

### 综合根因（修正后）
定位不是主因。真正的问题是**几何够不到 + 检测稀疏**：
1. **手蹲不够低**：手基座最低 ~0.40m，地面物体 z≈0.05 → 垂直差 ~0.35m，单这一项就超 0.20m 阈值。
2. **standoff 太远**：planner approach 停在离物体 ~0.42m，加上人形蹲姿手臂前伸地面可达范围很小，水平也够不到。要让物体几乎到脚边（pelvis 近乎在物体正上方）。
3. **检测稀疏 + eval 过早下蹲**：一进场出现 ≤0.55m 检测就下蹲（ARRIVE_DIST 太宽松），且头相机只看前方窄锥、转身即丢，可用目标少。

### 建议首轮调参（一次只动一个，每次都跑 probe 复测；进入前先和用户对思路）
按收益/风险排序：
1. **蹲更低 + 手更下压**：`SQUAT_TARGET_HEIGHT` 0.40→0.30、`GroundSweepArmController.REACH_SH_PITCH/REACH_ELBOW` 加大（先让手基座 z 进到 ~0.15）。验证：probe 的 squat 段 `*_hand_z` min。
2. **停更近**：`_approach_output` standoff 0.42→0.20、`ARRIVE_DIST` 收紧但配合检测稳定性。验证：probe squat 段 nearest dist min。
3. **检测更密**：放宽 `TaskBRgbdPerception` 阈值 / 增大 search 扫视，让一帧多检几个、approach 中不丢目标。

> 注意（用户约定）：蹲姿目前**稳定不摔**，调蹲深/前压时要小步、每步复测，避免把稳定性调没。

## Task 9 Run 4-5（"停更近"尝试，已回退）

试了"蹲更低(height 0.40→0.35)+停更近(standoff 0.42→0.20, ARRIVE_DIST 0.55→0.30)+手更下压(reach 0.7/0.6→1.0/0.95)"：

- **v2**（gate 仍用 `det.distance` 相机深度）：**完全不蹲**（approach 398 步）。因为相机到地面物体的深度对人形头相机始终 ≥~相机高度，永远 ≤0.30 不成立。
- **v3**（把 gate 改成 odometry 地面距离 `pose.distance_to(object_world)`）：**仍完全不蹲**。机器人物理上停在离物体 ~0.5m（真值最近 0.595m）就**走不近了**，到不了 0.30m 阈值。

### 根因（关键）
**感知把物体世界坐标定得太远 → 导航永远"到不了"。** `TaskBRgbdPerception._pixel_to_robot_xy` 用 `rel_x = depth`（相机斜距）当作前向地面距离。人形头相机是抬高+俯视的，看地面物体时**斜距 >> 地面水平距离**（近距离误差最大）。所以：
- 物体被估计在前方 ~0.7–1.3m（实际只有 0.5m）；
- `_approach_output` 朝这个偏远目标走，机器人走到真物体快到脚下时物体已出相机视野/深度更大，估计点始终在前方 → 永远差 ~0.5m 收不拢。
- 这也解释了 objects 探针里"远处检测误差只 0.18m 但近处够不到"——偏差随距离变化，近处最大。

→ 已回退到 baseline（height0.40/standoff0.42/reach0.7,0.6/det.distance gate，能蹲但手离物体 0.56m）。

### 下一步是设计岔路（待用户定）
- **A. 修感知地面投影**（最高杠杆）：用头相机俯仰角+高度把 depth 投影到地面，得到真实 ground 距离/世界坐标，导航才能真正贴近物体。需要相机外参（可从 env cfg 或探针量）。
- **B. 接受停在 ~0.5m**：把"到达"改成"接近停滞即蹲"，并把手臂拉到极限前伸下压 + 蹲到 0.35 + 躯干前倾，看 0.5m 外能否物理够到（probe 量）。
- **C. 提高接近速度下限**：走路策略在目标附近速度指令太小可能不迈步，给一个最小前进速度让它真正贴上去。
- **D. 重新考虑**地面拾取这条路是否够稳拿分。

## Task 9 Run 6（修感知地面投影 = 选项A）✅ 定位修好了

测得头相机外参（`scripts/probe_task_b_g1_camcalib.py`）：**俯仰 47.6° 朝下**、高度 1.37m（base 上方 0.56m）、前向偏移 ~0.03m；内参 fx=fy=733(=1.1453·W)、cx0=W/2、cy0=H/2、HFOV 47.2°/VFOV 36.3°。

把 `TaskBRgbdPerception._pixel_to_robot_xy`（错误地把斜距 depth 当前向地面距离、且忽略竖直像素）换成正确的**针孔反投影 + 相机俯仰旋转**（`_project_pixel_to_base`，用上 cx、cy、depth + 47.6° pitch + 相机高度/偏移），`Detection.distance` 改为真实地面距离 `hypot(forward,left)`。

- **detection-vs-truth 定位误差：0.18m → 0.03–0.05m（近处 4× 提升）**。投影修对了。
- 纯函数投影有本地单测 `HeadCameraGroundProjectionTest`（3 例，不需 torch）。

### 但暴露了一个结构性约束：头相机看不见近处物体
相机俯仰 47.6°、高 1.37m → 视线中心打在地面 ~1.25m 前，VFOV 36° → **只看得到前方约 0.7–2.5m 的地面带**。物体一旦近于 ~0.7m 就**掉出视野**。机器人恰好在最需要精确贴近时"瞎了"。

→ 这解释了 Run 6 squat 几何仍差（蹲点离真物体 1.48m）：机器人贴近到 ~0.7m 丢失目标，之后没有视觉伺服。

### 下一步（自然延伸，复用 v3 思路）
定位已准 → **记住物体的准确世界坐标，丢失视野后用里程计盲走最后 ~0.7m 贴上去，再按里程计地面距离触发蹲下**。v3 当时失败只因定位是错的；现在定位准了，盲走+里程计 gate 应该能让机器人真正站到物体上方再蹲。需要：odometry 地面距离 gate + 延长 approach 的目标记忆（盲走窗口）+ 收紧 standoff。
