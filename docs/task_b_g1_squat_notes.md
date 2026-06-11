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

## Task 9 Run 7-8（手部相机探查 = 选定方案前置）✅ 手眼可行

`scripts/probe_task_b_g1_handcam.py`：用真实 AlgSolution 驱动到蹲下，dump 手相机所见 + 相机相对手基座外参。

**传感器确认**：
- `ee_rgb/ee_depth`（左手）、`ee_dual_rgb/ee_dual_depth`（右手），各 **480×640 RGB-D**。
- 外参：相机在手基座**上方 ~0.19m**、前向 ~0.05m、绕 y 俯仰 ~50°（quat≈(0.906,0,0.423,0)）。

**关键结论：手相机蹲下时清楚看得到地面物体**
- 近场门控（colored ∩ depth∈[0.08,0.5]m）：**右手 251/271 帧（93%）有近物 blob**，左手 53/271。
- 保存的最近帧图像 `closest_{L,R}_rgb.png`：黄色 Domino 糖盒占据画面、清晰可辨，机器人手指在画面底部。相机略朝上前方看，物体出现在画面上/中部。
- 开环 sweep 下手最近到物体 ~0.35m（阈值 0.20m）—— 需要闭环伺服把这 0.15m 补上。

→ 手眼伺服可行。下一步设计：右手相机近场门控检测物体 blob（cx,cy,depth）→ 比例伺服右臂（肩/肘）让 blob 趋向"抓取点"（画面底部手指处）且 depth 减小，直到 hand_base_link 进 0.20m。需要标定关节→像素运动方向（或启发式 + 实测调）。

## Task 9 Run 9-10（右臂关节↔手相机标定）⚠️ 手眼伺服偏弱

`scripts/probe_task_b_g1_jointcal.py`（delta 0.2）：

Jacobian（每单位 action）：
- 肩pitch(22): dcx=-5.7, dcy=+3.3, **ddepth=-0.004**
- 肩roll(23): dcx=**+45.8**, dcy=-8.2, **ddepth=+0.003**
- 肩yaw(24): dcx=+37.1, dcy=+31.1, **ddepth=-0.008**
- 肘(25): 扰动即丢 blob

**两个硬发现：**
1. **手臂关节只"平移相机视角"(±几十像素),几乎不改变物体深度(±0.004–0.008m)**。即伺服臂只能把物体在画面里挪来挪去,无法把手真正"挪近"物体——要拉近 hand_base_link↔物体距离,得靠身体把物体送进臂工作空间,不是靠转臂。
2. **稳定深蹲+伸臂基线下,右手相机其实没对着物体**：保存的 `baseline_R_rgb.png` 显示画面大部分是空地/天空,两根黑手指在右下,真正的黄色物体又小又偏在左下;那个 depth=0.144m 的"近 blob"(cx525,cy412)是手指旁的零碎彩色像素,不是目标。之前"物体占满画面"的 closest 图是开环扫地里的瞬态,稳定姿态保持不住。

→ 之前 93% 近可见率是被零碎/边缘彩色像素夸大了。**手眼伺服对这个机器人/几何偏弱**:相机近场盲区 + 臂转不平移 + 蹲姿手不稳定对物。

## 修正后的建议（重要）
定位已修准(0.04m)后,**主杠杆其实是"用准确世界坐标把身体精确停到物体边",不是手眼伺服**。
- v3 里程计盲走当初失败只因定位错;**现在定位准了,里程计盲走最后 0.7m 把身体停到物体正前 ~0.2m,再开环蹲扫**,大概率比手眼伺服更直接有效。
- 手眼伺服可作为身体停准后的"最后几 cm 微调",但不能当主力(它改不了深度)。

建议下一步:**回到"准确定位 + 里程计盲走精确停位 + 调好的开环蹲扫"**,而不是继续投入手眼伺服。

## Task 9 Run 11（适配 mini 全身控制器 policy18.onnx）✅✅ 重大进展

`mini/` 是一个全身 loco-manip ONNX 控制器,接口比 OpenWBT 好太多:
- **num_actions=29(全身:腿+腰+臂)**,命令含:base 速度(3)、**base 高度(蹲)**、腰 rpy(3)、**左右手位姿(各7:xyz+四元数,base 系)**、waist_weight。
- **kp/kd 几乎与 ATEC 一致**(腿/腰完全相同;臂 40 vs ATEC 100/50/40,小差),**默认角与 ATEC 一致**。obs=115、his_len=5、action_scale=0.25、50Hz。
- ONNX: obs(115)+obs_history(5×115)+ik_input(15) → actions(29)+ik_output(17,上身IK)。
- 关节序:policy 用 isaaclab 交错序;mini 的 "mujoco" 序==ATEC dex1 前29序 → 用置换桥接。

适配实现:`scripts/mini_wbc_lib.py`(`MiniWBC`,纯 numpy+onnx)。从 ATEC proprio 建 obs/history/ik_input,action(29 policy序)→ 复刻 apply_action(含 ik_output 替换上身默认+0.4缩放)→ ATEC 33 动作。

**冒烟测试(`scripts/eval_miniwbc.py`,ATEC-TaskB-G1,30s,全程不摔 fell=False):**
- 站立 0-3s:base_z~0.80,tilt~0.07 稳。
- 行走 3-8s:前进 +1.67m(~0.33m/s),稳。
- 蹲+伸右手 12-30s:base_z 降到 **~0.48**,前倾 tilt~0.25,**不摔**。

视频:`logs/videos/miniwbc/run.mp4`(30s,跟随第三人称+头相机+右手相机)。机器人清晰可见,站→走→蹲+右臂下伸。

### 意义 & 下一步
这一个控制器就**替代了 走路+蹲下+扫地+手眼伺服 四件事**,且 kp/kd/默认角对齐 ATEC、能用手位姿命令做全身 IK 触达。
→ 下一步:把 MiniWBC 接进 AlgSolution。用已修准的感知(0.04m)拿到物体在 base 系坐标 → 命令 `right_hand_pose` 到物体处 + 降 base 高度 → WBC 全身触达 → hand_base_link 进 0.20m 得分。这才是真正能拿分的路径。

## 触碰策略优化 v2（targeted descending sweep）
之前 reach 是静态单点,易被定位误差(~0.1m 叠加)甩出 0.20m 球。改为:
- 蹲更低 `REACH_BASE_HEIGHT 0.45→0.38`(手更易够到地面 0.12m);
- 手 z 压到物体高度略下 `z = OBJECT_Z - 0.38 - 0.03 = -0.29`(world≈0.11);
- **近侧手在物体估计点周围做 Lissajous 小扫**(±0.13m,周期 50/33 步)覆盖一片 ~0.25m 区域,4.4s 窗口内反复扫过物体邻域;
- PostureGuard(tilt>0.35→recover)+ planner stand_up 作深蹲失稳的安全网。

## 感知确认 + 触碰逻辑 v3（按用户指定：识别→接近→蹭一步→蹲0.3→双手平扫）

### 感知确认（scripts/confirm_detection.py，原地转扫 500 步）
- ✅ **能识别垃圾**：标注图 `outputs/confirm_detection/detect_*.png` 显示绿框准确框住彩色垃圾。
- 局限:
  - **视距受限**:头相机俯视带只覆盖前方 ~0.7–2.5m,一次只看到附近几个;转一圈只检出 **3/18 distinct**(远处 >2.5m 看不到,需导航靠近)。
  - **定位误差(用里程计):median 0.24m / mean 0.54m / p90 1.27m**。感知本身准(真值位姿下 ~0.04m),误差主要来自 dead-reckoning **里程计漂移**(快速转向时最差)。双手平扫(~0.4m 覆盖)用来吸收这个误差。

### WBC 命令有效范围(来自 mini/command_gui.py,关键!)
base height **[0.3,0.9]**;hand x[-0.2,0.6];hand z **[-0.2,0.65]**;left y[-0.1,0.6];right y[-0.6,0.1]。
→ **base 0.3 + hand z -0.2 = 手到地面(world~0.12)**。base 0.38 时 hand z 需 -0.26(超范围)够不到地面——所以必须蹲到 0.3。之前 v2 的 z=-0.29 越界了。

### 触碰逻辑 v3
squat_sweep 阶段:先 `CREEP_STEPS=30` 步以 0.22m/s 朝物体蹭近(站立)→ 蹲到 `SQUAT_HEIGHT=0.3` → **双手在物体两侧 floor 高度(z=-0.2)做横向平扫**(各 ±0.14m,周期 40/27,加前向 dither),覆盖物体邻域。命令全部在 WBC 有效范围内。

## 行为流程 v4（旋转扫描搜索 + 提速）
- **搜索**:看不到物体就**原地旋转扫描**(yaw 1.3 rad/s);转一圈(185步)还没检测到就**前进 75 步换块地再转**(因头相机只看 0.7–2.5m,纯原地转会漏远处)。检测到→planner 转 approach。
- **提速**:approach 速度命令 ×1.9(钳到 vx 0.6 / vy 0.4 / wz 1.6),creep 0.22→0.35。
- 整圈循环:旋转找物体→接近→蹭近→蹲0.3→双手平扫→起身→(_search_step 复位)重新旋转找下一个,直到超时。

## 2026-06-11 Refactor validation (unified planner + blind-walk arrival)

**Verdict: BLOCKED — environment-level RTX renderer crash, no validation runs could execute.**

### What the harness scripts drive (Step 0 — fixed)
Both probe/eval scripts were importing the LEGACY pipeline. Repointed to the refactored `demo.solution.AlgSolution`:
- `scripts/probe_task_b_g1_squat.py:36` — was `from demo.solution_task_b_g1 import AlgSolution` → now `from demo.solution import AlgSolution`.
- `scripts/eval_task_b_g1_wbc.py:40` — was `from demo.solution_task_b_g1_wbc import AlgSolution` → now `from demo.solution import AlgSolution`. Also: the new unified `TaskBPlanner` exposes `scored` (set of objects that produced a score increase) instead of the legacy `touched_track_ids`; updated the eval's two reads to `getattr(planner, "scored", getattr(planner, "touched_track_ids", set()))` so it prints the right set and never crashes on either planner.
- Compatibility verified WITHOUT Isaac: `predicts(obs, score) -> {"action","giveup"}` signature, `obs["image"]` keys (`head_rgb`/`head_depth`/`ee_dual_rgb`) consumed by perception+video, and `planner.phase` all match. `python -m unittest discover -s tests/task_b` = **126 passed (29 torch/onnx skipped in bare shell)**. Both scripts `py_compile`-clean.

### Blocker: Isaac Sim crashes on every `--enable_cameras` launch (SIGSEGV in RTX renderer)
Every camera-enabled run dies ~0.3s into `AppLauncher(...)`, **before any solution code is imported or run** (py-spy main-thread top frame: `__enable_hydra_engine` → `createHydraEngine`; native fault in `librtx.scenedb.plugin.so` `carbOnPluginStartup`, signal 11 / exit 139). Isolated with a 12-line minimal repro containing none of this project's code:
- `AppLauncher(['--headless'])` → **starts OK** (exit 0, `APP_STARTED_OK`).  [log: `outputs_minimal_nocam.log`]
- `AppLauncher(['--headless','--enable_cameras'])` → **SIGSEGV in `librtx.scenedb.plugin.so`** (exit 139).  [log: `outputs_minimal_cam.log`]

The probe crashed identically twice in a row (`outputs_squat_probe_refactor.log`). The crash is the RTX ray-tracer failing to initialize, which `--enable_cameras` forces on. GPU itself is healthy (RTX 4090, driver 595.71.05, 35°C, idle, ~1.5G used by the desktop session only; no zombie Isaac procs). The Task B perception pipeline REQUIRES cameras, so probe (Step 1) and the 3×300s scored evals (Step 2) **could not be run** — squat-phase nearest-distance numbers, per-run scores, and fall checks are all unavailable.

Note: prior camera runs on **2026-06-10** worked (`outputs_squat_probe_v4.log` reached env-creation + DLSS rendering; `outputs_eval_smoke.log` printed a score), so this RTX init failure is a state change on the machine since yesterday, NOT caused by the refactor. The OV shader cache (`~/.cache/ov`, dated 2026-04-09) predates the working runs, so stale-cache corruption is not clearly the cause.

**Not self-fixed (out of scope + user wants to be consulted before debugging):** clearing the 3.4G `~/.cache/ov` cache, GPU driver/renderer changes, or display-session changes are environment-/system-level actions outside "empirical validation + small ratified tuning." Needs the user to recover the RTX renderer (e.g. confirm display session / try a fresh `~/.cache/ov`), after which the probe + evals can run unchanged.
