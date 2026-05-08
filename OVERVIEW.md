# ATEC2026 仿真挑战赛 · 速查手册

> 基于官方《选手指南 v1.4.1》+ 本仓库 `readme.md` / `example.md` / `demo/` 整理。所有任务必须由机器人完全自主完成。

---

## 1. 赛道分类

| 赛道 | 侧重 | L0 入门任务 | L1 高级任务（共享） |
|---|---|---|---|
| **赛道 1** Track 1 | 运动 Locomotion | **任务 A** 机器人徒步 | 任务 B + 任务 D |
| **赛道 2** Track 2 | 操作 Manipulation | **任务 E** 桌面整理 | 任务 B + 任务 D |

- **L0**：单点能力评测，每任务独立排行榜。任一 L0 任务得分 **≥ 0.01** 即解锁 L1，**L0 不计入最终排名**。
- **L1**：系统级能力评测，**最终排名以 L1 综合成绩为准**。
- 总成绩：`Final Score = Score(Task B) + Score(Task D)`
- 同分排序：`B 分 → D 分 → 完成时间（短者优先）→ 提交时间戳`

---

## 2. 任务要求

### 2.1 机器人平台（5 款）

| 平台 | 简称 | 自由度 | 适用任务 |
|---|---|---|---|
| 固定基机械臂（AgileX Piper） | `Piper` | 8 DoF | **E** |
| 四足 + 机械臂（Unitree B2 + Piper） | `B2Piper` | 20 DoF | A / B / D |
| 四轮足 + 机械臂（B2W + Piper） | `B2wPiper` | 24 DoF | A / B / D |
| 二轮足 + 机械臂（Tron1 + Piper） | `Tron1Piper` | 16 DoF | A / B / D |
| 人形机器人（Unitree G1，含双指夹爪） | `G1` | 33 DoF | A / B / D |

**统一传感器**：1× LiDAR + 1× 头部 RGB-D（eye-to-hand）+ 1× 末端 RGB-D（eye-in-hand，G1 为双目）+ 接触传感器。

### 2.2 Gym 环境矩阵

| 任务 \ 机器人 | G1 | Tron1Piper | B2Piper | B2wPiper | Piper |
|---|---|---|---|---|---|
| 任务 A | `ATEC-TaskA-G1` | `ATEC-TaskA-Tron1Piper` | `ATEC-TaskA-B2Piper` | `ATEC-TaskA-B2wPiper` | — |
| 任务 B | `ATEC-TaskB-G1` | `ATEC-TaskB-Tron1Piper` | `ATEC-TaskB-B2Piper` | `ATEC-TaskB-B2wPiper` | — |
| 任务 D | `ATEC-TaskD-G1` | `ATEC-TaskD-Tron1Piper` | `ATEC-TaskD-B2Piper` | `ATEC-TaskD-B2wPiper` | — |
| 任务 E | — | — | — | — | `ATEC-TaskE-Piper` |

> 提供的环境**仅用于评估和提交**，不支持并行训练。训练需要自己写 wrapper 或用外部框架。

### 2.3 任务定义与评分

#### 任务 A · 机器人徒步（Off-road Navigation） — L0
- **场景**：~400 m 赛道，含平地 / 崎岖 / 上下坡 / 多阶楼梯
- **评分**（满分 **26**）：平地 +2 · 崎岖 +4 · 斜坡 +8 · 楼梯 +8 · 到达终点 +4

#### 任务 E · 桌面整理（Tabletop Manipulation） — L0
- **场景**：3 类物体，需放入指定粉色容器
- **评分**（满分 **18**）：每次成功抓取 +3 · 正确放置 +3

#### 任务 B · 垃圾拾取与投放（Garbage Collection） — L1
- **场景**：20×20 m 工作区，**18 个目标物体**
- **评分**（满分 **36**）：每件物体 拾取 +1 · 放置正确 +1

#### 任务 D · 越障（Obstacle Traversal） — L1
- **场景**：高架平台 / 沟壑，可使用箱子改造环境
- **评分**（满分 **36**）：到达起点 +2 · 成功穿越 +20 · 利用箱子改造环境 +14

> 各任务分数不可直接比较；只有 L1 的 B+D 决定排名。

### 2.4 观测与动作接口

```python
obs["proprio"]  # base_lin_vel, base_ang_vel, velocity_commands,
                # projected_gravity, joint_pos, joint_vel, actions
obs["image"]    # head_rgb, head_depth, ee_rgb, ee_depth,
                # ee_dual_rgb / ee_dual_depth (G1 only),
                # video_rgb / video_depth (Task E only)
obs["extero"]   # lidar_scan
```

**动作空间**（统一）：

```python
mdp.JointPositionActionCfg(
    asset_name="robot", joint_names=[".*"],
    scale=0.5, use_default_offset=True, preserve_order=True,
)
```

- 控制量按 **0.5** 缩放后下发
- **关节顺序固定**，每款机器人特定排列（见 PDF 第 13 页或仓库 `tasks/` 下定义）
- `predicts()` 返回 `{"action": List[float], "giveup": bool}`

---

## 3. 提交与评审

### 3.1 选手必须交付

| 文件 | 必需 | 说明 |
|---|---|---|
| `demo/solution.py` | ✅ | 类 `AlgSolution`，方法 `predicts(obs, current_score) → {"action": List[float], "giveup": bool}` |
| `policy.pt` 等权重 | 视方案而定 | 模型文件 |
| `requirements.txt` | 视方案而定 | 追加依赖（不要覆盖基础镜像核心包） |

**禁止**上传 `run.sh` / `server.py`（系统自动注入）。

### 3.2 提交方式（二选一）

| 方式 | 流程 | 限制 |
|---|---|---|
| **源码上传** | 选机器人构型 → 上传文件 → 系统自动构建镜像并打分 | 文件数 ≤ 300 |
| **推送镜像** | 本地 `docker build` → 用平台临时凭证 `docker login/push` → 系统自动打分 | 镜像 ≤ 30 GB |

### 3.3 评测流程

`/health` 探活（300 s 内 ready）→ `env.reset()` → 循环 `/step`（obs → action）→ 30 min 墙钟终止 → `/quit`。

### 3.4 排行榜与配额

- **排名规则**：L1 排行榜 = `Score(B) + Score(D)`，仅保留历史最高分；L0 仅作晋级资格
- **同分排序**：B 分 → D 分 → 完成时间 → 提交时间戳
- **每日额度**（UTC+8 10:00 重置）：

| 指标 | 上限 | 何时扣减 |
|---|---|---|
| 提交次数 | 10 | 构建/推送成功即扣 |
| 出分次数 | 3 | 仅成功评测产出分数才扣 |

> 构建失败 / 服务启动失败 → 不扣出分次数。

---

## 4. 运行 / 资源限制

| 项目 | 限制 |
|---|---|
| GPU | RTX 5880 单卡（总显存 48 GB） |
| 可用显存 | **~34 GB**（仿真占 ~14 GB） |
| CPU / 内存 / 磁盘 | 16 核 / 24 GB / 500 GB |
| 网络 | 选手容器**无外网**，依赖必须打包进镜像 |
| 启动时间 | `/health` 必须在 **300 秒**内响应 |
| 单次任务 | 物理墙钟 **30 分钟** |
| 同时运行 | 每队每榜 **1** 个任务 |
| 基础镜像 | Python 3.12 + PyTorch 2.7.1 + CUDA 12.8.1 + Alinux 3 |

> 完整依赖清单见 PDF 附录 / 仓库 `demo/requirements.txt`。

---

## 5. Demo 现状 — 已有什么 / 还要做什么

### 5.1 仓库已提供

`demo/` 目录：

| 文件 | 角色 |
|---|---|
| `run.sh` | 平台打分启动脚本，**不要改、不要传** |
| `server.py` | HTTP 服务（/health, /step, /quit），**不要改、不要传** |
| `solution.py` | 入口（默认是 zero baseline），**必须实现** |
| `solution_zero.py` | 全 0 输出参考 |
| `solution_rl.py` + `policy.pt` | RL baseline（B2 平地行走，PPO） |
| `solution_act.py` + `act/` + `policy_act.pt` | ACT 模仿学习 baseline（Task E） |
| `Dockerfile` | 镜像构建参考 |
| `requirements.txt` | 依赖参考（可追加） |
| `__init__.py` | 本地 python package 调试用 |

`scripts/`：

- `list_envs.py` — 检查 ATEC-* 环境注册
- `view_robots.py` / `view_task_{a,b,d,e}.py` — 可视化机器人与任务
- `play_atec_task.py` — **本地调试**评测入口（加载 `demo/solution.py`）
- `rsl_rl/train.py` / `rsl_rl/play.py` — PPO 训练 / 回放
- `act/collect_demos_task_e.py` / `filter_demos.py` / `baseline.sh` — ACT 流水线

`source/atec_rl_lab/`：

- `tasks/task_{a,b,d,e}/` — 4 个任务的环境定义
- `assets/robots/` `assets/objects/` — USD 资产
- `train/locomotion/` `train/act/` — 训练配置

`atec_robot_model/baseline/`：RL（`unitree_b2_flat/policy.pt`）和 ACT（`act/policy.pt`）的预训练权重。

### 5.2 选手需要自己做

| # | 待办 | 备注 |
|---|---|---|
| 1 | 选 baseline → 改名为 `solution.py` | `cp demo/solution_rl.py demo/solution.py` 或 `solution_act.py` |
| 2 | 在 `AlgSolution.predicts(obs, current_score)` 里写策略 | 返回 `{"action": List[float], "giveup": False}` |
| 3 | 训练自己的策略 | 多样地形 / 域随机化（A）；多物体泛化（E） |
| 4 | **任务 B 自研**（仓库无 baseline） | 推荐：端到端 RL（whole body control）/ RL + IK/MPC 混合 / 全身 MPC |
| 5 | **任务 D 自研**（仓库无 baseline） | 推荐：图像+深度的端到端控制（参考 extreme parkour） |
| 6 | 准备 `requirements.txt` 与权重文件（如 `policy.pt`） | 注意基础镜像已有依赖，不要覆盖核心包 |
| 7 | 用 `play_atec_task.py` 跑通本地完整流程 | `--debug` 看分数；至少跑一次再提交 |
| 8 | 控制初始化 < 300 s、单步推理速度、显存 | 否则触发 /health 超时、step 超时、OOM |

> **进阶方向**（PDF 提示）：VLA（视觉-语言-动作）、VLM（视觉-语言模型）。

---

## 6. 本地验证 & 训练命令

### 6.1 安装

```bash
# Isaac Lab v2.3.2 必须先装好
conda activate isaaclab

cd ATEC2026_Simulation_Challenge/source/atec_rl_lab
pip install -e .

# 机器人模型（含 baseline 权重）
git clone https://github.com/skywoodsz/atec_robot_model.git
cd atec_robot_model && git lfs pull
```

### 6.2 检查与可视化

```bash
python scripts/list_envs.py                      # 列出所有 ATEC-* 环境
python scripts/view_robots.py    --enable_cameras
python scripts/view_task_a.py    --enable_cameras
python scripts/view_task_b.py    --enable_cameras
python scripts/view_task_d.py    --enable_cameras
python scripts/view_task_e.py    --enable_cameras
```

### 6.3 本地评测（加载 `demo/solution.py`）

```bash
python scripts/play_atec_task.py \
    --task ATEC-TaskA-B2Piper \
    --enable_cameras \
    --debug
```

- `--task`：从环境矩阵选一个
- `--debug`：打印运行时和得分

### 6.4 RL 训练（任务 A baseline）

```bash
# 训练（RTX 5090 约 90 分钟）
python scripts/rsl_rl/train.py \
    --task ATEC-Isaac-Velocity-Flat-Unitree-B2-v0 \
    --headless --video

# 回放
python scripts/rsl_rl/play.py \
    --task ATEC-Isaac-Velocity-Flat-Unitree-B2-v0
```

提交前：把训练得到的 `policy.pt` 放到 `demo/`，并把 `demo/solution_rl.py` 改名为 `demo/solution.py`。

### 6.5 ACT 训练（任务 E baseline）

```bash
# 1) 采集专家轨迹
python scripts/act/collect_demos_task_e.py \
    --pick_objects 3 --num_demos 100 \
    --headless --enable_cameras --save_images

# 2) 过滤近 0 动作
python scripts/act/filter_demos.py \
    --input  datasets/atec_task_e/trajectory.hdf5 \
    --output datasets/atec_task_e/trajectory_filtered.hdf5 \
    --threshold 0.001

# 3) 训练
cd scripts/act && bash baseline.sh

# 4) 本地验证（先把 solution_act.py 改名为 solution.py）
python scripts/play_atec_task.py --task ATEC-TaskE-Piper --enable_cameras --debug
```

权重路径：`atec_robot_model/baseline/act/policy.pt` → 拷贝到 `demo/policy_act.pt`。

---

## 7. Locomotion 赛道解题思路（赛道 1）

> 赛道 1 = L0 任务 A（资格门票） + L1 任务 B & D（决定排名）。

### 7.1 任务 A · 机器人徒步（L0）

- **方法**：PPO + 地形 curriculum，沿用 `solution_rl.py` 路线
- **机器人选型**：
  - **B2Piper / B2wPiper** ← 推荐（生态成熟、稳定）
  - Tron1Piper：二轮足平衡难度高
  - G1：人形在楼梯/斜坡上稳定性差，不建议作 L0 用车
- **关键技巧**：
  - 地形 curriculum：平地 → 崎岖 → 斜坡 → 楼梯，渐进解锁
  - Domain randomization：摩擦、推力扰动、传感器噪声、初始姿态
  - 必须用 `lidar_scan` / height scan，纯 proprio 上不去楼梯
  - Reward：tracking_lin_vel + base_height + 防滑 + 越障 bonus
- **难点**：仓库 baseline 只在平地训练，**楼梯/斜坡需要扩展任务定义**（参考 `robot_lab` 项目的 rough terrain config）

### 7.2 任务 B · 垃圾拾取与投放（L1）

- **本质**：长时序 loco-manipulation，30 min 内完成 18 个物体的 navigate→grasp→carry→place
- **三种方案**：

  | 方案 | 描述 | 优点 | 缺点 |
  |---|---|---|---|
  | **A. 分层** | 感知（YOLO/Grounding-DINO）+ FSM 规划 + nav policy + manip policy | 模块独立可验证 | 误差累积 |
  | **B. 端到端 WBC** | 单 policy 同时输出腿+臂（参考 deep WBC / HumanPlus） | 协调性最好 | reward 工程极难 |
  | **C. RL+IK 混合** ⭐ | RL 训腿（复用 A 的 policy）+ 机械臂用 IK/MPC + 视觉做物体定位 | 开发快、易 debug | 上限低于 B |

- **建议路线**：先用 **C** 拿基线分（哪怕只完成 5-10 个物体也是有效得分），行有余力再上 B
- **注意**：B2Piper = 12 腿关节 + 8 臂关节（含夹爪），action 拼接顺序不能错；时间预算平均每个物体 ~100 秒

### 7.3 任务 D · 越障（L1）

- **本质**：视觉 parkour + **环境改造**（用箱子填沟/垫高，占 14/36 分）
- **分阶段拿分**：
  1. **基础穿越（+22 分）**：到达起点(+2) + 视觉 parkour 穿越(+20)。参考 *Extreme Parkour with Legged Robots*（Cheng et al. 2023），输入 `head_depth + proprio + lidar`，端到端输出关节动作
  2. **箱子改造（+14 分）**：本质是 loco-manipulation 子集，可复用 B 任务的 push/manipulate 能力。空间推理可先用启发式（沟宽阈值），进阶用 VLM
- **机器人选型**：B2wPiper 跨沟更灵活；G1 上限最高但训练代价大

### 7.4 训练显卡需求评估

| 任务 | 最低 | 推荐 | 训练时长 | 备注 |
|---|---|---|---|---|
| **A** flat | 1× RTX 4090/5090（24 GB） | 1× A100 40 GB | 1-15 h | 单卡足够 |
| **A** rough+stairs | 1× A100 40 GB | 1× A100 80 GB | 6-15 h | 显存富裕可拉到 8192 envs |
| **B** | 1× A100 80 GB | 1× H100 80 GB 或 2× A100 80 GB | 1-3 天 | 视觉显存吃紧，**80 GB 起步** |
| **D** | 1× A100 80 GB | 1× H100 80 GB | 2-5 天 | 深度图 encoder 占显存 |

**总体建议**：

| 配置 | 总训练耗时（A+B+D） | 适合 |
|---|---|---|
| 1× A100/H100 80 GB | 5-10 天 | 个人/小队最低门槛 |
| **2× H100 80 GB** ⭐ | 3-5 天 | 推荐：一卡训 RL，一卡训视觉/做实验 |
| 4× H100 80 GB | 收益递减 | 适合多实验并行而非单实验加速 |

**不推荐**：
- RTX 3090/4090（24 GB）：B/D 加视觉后会 OOM，envs 被迫降到 < 1k
- 多张 V100（32 GB）：架构旧、FP16 性能差

**关键提醒**：
- 评测用 **RTX 5880（48 GB）**，仿真占 14 GB，**部署模型显存预算 ~30 GB**。训练时显存可大，但部署要瘦身
- 训练环境最好对齐 PyTorch 2.7.1 + CUDA 12.8，避免兼容问题
- Isaac Lab 单 GPU 内已开 thousands of envs，多卡更多用于「实验并行」而非「单实验加速」

---

## 8. 提交前自检清单

- [ ] `demo/solution.py` 存在且类名 `AlgSolution`、方法 `predicts(obs, current_score)`
- [ ] action 维度与所选机器人关节数一致，控制值已裁剪
- [ ] 初始化（模型加载等）能在 300 s 内完成
- [ ] 单步推理稳定，避免 /step 超时
- [ ] 显存峰值 < 34 GB（OOM 直接零分）
- [ ] 无外网依赖，所有模型/资源已打包进镜像
- [ ] 没有上传 `run.sh` / `server.py`
- [ ] 至少完整跑过一次 `play_atec_task.py --debug`
