# ATEC 比赛 README

> 本文面向当前仓库 `(https://github.com/wang-sm520/ATEC.git)`，整理安装步骤，以及 Task A / B / D 当前方案的实现思路、关键文件和本地复现指令。
>
> 重要约定：`scripts/play_atec_task.py` 固定加载 `demo/solution.py`。因此同一时间只能有一个任务的 solution 处于 active 状态。切换 Task A/B/D 前，先确认并备份当前 `demo/solution.py`。

---

## 目录

- [1. 安装步骤](#1-安装步骤)
- [2. 通用运行与调试约定](#2-通用运行与调试约定)
- [3. Task A：G1 AMP 越野行走](#3-task-ag1-amp-越野行走)
- [4. Task B：G1 RGB-D 搜索 + 接触 / 推扫 baseline](#4-task-bg1-rgb-d-搜索--接触--推扫-baseline)
- [5. Task D：G1 推箱改造环境 baseline](#5-task-dg1-推箱改造环境-baseline)
- [6. 提交打包注意事项](#6-提交打包注意事项)

---

## 1. 安装步骤

### 1.1 硬件与系统要求

建议配置：

| 项目 | 建议 |
|---|---|
| GPU | NVIDIA RTX 系列，显存至少 8GB 可做本地冒烟，24GB+ 更适合训练 |
| 驱动 | 当前本机验证可用：`580.159.03`； 595版本，Isaac camera 曾出现兼容问题 |
| 系统 | Ubuntu 22.04 / 24.04 |
| 内存 | 32GB+ |
| 磁盘 | 50GB+，如果采集 RGB demo 或训练，需要更多空间 |

### 1.2 系统依赖

```bash
sudo apt update
sudo apt install -y git git-lfs build-essential cmake \
  libglu1-mesa libxrender1 libxi6 libxrandr2 libxcursor1 libxinerama1 \
  libgl1 libegl1 libvulkan1 vulkan-tools

git lfs install
```

确认 Vulkan 可用：

```bash
vulkaninfo | head
```

### 1.3 Python / IsaacLab 环境

本机已有 conda 环境：

```bash
conda activate atec
```

当前本地常用命令统一写成：

```bash
cd /ATEC
conda activate atec
```

如果在新机器重建环境，建议遵循 IsaacLab 官方安装流程，然后安装本仓库扩展：

```bash
# 1) 创建 conda 环境
conda create -n atec python=3.11 -y
conda activate atec
python -m pip install --upgrade pip

# 3) 安装 Isaac Sim / IsaacLab
# 推荐按 IsaacLab 官方文档安装‘https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/pip_installation.html’，isaacsim版本选择5.1,isaaclab版本选择2.3.2，并确保 AppLauncher 能启动。


# 4) 安装本仓库扩展
cd /ATEC/source/atec_rl_lab
python -m pip install -e .
```

常用额外依赖：

```bash
# Task B 的 MiniWBC ONNX 推理需要
python -m pip install onnxruntime

# Task E ACT 训练需要；A/B/D 不强制需要
python -m pip install diffusers tyro h5py tensorboard wandb imageio
```

### 1.4 资产与 LFS

如果是重新 clone 的仓库，确认 LFS 资产完整：

```bash
cd /home/amiao/wsm/ATEC
git lfs pull
```

### 1.5 环境自检

列出已注册环境：

```bash
cd /home/amiao/wsm/ATEC
python scripts/list_envs.py | grep ATEC
```

查看任务环境：

```bash
python scripts/view_task_a.py --enable_cameras
python scripts/view_task_b.py --enable_cameras
python scripts/view_task_d.py --enable_cameras
```

Task E / 带 camera 的环境必须加 `--enable_cameras`，否则会报：

```text
RuntimeError: A camera was spawned without the --enable_cameras flag.
```

---

## 2. 通用运行与调试约定

### 2.1 本地评测入口

本地端到端评测统一走：

```bash
PYTHONPATH=. python scripts/play_atec_task.py \
  --task <ATEC-TaskX-Robot> \
  --num_envs 1 \
  --enable_cameras \
  --debug
```

例如：

```bash
PYTHONPATH=. python scripts/play_atec_task.py \
  --task ATEC-TaskD-G1 \
  --num_envs 1 \
  --enable_cameras \
  --debug
```

### 2.2 `demo/solution.py` 是唯一 active 入口

`scripts/play_atec_task.py` 中固定：

```python
from demo.solution import AlgSolution
```

因此：

- Task A 复现时，需要让 `demo/solution.py` 是 Task A 版，同时demo下包含提交时所需文件；
- Task B 复现时，需要让 `demo/solution.py` 是 Task B 版，同时demo下包含提交时所需文件；
- Task D 当前主仓库 active 的 `demo/solution.py` 是 Task D 版，同时demo下包含提交时所需文件。

切换前建议手动改solution文件

---

## 3. Task A：G1 AMP 越野行走

### 3.1 当前实现思路

Task A 当前路线是 **G1 AMP locomotion**：

```text
motion data
  -> AMP-PPO 训练 G1 行走策略
  -> 导出 TorchScript policy.pt
  -> demo/solution_a.py 在评测时重建训练期 observation layout
  -> policy 输出 29 维 body action
  -> padding 到 G1 完整 action 维度
```

核心点：

1. **训练算法**：PPO + AMP（Adversarial Motion Priors）。AMP 用 motion data 约束步态风格，PPO 优化任务 reward。
2. **输入历史**：策略期望 960 维 term-major history：
   ```text
   ang_vel history + cmd history + gravity history + joint_pos history + joint_vel history + last_action history
   ```
3. **航向控制**：`demo/solution_a.py` 内部积分 yaw rate，用 P-controller 生成 `cmd[2]`，避免走着走着偏航。
4. **action scale 补偿**：训练时使用 per-joint action scale，ATEC 评测环境使用统一 `0.5` scale，因此推理时要乘上补偿系数。

关键文件：

| 路径 | 作用 |
|---|---|
| `docs/task_a_amp_training.md` | Task A AMP 训练说明：训练命令、AMP 参数、地形 curriculum、checkpoint / 导出说明 |
| `docs/task_a_motion_data_pipeline.md` | Task A motion 数据流水线：AMASS/GMR retarget、AMP JSON 转换、motion 数据维护 |
| `scripts/rsl_rl/train_amp.py` | AMP 训练入口 |
| `scripts/rsl_rl/play_amp.py` | 回放并导出 `policy.pt` / `policy.onnx` |
| `motion_data/g1_29dof/` | AMP motion 数据 |
| `demo/solution_a.py` | Task A 提交入口实现 |
| `demo/archieve/TASK_A/5.31--2.6/` | 历史 Task A 归档版本 |


### 全量训练

```bash
tmux new -s amp
conda activate atec
cd /home/amiao/wsm/ATEC

python scripts/rsl_rl/train_amp.py \
  --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 \
  --num_envs=4096 \
  --max_iterations=100000 \
  --headless \
  --amp_motion_files motion_data/g1_29dof/*.json
```

监控：

```bash
tensorboard --logdir logs/rsl_rl_amp/g1_amp_rough --port 6006
```

### 3.5 回放与导出

```bash
python scripts/rsl_rl/play_amp.py \
  --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 \
  --num_envs=16 \
  --real-time
```

`play_amp.py` 会导出：

```text
<run_dir>/exported/policy.pt
<run_dir>/exported/policy.onnx
```

### 3.6 Task A 本地复现

> 注意：这会把 active solution 切成 Task A。不要在需要保留 Task D active 状态时直接覆盖。

```bash
cd /home/amiao/wsm/ATEC

# 使用训练导出的 policy
cp logs/rsl_rl_amp/g1_amp_rough/<run>/exported/policy.pt demo/policy.pt

# 切换 active solution
cp demo/solution_a.py demo/solution.py

# 本地评测
PYTHONPATH=. python scripts/play_atec_task.py \
  --task ATEC-TaskA-G1 \
  --num_envs 1 \
  --enable_cameras \
  --real-time \
  --debug
```

---

## 4. Task B：G1 RGB-D 搜索 + 接触 / 推扫 baseline

### 4.1 当前实现思路

Task B 当前路线是 **分层 G1 loco-manip baseline**，目前找了一个WBC的policy，可以控制双手运动，各方向运动，包括z方向蹲下，目标是先稳定拿接触分，再尝试少量放置分：

```text
obs["proprio"]
  -> DeadReckoningOdometry 估计机器人 pose

obs["image"]["head_rgb/head_depth"]
  -> TaskBRgbdPerception 检测彩色物体并投影到地面坐标

pose + detections + current_score
  -> TaskBPlanner 状态机
  -> 输出 WBCCommand：底盘速度 / 身高 / 腰部 / 双手目标

WBCCommand + proprio
  -> MiniWBC(policy18.onnx)
  -> 33 维 G1 action
```

关键策略：

1. **不依赖固定物体坐标**：通过 RGB-D 检测物体。
2. **里程计粗定位**：从 proprio 的 base velocity 和 projected gravity 积分估计 `(x, y, yaw)`。
3. **搜索状态机**：旋转扫描 + 前进换点，覆盖物体可能出现区域。
4. **接触得分优先**：Task B 中手部靠近物体可触发拾取分，因此第一阶段用低位双手接触 / 推扫。
5. **MiniWBC 控制**：使用 `policy18.onnx` 的全身控制器，同时控制底盘速度、身体高度和双手位姿。

当前归档文件集（同时也是需要提交的文件）：

```text
demo/archieve/TASK_B/6.13_7/
  solution.py
  mini_wbc.py
  task_b_nav.py
  task_b_perception.py
  task_b_planner.py
  policy18.onnx
  requirements.txt
```

关键文件说明：

| 文件 | 作用 |
|---|---|
| `solution.py` | `AlgSolution` 入口，只做模块 wiring |
| `task_b_nav.py` | dead-reckoning odometry + posture guard |
| `task_b_perception.py` | RGB-D 彩色物体检测、深度反投影、track 分配 |
| `task_b_planner.py` | 搜索 / approach / creep / squat / sweep 状态机 |
| `mini_wbc.py` | ONNX MiniWBC adapter，输出 G1 33 维 action |
| `policy18.onnx` | MiniWBC 权重 |

### 4.2 Task B 本地复现

```bash
cd /home/amiao/wsm/ATEC
conda activate atec

# 确保依赖
python -m pip install onnxruntime

# 本地评测
PYTHONPATH=. python scripts/play_atec_task.py \
  --task ATEC-TaskB-G1 \
  --num_envs 1 \
  --enable_cameras \
  --debug
```

### 4.3 Task B 提交文件集

平台上传时使用 flat 文件集：

```text
solution.py
mini_wbc.py
task_b_nav.py
task_b_perception.py
task_b_planner.py
policy18.onnx
requirements.txt
```

不要上传：

```text
run.sh
server.py
```

平台会自动注入服务入口。

### 4.4 已知限制与后续方向

当前 Task B 是 baseline，不是完整抓取搬运方案：

- RGB-D 传统检测对 IsaacLab / camera calibration 版本敏感；升级环境后需要重新校准。
- 主要目标是接触得分，放置分依赖推扫运气和物体位置。
- 下一阶段可以加入真实 grasp / carry、轻量 detector、目标排序和局部 interaction policy。

---

## 5. Task D：G1 推箱改造环境 baseline

### 5.1 当前实现思路

当前主仓库 active 的 `demo/solution.py` 是 Task D 方案。

路线：冻结 Task A G1 locomotion policy，把它当作速度跟踪器；上层用纯 Python 状态机规划推箱：

```text
obs["proprio"]
  -> _Odometry 里程计估计机器人位姿
  -> _WallPushController 推箱状态机生成速度命令 (vx, vy, wz)
  -> _G1VelocityPolicyBridge 转成 G1 关节 action
  -> env.step(action)
```

当前状态机核心阶段：

```text
warmup
  -> approach_back
  -> push_x
  -> around_top
  -> push_y
  -> around_back2
  -> push_x_pit
  -> forward
```

得分思路：

1. 利用箱子起始位置确定、场景几何确定的特点；
2. 先把箱子推到平台边 / 坑附近，拿环境改造相关分；
3. 机器人自己越过 `x=-1.4` 线，拿基础前进分；
4. 进入 `forward` 后尝试切换到 `policy_climb.pt`，继续向终点方向走。
5. 目前已经可以把箱子推进沟壑，还需要一个可以爬楼梯的策略
关键文件：

| 路径 | 作用 |
|---|---|
| `demo/solution.py` | 当前 active Task D solution，包含里程计、policy bridge、推箱状态机 |
| `demo/policy_a.pt` | G1 行走策略，960 维输入、29 维 action |
| `demo/policy_climb.pt` | 爬楼 / 爬越策略，进入 `forward` 阶段后尝试切换 |
| `docs/task_d_submission.md` | Task D 提交 / 复刻说明 |
| `docs/superpowers/specs/2026-06-07-task-d-dual-policy-handoff-design.md` | 双策略接力设计说明 |
| `scripts/probe_task_d_groundtruth.py` | 本地 probe，查看真值与分数 |
| `scripts/probe_task_d_wallpush.py` | 推箱方案调试脚本 |

### 5.2 Task D 本地复现

当前 active solution 如果没有被覆盖，可以直接跑：

```bash
cd /home/amiao/wsm/ATEC
conda activate atec

PYTHONPATH=. python scripts/play_atec_task.py \
  --task ATEC-TaskD-G1 \
  --num_envs 1 \
  --enable_cameras \
  --debug
```

### 5.3 Task D 提交打包

本地 `demo/solution.py` 优先加载：

```text
policy_a.pt
policy.pt
```

平台提交时如果只传 `policy.pt`，需要先把正确的 Task A 行走权重复制成 `policy.pt`：

推荐提交文件：

```text
solution.py
policy.pt
policy_climb.pt
requirements.txt
```

如果不提交 `policy_climb.pt`，当前代码会优雅降级为只用 walk policy，但过坑 / 继续前进能力会受影响。

