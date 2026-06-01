# 部署到新机器 · 训练上手手册

把整个 `ATEC2026_Simulation_Challenge/` 仓库拷贝到一台新机器后,按本文从硬件检查 → 系统依赖 → Python 环境 → 仓库自检 → 训练 → 提交,一路走通。

---

## 一、硬件 / 系统前置

| 项 | 要求 |
|---|---|
| GPU | NVIDIA RTX 系列,显存 ≥ 24 GB(RTX 5090 / 5880 / 4090 / A6000 都验证过) |
| 驱动 | NVIDIA Driver ≥ 535,CUDA Runtime ≥ 12.4(本机用的是 12.8) |
| 系统 | Ubuntu 22.04 / 24.04 / Alinux 3(其他 Linux 同理,Windows/WSL 未验证) |
| 磁盘 | ≥ 50 GB(IsaacSim USD 缓存 + logs;不含训练 checkpoint) |
| 内存 | ≥ 32 GB |

验证 GPU 可用:

```bash
nvidia-smi
```

---

## 二、系统层依赖

```bash
sudo apt update
sudo apt install -y git git-lfs build-essential cmake \
    libglu1-mesa libxrender1 libxi6 libxrandr2 libxcursor1 libxinerama1 \
    libgl1 libegl1 libvulkan1 vulkan-tools
git lfs install
```

`vulkan-tools` 装完跑一下 `vulkaninfo | head` 应该有输出,IsaacSim 没 Vulkan 起不来。

---

## 三、Python 环境(Conda)

仓库里的训练栈 = **IsaacLab 2.3.2 + Isaac Sim 4.5 + rsl-rl-lib + PyTorch 2.7 / CUDA 12.8**,本机的环境名叫 `atec`,新机器照着重建即可。

### 3.1 装 Miniconda(已装可跳)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
conda init zsh    # 或 bash
```

### 3.2 创建 atec 环境

```bash
conda create -n atec python=3.10 -y
conda activate atec
pip install --upgrade pip
```

### 3.3 装 PyTorch(CUDA 12.8 wheel)

```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

验证:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3.4 装 Isaac Sim + Isaac Lab

**严格按官方 pip 安装流程**: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

简版命令(2026/05 时点可用):

```bash
# Isaac Sim 4.5 (pip 版)
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# Isaac Lab 2.3.2(克隆到任意路径,这里放 ~/IsaacLab)
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2
./isaaclab.sh --install
```

第一次 `isaaclab.sh` 会编译 shader / 下 USD,需要 5–15 分钟。

### 3.5 装 rsl-rl-lib

```bash
pip install "rsl-rl-lib>=5.0.0"
```

> 仓库 README 里写的 "3.0.1" 是旧版,实际本机跑的是 **5.0.1**,新 API 已经在 `scripts/rsl_rl/train_amp.py` 里适配过。

### 3.6 装本仓库扩展(`atec_rl_lab`)

```bash
cd /path/to/ATEC2026_Simulation_Challenge/source/atec_rl_lab
pip install -e .
```

这一步会注册所有 `ATEC-*` 和 `ATEC-Isaac-AMP-Unitree-G1-TaskA-*` Gym 环境,并自动装 `tensorboard / wandb / h5py / diffusers / tyro / torchvision / psutil`。

---

## 四、仓库内容自检

仓库已经自带:

| 路径 | 内容 | 大小 |
|---|---|---|
| `atec_robot_model/` | G1 / B2 / Tron1 / Piper USD 资产(git LFS,已 pull) | ≈ 480 MB |
| `motion_data/g1_29dof/` | 13 段 73-dim AMP JSON(LAFAN1 + ACCAD) | ≈ 49 MB |
| `source/atec_rl_lab/` | 训练扩展(env / mdp / amp 算法) | ≈ 1.6 MB |
| `scripts/` | 训练 / 播放 / 调试入口 | — |
| `submission/` | 提交模板 `solution.py` + `policy.pt` | ≈ 2.6 MB |

如果是用 `git clone`(而不是 scp/rsync 整个目录)而且 `atec_robot_model/` 是空的,跑:

```bash
cd /path/to/ATEC2026_Simulation_Challenge
git lfs pull
```

环境注册自检:

```bash
conda activate atec
cd /path/to/ATEC2026_Simulation_Challenge
python scripts/list_envs.py | grep -E "TaskA|AMP"
```

应看到 `ATEC-Isaac-AMP-Unitree-G1-TaskA-v0` / `Flat-v0` / `Rough-v0` 与 `ATEC-TaskA-G1` 等。

---

## 五、冒烟测试(2 分钟)

确认 GPU / IsaacSim / 扩展 / rsl-rl 全链路通:

```bash
conda activate atec
cd /path/to/ATEC2026_Simulation_Challenge

python scripts/rsl_rl/train_amp.py \
    --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 \
    --num_envs=16 --max_iterations=2 --headless
```

观察点:
- 打印 `[train_amp] amp_motion_files is empty -> running in degenerate PPO mode` → 算法路径 OK
- 跑完 2 个 iter 不报错退出 → IsaacSim + env + 网络 OK
- `logs/rsl_rl_amp/g1_amp_rough/<时间戳>/` 有新目录 → 日志路径 OK

---

## 六、正式训练

### 6.1 中规模联调(512 envs / 100 iter,5–10 min)

```bash
python scripts/rsl_rl/train_amp.py \
    --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 \
    --num_envs=512 --max_iterations=100 --headless \
    --amp_motion_files motion_data/g1_29dof/*.json
```

预期 `Train/episode_length` 由 ~12 升至 800+,`Train/fall_rate` 由 100% 降到 ~30%。

### 6.2 全量训练(4096 envs / 30000 iter,RTX 5090 ≈ 12–20 h)

放 tmux 长跑:

```bash
tmux new -s g1amp
conda activate atec
cd /path/to/ATEC2026_Simulation_Challenge

python scripts/rsl_rl/train_amp.py \
    --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 \
    --num_envs=4096 --max_iterations=30000 --headless \
    --amp_motion_files motion_data/g1_29dof/*.json
```

监控(另起一个 shell):

```bash
tensorboard --logdir logs/rsl_rl_amp/g1_amp_rough --port 6006
```

### 6.3 续训

```bash
python scripts/rsl_rl/train_amp.py \
    --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 \
    --num_envs=4096 --max_iterations=30000 --headless \
    --resume True --load_run -1 --checkpoint -1 \
    --amp_motion_files motion_data/g1_29dof/*.json
```

> 关键 CLI 参数(`--amp_reward_coef`, `--amp_task_reward_lerp`, ...)的含义见 [`docs/task_a_amp_training.md`](task_a_amp_training.md) §四。

---

## 七、回放 & 导出 policy

```bash
# 训练阵地形 + GUI 实时查看
python scripts/rsl_rl/play_amp.py \
    --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 \
    --num_envs=16 --real-time

# 无 GUI + 录视频
python scripts/rsl_rl/play_amp.py \
    --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 \
    --num_envs=16 --video --video_length=600 --headless
```

`play_amp.py` 会把 actor 导出成 `<run_dir>/exported/policy.pt`(TorchScript)和 `policy.onnx`,可直接用于提交。

ATEC 评测 env 端到端验证(过 `demo/solution.py` 入口):

```bash
PYTHONPATH=. python scripts/play_atec_task.py \
    --task=ATEC-TaskA-G1 --num_envs=1 \
    --enable_cameras --real-time --debug
```

---

## 八、提交准备

1. 训练完用 `play_amp.py` 导出最新 `<run_dir>/exported/policy.pt`
2. 拷到 demo:`cp logs/rsl_rl_amp/g1_amp_rough/<run>/exported/policy.pt demo/policy.pt`
3. `demo/solution.py` 已写好 term-major 960-dim history,通常无需改
4. 本地终检:`PYTHONPATH=. python scripts/play_atec_task.py --task=ATEC-TaskA-G1 --debug`
5. 按 ATEC 平台规则上传 `demo/` 目录或 docker 镜像

---

## 九、常见坑

| 症状 | 排查 |
|---|---|
| `import isaacsim` 报 `libcudart` 找不到 | 重装 NVIDIA driver + `pip install isaacsim[all]` 用 `--extra-index-url https://pypi.nvidia.com` |
| 首次 IsaacSim 启动卡 5+ 分钟 | 正常,在缓存 shader,后续启动 < 30s |
| `vulkaninfo` 报 GPU 找不到 | 确认非 headless 服务器装了 `nvidia-utils-XXX`,headless 跑用 `--headless` |
| `KeyError: 'class_name'` | 升级到当前 `train_amp.py`(已 `handle_deprecated_rsl_rl_cfg`) |
| `fall_rate` 100% 一直降不下来 | 确认 `--amp_motion_files` 真的传进去了;先在 `Flat-v0` 预训练再切 `TaskA-v0` |
| `AMP/discr_pred_real` 一直 +1 / −1 | 调小 `--amp_loss_coef` 或加大 `--amp_grad_pen_lambda` |
| 显存 OOM | 把 `--num_envs` 由 4096 砍到 2048 / 1024,再不行降到 512 |
| `atec_robot_model/robot/g1/*.usd` 缺失 | `cd atec_robot_model && git lfs pull` |

---

## 十、目录速查

```
ATEC2026_Simulation_Challenge/
├── atec_robot_model/        # USD 资产(G1/B2/Tron1/Piper),已 LFS
├── motion_data/g1_29dof/    # 13 段 73-dim AMP JSON
├── source/atec_rl_lab/      # 训练扩展,pip install -e .
│   └── atec_rl_lab/
│       ├── algorithms/amp/  # AMP-PPO + Discriminator + MotionLoader
│       └── tasks/.../unitree_g1/  # env cfg + task_a 地形
├── scripts/
│   ├── rsl_rl/train_amp.py  # 训练入口
│   ├── rsl_rl/play_amp.py   # 播放 + 导出 policy.pt/onnx
│   ├── play_atec_task.py    # 在 ATEC 评测 env 跑 solution.py
│   └── list_envs.py         # 列出全部已注册环境
├── demo/                    # 提交模板(solution.py / policy.pt)
├── submission/              # 备用提交包
├── logs/                    # 训练 checkpoint + tensorboard(不提交)
└── docs/
    ├── deployment.md        # 本文
    └── task_a_amp_training.md  # AMP 训练细节 & 参数
```
