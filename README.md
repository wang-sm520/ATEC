# ATEC2026 G1 AMP — Task A

把 bxi humanoid 的 AMP (Adversarial Motion Priors) 训练方法迁移到 ATEC2026 Simulation Challenge 的 Unitree G1 humanoid (Task A 越野导航)。

- 训练框架: IsaacLab 2.3.2 + rsl-rl-lib 3.0.1
- 算法: PPO + AMP (在 `atec_rl_lab/algorithms/amp/` 下重写,适配新 rsl-rl API)
- 机器人: Unitree G1 33 DoF (训练只用 29 body joints,4 finger joints 固定为 default)
- 任务: Task A —— 300m 走廊 (flat / rough / stairs 混合)

完整文档见 [`docs/task_a_amp_training.md`](docs/task_a_amp_training.md)。

## 一、环境

```bash
conda activate atec   # IsaacLab 2.3.2 + rsl-rl-lib 3.0.1 + torch
cd ATEC2026_Simulation_Challenge
```

## 二、训练 Task A AMP

启动 G1 在 Task A 同款地形上训 AMP-PPO,4096 envs / 30000 iter / RTX 5090 ≈ 20-24h:

```bash
python scripts/rsl_rl/train_amp.py \
    --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 \
    --num_envs=4096 --max_iterations=30000 --headless \
    --amp_motion_files motion_data/g1_29dof/*.json
```

监控:

```bash
tensorboard --logdir logs/rsl_rl_amp/unitree_g1_amp_rough
```

退化为标准 PPO (调试管线用,不喂 motion):

```bash
python scripts/rsl_rl/train_amp.py \
    --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 \
    --num_envs=16 --max_iterations=2 --headless
```

## 三、播放 / 检查策略

在训练 env 里看步态 (我们自己的地形,带 GUI):

```bash
python scripts/rsl_rl/play_amp.py \
    --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 \
    --num_envs=16 --real-time
```

`play_amp.py` 同时把 actor 导出到 `<run_dir>/exported/policy.pt` (TorchScript) 和 `policy.onnx`。

在 ATEC 真评测 env 里跑 (通过 `demo/solution.py` 接口):

```bash
PYTHONPATH=. python scripts/play_atec_task.py \
    --task=ATEC-TaskA-G1 --num_envs=1 \
    --enable_cameras --real-time --debug
```

## 四、提交

1. 让 `play_amp.py` 把最新 actor 导出到 `<run_dir>/exported/policy.pt`
2. 拷贝到 demo:`cp logs/rsl_rl_amp/unitree_g1_amp_rough/<run>/exported/policy.pt demo/policy.pt`
3. `demo/solution.py` 已经写好,无需改动 (term-major 960-dim history,按 ATEC proprio 抽取 29 body joints,padding 到 33 dim)

## 五、Motion 数据

`motion_data/g1_29dof/` 当前包含 13 段 G1-retargeted motion (4 LAFAN1 forward walk + 9 ACCAD curated: walk_back / left_45/90/135 / right_45/90 / around / sidestep_l/r),73-dim 帧格式 (`jp_29 + jv_29 + ee_pos_in_pelvis_frame_15`)。

新增 motion (e.g. 爬楼梯) 流程: AMASS NPZ → GMR retarget → `motion_data/gmr_to_amp_json.py` → 73-dim AMP JSON,详情见 `docs/task_a_amp_training.md` §六。

## 六、关键文件

| 路径 | 内容 |
|---|---|
| `scripts/rsl_rl/train_amp.py` | 训练入口 |
| `scripts/rsl_rl/play_amp.py` | 播放 + jit/onnx 导出 |
| `source/atec_rl_lab/atec_rl_lab/algorithms/amp/` | AMP-PPO + Discriminator + MotionLoader |
| `source/.../config/humanoid/unitree_g1/rough_env_cfg.py` | 主 env cfg (reward 与 bxi 对齐) |
| `source/.../config/humanoid/unitree_g1/task_a_env_cfg.py` | Task A 风格训练地形 |
| `source/.../config/humanoid/unitree_g1/agents/rsl_rl_amp_cfg.py` | PPO 超参 |
| `source/.../mdp/observations.py` | 含 `amp_obs_g1` (73-dim) |
| `motion_data/gmr_to_amp_json.py` | GMR pkl → 73-dim AMP JSON |
| `demo/solution.py` | ATEC submission 接口 |
| `demo/policy.pt` | 训练好的 actor TorchScript |

## 致谢

- bxi (`bx_lab_amp`) — AMP 算法实现参考
- GMR (General Motion Retargeting) — AMASS → G1 retargeting 流水线
- LAFAN1 / ACCAD (AMASS) — motion 数据来源
