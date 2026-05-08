# Task A — G1 AMP 训练使用文档

ATEC2026 Task A (越野导航, 300m 多地形走廊) 上训练 Unitree G1 humanoid 的 AMP (Adversarial Motion Priors) 策略。

## 环境

- conda env: `atec`
- 工作目录: `ATEC2026_Simulation_Challenge/`
- 注册的 task id: `ATEC-Isaac-AMP-Unitree-G1-TaskA-v0`
  - 同时存在 `Flat-v0` 和 `Rough-v0` 两个 task,适合预训练或调试
- 训练入口: `scripts/rsl_rl/train_amp.py`
- 算法: `atec_rl_lab.algorithms.amp.amp_ppo.AMPPPO` (AMP-PPO,新 rsl-rl-lib 3.0.1 API)

## 一、冒烟测试 (16 envs / 2 iter)

确认管道无 shape mismatch,先跑空 motion (退化为标准 PPO):

```bash
conda activate atec
cd ATEC2026_Simulation_Challenge
python scripts/rsl_rl/train_amp.py \
    --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 \
    --num_envs=16 --max_iterations=2 --headless
```

打印 `[train_amp] amp_motion_files is empty -> running in degenerate PPO mode` 即为 OK。

## 二、中等规模联调 (512 envs / 100 iter)

加上目前已重定向的 LAFAN1 walk motion,大约 5–10 分钟出结果:

```bash
python scripts/rsl_rl/train_amp.py \
    --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 \
    --num_envs=512 --max_iterations=100 --headless \
    --amp_motion_files motion_data/g1_29dof/*.json
```

通配 `*.json` 一次性吃掉:

- 4 段 LAFAN1 平地前进 (`walk1_subject1/2`, `walk2_subject1`, `walk3_subject1`)
- 9 段 ACCAD Male2Walking 重定向 (`accad_walk_back/around/left_45/90/135/right_45/90/sidestep_left/right`) — 对齐 bxi production curation

预期:
- `Train/episode_length` 从 ~12 上升到 ~800+
- `Train/fall_rate` 从 100% 降到 ~30%
- `AMP/discr_pred_real` 与 `AMP/discr_pred_fake` 围绕 0 收敛 (对抗平衡)

## 三、全量训练 (4096 envs / 30000 iter)

RTX 5090 上约 12–20 小时。建议放 `tmux` 内长跑:

```bash
tmux new -s g1amp
conda activate atec
cd ATEC2026_Simulation_Challenge

python scripts/rsl_rl/train_amp.py \
    --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 \
    --num_envs=4096 --max_iterations=30000 --headless \
    --amp_motion_files motion_data/g1_29dof/*.json
```

监控:
```bash
tensorboard --logdir logs/rsl_rl_amp/g1_amp_rough --port 6006
```

## 四、关键 CLI 参数

| 参数 | 默认 (bxi 对齐) | 说明 |
|---|---|---|
| `--amp_reward_coef` | 0.3 | AMP 奖励缩放 |
| `--amp_task_reward_lerp` | 0.6 | task vs AMP 奖励混合 (1.0 = 纯 task) |
| `--amp_replay_buffer_size` | 100000 | policy 经验回放容量 |
| `--amp_num_preload_transitions` | 200000 | 启动时从 motion 预填的样本数 |
| `--amp_loss_coef` | 1.0 | discriminator BCE 权重 |
| `--amp_grad_pen_lambda` | 10.0 | R1 gradient penalty |
| `--amp_obs_dim` | 73 | 自动从 env 的 `obs["amp"]` 推断,通常无需改 |
| `--amp_motion_files` (空) | — | 留空 ⇒ 退化为标准 PPO (调试用) |

## 五、地形说明

`task_a_env_cfg.py` 把 Task A 的 6 个 sub-terrain 同比例铺满 10×20 的训练阵 (200m × 400m):

- `flat` (10%) / `random_rough` (10%) / `slope ± inv_slope` (40%) / `stairs ± inv_stairs` (40%)
- slope curriculum: `(0.10, 0.40)` —— 起步温和,逐级上升到 Task A 评测难度
- stairs curriculum: `step_height_range=(0.05, 0.20)` —— 起步 5cm 台阶,封顶 20cm

`max_init_terrain_level=0` 表示所有 env 从最低难度起步,IsaacLab 内置的 terrain-level curriculum 会按表现升降难度。

## 六、新增 motion 数据 (e.g. 爬楼梯)

1. 把候选 AMASS NPZ 放进 `~/wsm/GMR/data/<dataset>/<subject>/`
2. 批量预览: `bash ~/wsm/GMR/batch_render_videos.sh <in_dir> /tmp/videos/<name>`
3. 选定后,用 GMR `smplx_to_robot.py` 重定向得到 G1 pkl
4. 再用 `motion_data/gmr_to_amp_json.py` 转成 73-dim AMP JSON
5. 把新 JSON 文件追加到 `--amp_motion_files`

## 七、日志与 checkpoint

- 输出根: `logs/rsl_rl_amp/g1_amp_rough/<日期>/`
- `model_*.pt`: 算法 + actor + critic + discriminator 完整 checkpoint
- `params/{env,agent}.yaml`: 还原训练所用 cfg
- `summaries/`: tensorboard

## 八、播放 / 可视化训练好的策略

最简 (跑最近一次训练的最新 checkpoint, 64 envs, 带 GUI):

```bash
python scripts/rsl_rl/play_amp.py \
    --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0
```

实时速度 + 少量 envs 看清楚:

```bash
python scripts/rsl_rl/play_amp.py \
    --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 \
    --num_envs=16 --real-time
```

录视频 (无 GUI, 输出到 `<run_dir>/videos/play/`):

```bash
python scripts/rsl_rl/play_amp.py \
    --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 \
    --num_envs=16 --video --video_length=600 --headless
```

指定某个 checkpoint:

```bash
python scripts/rsl_rl/play_amp.py \
    --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 --num_envs=16 \
    --checkpoint logs/rsl_rl_amp/g1_amp_rough/2026-05-08_*/model_5000.pt
```

`play_amp.py` 同时会把 actor 导出成 `<run_dir>/exported/policy.pt` 和 `policy.onnx`,后续 §九做 submission 直接用。

## 九、续训

```bash
python scripts/rsl_rl/train_amp.py \
    --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 \
    --num_envs=4096 --max_iterations=30000 --headless \
    --resume True --load_run -1 --checkpoint -1 \
    --amp_motion_files motion_data/g1_29dof/*.json
```

## 九、常见问题

- **`KeyError: 'class_name'`**: 升级到当前 `train_amp.py` 即可 (已调用 `handle_deprecated_rsl_rl_cfg`)
- **fall_rate 长期 100%**: motion 太少或与地形不匹配,先在 `Flat-v0` 上预训练再切到 `TaskA-v0`
- **AMP/discr_pred 失衡 (一直 +1 或 -1)**: 调小 `--amp_loss_coef` 或加大 `--amp_grad_pen_lambda`
