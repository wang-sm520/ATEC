# Task A — AMP Motion 数据集重定向与转换 Pipeline

本文是给**接手者**的一份可复现手册：如何把人类动作捕捉数据（LAFAN1 BVH / AMASS SMPL-X）经
**GMR 重定向**到 Unitree G1，再转成 Task A AMP 训练直接可吃的 **79-dim motion JSON**。

> 训练/播放/调参看 [`task_a_amp_training.md`](./task_a_amp_training.md)。本文只管「原始 mocap → 可用 motion 数据」这一段。

---

## 0. 全局数据流

```
                 ┌──────────────── gmr conda env (mujoco) ────────────────┐
人类 mocap        │  GMR 重定向 (IK)              本仓库转换脚本           │   atec conda env
─────────         │  ───────────────             ───────────────          │   ──────────────
LAFAN1 .bvh  ─────┼─▶ bvh_to_robot.py  ─┐                                  │
                  │   --robot unitree_g1 │                                 │
                  │                      ├─▶ G1 29-dof .pkl ─▶ gmr_to_amp_json.py ─▶ 79-dim .json ─▶ MotionLoaderG1
AMASS .npz   ─────┼─▶ smplx_to_robot.py ─┘   (root_pos/rot/dof_pos/fps)    │   (AMP discriminator)
(SMPL-X)          │   --robot unitree_g1                                   │
                  └─────────────────────────────────────────────────────────┘
```

两个 conda 环境，分工明确：

| env | 用途 | 关键依赖 |
|---|---|---|
| `gmr` | 重定向 + pkl→json 转换 | mujoco 3.7.0、GMR (`pip install -e .`) |
| `atec` | AMP 训练 / 评测 | IsaacLab 2.3.2、rsl-rl-lib 3.0.1、Isaac Sim 5.1 |

**重定向和 JSON 转换全程在 `gmr` env 里跑**（转换脚本用 mujoco 做 FK 算末端位置）。训练才切到 `atec`。

---

## 1. 目录与角色

### GMR 侧

| 路径 | 角色 |
|---|---|
| `scripts/bvh_to_robot.py` | LAFAN1 BVH → G1 pkl（单条，带可视化窗口） |
| `scripts/smplx_to_robot.py` | AMASS SMPL-X npz → G1 pkl（单条） |
| `assets/unitree_g1/g1_mocap_29dof.xml` | G1 的 MuJoCo MJCF，IK 求解 + 转换脚本 FK 都读它 |
| `data/lafan1/*.bvh` | LAFAN1 原始 BVH（直走素材） |
| `data/Male2Walking_c3d/*.npz` | AMASS Male2Walking SMPL-X（转身/侧步素材） |
| `data/g1_walks_pkl/*.pkl` | ACCAD 转身/侧步重定向后的 **29-dof** G1 pkl（已保留） |
| `batch_retarget_to_pkl.sh` | 批量把 bxi 精选的 9 段 Male2Walking npz 重定向成 g1 pkl |
| `batch_vis_g1_walks.sh` | 批量把上面的 pkl 渲染成 mp4 人眼验收 |
| `RETARGET_PIPELINE.md` | GMR 重定向通用手册（新机器人注册、IK 配置、排错） |

### 本仓库侧（`ATEC2026_Simulation_Challenge`）

| 路径 | 角色 |
|---|---|
| `motion_data/gmr_to_amp_json.py` | **核心转换器**：G1 pkl → 79-dim AMP JSON |
| `motion_data/g1_29dof/*.json` | 13 段最终训练数据（4 直走 + 9 转身/侧步） |
| `motion_data/README.md` | 旧版速记（注意：里面写的 `73-dim` 已过时，实际是 **79**） |
| `source/atec_rl_lab/atec_rl_lab/algorithms/amp/motion_loader_g1.py` | 训练时读 JSON 的 loader，定义 `FRAME_DIM = 79` |

> ⚠️ `data/lafan1_air/*.pkl` 是 **27-dof 的 "air" 机器人**早期实验产物，**与 G1 无关**，转换脚本不吃它（它要 29-dof）。别被目录名误导。

---

## 2. 79-dim 帧格式（必须严格对齐）

`MotionLoaderG1.FRAME_DIM = 79`，每帧布局：

| 区间 | 内容 | 维度 | 说明 |
|---|---|---|---|
| `[0:6)` | `root_vel_b` | 6 | pelvis 坐标系下 `[lin_vel(3), ang_vel(3)]` |
| `[6:35)` | `joint_pos` | 29 | **RAW 关节角**，顺序 = `G1_BODY_29_JOINT_NAMES` = GMR mujoco 关节顺序 |
| `[35:64)` | `joint_vel` | 29 | `joint_pos` 有限差分 × fps |
| `[64:79)` | `ee_pos_in_base` | 15 | 5 个末端 body × 3，pelvis 坐标系 |

5 个末端 body（顺序必须匹配 `rough_env_cfg.py` 里的 `G1_AMP_EE_BODIES`）：

```
left_ankle_roll_link, right_ankle_roll_link, waist_yaw_link,
left_wrist_yaw_link, right_wrist_yaw_link
```

这套布局必须和训练 env 里在线计算的 AMP obs（`mdp/observations.py:amp_obs_g1`）逐位对齐，
否则 discriminator 学到的是噪声。**改任何一端都要同步改另一端。**

JSON 顶层字段：

```json
{
  "LoopMode": "Wrap",
  "FrameDuration": 0.0333,          // = 1/fps
  "EnableCycleOffsetPosition": true,
  "EnableCycleOffsetRotation": true,
  "MotionWeight": 3.0,              // 采样权重（见 §6）
  "Frames": [[...79...], ...]
}
```

---

## 3. 当前 13 段数据来源映射

转换时 LAFAN1 的 G1 pkl 写到 `/tmp`（用完即弃），ACCAD 的 pkl 保留在 `data/g1_walks_pkl/`。

### 直走 — LAFAN1 BVH（`data/lafan1/`，权重 3.0）

| 输出 JSON | 源 BVH | 帧数 |
|---|---|---|
| `walk1_subject1.json` | `walk1_subject1.bvh` | 7839 |
| `walk1_subject2.json` | `walk1_subject2.bvh` | 7839 |
| `walk2_subject1.json` | `walk2_subject1.bvh` | 7145 |
| `walk3_subject1.json` | `walk3_subject1.bvh` | 7398 |

### 转身 / 侧步 — AMASS Male2Walking SMPL-X（bxi 精选，权重 0.3 / 0.5）

| 输出 JSON | 源 npz (`data/Male2Walking_c3d/`) | 权重 |
|---|---|---|
| `accad_walk_back.json` | `B5_-__Walk_backwards_stageii.npz` | 0.3 |
| `accad_walk_left_90.json` | `B9_-__Walk_turn_left_90_stageii.npz` | 0.3 |
| `accad_walk_left_45.json` | `B10_-__Walk_turn_left_45_stageii.npz` | 0.3 |
| `accad_walk_left_135.json` | `B11_-__Walk_turn_left_135_stageii.npz` | 0.3 |
| `accad_walk_right_90.json` | `B13_-__Walk_turn_right_90_stageii.npz` | 0.3 |
| `accad_walk_right_45.json` | `B14_-__Walk_turn_right_45_stageii.npz` | 0.3 |
| `accad_walk_around.json` | `B15_-__Walk_turn_around_stageii.npz` | 0.3 |
| `accad_sidestep_left.json` | `B22_-__side_step_left_stageii.npz` | 0.5 |
| `accad_sidestep_right.json` | `B23_-__side_step_right_stageii.npz` | 0.5 |

> 权重不是 1.0 是有意为之：直走 3.0 / 转身 0.3 / 侧步 0.5，让 expert 分布约 80% 是直走，
> 避免 policy 习得「自带转弯」的步态先验（曾导致 Task A eval 里走圆圈）。详见 §6。

---

## 4. 从零复现（接手者照抄）

### 4.0 一次性准备

```bash
conda activate gmr
cd /home/hpf/wsm/GMR
pip install -e .                       # 若未装过 GMR

# 批量脚本会读这两个「难动作黑名单」，缺了会 FileNotFoundError
mkdir -p assets/hard_motions && touch assets/hard_motions/0.txt assets/hard_motions/1.txt
```

SMPL-X body model（仅处理 AMASS/SMPL-X 数据时需要）放在 `assets/body_models/smplx/`
（`SMPLX_{NEUTRAL,MALE,FEMALE}.pkl`，从 https://smpl-x.is.tue.mpg.de/ 下载）。

### 4.1 LAFAN1 直走（4 段）

每段两步：BVH→pkl→json。

```bash
conda activate gmr
cd /home/hpf/wsm/GMR

for name in walk1_subject1 walk1_subject2 walk2_subject1 walk3_subject1; do
  # 步骤 1：重定向到 G1（会弹 mujoco 窗口验收；--robot unitree_g1 才是 29-dof）
  python scripts/bvh_to_robot.py \
      --bvh_file data/lafan1/${name}.bvh \
      --robot unitree_g1 \
      --format lafan1 \
      --save_path /tmp/${name}_g1.pkl

  # 步骤 2：pkl → 79-dim JSON（直走权重 3.0）
  python /home/hpf/atec/ATEC2026_Simulation_Challenge/motion_data/gmr_to_amp_json.py \
      --pkl /tmp/${name}_g1.pkl \
      --output /home/hpf/atec/ATEC2026_Simulation_Challenge/motion_data/g1_29dof/${name}.json \
      --motion_weight 3.0
done
```

无头机器跑可加 `MUJOCO_GL=egl` 前缀跳过窗口。

### 4.2 AMASS 转身 / 侧步（9 段）

重定向已脚本化：

```bash
conda activate gmr
cd /home/hpf/wsm/GMR
bash batch_retarget_to_pkl.sh          # → data/g1_walks_pkl/*.pkl（9 段，29-dof）
bash batch_vis_g1_walks.sh             # 可选：渲染 mp4 到 /tmp/g1_walks_videos 人眼验收
```

再逐个转 JSON（按 §3 的名字映射 + 权重）：

```bash
G1JSON=/home/hpf/atec/ATEC2026_Simulation_Challenge/motion_data/gmr_to_amp_json.py
OUT=/home/hpf/atec/ATEC2026_Simulation_Challenge/motion_data/g1_29dof
PKL=/home/hpf/wsm/GMR/data/g1_walks_pkl

# 转身段 weight 0.3
python $G1JSON --pkl $PKL/B5_-__Walk_backwards_stageii.npz.pkl  --output $OUT/accad_walk_back.json     --motion_weight 0.3
# ↑ 注意实际 pkl 文件名是 B5_-__Walk_backwards_stageii.pkl（basename 去掉 .npz），按 data/g1_walks_pkl/ 里的真实名填
python $G1JSON --pkl $PKL/B9_-__Walk_turn_left_90_stageii.pkl   --output $OUT/accad_walk_left_90.json  --motion_weight 0.3
python $G1JSON --pkl $PKL/B10_-__Walk_turn_left_45_stageii.pkl  --output $OUT/accad_walk_left_45.json  --motion_weight 0.3
python $G1JSON --pkl $PKL/B11_-__Walk_turn_left_135_stageii.pkl --output $OUT/accad_walk_left_135.json --motion_weight 0.3
python $G1JSON --pkl $PKL/B13_-__Walk_turn_right_90_stageii.pkl --output $OUT/accad_walk_right_90.json --motion_weight 0.3
python $G1JSON --pkl $PKL/B14_-__Walk_turn_right_45_stageii.pkl --output $OUT/accad_walk_right_45.json --motion_weight 0.3
python $G1JSON --pkl $PKL/B15_-__Walk_turn_around_stageii.pkl   --output $OUT/accad_walk_around.json   --motion_weight 0.3

# 侧步段 weight 0.5
python $G1JSON --pkl $PKL/B22_-__side_step_left_stageii.pkl     --output $OUT/accad_sidestep_left.json  --motion_weight 0.5
python $G1JSON --pkl $PKL/B23_-__side_step_right_stageii.pkl    --output $OUT/accad_sidestep_right.json --motion_weight 0.5
```

---

## 5. `gmr_to_amp_json.py` 内部做了什么

输入 pkl（GMR 输出）含 `dof_pos (N,29)`、`root_pos (N,3)`、`root_rot (N,4) xyzw`、`fps`。脚本：

1. **四元数转换**：GMR 存 `xyzw`，mujoco 要 `wxyz`，重排 `[3,0,1,2]`。
2. **joint_vel**：`dof_pos` 相邻帧有限差分 / dt，首帧复制次帧。
3. **末端位置 FK**：加载 `g1_mocap_29dof.xml`，逐帧 `mj_kinematics`，取 5 个 EE body 的世界坐标，
   用 pelvis 旋转矩阵转置投影到 pelvis 系 → 15 维。
4. **root 速度**：世界系线速度有限差分后 `R^T` 转 body 系；角速度用相邻帧相对四元数小角近似。
5. 拼成 `(N,79)`，连同 `LoopMode/FrameDuration/MotionWeight` 写 JSON。

`--gmr_root` 默认 `/home/hpf/wsm/GMR`（用来定位 mujoco XML），换机器要改。

---

## 6. MotionWeight 调参（无需重跑 GMR）

`MotionWeight` 只影响 AMP 训练时该 clip 被采样的相对概率。**改它不用重新重定向**——
直接编辑 13 个 JSON 顶层的 `"MotionWeight"` 字段即可。

当前配比（直走 3.0 / 转身 0.3 / 侧步 0.5）让 expert 分布 ~80% 直走，是为修「Task A eval 走圆圈」
而调的：所有 clip 都 1.0 时，69% 的 expert 是转身动作，AMP 把 policy 拉向自带转弯的步态。

快速批量改某类权重（示例：把转身段统一设 0.2）：

```bash
conda activate gmr
cd /home/hpf/atec/ATEC2026_Simulation_Challenge/motion_data/g1_29dof
python - <<'PY'
import json, glob
for f in glob.glob('accad_walk_*.json'):
    d = json.load(open(f)); d['MotionWeight'] = 0.2
    json.dump(d, open(f, 'w'))
    print('set', f)
PY
```

---

## 7. 验收

```bash
# 1) loader 自检（atec env）：确认 79-dim、能采样 (s, s_next) 对
conda activate atec
cd /home/hpf/atec/ATEC2026_Simulation_Challenge
python -m atec_rl_lab.algorithms.amp.motion_loader_g1

# 2) 逐文件巡检 frames/dim/fps/weight（gmr 或任意 python 都行）
python - <<'PY'
import json, glob, os
os.chdir('motion_data/g1_29dof')
for f in sorted(glob.glob('*.json')):
    d = json.load(open(f)); fr = d['Frames']
    print(f'{f:32s} frames={len(fr):5d} dim={len(fr[0])} '
          f'fps={1/d["FrameDuration"]:.1f} w={d["MotionWeight"]}')
PY
```

期望：每个文件 `dim=79`；直走 4 段 w=3.0、转身 7 段 w=0.3、侧步 2 段 w=0.5。

重定向质量人眼四查（GMR `vis_robot_motion.py` 或 `batch_vis_g1_walks.sh` 出的 mp4）：
脚底贴地、左右对称、躯干直立、手臂跟随。不过关回 GMR `RETARGET_PIPELINE.md` §3 调 IK。

---

## 8. 新增一段 motion（如爬楼梯）

1. 候选数据放进 `GMR/data/<dataset>/`（AMASS `.npz` 或 LAFAN `.bvh`）。
2. 单条重定向：`smplx_to_robot.py`（npz）或 `bvh_to_robot.py`（bvh），`--robot unitree_g1`，存 pkl。
3. 人眼验收（`vis_robot_motion.py`）。
4. `gmr_to_amp_json.py --pkl ... --output motion_data/g1_29dof/<new>.json --motion_weight <w>`。
5. 训练时 `--amp_motion_files motion_data/g1_29dof/*.json` 通配会自动吃进新文件。

---

## 9. 接手 checklist

- [ ] `gmr` env 可用，`pip install -e .` 过，`assets/hard_motions/{0,1}.txt` 存在
- [ ] 处理 SMPL-X 数据：`assets/body_models/smplx/` 三个 pkl 就位
- [ ] `assets/unitree_g1/g1_mocap_29dof.xml` 存在（转换脚本 FK 依赖）
- [ ] 重定向务必 `--robot unitree_g1`（29-dof）；别用 `lafan1_air`（27-dof air 机器人）
- [ ] 转换后跑 §7 验收，确认 `dim=79` 且权重符合 §3
- [ ] 训练命令见 [`task_a_amp_training.md`](./task_a_amp_training.md)
