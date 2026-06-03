# Task D MAPush Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `ATEC2026_Simulation_Challenge` 的 Task D 中复用 Task A 训练出的 G1 locomotion 策略，并借鉴 MAPush 的层级推箱框架，让机器人完成“推箱入沟壑、踩箱越沟、到达终点”的得分流程。

**Architecture:** 保持 `demo/policy_a.pt` 作为冻结低层速度命令跟踪器，不直接改 policy 网络；在 `source/atec_rl_lab/atec_rl_lab/train/task_d/` 新建 Task D 专用模块，实现 MAPush 风格的高层任务控制器。当前主线采用方案 B：用 LiDAR 先建立“大障碍物/沟壑”的局部坐标系，再估计箱子相对大障碍物的坐标，最后闭环生成 `(vx, vy, wz)` 速度指令推箱。

**Tech Stack:** ATEC 侧使用 IsaacLab 2.3.2、PyTorch/TorchScript、G1 33 DoF submission API；MAPush 侧作为算法参考，核心来自 IsaacGym、OpenRL、`mqe/envs/wrappers/*`、`TrajectoryPlanner` 和 `KinodynamicRRT`。

---

## 1. 已梳理的框架结论

### 1.1 ATEC Task D 侧

关键文件：

- `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/tasks/task_d/env_cfg.py`
- `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/tasks/task_d/terrain.py`
- `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/tasks/task_d/mdp/rewards.py`
- `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/tasks/task_d/mdp/terminations.py`
- `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/tasks/task_base/envs_base_cfg.py`
- `/home/hpf/atec/ATEC2026_Simulation_Challenge/demo/solution_a.py`
- `/home/hpf/atec/ATEC2026_Simulation_Challenge/demo/solution_d.py`

Task D 的确定性结构：

- G1 初始位置：`(-3, 0, 0.8)`。
- 箱子初始位置：`(-3, 1.6, 0.5)`。
- 箱子尺寸和物理：`0.8 x 1.0 x 0.6 m`，质量 `8 kg`，静摩擦 `0.9`，动摩擦 `0.8`。
- G1 Task D 沟壑宽度：`0.9-1.0 m`；平台高度：`0.9-1.0 m`。
- 地形生成固定 `seed=0`，平台固定在 `+y` 侧。
- 机器人 crossing reward：
  - root x > `-1.4` 时一次性 `+2`。
  - root x > `2.0` 时一次性 `+20`。
- 箱子改造 reward：
  - box x 在 `[-1.4, 0.7]` 的并集范围内一次性 `+14`。
  - 当前 reward 只检查箱子 x，不检查 y 或姿态；但物理通关仍需要箱子位置能支撑机器人跨沟。
- 终止：
  - root x > `3.5` 触发完成。
  - fall 最小高度阈值被设为 `0.25`。

Submission 接口：

- `AlgSolution.predicts(obs, current_score)` 返回 `{"action": List[float], "giveup": bool}`。
- `obs["proprio"]` 排列为：
  - `base_lin_vel(3)`
  - `base_ang_vel(3)`
  - `velocity_commands(3)`
  - `projected_gravity(3)`
  - `joint_pos(N)`
  - `joint_vel(N)`
  - `last_action(N)`
- G1 当前 full action dim 由 `(proprio_dim - 12) / 3` 推出，通常是 `33`。
- Task A AMP policy 只控制 29 个 body joints，4 个 finger joints 保持 0。
- ATEC eval action 是 `JointPositionActionCfg(scale=0.5, use_default_offset=True)`，现有 demo 已用 `TRAINING_ACTION_SCALE_29 / 0.5` 做了动作尺度补偿。

LiDAR / extero 接口：

- Task D 中 `self.observations.extero.enable_corruption = False`，可以直接使用 `obs["extero"]`。
- base env 的 LiDAR 配置在 `tasks/task_base/envs_base_cfg.py`：
  - `RayCasterCfg`
  - `vertical_fov_range=(-20.0, 20.0)`
  - `horizontal_fov_range=(-180.0, 180.0)`
  - `horizontal_res=1.0`
  - `channels=16`
  - `max_distance=10.0`
  - `update_period=0.1`
- G1 的 `lidar_sensor_link_name="torso_link"`，所以 LiDAR 坐标系随 torso/base 运动。
- observation term 是 `mdp.height_scan(lidar_sensor)`，会被 flatten 后通过 `obs["extero"]` 传入 `predicts()`。
- 重要限制：当前 `RayCasterCfg.mesh_prim_paths=["/World/ground"]`。这意味着 LiDAR 对沟壑、平台等地形很合适，但未必直接命中动态箱子。方案 B 的感知设计需要先用 LiDAR 建立大障碍物坐标系；箱子位置优先尝试从 LiDAR 的非地形异常中估计，如果实际 observation 不含箱子，则切换到 RGB-D 或训练侧增强传感器 mesh paths。

现有 Task A policy 接入方式：

- `/home/hpf/atec/ATEC2026_Simulation_Challenge/demo/policy_a.pt` 存在。
- `/home/hpf/atec/ATEC2026_Simulation_Challenge/demo/policy.pt` 也存在。
- 历史 `demo/solution.py` / 当前 `demo/solution_a.py` 是 Task A walking solution，复现了 960 维 term-major history：
  - `[ang_vel_t0..t9, cmd_t0..t9, gravity_t0..t9, jp_t0..t9, jv_t0..t9, lastact_t0..t9]`
- 历史 `demo/solution_d.py` 曾在 Task A policy 上加 Task D 开环高层状态机：
  - 用 `base_ang_vel` 投影到 `projected_gravity` 的 up axis 估计世界 yaw。
  - 用 body-frame `base_lin_vel` 做 2D dead-reckoning。
  - 通过分段脚本生成 `(vx, vy, wz)`。
  - 当前 `demo/solution_d.py` 已改为薄入口，委托 `atec_rl_lab.train.task_d.solution_adapter.AlgSolution`。

### 1.2 MAPush 侧

关键文件：

- `/home/hpf/atec/MAPush/README.md`
- `/home/hpf/atec/MAPush/mqe/envs/go1/go1.py`
- `/home/hpf/atec/MAPush/mqe/envs/wrappers/go1_push_mid_wrapper.py`
- `/home/hpf/atec/MAPush/mqe/envs/wrappers/go1_push_upper_wrapper.py`
- `/home/hpf/atec/MAPush/mqe/envs/wrappers/utils/trajectory.py`
- `/home/hpf/atec/MAPush/mqe/envs/wrappers/utils/rrt.py`
- `/home/hpf/atec/MAPush/mqe/envs/configs/go1_push_mid_config.py`
- `/home/hpf/atec/MAPush/mqe/envs/configs/go1_push_upper_config.py`

MAPush 的层级结构：

- `Go1.step(action)` 在 `control_type == "C"` 时把上层 action 当成 locomotion command。
- `Go1.preprocess_action()` 将 3 维速度命令写入 locomotion obs：
  - `lin_vel_x`
  - `lin_vel_y`
  - `ang_vel_z`
- MAPush 底层 locomotion policy 输出 12 DoF Go1 joint action。
- `Go1PushMidWrapper` 是中层推箱控制器：
  - observation：目标位置、箱子位置、箱子 yaw、其他 agent 相对位姿，全部转到机器人局部坐标。
  - action：3 维 `[-1, 1]`，经过 `action_scale=[0.5, 0.5, 0.5]` 后作为速度命令。
  - reward：接近目标、推动箱子、箱子目标方向/接触法线一致性、到达目标、异常惩罚。
- `Go1PushUpperWrapper` 是上层控制器：
  - action：2 维子目标位置。
  - 内部把子目标转成 mid-level observation。
  - 调用 `command_module.act()` 得到 3 维中层 velocity command。
  - `TrajectoryPlanner` 生成 13 个 2D waypoint。
  - `KinodynamicRRT` 可在 2D 中绕障规划箱子路径。

MAPush 可直接借鉴的思想：

- 不把推箱问题交给低层 gait policy，而是让低层只跟踪速度命令。
- 以“箱子目标点”作为中层接口，而不是直接输出关节动作。
- 将目标、箱子、机器人关系转成机器人局部坐标，降低策略对绝对坐标的依赖。
- 上层规划不需要很复杂，给箱子生成一串 waypoint 即可。
- 推箱接触方向很重要，MAPush 的 OCB reward 本质是在鼓励机器人站在正确接触面，用正确法向推箱。

MAPush 不能直接拷贝的部分：

- MAPush 是 IsaacGym + Python 3.8 + OpenRL + Go1；ATEC 是 IsaacLab + Python 3.12 + G1。
- MAPush 训练环境能直接读取 `root_states_npc` 的真实箱子位姿；ATEC submission 只能拿到 `proprio`、`image`、`extero`。
- MAPush 是多四足协作，Task D 现在是单 G1。
- MAPush mid-level network 输出的 velocity command 适配 Go1 locomotion，不适配 G1 AMP policy。

结论：短期不应该把 MAPush 的模型或环境硬迁移到 ATEC；应该迁移它的层级控制接口和推箱几何逻辑。

---

## 2. 设计方案对比

### 方案 A：确定性高层状态机 + Task A policy

做法：

- 保持 `policy_a.pt` 冻结。
- 在 `solution_d.py` 里把现有 open-loop `SEGMENTS` 改成更清晰的阶段控制：
  - warmup
  - 走到箱子后方
  - 对齐箱子
  - 推箱到沟壑中心
  - 后撤/重定位
  - 踩箱跨沟
  - 走到终点
- 状态估计只用 known spawn + proprio dead-reckoning。

优点：

- 最快能落地。
- 不需要训练新策略。
- 依赖最少，适合先拿可提交分数。

缺点：

- 箱子实际位置不可观测，推箱失败后无法纠偏。
- 转弯、后退、横移如果超出 Task A policy 分布，稳定性要靠实测调参。
- 对碰撞、脚踩箱子的细节很敏感。

适用目标：

- 快速争取 `+2`、`+14`、部分 `+20`，验证物理可行性。

### 方案 B：MAPush 式子目标推箱闭环 + LiDAR 大障碍物坐标系

做法：

- 保持 `policy_a.pt` 冻结。
- 在 `source/atec_rl_lab/atec_rl_lab/train/task_d/` 新增高层控制模块，把 Task D 分成 box waypoint tracking：
  - 估计 robot pose。
  - 用 LiDAR 检测大障碍物/沟壑边缘，建立 obstacle frame。
  - 估计 box pose in obstacle frame。
  - 计算 desired contact pose。
  - 生成 body-frame velocity command。
- 箱子位姿来源按优先级：
  - LiDAR 可见时：由 LiDAR 聚类/几何边缘得到 box relative to obstacle。
  - LiDAR 只看到地形时：用 LiDAR 得到 obstacle frame，用 odometry + 已知初始化 + 推箱接触模型估计 box relative to obstacle。
  - 如 LiDAR 无法稳定看箱子：用 head RGB-D 做箱子校正。
- 使用 MAPush 的 `TrajectoryPlanner` 思路生成箱子目标：
  - box start `(-3, 1.6)`
  - bridge target `(-0.9, 1.6)` 或实测后修正到踩踏最稳的位置。
  - cross target `(3.6, 1.6)`。

优点：

- 比纯脚本更鲁棒。
- 不需要训练 mid-level network，就能获得 MAPush 的闭环控制收益。
- 适合逐步加感知，不影响已有 policy 输入格式。

缺点：

- 需要实现并调试状态估计。
- LiDAR 当前可能只扫到 `/World/ground`，动态箱子是否可见需要实测确认。
- RGB-D/LiDAR 的箱子检测要处理视角、遮挡和坐标系标定。
- 仍然受限于 Task A locomotion policy 的推箱和踩箱能力。

适用目标：

- 推荐主线。先用几何闭环拿稳定分，再按失败日志加感知校正。

### 方案 C：在 ATEC/IsaacLab 中重训 G1 mid-level push policy

做法：

- 参考 MAPush `Go1PushMidWrapper`，在 ATEC 侧写一个训练 wrapper。
- 训练输入：
  - G1 proprio history。
  - robot/box/target 相对位姿。
  - 可选 depth/LiDAR。
- 输出：
  - `vx, vy, wz` 给已有 Task A locomotion policy。
- reward 参考 MAPush：
  - 接近箱子。
  - 推动箱子。
  - 箱子靠近目标 x/y。
  - 接触法向与目标方向一致。
  - 机器人不过早过沟/不跌倒。

优点：

- 长期最接近 MAPush 的能力边界。
- 可学习接触、推力、踩箱时机，不靠手调。

缺点：

- ATEC 官方 env 不支持并行训练，需要自己封装或另建训练环境。
- 训练成本高，不适合作为第一条提交路径。
- 仍需要保证 policy_a 对高层 velocity command 的分布支持。

适用目标：

- 在方案 B 能稳定拿分后，用它生成专家/轨迹，再考虑训练。

推荐：先执行方案 B 的最小闭环版本，同时保留方案 A 作为 fallback。方案 C 作为后续增强，不阻塞第一次可得分提交。

---

### 2.4 当前采用的定版方向

当前执行方案 B，具体约束如下：

- Task D 相关实现文件统一放在：
  - `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/`
- `demo/solution_d.py` 作为 Task D submission 入口和薄 adapter，尽量不堆任务逻辑。
- 感知坐标系优先使用 obstacle frame，而不是纯世界坐标：
  - `obstacle_frame.x`：沿赛道前进方向。
  - `obstacle_frame.y`：横向，平台/箱子初始侧为 `+y`。
  - `obstacle_origin`：由 LiDAR 检测到的沟壑近边缘或地形中心线定义。
- 控制闭环以“箱子相对大障碍物的位置”为核心：
  - robot in obstacle frame。
  - box in obstacle frame。
  - desired box bridge pose in obstacle frame。
  - desired robot contact pose in obstacle frame。
- velocity command 仍是对 `policy_a.pt` 的输入，不直接输出关节动作。

---

## 3. 推荐控制架构

### 3.1 模块边界

建议在 ATEC repo 中拆成以下文件。按你的要求，Task D 逻辑统一放在 `source/atec_rl_lab/atec_rl_lab/train/task_d/`；`demo/solution_d.py` 只保留 Task D 提交入口。

- 修改：`/home/hpf/atec/ATEC2026_Simulation_Challenge/demo/solution_d.py`
  - 作为平台实际加载入口。
  - import `atec_rl_lab.train.task_d.solution_adapter.AlgSolution` 或直接委托给该模块。
- 创建：`/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/__init__.py`
  - 导出 Task D controller / adapter。
- 创建：`/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/policy_bridge.py`
  - 负责加载 `policy_a.pt`。
  - 负责 960 维 term-major history。
  - 负责 29 body joints 到 33 full action 的 padding。
  - 负责 action scale compensation。
- 创建：`/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/state_estimator.py`
  - 基于 proprio 的 yaw 去倾斜积分。
  - robot odometry。
  - obstacle frame 下的 robot pose。
- 创建：`/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/lidar_perception.py`
  - 将 `obs["extero"]` 还原成 LiDAR scan grid。
  - 检测沟壑/平台/大障碍物位置。
  - 尝试检测箱子 cluster。
  - 输出 obstacle frame 和 box measurement。
- 创建：`/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/box_tracker.py`
  - 融合 LiDAR、odometry、known init 和推箱接触模型。
  - 输出 box pose in obstacle frame。
- 创建：`/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/waypoint_planner.py`
  - MAPush 式 box waypoint。
  - desired contact pose。
- 创建：`/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/velocity_controller.py`
  - 根据 robot/box/target 的 obstacle-frame 误差生成 `(vx, vy, wz)`。
- 创建：`/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/phase_machine.py`
  - 管理 Task D phase。
  - 根据 score、pose、timeout、box progress 切换阶段。
- 创建：`/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/controller.py`
  - 串联 perception、tracking、planning、velocity control。
- 创建：`/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/solution_adapter.py`
  - 提供 `AlgSolution`，供 `demo/solution_d.py` 调用。
- 创建：`/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/debug.py`
  - 只放本地调试 print、phase log、score log。
  - 提交时可以保留轻量日志，也可以关闭。

MAPush repo 中建议只新增文档和实验记录，不直接改训练代码，除非进入方案 C。

### 3.2 数据流

每个 `predicts()` step：

1. 从 `obs["proprio"]` 解析：
   - `base_lin_vel`
   - `base_ang_vel`
   - `projected_gravity`
   - `joint_pos`
   - `joint_vel`
   - `last_action`
2. 从 `obs["extero"]` 解析 LiDAR：
   - 如果 `obs["extero"] is None`，进入 odometry fallback。
   - 否则按 `16 channels x 360 azimuth` 的初始假设恢复 scan；实际 shape 以第一次打印为准。
   - 通过 scan 的距离/高度突变检测沟壑近边缘、平台边界和大障碍物轮廓。
3. `TaskDLidarPerception.update()`：
   - 输出 obstacle frame。
   - 输出可选 box measurement。
   - 标记 box 是否 LiDAR-visible。
4. `TaskDStateEstimator.update()`：
   - 用 gravity 投影积分 yaw。
   - 将 body velocity 转成 world velocity。
   - 积分 robot `(x, y, yaw)`。
   - 使用 known spawn 初始化。
   - 将 robot pose 投影到 obstacle frame。
5. `TaskDBoxTracker.update()`：
   - LiDAR 检测到箱子时，用 measurement 校正。
   - LiDAR 未检测到箱子时，用 known init + 推箱接触模型估计。
   - 输出 box pose in obstacle frame。
6. `TaskDPhaseMachine.update()`：
   - 根据 robot pose、box estimate、current_score、接触/进度判断 phase。
7. `TaskDWaypointPlanner.target()`：
   - 输出当前 robot target pose 或 box target pose。
8. `TaskDVelocityController.command()`：
   - 把 world-frame target 转成 body-frame `(vx, vy, wz)`。
   - 限幅到 Task A policy 支持的范围。
9. `TaskDPolicyBridge.act()`：
   - 把 `(vx, vy, wz)` 写入 policy input 的 command term。
   - 调用 `policy_a.pt`。
   - 输出 full action list。

### 3.3 Phase 设计

建议使用以下 phase，而不是一长串硬编码 `SEGMENTS`：

1. `WARMUP`
   - 目的：让 history buffer 从 0 逐步变成真实观测。
   - 命令：`vx=0.3` 或 `vx=0.5`，`vy=0`，`wz=heading hold`。
   - 退出：`step_count >= 20`。

2. `MOVE_TO_BOX_LANE`
   - 目的：从 `y=0` 移动到箱子 lane `y=1.6` 附近。
   - 推荐优先用 `vy` 横移，因为用户说明 `policy_a.pt` 支持左移/右移。
   - 如果实测横移不稳，fallback 到 turn-arc 方式。
   - 退出：`abs(y_est - 1.6) < 0.15`。

3. `ALIGN_BEHIND_BOX`
   - 目的：站到箱子后方，准备沿 `+x` 推。
   - 目标 robot pose：`x = box_x - 0.75`，`y = box_y`，`yaw = 0`。
   - 退出：`x/y/yaw` 都进入容差，或超时后进入小速度推箱。

4. `PUSH_BOX_TO_BRIDGE`
   - 目的：把箱子 x 推到沟壑/得分范围。
   - 目标 box x：初值建议 `-0.9`。
   - 命令：`vx=0.5-1.0`，`vy` 用 lane error 小幅修正，`wz` 保持 `yaw=0`。
   - 退出：
     - `current_score` 增加到包含 `+14`，或
     - `box_x_est >= -0.9`，或
     - 推箱超时后进入重定位。

5. `BACK_OFF_AND_CENTER`
   - 目的：从箱子附近脱离，避免卡在箱子边缘，重建踩箱路径。
   - 命令：短时间 `vx=-0.3` 或用 turn-arc 回退。
   - 退出：后退距离 `0.3-0.5 m` 或 `0.8 s`。

6. `CROSS_ON_BOX`
   - 目的：沿箱子 lane 前进，通过沟壑。
   - 命令：`vx=0.7-1.0`，`vy` 小幅保持 lane，`wz` 保持 `yaw=0`。
   - 退出：`current_score` 包含 `+20` 或 `x_est > 2.1`。

7. `FINISH`
   - 目的：继续走到 `x > 3.5`。
   - 命令：`vx=1.0-1.2`，`vy` 朝安全 lane 轻微回中。
   - 退出：环境终止或 `current_score >= 35`。

### 3.4 速度命令限幅

初始建议：

- `vx_forward`: `0.5-1.2`
- `vx_push`: `0.5-0.8`
- `vx_backward`: `-0.2` 到 `-0.5`
- `vy`: `-0.6` 到 `0.6`
- `wz`: `[-1.57, 1.57]`
- heading stiffness：
  - walking：`0.5`
  - turning/recover：`1.0-1.5`

调参原则：

- 先保证不摔，再提高推箱效率。
- 如果 `vx=0` 原地转会卡住，用 `TURN_FWD_SPEED=0.3-0.5` 的小弧线转向。
- 如果横移造成 gait 不稳，改成“面向目标点 + 前进”的 pure pursuit。
- 推箱阶段优先低速稳定接触，不追求快。

---

## 4. 文件级实施计划

### Task 1: 固化 Task A policy bridge

**Files:**

- Create: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/__init__.py`
- Create: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/policy_bridge.py`
- Create: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/solution_adapter.py`
- Modify: `/home/hpf/atec/ATEC2026_Simulation_Challenge/demo/solution_d.py`

- [ ] **Step 1: 提取 policy 加载和 960 维 history 逻辑**

从历史 `demo/solution.py` 和当前 `demo/solution_d.py` 中提取：

- `BODY_29_IDX`
- `ACTION_DIM_BODY`
- `HISTORY_LEN`
- term dim 常量
- `TRAINING_ACTION_SCALE_29`
- `action_scale_ratio`
- `_make_buffer()`
- `_push()`
- term-major concat
- 29 body action padding 到 full action

目标接口：

```python
class G1VelocityPolicyBridge:
    def __init__(self, policy_path: str, device: str = "cuda"):
        ...

    def reset(self) -> None:
        ...

    def act(self, proprio: torch.Tensor, velocity_commands: torch.Tensor) -> list[float]:
        ...
```

- [ ] **Step 2: 在 `solution_adapter.py` 调用 bridge**

`AlgSolution.predicts()` 中只保留：

```python
proprio = obs["proprio"].to(self.device, dtype=torch.float32)
velocity_commands = self.controller.update(obs, current_score)
action = self.bridge.act(proprio, velocity_commands)
return {"action": action, "giveup": False}
```

- [ ] **Step 3: 让 `demo/solution_d.py` 成为薄入口**

```python
from atec_rl_lab.train.task_d.solution_adapter import AlgSolution
```

- [ ] **Step 4: 本地验证 Task A 直走行为不变**

Run:

```bash
cd /home/hpf/atec/ATEC2026_Simulation_Challenge
PYTHONPATH=. python scripts/play_atec_task.py --task ATEC-TaskA-G1 --num_envs 1 --debug --real-time
```

Expected:

- 不出现 policy input shape 错误。
- G1 能稳定沿 +x 行走。
- action dim 与 env full action dim 匹配。

### Task 2: 实现 Task D state estimator

**Files:**

- Create: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/state_estimator.py`
- Create: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/types.py`

- [ ] **Step 1: 实现 yaw 去倾斜积分**

保留现有可靠逻辑：

```python
up_body = -projected_gravity / (projected_gravity.norm() + 1e-6)
yaw_rate_world = torch.dot(base_ang_vel, up_body)
heading_est += float(yaw_rate_world.item()) * step_dt
```

- [ ] **Step 2: 实现 2D 里程计**

```python
cos_h = math.cos(heading_est)
sin_h = math.sin(heading_est)
x_est += (bvx * cos_h - bvy * sin_h) * step_dt
y_est += (bvx * sin_h + bvy * cos_h) * step_dt
```

- [ ] **Step 3: 增加 reset 初始化**

```python
robot_x = -3.0
robot_y = 0.0
robot_yaw = 0.0
box_x = -3.0
box_y = 1.6
```

- [ ] **Step 4: 输出 obstacle-frame robot pose**

当 LiDAR perception 已经给出 obstacle frame 时：

```python
robot_pose_obstacle = obstacle_frame.world_to_local(robot_pose_world)
```

如果 obstacle frame 尚未初始化，则使用 known Task D frame fallback。

- [ ] **Step 5: 记录 phase 日志**

每 25 step 打印一次：

```text
[TaskD] phase=PUSH_BOX_TO_BRIDGE robot=(-2.30,1.55,2deg) box=(-1.20,1.58) score=16.0 cmd=(0.65,0.02,-0.05)
```

### Task 3: 实现 LiDAR 大障碍物感知

**Files:**

- Create: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/lidar_perception.py`
- Modify: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/types.py`

- [ ] **Step 1: 解析 `obs["extero"]`**

初始假设：

```python
LIDAR_CHANNELS = 16
LIDAR_AZIMUTH_BINS = 360
```

如果实际 `obs["extero"].numel()` 不等于 `16 * 360`，第一版先记录 shape，并按实际长度自动 reshape 或退化为 1D scan。

- [ ] **Step 2: 检测沟壑/平台边界**

目标：

- 从 LiDAR/height scan 的前向扇区中找到距离或高度突变。
- 输出 obstacle frame：
  - `origin`: 沟壑近边缘或大障碍物中心线。
  - `yaw`: 赛道方向，初始用 robot yaw / known +x。
  - `confidence`: `0.0-1.0`。

- [ ] **Step 3: 尝试检测箱子 cluster**

检测逻辑：

- 在 `+y` 侧、机器人附近筛选非地形异常点。
- 聚类得到候选箱子中心。
- 与已知箱子尺寸 `0.8 x 1.0 x 0.6` 做宽度/高度一致性检查。

重要 caveat：

- 当前 RayCaster 只配置了 `mesh_prim_paths=["/World/ground"]`，如果实测 LiDAR 不包含动态箱子，`box_measurement.valid=False`，由 `box_tracker.py` 使用 odometry/contact fallback。

### Task 4: 实现 MAPush 式 box tracker 与 waypoint planner

**Files:**

- Create: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/box_tracker.py`
- Create: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/waypoint_planner.py`
- Modify: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/types.py`

- [ ] **Step 1: 定义 obstacle-frame landmarks**

```python
BOX_START_OBS = (-3.0, 1.6)
BOX_BRIDGE_TARGET_OBS = (-0.9, 1.6)
ROBOT_START_OBS = (-3.0, 0.0)
ROBOT_FINISH_X_OBS = 3.5
SCORE_X_REACHED_OBS = -1.4
SCORE_X_CROSSED_OBS = 2.0
```

- [ ] **Step 2: 融合 LiDAR box measurement 与推箱接触模型**

```python
if box_measurement.valid:
    box_pose = blend(box_pose, box_measurement.pose_obstacle, alpha=0.35)
elif phase == "PUSH_BOX_TO_BRIDGE" and contact_likely:
    box_pose.x = max(box_pose.x, robot_pose.x + contact_offset)
    box_pose.y = 0.95 * box_pose.y + 0.05 * robot_pose.y
```

- [ ] **Step 3: 用 score 反推 box 已进得分范围**

如果 `current_score >= 16.0` 且之前没有记录 box reward：

```python
box_pose.x = min(max(box_pose.x, -1.4), -0.9)
box_reward_seen = True
```

- [ ] **Step 4: 实现 robot target pose 计算**

对每个 phase 输出：

- `MOVE_TO_BOX_LANE`: `(-3.0, 1.35, yaw=0)`
- `ALIGN_BEHIND_BOX`: `(box_x - 0.75, box_y, yaw=0)`
- `PUSH_BOX_TO_BRIDGE`: `(box_x - 0.65, box_y, yaw=0)`，目标是推动 box。
- `BACK_OFF_AND_CENTER`: `(robot_x - 0.4, box_y, yaw=0)`
- `CROSS_ON_BOX`: `(2.2, box_y, yaw=0)`
- `FINISH`: `(3.8, 1.2, yaw=0)`

- [ ] **Step 5: 增加 phase 退出条件**

不只依赖时间，至少依赖：

- pose error。
- `current_score`。
- 估计推动距离。
- phase timeout。

### Task 5: 实现 velocity controller

**Files:**

- Create: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/velocity_controller.py`

- [ ] **Step 1: world target 转 body-frame error**

```python
dx = target_x - robot_x
dy = target_y - robot_y
ex_body = math.cos(-yaw) * dx - math.sin(-yaw) * dy
ey_body = math.sin(-yaw) * dx + math.cos(-yaw) * dy
```

- [ ] **Step 2: pose tracking 生成 velocity command**

```python
vx = clamp(kx * ex_body, vx_min, vx_max)
vy = clamp(ky * ey_body, -vy_max, vy_max)
wz = clamp(kw * wrap_to_pi(target_yaw - yaw), -1.57, 1.57)
```

- [ ] **Step 3: phase-specific 限幅**

推箱阶段：

```python
vx = clamp(vx, 0.35, 0.8)
vy = clamp(vy, -0.25, 0.25)
```

走位阶段：

```python
vx = clamp(vx, -0.5, 1.0)
vy = clamp(vy, -0.6, 0.6)
```

跨沟阶段：

```python
vx = clamp(vx, 0.6, 1.0)
vy = clamp(vy, -0.2, 0.2)
```

### Task 6: 实现 phase machine 与 controller 串联

**Files:**

- Create: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/phase_machine.py`
- Create: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/controller.py`
- Create: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/debug.py`

- [ ] **Step 1: 实现 phase 切换**

phase 列表：

- `WARMUP`
- `MOVE_TO_BOX_LANE`
- `ALIGN_BEHIND_BOX`
- `PUSH_BOX_TO_BRIDGE`
- `BACK_OFF_AND_CENTER`
- `CROSS_ON_BOX`
- `FINISH`

- [ ] **Step 2: 串联 perception -> tracker -> planner -> velocity**

```python
robot_state = state_estimator.update(proprio)
obstacle_state = lidar_perception.update(extero, robot_state)
box_state = box_tracker.update(robot_state, obstacle_state, current_score, phase)
phase = phase_machine.update(robot_state, box_state, current_score)
target = waypoint_planner.target(phase, robot_state, box_state)
velocity_commands = velocity_controller.command(robot_state, target)
```

- [ ] **Step 3: 如果推箱超时，进入恢复阶段**

恢复策略：

- 后退 `0.3-0.5 m`。
- 回到 `ALIGN_BEHIND_BOX`。
- 降低推箱 `vx`。

### Task 7: 增加本地 Task D 验证脚本流程

**Files:**

- Modify: `/home/hpf/atec/ATEC2026_Simulation_Challenge/demo/solution_d.py`
- Modify: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/solution_adapter.py`

- [ ] **Step 1: 让 `demo/solution_d.py` 指向 Task D controller**

本地验证前，把 Task D 方案作为平台入口：

```python
from atec_rl_lab.train.task_d.solution_adapter import AlgSolution
```

- [ ] **Step 2: 确认加载 `policy_a.pt`**

因为 `solution_adapter.py` 位于 `source/.../train/task_d/`，不能再用 adapter 文件目录直接找权重。建议由 `demo/solution_d.py` 把 demo 目录下的权重路径传入，或在 adapter 中显式计算仓库根目录。最简单入口：

```python
import os
from atec_rl_lab.train.task_d.solution_adapter import AlgSolution as TaskDAlgSolution

class AlgSolution(TaskDAlgSolution):
    def __init__(self):
        demo_dir = os.path.dirname(os.path.abspath(__file__))
        super().__init__(policy_path=os.path.join(demo_dir, "policy_a.pt"))
```

如果要兼容已有提交脚本，也可以保留 fallback：

```python
policy_a = os.path.join(demo_dir, "policy_a.pt")
policy = os.path.join(demo_dir, "policy.pt")
super().__init__(policy_path=policy_a if os.path.exists(policy_a) else policy)
```

- [ ] **Step 3: 运行 Task D debug**

Run:

```bash
cd /home/hpf/atec/ATEC2026_Simulation_Challenge
PYTHONPATH=. python scripts/play_atec_task.py --task ATEC-TaskD-G1 --num_envs 1 --enable_cameras --debug --real-time
```

Expected milestones:

- `score >= 2`: 机器人 root x 过 `-1.4`。
- `score >= 16`: 箱子 x 进入 `[-1.4, 0.7]`。
- `score >= 36`: 箱子改造分 + 机器人过 `2.0` + 到达终点。

- [ ] **Step 4: 录制视频复盘**

Run:

```bash
cd /home/hpf/atec/ATEC2026_Simulation_Challenge
PYTHONPATH=. python scripts/play_atec_task.py --task ATEC-TaskD-G1 --num_envs 1 --enable_cameras --video --video_length 1200 --debug
```

Expected:

- 视频中能看清是否接触箱子、是否把箱子推入沟壑、是否踩箱跨沟。
- 日志 phase 与视频动作一致。

### Task 8: 感知闭环增强

**Files:**

- Modify: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/lidar_perception.py`
- Create: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/depth_perception.py`

- [ ] **Step 1: 先用 LiDAR/height scan 做沟壑边缘和大障碍物定位**

输入：

- `obs["extero"]`

目标：

- 得到 obstacle frame。
- 得到 robot relative to obstacle。
- 在该坐标系下判断推箱目标和跨沟目标。

- [ ] **Step 2: 尝试用 LiDAR 检测箱子位置**

目标：

- 输出 box relative to obstacle。
- 如果当前 LiDAR 不返回箱子，明确 `valid=False`，不污染 tracker。

- [ ] **Step 3: 使用 head depth 检测近处箱子平面**

输入：

- `obs["image"]["head_depth"]`

目标：

- 在前方 `0.3-3.0 m` 范围内检测箱子前表面。
- 输出 body-frame `box_front_x`, `box_front_y`。

实现原则：

- 不引入重依赖。
- 用 depth 阈值 + 中心区域聚类即可。
- 检测失败时回退到里程计估计。

- [ ] **Step 4: 将检测结果融合到 tracker**

融合方式：

```python
if perception_valid:
    box_x_est = 0.7 * box_x_est + 0.3 * measured_box_x
    box_y_est = 0.7 * box_y_est + 0.3 * measured_box_y
```

### Task 9: 中期 MAPush-style 训练路线

**Files:**

- Create: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/env_wrapper.py`
- Create: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/rewards.py`
- Create: `/home/hpf/atec/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/train/task_d/train_push_policy.py`

训练任务拆分：

- [ ] **Step 1: 写 IsaacLab wrapper 暴露 privileged box state**

训练时可以使用真实 box pose，submission 时换成 estimator。

- [ ] **Step 2: 固定 `policy_a.pt`，训练高层 velocity command policy**

输入：

- robot proprio summary。
- robot pose in task frame。
- box pose relative to robot。
- target box pose relative to robot。
- phase id。

输出：

- `vx, vy, wz`。

- [ ] **Step 3: reward 参考 MAPush**

包含：

- approach box reward。
- push box velocity reward。
- box target x/y reward。
- OCB 接触方向 reward。
- fall penalty。
- box bridge placement one-time reward。
- robot crossing reward。

- [ ] **Step 4: 用方案 B 的闭环控制生成初始专家**

先跑脚本生成成功/半成功轨迹，再做 imitation 或 reward shaping warm start。

---

## 5. 调参顺序

建议按以下顺序调，不要同时改太多参数。

1. 只跑 `MOVE_TO_BOX_LANE`，确认横移是否稳定。
2. 只跑 `ALIGN_BEHIND_BOX`，确认能站到箱子后方且不撞倒。
3. 只跑 `PUSH_BOX_TO_BRIDGE`，确认箱子能从 `x=-3` 推到 `x≈-0.9`。
4. 接入 score 判断，确认 `current_score` 能从 `2` 变到 `16`。
5. 只跑 `CROSS_ON_BOX` 的后半段，人工把前面阶段用脚本完成，观察是否能踩箱跨沟。
6. 串联全流程。
7. 开启 video，逐帧看失败点。

每轮记录：

- 成功最高分。
- phase 转换 step。
- robot `(x,y,yaw)` 估计。
- box `(x,y)` 估计。
- 摔倒位置。
- 卡箱位置。
- 是否拿到 `+14`。
- 是否过 `x=2.0`。

---

## 6. 关键风险与缓解

### 风险 1：policy_a 的横移/后退命令虽然训练过，但推箱时不稳定

缓解：

- 横移只用于走位，不在推箱阶段大幅横移。
- 推箱时保持 `vy` 很小。
- 后退只短时使用，必要时改为转弯绕出。

### 风险 2：箱子无真实位姿，开环估计漂移

缓解：

- 用 `current_score` 反推箱子进入目标 x range。
- 用 phase timeout 触发重定位。
- 中期加入 head depth/LiDAR 校正。

### 风险 3：箱子进了得分 x range，但没有形成可踩桥

缓解：

- 把桥目标从 `box_x=-0.9` 做网格搜索：
  - `-1.20`
  - `-1.05`
  - `-0.90`
  - `-0.75`
- 对每个点分别测试 crossing 成功率。
- 如果 y=1.6 不适合踩箱，搜索 y lane：
  - `1.2`
  - `1.4`
  - `1.6`
  - `1.8`

### 风险 4：踩箱/沟壑是 Task A policy 的分布外动作

缓解：

- 跨沟速度不要太低，避免停在边缘。
- 进入箱子前保持 yaw 对齐。
- 如果 policy 上箱子不稳，尝试更高 `vx=1.0-1.2` 的动态通过。
- 后续训练加入 box/step terrain motion 或高层 crossing phase。

### 风险 5：MAPush 代码直接迁移成本过高

缓解：

- 只迁移算法结构，不迁移 IsaacGym 环境。
- 不使用 OpenRL 作为 submission 依赖。
- 所有 submission 逻辑保持 PyTorch + 标准库。

---

## 7. 最小可提交路径

最小版本只需要完成：

1. `policy_a.pt` bridge 稳定加载。
2. `TaskDStateEstimator`。
3. phase machine。
4. waypoint velocity controller。
5. `solution_d.py` 指向 Task D controller。
6. 本地 Task D 能稳定达到至少 `score >= 16`。

提交前检查：

```bash
cd /home/hpf/atec/ATEC2026_Simulation_Challenge
PYTHONPATH=. python scripts/play_atec_task.py --task ATEC-TaskD-G1 --num_envs 1 --enable_cameras --debug
```

如果本地无法稳定 `>=16`，不要急着提交；先用 video 确认失败是“没推到箱子”、“推歪”、“箱子位置不适合踩”、“机器人跨沟失败”中的哪一种。

---

## 8. Self-review

- Placeholder scan: 文档没有未填内容或空步骤；每个阶段都有明确输入、输出和退出条件。
- Scope check: 本规划聚焦 Task D，不扩展到 Task B；MAPush 训练迁移只作为中期路线。
- Consistency check: 推荐主线始终保持 `policy_a.pt` 冻结，上层只生成 `(vx, vy, wz)`；这与现有 ATEC Task A solution 的 960 维输入格式一致。
- Risk check: 已明确 MAPush 不可直接迁移的原因，并给出短期闭环、感知增强、长期训练三层路径。
