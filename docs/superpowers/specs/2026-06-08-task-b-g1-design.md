# Task B G1 视觉搜索 + 接触/推扫方案设计

日期：2026-06-08  
状态：设计稿，待用户 review

## 背景

ATEC2026 Task B 是 L1 垃圾拾取与投放任务。场景为 20×20m 工作区，18 个目标物体，满分 36 分：每个物体“拾取”+1，“正确放置”+1。

当前仓库没有 Task B baseline。已有可复用资产主要是 G1 Task A/D 的 locomotion policy wrapper 与速度命令接口经验。用户选择 Task B 使用 G1，并明确要求按“物体位置不可预测”处理，因此方案不能依赖固定 seed 或硬编码物体坐标。

从环境源码可知：

- 目标区域中心是 `(-3.0, -10.0)`，半径 `1.0`。
- 物体初始分布在 `x,y ∈ [-15.0, -5.0]`。
- 物体共 18 个：6 个 sugar box、6 个 mustard bottle、6 个 banana。
- G1 的拾取判分不是严格抓取，而是 `left_hand_base_link` 或 `right_hand_base_link` 距离物体 root 小于等于 `0.20m` 即一次性得分。
- 放置判分是物体 root 进入目标圆且 z 在 `[0.0, 0.5]`。

这些规则允许先做“视觉搜索 + 末端接触 + 推扫”的可提交 baseline，再逐步升级为更高分方案。

## 目标

### 第一阶段目标

建立一个 G1 Task B baseline，能在未知物体位置下自主搜索、靠近物体、触发若干接触分，并尝试把部分物体推入目标区。

阶段性里程碑：

1. 能保存并解释 Task B G1 的 RGB-D 观测。
2. G1 能在 Task B 工作区按 waypoint 巡航，不因上层控制明显摔倒。
3. 能检测或定位至少一个物体，并靠近它。
4. 能通过 G1 手部末端靠近触发至少一个接触分。
5. 能稳定拿到多个接触分。
6. 能把目标区附近或路径简单的物体推/扫进目标圆，获得少量放置分。

### 第二阶段目标

在第一阶段 baseline 可得分后，提升放置成功率和总分：

- 改进 RGB-D 物体检测与跟踪。
- 优化物体排序和路径规划。
- 训练或搜索局部 interaction policy，用于更稳定地推扫/半抓取物体。
- 对 sugar box、mustard、banana 分别调不同的局部策略。

## 非目标

第一阶段不做以下内容：

- 不直接训练端到端 RGB-D 到全身动作的大策略。
- 不要求真实抓住并搬运每个物体。
- 不依赖固定 seed、固定物体坐标或本地 env.scene 真值作为提交时输入。
- 不重写 G1 底层 locomotion；优先复用现有速度跟踪策略。

## 推荐架构

Task B G1 solution 拆成五个模块：

```text
AlgSolution
  ├─ G1VelocityPolicyBridge
  ├─ DeadReckoningOdometry
  ├─ TaskBRgbdPerception
  ├─ TaskBPlanner
  └─ LocalObjectInteraction
```

### G1VelocityPolicyBridge

复用现有 G1 Task A/D 的 960 维 term-major history policy wrapper。输入期望速度命令 `(vx, vy, yaw_rate)`，输出完整 G1 action。

设计要求：

- 保持和训练时一致的 observation history layout。
- 支持 reset 时清空 history。
- 默认使用保守速度，降低物体交互阶段摔倒风险。
- 局部交互模块可对手臂/手部关节做小范围 override，但腿部仍由 locomotion policy 控制。

### DeadReckoningOdometry

从 `obs["proprio"]` 的 base linear velocity、base angular velocity、projected gravity 积分估计机器人 `(x, y, yaw)`。

用途：

- 执行固定搜索 waypoint。
- 将视觉检测结果融合到粗略世界坐标。
- 估计机器人到目标区、物体、巡航点的关系。

第一版接受一定漂移，通过视觉重新观测和保守 waypoint 修正行为。

### TaskBRgbdPerception

从 `obs["image"]` 中读取 head RGB-D，检测 Task B 物体，输出候选物体相对位置。

第一版使用传统 RGB-D 检测：

- RGB 颜色/纹理阈值提取 banana、mustard、sugar box 候选区域。
- depth 过滤地面附近小物体。
- 连通域/轮廓筛选去掉明显误检。
- 通过相机内参反投影到相机坐标，再转换到机器人局部或粗略世界坐标。
- 多帧融合维护 object track，减少瞬时误检。

如果传统检测不稳定，第二阶段再用合成截图训练轻量 detector。

### TaskBPlanner

负责搜索、目标选择、状态机转移和速度命令生成。

核心状态：

- `search`：按固定栅格路线巡航并扫描物体。
- `approach_object`：导航到已检测物体附近。
- `touch_object`：让手部末端靠近物体 root，触发接触分。
- `push_to_goal`：将物体沿目标方向推/扫。
- `verify_or_next`：根据 `current_score` 变化和视觉结果确认是否继续当前物体或切换目标。

搜索路线覆盖物体可能出现区域 `[-15,-5] × [-15,-5]`，例如：

```text
(-10,-10)
→ 扫描行 y=-14, x=-14 到 -6
→ 扫描行 y=-12, x=-6 到 -14
→ 扫描行 y=-10, x=-14 到 -6
→ 扫描行 y=-8,  x=-6 到 -14
→ 扫描行 y=-6,  x=-14 到 -6
```

导航策略：

- 转向对准 waypoint 或物体。
- 以低速前进为主，谨慎使用横移。
- 视觉发现物体后打断巡航，优先处理近距离、易接触、靠近目标区的物体。

### LocalObjectInteraction

处理最后 0.5m 到 1.5m 的接触和推扫动作。

第一版策略：

- 固定一只手或双手的低位前伸姿态。
- 机器人低速接近物体，使 `left_hand_base_link` 或 `right_hand_base_link` 进入 20cm 判分范围。
- 如果 `current_score` 增加，则标记当前 track 已接触。
- 对目标区附近或方向简单的物体，保持手臂/身体低速前推。
- 推扫方向为物体当前位置指向 `(-3.0, -10.0)`。

局部交互以稳定为第一优先级。若手臂 override 破坏步态，则手臂只在低速或停止时伸出。

## 数据流

每个 `predicts(obs, current_score)` 执行以下流程：

```python
proprio = obs["proprio"]
images = obs.get("image", {})

pose = odometry.update(proprio)
detections = perception.update(images, pose)
phase, cmd, arm_pose = planner.step(pose, detections, current_score)

action = locomotion_bridge.act(proprio, cmd)
action = local_interaction.apply_arm_override(action, phase, arm_pose)

return {"action": action, "giveup": False}
```

`current_score` 用于检测接触或放置是否成功。状态机必须记录上一步分数，计算 score delta，并将成功的 object track 标记为已接触或已放置。

## 文件结构

为避免影响当前 Task D solution，先新增 Task B 专用文件：

```text
demo/
  solution_task_b_g1.py
  policy_a.pt
scripts/
  probe_task_b_g1_camera.py
  probe_task_b_g1_objects.py
  eval_task_b_g1.py
tests/task_b/
  test_solution_task_b_g1.py
```

职责：

- `solution_task_b_g1.py`：Task B G1 可提交入口，包含上述五个模块。
- `probe_task_b_g1_camera.py`：启动 `ATEC-TaskB-G1 --enable_cameras`，保存 RGB-D、打印 image keys、确认相机视野和深度单位。
- `probe_task_b_g1_objects.py`：本地调试用，读取 env.scene 真值对比视觉检测误差；不进入提交文件。
- `eval_task_b_g1.py`：跑完整或短时评测，打印 phase、score、检测数量、耗时。
- `tests/task_b/test_solution_task_b_g1.py`：不启动 Isaac 的单元测试，覆盖 proprio 解析、policy input 维度、状态机和几何逻辑。

## 测试计划

### 1. 无 Isaac 单元测试

覆盖：

- proprio 维度解析。
- G1 policy 输入 960 维拼接。
- odometry yaw wrap 与速度积分。
- waypoint 选择和状态机转移。
- score delta 对 object track 标记的影响。

### 2. 相机探针

运行 Task B G1，保存 head RGB-D 和必要 metadata：

- `obs["image"]` key。
- RGB/depth shape、dtype、数值范围。
- 物体在图像中的实际颜色、大小、可见距离。
- 视觉检测结果与 env.scene 物体真值的误差。

### 3. 巡航测试

不开物体交互，只执行搜索 waypoint：

- 检查 G1 是否稳定行走。
- 检查 odometry 漂移是否可接受。
- 检查 fixed route 是否能覆盖大部分物体区域。

### 4. Oracle 单物体接触测试

本地脚本临时使用 env.scene 真值选最近物体：

- 验证靠近策略。
- 验证手部固定姿态能否触发 +1。
- 记录成功距离、速度、姿态参数。

该 oracle 只用于验证控制，不进入提交 solution。

### 5. 视觉单物体接触测试

将 oracle 替换为 RGB-D 检测：

- 至少找到一个物体。
- 导航到物体附近。
- 触发一次接触分。

### 6. 完整 baseline 测试

运行：

```bash
PYTHONPATH=. python scripts/play_atec_task.py \
  --task ATEC-TaskB-G1 \
  --enable_cameras \
  --debug
```

记录：

- 总分。
- 接触分和放置分来源。
- 每个 phase 的耗时。
- 失败原因：摔倒、检测丢失、推扫失败、超时等。

## 风险与缓解

### G1 locomotion 在物体交互时不稳定

缓解：低速接触；先拿接触分，减少长距离推物；手臂 override 只在低速或停止时启用。

### RGB-D 坐标转换不明确

缓解：先写 camera/object probe；用 env.scene 真值离线标定误差；提交时只保留基于观测的检测逻辑。

### 传统视觉误检或漏检

缓解：多帧融合；结合 depth 和几何约束；优先近距离候选；必要时第二阶段训练轻量 detector。

### 推扫效率低

缓解：第一阶段以接触分为主；只对靠近目标区或方向简单的物体尝试放置；长期用局部策略训练提高成功率。

### 里程计漂移

缓解：搜索路线使用相对宽松 waypoint；检测到物体后以视觉闭环为主；不要依赖长时间精确世界坐标。

## 实施顺序

1. 新增 Task B G1 camera/object 探针，弄清观测和坐标转换。
2. 新增 `demo/solution_task_b_g1.py` skeleton，复用 G1 locomotion bridge，先巡航不摔。
3. 做 oracle 接触 baseline，验证接触得分和手臂姿态。
4. 实现 RGB-D 传统检测 baseline，替换 oracle。
5. 跑完整 Task B G1 baseline，目标是稳定获得多个接触分。
6. 加入目标区附近推扫逻辑，争取获得部分放置分。
7. 根据失败日志选择升级方向：检测器训练、局部交互策略训练或物体排序优化。

## 验收标准

第一阶段完成的最低验收标准：

- `solution_task_b_g1.py` 能在 `ATEC-TaskB-G1` 下启动并返回合法 33 维 action。
- camera probe 能保存 Task B G1 RGB-D，并说明 image key、shape、depth 单位/范围。
- 无 Isaac 单元测试通过。
- 本地至少一次完整或短时运行中，G1 能自主检测/接近物体并触发接触分。

推荐继续迭代的验收标准：

- 多次本地运行平均能拿到多个接触分。
- 至少一次成功把物体推入目标圆获得放置分。
- 日志能解释每个得分来源和主要失败原因。
