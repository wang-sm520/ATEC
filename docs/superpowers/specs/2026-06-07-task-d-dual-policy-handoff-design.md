# Task D 双策略接力（走路 → 爬楼）设计

日期：2026-06-07
作用域：仅 `demo/solution.py`

## 背景

Task D 当前方案 `_WallPushController` 用冻结的 G1 行走策略 `policy_a.pt`（作为速度跟踪器，
经 `_G1VelocityPolicyBridge` 把速度指令转成关节动作）完成「靠墙推箱入坑」7 段机动，
最后在 `forward` 相位一直发前进速度走向终点。

但箱子填入坑后，机器人要越过箱子/台阶才能继续到终点，纯行走策略难以爬越。已新训练出
爬楼策略 `demo/policy_climb.pt`（v4 楼梯地形 AMP run，2026-06-04_14-26-56，model_73200 导出）。

## 目标

让 solution 在前段照常用 `policy_a` 走路+推箱；**一旦状态机进入 `forward` 相位**
（判定箱子已入坑、机器人到达坑边 `rx ≥ PIT_EDGE_X`），bridge 立刻切换到 `policy_climb`，
继续发前进速度指令直到终点。单向切换，不再切回。

## 关键事实（已验证）

- `policy_climb.pt` 与该爬楼 run 导出的 `exported/policy.pt` **字节完全一致**（md5 相同）。
- `policy_a.pt` 与 `policy_climb.pt` 输入/输出签名一致：**960 维 obs → 29 维 action**。
- 因此 `_G1VelocityPolicyBridge` 的历史缓冲布局、action-scale、速度指令接口对两者通用。
- 10 步本体感受历史缓冲是与策略无关的原始 proprio（ang_vel/cmd/gravity/jp/jv/last_act），
  切换时缓冲依然有效 → **坑边切换无冷启动踉跄**。

## 设计

全部改动在 `demo/solution.py`。

### 1. 策略路径发现

在现有 `_POLICY_PATH`（`policy_a.pt` → `policy.pt`）之外，新增 `_CLIMB_POLICY_PATH`，
查找 `demo/policy_climb.pt`；找不到则为 `None`。

### 2. `_G1VelocityPolicyBridge` 改为双策略

- `__init__(self, policy_path, climb_path=None, device="cuda")`
- 内部保存 `self._paths = {"walk": policy_path, "climb": climb_path}`，
  `self._modules = {"walk": None, "climb": None}`，`self._active = "walk"`。
- `select(self, name)`：把 `self._active` 切到 `name`；**若 `self._paths[name]` 为 `None` 则空操作**
  （优雅降级——没带 climb 权重时退化为只用 walk）。
- `_load(self)`：惰性加载并返回 `self._active` 对应模块。
- `reset(self)`：清空共享历史缓冲（不变），并把 `self._active` 复位回 `"walk"`。
- `act()` 主体逻辑不变（仍用同一个共享缓冲）。

### 3. `AlgSolution`

- 构造时传入 `climb_path=_CLIMB_POLICY_PATH`。
- `predicts` 中 `cmd = self.controller.update(row)` 之后、`act` 之前：
  ```python
  if self.controller.phase == "forward":
      self.bridge.select("climb")
  ```

## 优雅降级

若提交容器未打包 `policy_climb.pt`，`select("climb")` 为空操作，行为与现状完全一致
（只用 `policy_a` 走 `forward`）。

## 打包注意

提交容器现在需额外打包 `demo/policy_climb.pt`。现有文件头注释只提到发 `policy.pt`，
需补充说明 climb 权重。

## 验证

- **静态**（不需要 Isaac，在 `amp` conda 环境）：实例化 `AlgSolution`，喂合成 obs 驱动
  状态机从早期相位走到 `forward`，断言：
  1. 在 `forward` 之前 `bridge._active == "walk"`；
  2. 进入 `forward` 后 `bridge._active == "climb"`；
  3. 两段 `predicts` 返回的 `action` 均为 29 维；
  4. `climb_path=None` 时进入 `forward` 也保持 `walk`（降级路径）。
- **仿真**（Isaac，由用户运行）：跑 task D 评测 / `scripts/probe_task_d_wallpush.py`，
  确认机器人确实爬越并走到终点。

## 非目标

- 不改 `_WallPushController` 的相位/几何/控制律。
- 不改打包脚本（仅在代码注释中提示需带 climb 权重）。
- 不重训任何策略。
