# Task D 提交 / 复刻指南(16 分方案)

## 方案是什么
冻结 Task A 的 G1 locomotion 策略(`policy_a.pt`,960 维 term-major 历史)当**速度跟踪器**,
上层用一个闭环状态机推箱过线:

```
warmup(站稳20步) → approach(里程计绕到箱子左后方) → push(vx0.9 顶箱子到台子根)
                                                  → advance(侧移到空旷平地,过 x=-1.4 线)
```
- 箱子被推进 `x∈[-1.4,0.7]` → **+14**
- 机器人 world x 过 `-1.4` → **+2**
- 合计 **16/36**(真值实测,见 `outputs/task_d_submit/trace.json`)

> +20(过沟到 x>3.5)拿不到:沟宽 1.3m、深 1.0m,单箱填不满、policy_a 不会跨缝爬坡。需另训过沟策略。

## 关键文件
| 文件 | 作用 |
|------|------|
| `demo/solution.py` | **自包含**入口(纯 torch+stdlib,无 atec_rl_lab 依赖)。含里程计+policy bridge+推箱状态机 |
| `demo/policy_a.pt` | 正确权重(960 维 AMP actor)。**注意**:`demo/policy.pt` 是另一套 1040 维模型,会 shape 报错 |
| `demo/server.py` `run.sh` `requirements.txt` `Dockerfile` | 评测服务/镜像构建,官方模板,基本不用改 |

## 本地复刻验证
```bash
cd /home/hpf/atec/ATEC2026_Simulation_Challenge
conda activate atec
PYTHONPATH=. python scripts/play_atec_task.py --task=ATEC-TaskD-G1 --num_envs=1 --enable_cameras
# 看画面 + 末尾 score;应到 ~16。去掉 --enable_cameras 会因相机报错,必须带。
# 想看真实 robot/box 坐标和分项得分,用探针:
PYTHONPATH=. python scripts/probe_task_d_groundtruth.py --task=ATEC-TaskD-G1 \
    --num_envs=1 --enable_cameras --num_steps 1000 --out outputs/task_d_submit
# 然后看 outputs/task_d_submit/trace.json 的 summary
```

## 提交步骤
solution 在容器里只会拿到 `solution.py` + `policy.pt`(见 Dockerfile 的 COPY)。
因为正确权重是 `policy_a.pt`,提交前必须把它复制成 `policy.pt`:

```bash
cd demo
cp policy_a.pt policy.pt          # 用正确的 960 维权重覆盖 stock policy.pt
```
(solution.py 本地优先加载 policy_a.pt;容器里只有 policy.pt,自动回退加载它。)

然后二选一(见 OVERVIEW.md 第 3 节):
- **源码上传**:上传 `demo/solution.py` + `demo/policy.pt` + `demo/requirements.txt`(文件数≤300)。
- **推送镜像**:`cd demo && docker build -t <tag> . && docker push <tag>`(Dockerfile 已配好 COPY solution.py/policy.pt)。

## 提交前自检
- [ ] `cp policy_a.pt policy.pt` 已执行(policy.pt 大小应为 2668358 字节,不是 1520504)
- [ ] `demo/solution.py` 无 `import atec_rl_lab`(已确认)
- [ ] 类 `AlgSolution`,有 `predicts(obs, current_score)` 和 `reset(**kwargs)`
- [ ] 本地 `play_atec_task.py` 跑出 ~16 分
- [ ] requirements.txt 不覆盖基础镜像核心包(torch 已在基础镜像里)
```
