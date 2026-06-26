# Task E 全流程操作手册（采集 → 训练 → 提交，手动版）

本手册记录 Task E 从采数据、训练到打包推送 docker 镜像的完整流程，含本次踩过的坑。
全部在 `atec` conda 环境、仓库根目录下执行。

## A. 数据采集 → 筛选 → 训练（产出 policy.pt）

产出当前提交 checkpoint（`act-task-e-rgb-full-order-123-20260624-1925`）的三条命令：

### A1. 采数据（full-order 1→2→3，带图像，只留成功）

```bash
python scripts/act/collect_demos_task_e.py \
  --full_order_123 \
  --optimized_grasp_flow \
  --only_success \
  --save_images \
  --num_demos 100 \
  --max_attempts 300 \
  --output_dir datasets/atec_task_e/full_order_123 \
  --headless --enable_cameras
```
产出 `datasets/atec_task_e/full_order_123/trajectory.hdf5`（含 `traj_N/obs`、`actions`、`images/rgb`）。
- `--full_order_123` 强制 object_1→2→3 顺序并自动开 `--only_success`。
- `--save_images` 需要相机（脚本会自动开 `--enable_cameras`）。
- 采数据前可选先扫抓取高度：`scripts/act/sweep_task_e_grasp_offsets.py`（见 `example.md`）。

### A2. 筛掉 0 动作（静止帧）

```bash
python scripts/act/filter_demos.py \
  --input  datasets/atec_task_e/full_order_123/trajectory.hdf5 \
  --output datasets/atec_task_e/full_order_123/trajectory_filtered.hdf5 \
  --threshold 0.001
```
判据：相邻帧 `max|action[t]-action[t-1]| < 0.001` 的帧删除（本次 40388→30181，删 ~25%）。

### A3. 训练（RGB ACT）

```bash
python scripts/act/train_task_e.py \
  --demo_path datasets/atec_task_e/full_order_123/trajectory_filtered.hdf5 \
  --include_rgb \
  --total_iters 100000 \
  --batch_size 256 \
  --log_freq 1000 \
  --save_freq 10000 \
  --seed 1 \
  --exp_name act-task-e-rgb-full-order-123-20260624-1925
```
产出 `runs/<exp_name>/checkpoints/`（`best_loss.pt`、周期 `N.pt`、`final.pt`）。
- `--exp_name` 决定 `runs/<名字>/`；不传会自动加时间戳。
- 提交打包用的是 `best_loss.pt`，但它只是“单 batch 训练 loss 最低”点；想更稳可试 `final.pt` 或后期 `N.pt`。
- 训练完把它拷成提交用权重：`cp runs/<exp_name>/checkpoints/best_loss.pt demo/policy.pt`。

---

## 0. 提交物在哪

提交需要的文件都在 `demo/` 目录（在哪个分支/worktree 改的，就用哪个 `demo/`）：

- `solution.py`  —— 自包含 Task E ACT 方案（必须定义 `AlgSolution`，不能 import `atec_rl_lab`/`scripts`）
- `policy.pt`    —— 训练好的 checkpoint（= runs/.../checkpoints/best_loss.pt 拷过来）
- `requirements.txt` —— 只列 base 镜像没有的依赖（**不要** pin `torchvision`，base 自带，见坑 3）
- `Dockerfile` / `server.py` / `run.sh` —— 平台固定 harness，**不要改**；Dockerfile 第 47-48 行已经会 COPY `solution.py` + `policy.pt`

> ⚠️ 一定用“改过新方案的那个 `demo/`”。本仓库主目录 `demo/` 和 worktree `.claude/worktrees/<name>/demo/` 是两份，别搞错。

## 1. 前置：确认 docker 可用

```bash
docker version
```

如果没装（Ubuntu）：

```bash
sudo apt-get update && sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo chmod 666 /var/run/docker.sock   # 让当前用户免 sudo 调 docker
```

## 2. 去官网生成凭证（每次提交都要，约 1 小时有效）

官网会给你三条命令（login / build / push），其中镜像 **tag 是官网生成的、和你的提交槽绑定**。直接用官网给的那一组，别自己编 tag。形如：

```
docker login --username=cr_temp_user -p '<临时token>' docker.atecup.com
docker build -t docker.atecup.com/atec2026/<repo>:<官网给的tag> .
docker push  docker.atecup.com/atec2026/<repo>:<官网给的tag>
```

## 3. 执行三步

```bash
cd <你的>/demo            # ★ 必须在 demo/ 里，下面 build 末尾的 . 就是这个目录
docker login --username=cr_temp_user -p '<token>' docker.atecup.com
docker build -t docker.atecup.com/atec2026/<repo>:<tag> .
docker push  docker.atecup.com/atec2026/<repo>:<tag>
```

- 第一次 build 会拉 base 镜像 `ac2-registry...pytorch:2.7.1-cuda12.8`（多 GB，几分钟）；之后有缓存，几十秒。
- 成功标志：push 末尾出现一行 `....: digest: sha256:... size: ...`。

## 4. 验证推送成功

```bash
docker manifest inspect docker.atecup.com/atec2026/<repo>:<tag> | head
```

能打印出 manifest（schemaVersion/layers）就说明镜像确实在仓库里。然后回**官网点提交/确认**（一般自动识别刚推的镜像；若有输入框就把 tag 粘进去）。

## 5. ★ 最大的坑：上传带宽 vs 1 小时 token

镜像约 **20GB**（几乎全是 base 镜像，瘦不下来）。算一笔账：

- 上行只有 ~9 Mbit/s（~1.1 MB/s）时，20GB 要 ~2-3 小时 > 1h token，**必失败**。
- 而且 docker push 默认 5 层并行，最大那个 ~3GB 的层分不到带宽、永远传不完，token 一过期从头来 → **永远收敛不了**。

应对（按优先级）：

1. **换快上行**：手机热点（5G 上行常有 50-60 Mbit/s）、公司/学校网、或云服务器。本次家宽 9Mbit 失败 → iPhone 热点 63Mbit，11 分钟传完。
   - 注意：插着网线时，有线默认路由(metric 100)会盖过热点(metric 600)，**要拔网线**热点才生效。`ip route show default` 看当前走哪条。
2. **断点续传**：push 中断后，重新 `docker login`（换新 token）+ 重跑 `docker push` 即可。已传完的层会显示 `Layer already exists` 跳过；**只有正在传的那层会重头**。即使 tag 变了，只要 repo 相同，blob 也复用。
3. **串行上传**（慢网兜底）：`/etc/docker/daemon.json` 加 `{"max-concurrent-uploads":1}` 再 `sudo systemctl restart docker`，让大层吃满全速、单层 ~49 分钟能卡进一个 1h token。

## 6. 常见坑清单

- **build 目录错**：必须 `cd demo` 再 build，否则 `COPY solution.py` 找不到文件。
- **solution.py 不自包含**：不能 import `atec_rl_lab`/`scripts`/`train_task_e`；模型代码要内嵌。检查：`grep -nE 'atec_rl_lab|scripts|train_task_e' demo/solution.py` 应无匹配。
- **torchvision pin 触发 torch 降级**：base 自带 `torch 2.7.1.8+cu128`/`torchvision`，requirements 里**别写 torchvision**，否则 pip 会卸载 vendor torch 换 PyPI 版（镜像还大 8GB）。
- **token 过期**：~1h。传不完就重新生成 + 重 login + 重 push（续传）。
- **kill 进程别用 `pkill -f 'docker push'`**：命令行自带这串会自杀。用 `ps -eo pid,args | grep '[d]ocker push'` 拿 PID 再 kill，或直接记下 PID。
- **GPU 进程残留**：本地 play_atec_task / Isaac 跑完要清，`conda run`+timeout 不杀子进程，用 `/home/air/miniconda3/envs/atec/bin/python` 直连 + `timeout -s KILL`。

## 一句话流程

`cd demo` → `docker login`（官网 token）→ `docker build -t <官网tag> .` → `docker push <官网tag>` → 见 `digest: sha256:` → 官网点提交。上传慢就**换热点/快网 + 重 push 续传**。
