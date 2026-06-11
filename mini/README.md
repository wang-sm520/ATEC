# G1 MuJoCo Mini Runner

这是从当前 `g1_loco_manip.py` 默认入口整理出的最小运行包。默认入口是 `main()`，不包含轨迹跟踪、PID 可视化、评估脚本和历史实验输出。

## 文件结构

```text
mini/
  g1_loco_manip.py                  # 默认运行入口
  g1_robot.py                       # G1 MuJoCo + ONNX 推理
  command_gui.py                    # tkinter 命令滑块 GUI
  mujoco_video_recorder.py          # 可选视频/数据录制
  utils.py                          # 关节顺序转换和观测工具
  configs/g1_loco_manip.yaml        # 默认运行配置
  g1_description/
    g1_29dof_with_hand_rev_1_0.xml  # 实际加载的 hand 版 XML
    meshes/*.STL                    # XML 引用的 49 个网格
  model/0116/policy18.onnx          # 实际加载的 ONNX 策略
```

## 环境

建议使用虚拟环境：

```bash
cd mini
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GUI 使用 Python 标准库 `tkinter`。如果系统缺少 Tk 支持，需要安装系统包，例如 Ubuntu/Debian:

```bash
sudo apt-get install python3-tk
```

## 运行

```bash
cd mini
python g1_loco_manip.py
```

运行后会打开 MuJoCo viewer 和命令控制 GUI。按 MuJoCo viewer 内的 `R` 键开始/停止录制，视频和 `MjData` 日志会写入 `mini/log/session_*/`。

## 当前默认资源

当前完整工程的 `g1_robot.py` 实际硬编码加载：

- `g1_description/g1_29dof_with_hand_rev_1_0.xml`
- `model/0116/policy18.onnx`

因此 mini 包里的 YAML 已改为这两个真实默认资源，而不是原 YAML 中旧的 `model/model_4900.pt` 和 `g1_29dof_rev_1_0.xml`。
