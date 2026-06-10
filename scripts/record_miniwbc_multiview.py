"""Record the mini WBC running in ATEC: following 3rd-person + head cam + right-hand cam.

Command schedule: stand -> walk -> stop -> squat + reach right hand down.

Usage:
  PYTHONPATH=. python scripts/record_miniwbc_multiview.py \
    --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless \
    --seconds 30 --out logs/videos/miniwbc/run.mp4
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record mini WBC multi-view in ATEC.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-G1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--capture_every", type=int, default=2)
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--mini_root", type=str, default="/home/hpf/atec/ATEC2026_Simulation_Challenge/mini")
parser.add_argument("--out", type=str, default="logs/videos/miniwbc/run.mp4")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401,E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from mini_wbc_lib import MiniWBC, DEFAULT_LEFT_HAND  # noqa: E402


def to_uint8(t):
    a = t.detach().cpu()
    if a.ndim == 4:
        a = a[0]
    return a[..., :3].clamp(0, 255).to(torch.uint8).numpy()


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, len(text) * 9 + 8, 20], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 0))
    return img


def main():
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    try:
        env_cfg.viewer.resolution = (1280, 720)
    except Exception:
        pass
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    robot = env.unwrapped.scene["robot"]
    wbc = MiniWBC(os.path.join(args_cli.mini_root, "model/0116/policy18.onnx"))
    obs, _ = env.reset()
    wbc.reset()

    writer = imageio.get_writer(args_cli.out, fps=args_cli.fps, macro_block_size=None)
    max_steps = int(args_cli.seconds / 0.02)
    written = 0
    try:
        for step in range(max_steps):
            if not simulation_app.is_running():
                break
            t = step * 0.02
            if t < 3.0:
                vel, h, rh, phase = [0.0, 0.0, 0.0], 0.75, [0.3, -0.2, 0.0, 1, 0, 0, 0], "stand"
            elif t < 9.0:
                vel, h, rh, phase = [0.4, 0.0, 0.0], 0.75, [0.3, -0.2, 0.0, 1, 0, 0, 0], "walk"
            elif t < 12.0:
                vel, h, rh, phase = [0.0, 0.0, 0.0], 0.75, [0.3, -0.2, 0.0, 1, 0, 0, 0], "stop"
            else:
                vel, h, rh, phase = [0.0, 0.0, 0.0], 0.45, [0.45, -0.15, -0.25, 1, 0, 0, 0], "squat+reach"

            row = obs["proprio"]
            action = wbc.act(row[0] if hasattr(row, "shape") and len(row.shape) >= 2 else row,
                             vel, h, [0, 0, 0], DEFAULT_LEFT_HAND, rh)
            a = torch.tensor(action, dtype=torch.float32, device=args_cli.device).view(1, -1)
            obs, _, terminated, truncated, _ = env.step(a)

            # follow camera
            bp = robot.data.root_pos_w[0].detach().cpu().tolist()
            try:
                env.unwrapped.sim.set_camera_view(eye=(bp[0] - 0.5, bp[1] - 3.2, bp[2] + 1.8),
                                                  target=(bp[0], bp[1], bp[2] - 0.2))
            except Exception:
                pass

            if step % args_cli.capture_every == 0:
                tp = env.render()
                if tp is None:
                    continue
                tp_img = Image.fromarray(np.asarray(tp)[..., :3].astype(np.uint8)).resize((960, 720))
                label(tp_img, f"3rd  {phase}  base_z={bp[2]:.2f}  t={t:.0f}s")
                img = obs.get("image", {})
                head = img.get("head_rgb")
                hand = img.get("ee_dual_rgb")
                head_img = Image.fromarray(to_uint8(head)).resize((480, 360)) if head is not None else Image.new("RGB", (480, 360))
                hand_img = Image.fromarray(to_uint8(hand)).resize((480, 360)) if hand is not None else Image.new("RGB", (480, 360))
                label(head_img, "head cam"); label(hand_img, "right-hand cam")
                canvas = Image.new("RGB", (1440, 720), (20, 20, 20))
                canvas.paste(tp_img, (0, 0)); canvas.paste(head_img, (960, 0)); canvas.paste(hand_img, (960, 360))
                writer.append_data(np.asarray(canvas))
                written += 1

            if (bool(terminated.any()) if hasattr(terminated, "any") else bool(terminated)) or \
               (bool(truncated.any()) if hasattr(truncated, "any") else bool(truncated)):
                print(f"[rec] terminated at t={t:.1f}s")
                break
    finally:
        writer.close()
        print(f"[rec] wrote {written} frames to {os.path.abspath(args_cli.out)}")
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
