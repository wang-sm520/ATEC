"""录制 Task B G1 一局：第三人称(自动框全场) + 头相机 + 右手相机，合成单个视频。

Usage:
  PYTHONPATH=. python scripts/record_task_b_g1_multiview.py \
    --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless \
    --seconds 60 --out logs/videos/task_b_g1_multiview/run.mp4
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record Task B G1 multi-view video.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-G1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seconds", type=float, default=60.0)
parser.add_argument("--capture_every", type=int, default=2)
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--out", type=str, default="logs/videos/task_b_g1_multiview/run.mp4")
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
from demo.solution_task_b_g1 import AlgSolution  # noqa: E402


def to_uint8_img(t):
    a = t.detach().cpu()
    if a.ndim == 4:
        a = a[0]
    a = a[..., :3].clamp(0, 255).to(torch.uint8).numpy()
    return a


def label(img_pil, text):
    d = ImageDraw.Draw(img_pil)
    d.rectangle([0, 0, len(text) * 9 + 8, 20], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 0))
    return img_pil


def main():
    out_dir = os.path.dirname(os.path.abspath(args_cli.out))
    os.makedirs(out_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    # high-res viewport for the third-person render
    try:
        env_cfg.viewer.resolution = (1280, 720)
    except Exception:
        pass

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    scene = env.unwrapped.scene
    robot = scene["robot"]

    obs, _ = env.reset()
    sol = AlgSolution()
    sol.reset()

    # auto-frame: bbox over robot + all objects
    pts = [robot.data.root_pos_w[0].detach().cpu().tolist()]
    for oi in range(1, 19):
        try:
            pts.append(scene[f"object_{oi}"].data.root_pos_w[0].detach().cpu().tolist())
        except Exception:
            pass
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    extent = max(max(xs) - min(xs), max(ys) - min(ys), 6.0)
    eye = (cx, cy - 1.1 * extent - 3.0, 0.85 * extent + 7.0)
    target = (cx, cy, 0.0)
    try:
        env.unwrapped.sim.set_camera_view(eye=eye, target=target)
        print(f"[rec] camera eye={eye} target={target} extent={extent:.1f}")
    except Exception as exc:
        print("[rec] set_camera_view failed:", exc)

    max_steps = int(args_cli.seconds / 0.02)
    writer = imageio.get_writer(args_cli.out, fps=args_cli.fps, macro_block_size=None)
    total = 0.0
    last_phase = None
    written = 0
    try:
        for step in range(max_steps + 5):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                resp = sol.predicts(obs, total)
            actions = torch.tensor(resp["action"], dtype=torch.float32, device=args_cli.device).view(1, -1)
            obs, reward, terminated, truncated, info = env.step(actions)
            r = reward.mean().item() if isinstance(reward, torch.Tensor) else float(reward)
            sd = info.get("Step_dt") if isinstance(info, dict) else None
            sd = (sd.item() if hasattr(sd, "item") else float(sd)) if sd is not None else 0.02
            total += (r / sd) if sd else 0.0
            phase = getattr(sol.planner, "phase", "?")
            if phase != last_phase:
                print(f"[rec] step={step} phase={phase} score={total:.2f}")
            last_phase = phase

            if step % args_cli.capture_every == 0:
                # third-person
                tp = env.render()
                if tp is None:
                    continue
                tp_img = Image.fromarray(np.asarray(tp)[..., :3].astype(np.uint8)).resize((960, 720))
                label(tp_img, f"3rd  phase={phase} score={total:.1f} t={step*0.02:.0f}s")
                img = obs.get("image", {})
                head = img.get("head_rgb")
                hand = img.get("ee_dual_rgb")
                head_img = Image.fromarray(to_uint8_img(head)).resize((480, 360)) if head is not None else Image.new("RGB", (480, 360))
                hand_img = Image.fromarray(to_uint8_img(hand)).resize((480, 360)) if hand is not None else Image.new("RGB", (480, 360))
                label(head_img, "head cam"); label(hand_img, "right-hand cam")
                canvas = Image.new("RGB", (960 + 480, 720), (20, 20, 20))
                canvas.paste(tp_img, (0, 0))
                canvas.paste(head_img, (960, 0))
                canvas.paste(hand_img, (960, 360))
                writer.append_data(np.asarray(canvas))
                written += 1

            done = (bool(terminated.any()) if hasattr(terminated, "any") else bool(terminated)) or \
                   (bool(truncated.any()) if hasattr(truncated, "any") else bool(truncated))
            if done:
                print(f"[rec] done at step={step}")
                break
    finally:
        writer.close()
        print(f"[rec] wrote {written} frames to {os.path.abspath(args_cli.out)} ; final score {total:.2f}")
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
