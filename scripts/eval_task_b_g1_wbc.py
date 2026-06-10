"""Evaluate the WBC Task B G1 solution (demo.solution_task_b_g1_wbc) + record multi-view video.

Runs a full episode, prints score / phase / touched tracks, and saves a
3rd-person(follow) + head-cam + right-hand-cam video.

Usage:
  PYTHONPATH=. python scripts/eval_task_b_g1_wbc.py \
    --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless \
    --seconds 90 --out logs/videos/task_b_g1_wbc/run.mp4
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval + record WBC Task B G1 solution.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-G1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seconds", type=float, default=90.0)
parser.add_argument("--capture_every", type=int, default=3)
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--out", type=str, default="logs/videos/task_b_g1_wbc/run.mp4")
parser.add_argument("--no_video", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401,E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from demo.solution_task_b_g1_wbc import AlgSolution  # noqa: E402

_VIDEO = not args_cli.no_video
if _VIDEO:
    import imageio
    from PIL import Image, ImageDraw


def _to_u8(t):
    a = t.detach().cpu()
    if a.ndim == 4:
        a = a[0]
    return a[..., :3].clamp(0, 255).to(torch.uint8).numpy()


def _label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, len(text) * 9 + 8, 20], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 0))
    return img


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    if _VIDEO:
        try:
            env_cfg.viewer.resolution = (1280, 720)
        except Exception:
            pass
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if _VIDEO else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    robot = env.unwrapped.scene["robot"]
    sol = AlgSolution()
    obs, _ = env.reset()
    sol.reset()

    writer = None
    if _VIDEO:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        writer = imageio.get_writer(args_cli.out, fps=args_cli.fps, macro_block_size=None)

    total = 0.0
    last_phase = None
    max_steps = int(args_cli.seconds / 0.02)
    try:
        for step in range(max_steps):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                resp = sol.predicts(obs, total)
            a = torch.tensor(resp["action"], dtype=torch.float32, device=args_cli.device).view(1, -1)
            obs, reward, terminated, truncated, info = env.step(a)
            r = reward.mean().item() if isinstance(reward, torch.Tensor) else float(reward)
            sd = info.get("Step_dt") if isinstance(info, dict) else None
            sd = (sd.item() if hasattr(sd, "item") else float(sd)) if sd is not None else 0.02
            total += (r / sd) if sd else 0.0

            phase = getattr(sol.planner, "phase", "?")
            touched = sorted(getattr(sol.planner, "touched_track_ids", set()))
            if phase != last_phase or step % 100 == 0:
                bz = float(robot.data.root_pos_w[0, 2])
                print(f"[wbc-eval] t={step*0.02:5.1f}s phase={phase:12s} score={total:.2f} touched={touched} base_z={bz:.2f}", flush=True)
            last_phase = phase

            if _VIDEO and step % args_cli.capture_every == 0:
                bp = robot.data.root_pos_w[0].detach().cpu().tolist()
                try:
                    env.unwrapped.sim.set_camera_view(eye=(bp[0] - 0.4, bp[1] - 3.0, bp[2] + 1.7),
                                                      target=(bp[0], bp[1], bp[2] - 0.2))
                except Exception:
                    pass
                tp = env.render()
                if tp is not None:
                    tp_img = Image.fromarray(np.asarray(tp)[..., :3].astype(np.uint8)).resize((960, 720))
                    _label(tp_img, f"3rd {phase} score={total:.1f} touched={len(touched)} t={step*0.02:.0f}s")
                    img = obs.get("image", {})
                    head, hand = img.get("head_rgb"), img.get("ee_dual_rgb")
                    head_img = Image.fromarray(_to_u8(head)).resize((480, 360)) if head is not None else Image.new("RGB", (480, 360))
                    hand_img = Image.fromarray(_to_u8(hand)).resize((480, 360)) if hand is not None else Image.new("RGB", (480, 360))
                    _label(head_img, "head cam"); _label(hand_img, "right-hand cam")
                    canvas = Image.new("RGB", (1440, 720), (20, 20, 20))
                    canvas.paste(tp_img, (0, 0)); canvas.paste(head_img, (960, 0)); canvas.paste(hand_img, (960, 360))
                    writer.append_data(np.asarray(canvas))

            done = (bool(terminated.any()) if hasattr(terminated, "any") else bool(terminated)) or \
                   (bool(truncated.any()) if hasattr(truncated, "any") else bool(truncated))
            if done:
                print(f"[wbc-eval] done/terminated at t={step*0.02:.1f}s", flush=True)
                break
    finally:
        if writer is not None:
            writer.close()
        print(f"score: {total:.2f}  touched={sorted(getattr(sol.planner,'touched_track_ids',set()))}", flush=True)
        if _VIDEO:
            print(f"[wbc-eval] video: {os.path.abspath(args_cli.out)}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
