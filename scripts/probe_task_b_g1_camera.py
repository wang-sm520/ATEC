"""Capture Task B G1 head RGB-D frames and print observation/sensor metadata.

Usage:
  PYTHONPATH=. python scripts/probe_task_b_g1_camera.py \
    --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --out outputs/task_b_g1_camera
"""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe Task B G1 camera observations.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-G1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=80)
parser.add_argument("--capture_every", type=int, default=20)
parser.add_argument("--out", type=str, default="outputs/task_b_g1_camera")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.capture_every <= 0:
    parser.error("--capture_every must be a positive integer")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

try:
    from PIL import Image  # noqa: E402
except Exception:  # pragma: no cover - optional debug dependency
    Image = None


def _tensor_summary(x):
    t = x.detach().float().cpu()
    finite = torch.isfinite(t)
    if bool(finite.any()):
        vals = t[finite]
        return {
            "shape": list(t.shape),
            "dtype": str(x.dtype),
            "min": float(vals.min().item()),
            "max": float(vals.max().item()),
            "mean": float(vals.mean().item()),
        }
    return {"shape": list(t.shape), "dtype": str(x.dtype), "min": None, "max": None, "mean": None}


def _save_rgb_png(rgb, path):
    if Image is None:
        return False
    arr = rgb.detach().cpu()
    if arr.ndim == 4:
        arr = arr[0]
    arr = arr[..., :3].clamp(0, 255).to(torch.uint8).numpy()
    Image.fromarray(arr).save(path)
    return True


def _save_depth_png(depth, path):
    if Image is None:
        return False
    d = depth.detach().float().cpu()
    if d.ndim == 4:
        d = d[0]
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    finite = torch.isfinite(d) & (d > 0)
    out = torch.zeros_like(d)
    if bool(finite.any()):
        vals = d[finite]
        out[finite] = ((d[finite] - vals.min()) / (vals.max() - vals.min() + 1e-6) * 255.0)
    Image.fromarray(out.clamp(0, 255).to(torch.uint8).numpy()).save(path)
    return True


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    if Image is None:
        print("[probe] Pillow is unavailable; PNG snapshots will be skipped.")
    if args_cli.num_envs > 1:
        print("[probe] PNG snapshots save env 0 only; .pt tensors contain all envs.")

    env = None
    try:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        env = gym.make(args_cli.task, cfg=env_cfg)
        obs, _ = env.reset()
        summaries = []
        print(f"[probe] obs groups: {list(obs.keys())}")
        print(f"[probe] image keys: {list(obs.get('image', {}).keys())}")

        for step in range(args_cli.num_steps):
            action_dim = int(env.unwrapped.action_space.shape[-1])
            actions = torch.zeros((args_cli.num_envs, action_dim), dtype=torch.float32, device=args_cli.device)
            obs, reward, terminated, truncated, info = env.step(actions)
            done = bool(terminated.any()) or bool(truncated.any())

            if step % args_cli.capture_every == 0:
                image_obs = obs.get("image", {})
                summary = {"step": step, "keys": list(image_obs.keys()), "terms": {}}
                for key, value in image_obs.items():
                    if hasattr(value, "shape"):
                        summary["terms"][key] = _tensor_summary(value)
                summaries.append(summary)
                print(json.dumps(summary, indent=2))
                if "head_rgb" in image_obs:
                    _save_rgb_png(image_obs["head_rgb"], os.path.join(args_cli.out, f"head_rgb_{step:04d}.png"))
                    torch.save(image_obs["head_rgb"].detach().cpu(), os.path.join(args_cli.out, f"head_rgb_{step:04d}.pt"))
                if "head_depth" in image_obs:
                    _save_depth_png(image_obs["head_depth"], os.path.join(args_cli.out, f"head_depth_{step:04d}.png"))
                    torch.save(image_obs["head_depth"].detach().cpu(), os.path.join(args_cli.out, f"head_depth_{step:04d}.pt"))
            if done:
                break

        with open(os.path.join(args_cli.out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"[probe] wrote {len(summaries)} captures to {args_cli.out}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
