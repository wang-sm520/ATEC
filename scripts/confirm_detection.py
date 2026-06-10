"""确认 G1 能识别垃圾位置：让 WBC 原地慢转扫视，把感知检测框画到头相机上，
并与 env.scene 真值比对（检测世界坐标 vs 最近真值物体的误差）。

Usage:
  PYTHONPATH=. python scripts/confirm_detection.py \
    --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless \
    --steps 500 --out outputs/confirm_detection
"""

from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Confirm garbage detection in ATEC-TaskB-G1.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-G1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--out", type=str, default="outputs/confirm_detection")
parser.add_argument("--mini_root", type=str, default="/home/hpf/atec/ATEC2026_Simulation_Challenge/mini")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401,E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from demo.mini_wbc import MiniWBC, DEFAULT_LEFT_HAND, DEFAULT_RIGHT_HAND  # noqa: E402
from demo.solution_task_b_g1 import DeadReckoningOdometry, TaskBRgbdPerception  # noqa: E402


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    scene = env.unwrapped.scene
    robot = scene["robot"]
    wbc = MiniWBC(os.path.join(args_cli.mini_root, "model/0116/policy18.onnx"))
    perception = TaskBRgbdPerception()
    odom = DeadReckoningOdometry(dt=0.02, x0=-10.0, y0=-10.0)
    obs, _ = env.reset()
    wbc.reset()

    def true_objects():
        out = []
        for oi in range(1, 19):
            try:
                p = scene[f"object_{oi}"].data.root_pos_w[0].detach().cpu().tolist()
                out.append((f"object_{oi}", p[0], p[1]))
            except Exception:
                pass
        return out

    errors = []
    detected_track_ids = set()
    saved = 0
    for step in range(args_cli.steps):
        if not simulation_app.is_running():
            break
        # WBC: stand and slowly rotate in place to scan
        row = obs["proprio"]
        row0 = row[0] if hasattr(row, "shape") and len(row.shape) >= 2 else row
        action = wbc.act(row0, [0.0, 0.0, 0.5], 0.75, [0.0, 0.0, 0.0], DEFAULT_LEFT_HAND, DEFAULT_RIGHT_HAND)
        a = torch.tensor(action, dtype=torch.float32, device=args_cli.device).view(1, -1)
        obs, _, term, trunc, _ = env.step(a)

        pose = odom.update(row0)
        dets = perception.update(obs.get("image", {}), pose)
        objs = true_objects()
        for d in dets:
            detected_track_ids.add(d.track_id)
            # nearest true object
            best = min(objs, key=lambda o: math.hypot(d.world_x - o[1], d.world_y - o[2])) if objs else None
            if best is not None:
                err = math.hypot(d.world_x - best[1], d.world_y - best[2])
                errors.append(err)

        if dets and saved < 6 and step % 12 == 0:
            rgb = obs.get("image", {}).get("head_rgb")
            if rgb is not None:
                arr = rgb.detach().cpu()
                if arr.ndim == 4:
                    arr = arr[0]
                img = Image.fromarray(arr[..., :3].clamp(0, 255).to(torch.uint8).numpy())
                dr = ImageDraw.Draw(img)
                for d in dets:
                    x0, y0, x1, y1 = d.bbox
                    dr.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
                    best = min(objs, key=lambda o: math.hypot(d.world_x - o[1], d.world_y - o[2])) if objs else None
                    err = math.hypot(d.world_x - best[1], d.world_y - best[2]) if best else -1
                    dr.text((x0, max(0, y0 - 12)), f"({d.world_x:.1f},{d.world_y:.1f}) e={err:.2f}", fill=(0, 255, 0))
                img.save(os.path.join(args_cli.out, f"detect_{step:04d}.png"))
                saved += 1

        if step % 50 == 0:
            print(f"[detect] step={step} dets_this_frame={len(dets)} distinct_tracks={len(detected_track_ids)} "
                  f"mean_err={np.mean(errors) if errors else float('nan'):.3f}", flush=True)
        if bool(term.any()) or bool(trunc.any()):
            print(f"[detect] terminated at step={step}")
            break

    summary = {
        "distinct_objects_detected": len(detected_track_ids),
        "total_true_objects": 18,
        "n_error_samples": len(errors),
        "mean_localization_error_m": float(np.mean(errors)) if errors else None,
        "median_localization_error_m": float(np.median(errors)) if errors else None,
        "p90_error_m": float(np.percentile(errors, 90)) if errors else None,
        "saved_annotated_images": saved,
    }
    with open(os.path.join(args_cli.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[detect] SUMMARY:", json.dumps(summary, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
