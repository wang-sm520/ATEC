"""探查 G1 手部相机（eye-in-hand）在蹲下/扫地时看到什么，并标定相机相对手基座的外参。

仅本地标定用（读 env.scene 真值）。回答：蹲下时手相机能否看到地面物体、物体在手相机里的
深度/像素、相机相对 *_hand_base_link 的位姿（供手眼伺服用）。

Usage:
  PYTHONPATH=. python scripts/probe_task_b_g1_handcam.py \
    --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless --num_steps 500
"""

from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe G1 hand (ee) cameras for Task B servoing.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-G1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=500)
parser.add_argument("--out", type=str, default="outputs/task_b_g1_handcam")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401,E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from demo.solution import AlgSolution  # noqa: E402


def colored_mask_stats(rgb_chw_last):
    """rgb: HxWx3 uint8-ish tensor. Return (count, cy, cx) of colored object pixels."""
    r = rgb_chw_last[..., 0].float()
    g = rgb_chw_last[..., 1].float()
    b = rgb_chw_last[..., 2].float()
    maxc = torch.maximum(torch.maximum(r, g), b)
    minc = torch.minimum(torch.minimum(r, g), b)
    sat = maxc - minc
    yellow = (r > 120) & (g > 90) & (b < 130)
    red_or = (r > 130) & (g > 45) & (b < 150) & (r > b + 35)
    bright = (maxc > 110) & (sat > 45)
    mask = yellow | red_or | bright
    n = int(mask.sum().item())
    if n == 0:
        return 0, None, None, None
    ys, xs = torch.nonzero(mask, as_tuple=True)
    return n, float(ys.float().mean()), float(xs.float().mean()), mask


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    scene = env.unwrapped.scene
    robot = scene["robot"]
    l_hand = int(robot.find_bodies("left_hand_base_link")[0][0])
    r_hand = int(robot.find_bodies("right_hand_base_link")[0][0])
    ee_cam = scene["ee_camera"] if "ee_camera" in scene.keys() else None       # left
    ee_dual = scene["ee_dual_camera"] if "ee_dual_camera" in scene.keys() else None  # right

    try:
        from PIL import Image
    except Exception:
        Image = None

    def save_rgb(rgb_t, path):
        if Image is None:
            return
        arr = rgb_t[..., :3].clamp(0, 255).to(torch.uint8).numpy()
        Image.fromarray(arr).save(path)

    def near_gated(rgb_t, depth_t, lo=0.08, hi=0.5):
        """colored pixels whose depth is in [lo,hi] (near the hand). Returns (count, cy, cx, med_depth)."""
        n, cy, cx, mask = colored_mask_stats(rgb_t[..., :3])
        if n == 0 or depth_t is None or mask is None:
            return 0, None, None, None
        d = depth_t
        if d.ndim == 3 and d.shape[-1] == 1:
            d = d[..., 0]
        near = mask & torch.isfinite(d) & (d > lo) & (d < hi)
        cnt = int(near.sum().item())
        if cnt == 0:
            return 0, None, None, None
        ys, xs = torch.nonzero(near, as_tuple=True)
        return cnt, float(ys.float().mean()), float(xs.float().mean()), float(d[near].median())

    sol = AlgSolution()
    obs, _ = env.reset()
    sol.reset()

    extrinsics_logged = False
    rows = []
    best_min = {"L": 1e9, "R": 1e9}
    try:
        for step in range(args_cli.num_steps):
            if not simulation_app.is_running():
                break
            resp = sol.predicts(obs, 0.0)
            actions = torch.tensor(resp["action"], dtype=torch.float32, device=args_cli.device).view(1, -1)
            obs, reward, terminated, truncated, info = env.step(actions)
            phase = getattr(sol.planner, "phase", "?")

            # one-time: image keys/shapes + camera-vs-hand extrinsics
            if not extrinsics_logged:
                img = obs.get("image", {})
                meta = {"image_keys": list(img.keys())}
                for k in ("ee_rgb", "ee_depth", "ee_dual_rgb", "ee_dual_depth", "head_rgb"):
                    if k in img and hasattr(img[k], "shape"):
                        meta[k + "_shape"] = list(img[k].shape)
                for name, cam, hand_idx in (("ee_left", ee_cam, l_hand), ("ee_right", ee_dual, r_hand)):
                    if cam is None:
                        continue
                    cpos = cam.data.pos_w[0].detach().cpu().tolist()
                    hpos = robot.data.body_pos_w[0, hand_idx].detach().cpu().tolist()
                    cq = getattr(cam.data, "quat_w_world", None)
                    cq = cq[0].detach().cpu().tolist() if cq is not None else None
                    meta[name] = {"cam_pos_w": cpos, "hand_pos_w": hpos,
                                  "cam_minus_hand": [cpos[i] - hpos[i] for i in range(3)],
                                  "cam_quat_w_world": cq}
                with open(os.path.join(args_cli.out, "handcam_meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)
                print("[handcam] meta:", json.dumps(meta, indent=2))
                extrinsics_logged = True

            # during squat: what does each hand cam see + true geometry
            if phase in ("squat_sweep", "stand_up"):
                img = obs.get("image", {})
                rec = {"step": step, "phase": phase}
                for name, key_rgb, key_d, hand_idx, cam in (
                    ("L", "ee_rgb", "ee_depth", l_hand, ee_cam),
                    ("R", "ee_dual_rgb", "ee_dual_depth", r_hand, ee_dual),
                ):
                    rgb = img.get(key_rgb)
                    depth = img.get(key_d)
                    if rgb is None:
                        continue
                    rgbt = rgb.detach().cpu()
                    if rgbt.ndim == 4:
                        rgbt = rgbt[0]
                    n, cy, cx, mask = colored_mask_stats(rgbt[..., :3])
                    hand_pos = robot.data.body_pos_w[0, hand_idx].detach().cpu().tolist()
                    # nearest true object to this hand
                    nd, nn = 1e9, None
                    for oi in range(1, 19):
                        opos = scene[f"object_{oi}"].data.root_pos_w[0].detach().cpu().tolist()
                        d = math.dist(hand_pos, opos)
                        if d < nd:
                            nd, nn = d, f"object_{oi}"
                    obj_depth = None
                    if depth is not None and n > 0 and mask is not None:
                        dt = depth.detach().cpu()
                        if dt.ndim == 4:
                            dt = dt[0]
                        if dt.ndim == 3 and dt.shape[-1] == 1:
                            dt = dt[..., 0]
                        vals = dt[mask]
                        vals = vals[torch.isfinite(vals) & (vals > 0.02) & (vals < 5)]
                        if vals.numel() > 0:
                            obj_depth = float(vals.median())
                    dt_full = None
                    if depth is not None:
                        dt_full = depth.detach().cpu()
                        if dt_full.ndim == 4:
                            dt_full = dt_full[0]
                    ngn, ngcy, ngcx, ngd = near_gated(rgbt[..., :3], dt_full)
                    rec[name] = {"colored_px": n, "blob_cy": cy, "blob_cx": cx,
                                 "blob_depth_m": obj_depth,
                                 "near_px": ngn, "near_cy": ngcy, "near_cx": ngcx, "near_depth_m": ngd,
                                 "hand_to_nearest_obj_m": nd, "nearest": nn,
                                 "hand_z": hand_pos[2]}
                    # save the closest-approach frame image per side
                    if nd < best_min[name]:
                        best_min[name] = nd
                        save_rgb(rgbt, os.path.join(args_cli.out, f"closest_{name}_rgb.png"))
                rows.append(rec)
                if step % 20 == 0:
                    print(f"[handcam] step={step} phase={phase} "
                          f"L={rec.get('L')} R={rec.get('R')}")

            if bool(terminated.any()) or bool(truncated.any()):
                print(f"[handcam] terminated at step={step}")
                break
    finally:
        with open(os.path.join(args_cli.out, "handcam_squat.json"), "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[handcam] wrote {len(rows)} squat rows to {args_cli.out}")
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
