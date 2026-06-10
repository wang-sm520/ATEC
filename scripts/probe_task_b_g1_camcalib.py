"""标定 G1 头相机外参（相对机器人 base 的高度/俯仰/前向偏移）并验证 depth 语义。

仅本地标定用（读 env.scene 真值）。输出供 TaskBRgbdPerception 烧写常量。

Usage:
  PYTHONPATH=. python scripts/probe_task_b_g1_camcalib.py \
    --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless
"""

from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Calibrate G1 head camera extrinsics for Task B.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-G1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--out", type=str, default="outputs/task_b_g1_camcalib")
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


def quat_to_rpy(w, x, y, z):
    # ZYX (yaw-pitch-roll) extraction
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sp = 2 * (w * y - z * x)
    sp = max(-1.0, min(1.0, sp))
    pitch = math.asin(sp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    scene = env.unwrapped.scene
    robot = scene["robot"]
    cam = scene["head_camera"]
    obs, _ = env.reset()

    # step a few times so sensors populate
    for _ in range(5):
        action_dim = int(env.unwrapped.action_space.shape[-1])
        obs, *_ = env.step(torch.zeros((args_cli.num_envs, action_dim), dtype=torch.float32, device=args_cli.device))

    base_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
    base_quat = robot.data.root_quat_w[0].detach().cpu().tolist()  # w,x,y,z
    cam_pos = cam.data.pos_w[0].detach().cpu().tolist()
    # camera world orientation (ros optical frame): try quat_w_ros then quat_w_world
    cam_quat_ros = getattr(cam.data, "quat_w_ros", None)
    cam_quat_world = getattr(cam.data, "quat_w_world", None)
    cam_quat_ros = cam_quat_ros[0].detach().cpu().tolist() if cam_quat_ros is not None else None
    cam_quat_world = cam_quat_world[0].detach().cpu().tolist() if cam_quat_world is not None else None

    base_rpy = quat_to_rpy(*base_quat)
    info = {
        "base_pos_w": base_pos,
        "base_quat_w_wxyz": base_quat,
        "base_rpy_deg": [math.degrees(a) for a in base_rpy],
        "cam_pos_w": cam_pos,
        "cam_height_above_ground_m": cam_pos[2],
        "cam_height_above_base_m": cam_pos[2] - base_pos[2],
        "cam_forward_offset_from_base_xy_m": [cam_pos[0] - base_pos[0], cam_pos[1] - base_pos[1]],
        "cam_quat_w_ros_wxyz": cam_quat_ros,
        "cam_quat_w_world_wxyz": cam_quat_world,
    }
    if cam_quat_world is not None:
        r, p, yw = quat_to_rpy(*cam_quat_world)
        info["cam_world_rpy_deg"] = [math.degrees(r), math.degrees(p), math.degrees(yw)]
    if cam_quat_ros is not None:
        r, p, yw = quat_to_rpy(*cam_quat_ros)
        info["cam_ros_rpy_deg"] = [math.degrees(r), math.degrees(p), math.degrees(yw)]

    # intrinsics from cfg
    W, H = 640, 480
    focal, h_ap = 24.0, 20.955
    v_ap = h_ap * H / W
    fx = focal / h_ap * W
    fy = focal / v_ap * H
    info["intrinsics"] = {"W": W, "H": H, "fx": fx, "fy": fy, "cx0": W / 2, "cy0": H / 2,
                          "hfov_deg": math.degrees(2 * math.atan(h_ap / (2 * focal))),
                          "vfov_deg": math.degrees(2 * math.atan(v_ap / (2 * focal)))}

    # depth semantics check: pick the nearest object, find its pixel, compare depth value
    # to euclidean cam->object distance and to the perpendicular (z) distance.
    image = obs.get("image", {})
    depth = image.get("head_depth")
    if depth is not None:
        d = depth.detach().float().cpu()
        if d.ndim == 4:
            d = d[0]
        if d.ndim == 3 and d.shape[-1] == 1:
            d = d[..., 0]
        finite = torch.isfinite(d) & (d > 0.05) & (d < 40)
        if bool(finite.any()):
            info["depth_stats"] = {"min": float(d[finite].min()), "max": float(d[finite].max()),
                                   "median": float(d[finite].median())}
    # object truths
    objs = []
    for oi in range(1, 19):
        try:
            opos = scene[f"object_{oi}"].data.root_pos_w[0].detach().cpu().tolist()
            objs.append({"name": f"object_{oi}", "pos": opos})
        except Exception:
            pass
    info["objects"] = objs
    if objs:
        cam_xyz = cam_pos
        nearest = min(objs, key=lambda o: math.dist(cam_xyz, o["pos"]))
        info["nearest_object_to_cam"] = {"name": nearest["name"], "pos": nearest["pos"],
                                         "euclid_cam_dist": math.dist(cam_xyz, nearest["pos"])}

    with open(os.path.join(args_cli.out, "camcalib.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    print(f"[camcalib] wrote {args_cli.out}/camcalib.json")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
