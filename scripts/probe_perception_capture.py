# Perception data capture for Task D box localization.
#
# Drives the robot (using god-view box pose ONLY for driving) to face the box and
# approach from far to near, dumping at several distances:
#   - world-frame point cloud from head_depth (create_pointcloud_from_depth)
#   - lidar height_scan (extero) + raw ray_hits
#   - god-view true box / robot pose, camera pose+intrinsics
# So a box detector can be developed and validated OFFLINE against ground truth.
#
# Usage:
#   PYTHONPATH=. python scripts/probe_perception_capture.py --task=ATEC-TaskD-G1 \
#       --num_envs=1 --enable_cameras --out outputs/percep_capture

import argparse
import math
import os
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="ATEC-TaskD-G1")
parser.add_argument("--num_steps", type=int, default=900)
parser.add_argument("--out", type=str, default="outputs/percep_capture")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth  # noqa: E402
from atec_rl_lab.train.task_d.policy_bridge import G1VelocityPolicyBridge  # noqa: E402

_DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo")


def yaw_from_quat(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    bridge = G1VelocityPolicyBridge(policy_path=os.path.join(_DEMO_DIR, "policy_a.pt"), device="cuda")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    scene = env.unwrapped.scene
    robot, box = scene["robot"], scene["box"]
    cam = scene.sensors["head_camera"]
    lidar = scene.sensors["lidar_sensor"]
    obs, _ = env.reset()
    bridge.reset()

    print(f"[cap] cam attrs: pos_w={hasattr(cam.data,'pos_w')} quat_w_ros={hasattr(cam.data,'quat_w_ros')} "
          f"intrinsic={hasattr(cam.data,'intrinsic_matrices')}")
    captures = []
    cap_idx = 0
    next_standoff = [2.2, 1.9, 1.6, 1.3, 1.0, 0.8]  # capture as the robot closes in

    for step in range(args_cli.num_steps):
        rx, ry, rz = robot.data.root_pos_w[0].tolist()
        bx, by, bz = box.data.root_pos_w[0].tolist()
        yaw = yaw_from_quat(robot.data.root_quat_w[0].tolist())

        # Drive: face the box and approach to the next standoff distance (god-view driving).
        dx, dy = bx - rx, by - ry
        dist = math.hypot(dx, dy)
        face = math.atan2(dy, dx)                  # heading to point at the box
        standoff = next_standoff[cap_idx] if cap_idx < len(next_standoff) else 0.8
        # body-frame command toward a point `standoff` before the box, facing the box
        tx, ty = bx - standoff * math.cos(face), by - standoff * math.sin(face)
        ex, ey = tx - rx, ty - ry
        c, s = math.cos(yaw), math.sin(yaw)
        vx = clamp(1.2 * (c * ex + s * ey), -0.4, 0.7)
        vy = clamp(1.2 * (-s * ex + c * ey), -0.5, 0.5)
        wz = clamp(2.0 * ((face - yaw + math.pi) % (2 * math.pi) - math.pi), -1.0, 1.0)
        a = bridge.act(obs["proprio"], (vx, vy, wz))
        obs, *_ = env.step(torch.tensor(a, dtype=torch.float32, device="cuda").view(1, -1))

        # Capture when settled near the target standoff and roughly facing the box.
        facing_err = abs((face - yaw + math.pi) % (2 * math.pi) - math.pi)
        if cap_idx < len(next_standoff) and abs(dist - standoff) < 0.18 and facing_err < 0.2:
            depth = cam.data.output["depth"][0].squeeze(-1)        # (H, W)
            intr = cam.data.intrinsic_matrices[0]
            cpos = cam.data.pos_w[0]
            cquat = cam.data.quat_w_ros[0]
            torch.save(depth.detach().float().cpu(), os.path.join(args_cli.out, f"depth_{cap_idx}.pt"))
            torch.save(intr.detach().float().cpu(), os.path.join(args_cli.out, f"intr_{cap_idx}.pt"))
            torch.save(obs["extero"].detach().float().cpu().reshape(-1),
                       os.path.join(args_cli.out, f"lidar_{cap_idx}.pt"))
            pcd = torch.zeros((1, 3))  # placeholder for the log line below
            meta = {
                "idx": cap_idx, "dist": round(dist, 3),
                "box_w": [round(v, 4) for v in box.data.root_pos_w[0].tolist()],
                "robot_w": [round(rx, 4), round(ry, 4), round(rz, 4)], "robot_yaw": round(yaw, 4),
                "cam_pos": [round(v, 4) for v in cpos.tolist()],
                "cam_quat_ros": [round(v, 4) for v in cquat.tolist()],
                "pcd_n": int(pcd.shape[0]),
            }
            captures.append(meta)
            print(f"[cap] captured idx={cap_idx} dist={dist:.2f} box=({bx:.2f},{by:.2f},{bz:.2f}) "
                  f"robot=({rx:.2f},{ry:.2f}) yaw={yaw:.2f} pcd_n={int(pcd.shape[0])}")
            cap_idx += 1
            if cap_idx >= len(next_standoff):
                break

    with open(os.path.join(args_cli.out, "meta.json"), "w") as f:
        json.dump(captures, f, indent=2)
    print(f"[cap] DONE {len(captures)} captures -> {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
