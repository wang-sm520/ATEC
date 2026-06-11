# Calibrate the head camera look direction: spin the robot in place and, at each yaw,
# build the world point cloud (IsaacLab create_pointcloud_from_depth + quat_w_ros) and
# count box-height points near the true box. The yaw that maximizes the count tells us
# where the camera actually looks relative to the base heading.

import argparse
import math
import os

from isaaclab.app import AppLauncher

_DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo")

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="ATEC-TaskD-G1")
parser.add_argument("--num_steps", type=int, default=900)
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


def yaw_from_quat(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main():
    bridge = G1VelocityPolicyBridge(policy_path=os.path.join(_DEMO_DIR, "policy_a.pt"), device="cuda")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    scene = env.unwrapped.scene
    robot, box = scene["robot"], scene["box"]
    cam = scene.sensors["head_camera"]
    obs, _ = env.reset()
    bridge.reset()

    best = (-1, None)
    for step in range(args_cli.num_steps):
        rx, ry, rz = robot.data.root_pos_w[0].tolist()
        bx, by, bz = box.data.root_pos_w[0].tolist()
        yaw = yaw_from_quat(robot.data.root_quat_w[0].tolist())
        # spin slowly in place
        a = bridge.act(obs["proprio"], (0.0, 0.0, 0.6))
        obs, *_ = env.step(torch.tensor(a, dtype=torch.float32, device="cuda").view(1, -1))

        if step % 8 == 0:
            depth = cam.data.output["depth"][0].squeeze(-1)
            intr = cam.data.intrinsic_matrices[0]
            P = create_pointcloud_from_depth(intr, depth, position=cam.data.pos_w[0],
                                             orientation=cam.data.quat_w_ros[0])
            near = (P[:, 0] - bx).abs() < 0.6
            near &= (P[:, 1] - by).abs() < 0.7
            near &= (P[:, 2] > -0.1) & (P[:, 2] < 0.7)
            n = int(near.sum())
            dir_to_box = math.atan2(by - ry, bx - rx)
            offset = (yaw - dir_to_box + math.pi) % (2 * math.pi) - math.pi
            if n > best[0]:
                best = (n, (round(yaw, 3), round(dir_to_box, 3), round(offset, 3)))
            if n > 200:
                print(f"step={step:03d} yaw={yaw:+.2f} dir_to_box={dir_to_box:+.2f} "
                      f"base_yaw-dir={offset:+.2f} near_box={n}")
    print(f"\nBEST near_box={best[0]} at (base_yaw, dir_to_box, base_yaw-dir_to_box)={best[1]}")
    print("=> camera look offset from base heading ~ -(base_yaw-dir_to_box) when best")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
