# Task D crossing-feasibility test (user's plan: box in the RIGHT pit, step across).
#
# Decouples "can we push box into pit" from "can robot cross stepping on box".
# Teleports the box into the right-side pit (against the far wall = most helpful),
# lets it settle, then drives the robot down the right lane to step across it.
#
# Geometry (world): right pit x in [-0.55, 0.75], depth 1.0 (floor z=-1.0),
# runs along -y (right). End-zone flat ground at x>0.75. Goal: robot x>0.75 at z~0.
#
# Usage:
#   PYTHONPATH=. python scripts/probe_task_d_cross.py \
#       --task=ATEC-TaskD-G1 --num_envs=1 --headless --enable_cameras --num_steps 1500

import argparse
import math
import os

from isaaclab.app import AppLauncher

_DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo")

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="ATEC-TaskD-G1")
parser.add_argument("--num_steps", type=int, default=1500)
parser.add_argument("--cross_y", type=float, default=-0.3, help="Right-lane y to cross at.")
parser.add_argument("--box_x", type=float, default=0.35, help="Box center x inside the pit.")
parser.add_argument("--no_box", action="store_true", help="Baseline: try to cross with NO box.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from atec_rl_lab.train.task_d.policy_bridge import G1VelocityPolicyBridge  # noqa: E402


def yaw_from_quat(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    bridge = G1VelocityPolicyBridge(policy_path=os.path.join(_DEMO_DIR, "policy_a.pt"), device="cuda")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    scene = env.unwrapped.scene
    robot, box = scene["robot"], scene["box"]
    obs, _ = env.reset()
    bridge.reset()

    # Teleport box into the right pit, resting on the pit floor (z_center = -1.0 + 0.3 = -0.7),
    # against the far wall (box far face at +0.75 -> center 0.75 - 0.4 = 0.35).
    if not args_cli.no_box:
        root = box.data.root_state_w.clone()
        root[0, 0] = args_cli.box_x
        root[0, 1] = args_cli.cross_y
        root[0, 2] = -0.70
        root[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=root.device)
        root[0, 7:] = 0.0
        box.write_root_state_to_sim(root)
        for _ in range(30):  # let it settle
            a = bridge.act(obs["proprio"], (0.0, 0.0, 0.0))
            obs, *_ = env.step(torch.tensor(a, dtype=torch.float32, device="cuda").view(1, -1))
        print(f"[cross] box settled at {[round(v,2) for v in box.data.root_pos_w[0].tolist()]}")

    CROSS_Y = args_cli.cross_y
    robot_max_x = -1e9
    robot_min_z = 1e9
    crossed = False
    for step in range(args_cli.num_steps):
        rp = robot.data.root_pos_w[0].tolist()
        yaw = yaw_from_quat(robot.data.root_quat_w[0].tolist())
        # Drive straight down the right lane toward +x, holding y=CROSS_Y and heading +x.
        ey = CROSS_Y - rp[1]
        vx = 1.0
        vy = clamp(1.0 * ey, -0.4, 0.4)
        wz = clamp(2.0 * (0.0 - yaw) + 1.0 * ey, -0.8, 0.8)
        a = bridge.act(obs["proprio"], (vx, vy, wz))
        obs, reward, terminated, truncated, info = env.step(
            torch.tensor(a, dtype=torch.float32, device="cuda").view(1, -1))
        robot_max_x = max(robot_max_x, rp[0])
        robot_min_z = min(robot_min_z, rp[2])
        if rp[0] > 0.9:
            crossed = True
        if step % 25 == 0:
            bp = box.data.root_pos_w[0].tolist()
            print(f"[cross] step={step:04d} robot=({rp[0]:+.2f},{rp[1]:+.2f},{rp[2]:+.2f}) yaw={yaw:+.2f} "
                  f"box=({bp[0]:+.2f},{bp[1]:+.2f},{bp[2]:+.2f}) crossed={crossed}")
        if terminated.item() or truncated.item():
            print(f"[cross] END step={step} term={terminated.item()} trunc={truncated.item()}")
            break

    rp = robot.data.root_pos_w[0].tolist()
    verdict = "CROSSED" if robot_max_x > 0.9 and rp[2] > 0.4 else ("FELL_IN_PIT" if robot_min_z < -0.3 else "STUCK")
    print(f"[cross] VERDICT={verdict} robot_max_x={robot_max_x:.2f} robot_min_z={robot_min_z:.2f} final=({rp[0]:.2f},{rp[1]:.2f},{rp[2]:.2f})")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
