# Decisive test: can the robot cross the pit by stepping on a box that is PINNED
# in place (immovable stepping stone)? Isolates crossing locomotion from box dynamics.
#
# Box is pinned each step (pose rewritten, velocity zeroed) flush against the pit's
# FAR wall, resting on the floor: pit x in [-0.55,0.75] -> box center x=0.35 (far
# face 0.75), z=-0.70 (top -0.40). Robot drives down the right lane to step onto it
# and climb out the far side (z->0 at x>0.75).
#
# Usage:
#   PYTHONPATH=. python scripts/probe_task_d_pinnedcross.py --task=ATEC-TaskD-G1 \
#       --num_envs=1 --enable_cameras --num_steps 900 --box_x 0.35 --cross_y -0.3 --vx 0.8

import argparse
import math
import os

from isaaclab.app import AppLauncher

_DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo")

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="ATEC-TaskD-G1")
parser.add_argument("--num_steps", type=int, default=900)
parser.add_argument("--box_x", type=float, default=0.35)
parser.add_argument("--box_z", type=float, default=-0.70)
parser.add_argument("--box_yaw_deg", type=float, default=0.0, help="Box yaw about z (90 = long 1.0m side along x).")
parser.add_argument("--cross_y", type=float, default=-0.3)
parser.add_argument("--vx", type=float, default=0.8)
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

    pinned = box.data.root_state_w.clone()
    pinned[0, 0] = args_cli.box_x
    pinned[0, 1] = args_cli.cross_y
    pinned[0, 2] = args_cli.box_z
    _half = math.radians(args_cli.box_yaw_deg) / 2.0
    pinned[0, 3:7] = torch.tensor([math.cos(_half), 0.0, 0.0, math.sin(_half)], device=pinned.device)
    pinned[0, 7:] = 0.0

    CY = args_cli.cross_y
    robot_max_x, robot_min_z = -1e9, 1e9
    crossed = False
    for step in range(args_cli.num_steps):
        box.write_root_state_to_sim(pinned.clone())  # pin the box every step

        rx, ry, rz = robot.data.root_pos_w[0].tolist()
        yaw = yaw_from_quat(robot.data.root_quat_w[0].tolist())
        ey = CY - ry
        vx = args_cli.vx
        # Steadier crossing: keep strictly on the box centre-line, hold heading hard,
        # commit forward once on the box; push harder near the far gap to climb out.
        vy = clamp(1.6 * ey, -0.3, 0.3)
        wz = clamp(3.0 * (0.0 - yaw), -0.6, 0.6)
        if rz < 0.55:                 # on the box / in the pit: commit forward to climb out
            vx = max(vx, 0.9)
        a = bridge.act(obs["proprio"], (vx, vy, wz))
        obs, reward, term, trunc, info = env.step(torch.tensor(a, dtype=torch.float32, device="cuda").view(1, -1))
        robot_max_x = max(robot_max_x, rx)
        robot_min_z = min(robot_min_z, rz)
        if rx > 0.9 and rz > 0.4:
            crossed = True

        if step % 20 == 0:
            print(f"[pinned] step={step:04d} robot=({rx:+.2f},{ry:+.2f},{rz:+.2f}) yaw={yaw:+.2f} "
                  f"box=({float(box.data.root_pos_w[0,0]):+.2f},{float(box.data.root_pos_w[0,1]):+.2f},"
                  f"{float(box.data.root_pos_w[0,2]):+.2f}) crossed={crossed}")
        if term.item() or trunc.item():
            print(f"[pinned] END step={step} term={term.item()} trunc={trunc.item()}")
            break

    rx, ry, rz = robot.data.root_pos_w[0].tolist()
    verdict = "CROSSED" if (robot_max_x > 0.9 and rz > 0.4) else ("FELL" if robot_min_z < -0.2 else "STUCK")
    print(f"[pinned] VERDICT={verdict} robot_max_x={robot_max_x:.2f} robot_min_z={robot_min_z:.2f} "
          f"final=({rx:.2f},{ry:.2f},{rz:.2f})")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
