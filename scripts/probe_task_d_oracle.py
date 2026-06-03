# Task D oracle closed-loop experiment.
#
# Uses PRIVILEGED true state (robot & box root_pos_w) as a perception oracle to
# decouple control-strategy risk from perception risk. Goal: prove the full
# "push box to pit -> step on box -> cross to x>3.5" strategy can score 36.
# Once this works, the oracle gets replaced by head_depth + lidar perception.
#
# Usage:
#   PYTHONPATH=. python scripts/probe_task_d_oracle.py \
#       --task=ATEC-TaskD-G1 --num_envs=1 --headless --enable_cameras --num_steps 3000

import argparse
import math
import os

from isaaclab.app import AppLauncher
from demo.solution import _DEMO_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="ATEC-TaskD-G1")
parser.add_argument("--num_steps", type=int, default=3000)
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


# Target box x to drop it squarely into the +14 range [-1.4, 0.7] and toward the pit.
BOX_TARGET_X = 0.3
CROSS_GOAL_X = 3.6


def control(robot, box, yaw, phase):
    """Return (vx, vy, wz, next_phase) in body frame. World +x is the goal direction."""
    rx, ry, rz = robot
    bx, by, bz = box

    if phase == "approach":
        # Get behind the box in x, centered on box y, facing +x.
        tx, ty = bx - 0.70, by
        if rx > bx - 0.55:  # in front of / overlapping: back off first
            tx = bx - 0.9
        ex, ey = tx - rx, ty - ry
        c, s = math.cos(yaw), math.sin(yaw)
        vx = clamp(1.4 * (c * ex + s * ey), -0.5, 0.9)
        vy = clamp(1.4 * (-s * ex + c * ey), -0.6, 0.6)
        wz = clamp(2.0 * (0.0 - yaw), -1.0, 1.0)
        if abs(ry - by) < 0.12 and (bx - 0.75) < rx < (bx - 0.5) and abs(yaw) < 0.1:
            return vx, vy, wz, "push"
        return vx, vy, wz, "approach"

    if phase == "push":
        # Stay centered behind box (track box y), heading +x, drive forward.
        ey = by - ry              # want robot y == box y
        c, s = math.cos(yaw), math.sin(yaw)
        vx = 0.9
        vy = clamp(1.2 * (c * 0 - s * 0 + (-s * 0 + c * ey)), -0.4, 0.4)  # body-y correction ~ world ey (yaw~0)
        wz = clamp(2.0 * (0.0 - yaw) - 1.0 * ey, -0.8, 0.8)
        # If robot lost contact (box too far ahead) re-approach.
        if bx - rx > 1.1:
            return vx, vy, wz, "approach"
        if bx >= BOX_TARGET_X:
            return 0.6, 0.0, wz, "cross"
        return vx, vy, wz, "push"

    if phase == "cross":
        # Box should now be at the pit; walk forward over it toward the far side.
        c, s = math.cos(yaw), math.sin(yaw)
        ey = by - ry
        vx = 1.0
        vy = clamp(0.8 * ey, -0.3, 0.3)
        wz = clamp(2.0 * (0.0 - yaw), -0.8, 0.8)
        return vx, vy, wz, "cross"

    return 0.0, 0.0, 0.0, phase


def main():
    bridge = G1VelocityPolicyBridge(policy_path=os.path.join(_DEMO_DIR, "policy_a.pt"), device="cuda")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    scene = env.unwrapped.scene
    robot, box = scene["robot"], scene["box"]
    obs, _ = env.reset()
    bridge.reset()

    phase = "approach"
    score = 0.0
    box_min_z = 1e9
    robot_max_x = -1e9
    for step in range(args_cli.num_steps):
        rp = robot.data.root_pos_w[0].tolist()
        bp = box.data.root_pos_w[0].tolist()
        yaw = yaw_from_quat(robot.data.root_quat_w[0].tolist())
        vx, vy, wz, phase = control(rp, bp, yaw, phase)

        action = bridge.act(obs["proprio"], (vx, vy, wz))
        action = torch.tensor(action, dtype=torch.float32, device="cuda").view(1, -1)
        obs, reward, terminated, truncated, info = env.step(action)
        sim_dt = info.get("Step_dt", 0.02)
        score += (reward.mean().item() if isinstance(reward, torch.Tensor) else float(reward)) / sim_dt
        box_min_z = min(box_min_z, bp[2])
        robot_max_x = max(robot_max_x, rp[0])

        if step % 25 == 0:
            print(f"[oracle] step={step:04d} ph={phase:8s} score={score:5.1f} "
                  f"robot=({rp[0]:+.2f},{rp[1]:+.2f},{rp[2]:+.2f}) box=({bp[0]:+.2f},{bp[1]:+.2f},{bp[2]:+.2f}) "
                  f"cmd=({vx:+.2f},{vy:+.2f},{wz:+.2f})")
        if terminated.item() or truncated.item():
            print(f"[oracle] END step={step} term={terminated.item()} trunc={truncated.item()}")
            break

    rp = robot.data.root_pos_w[0].tolist()
    bp = box.data.root_pos_w[0].tolist()
    print(f"[oracle] FINAL score={score:.1f} robot_max_x={robot_max_x:.2f} "
          f"box=({bp[0]:.2f},{bp[1]:.2f},{bp[2]:.2f}) box_min_z={box_min_z:.2f} robot=({rp[0]:.2f},{rp[1]:.2f},{rp[2]:.2f})")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
