# Task D push-feasibility test.
#
# Bypasses the (broken) TaskDController. Drives the frozen policy_a bridge with
# HAND-SCRIPTED velocity commands, using privileged true state only as a closed
# loop for this experiment, to answer one question:
#   Can policy_a actually push the 8 kg box in +x?
#
# Strategy: (1) side-step in +y to the box lane (y ~ 1.6) while staying at x~-3.9
#           (behind the box in x), facing +x; (2) drive vx hard in +x and watch
#           the TRUE box x move.
#
# Usage:
#   PYTHONPATH=. python scripts/probe_task_d_pushtest.py \
#       --task=ATEC-TaskD-G1 --num_envs=1 --headless --enable_cameras --num_steps 500

import argparse
import math
import os

from isaaclab.app import AppLauncher
from demo.solution import _DEMO_DIR  # reuse path logic

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="ATEC-TaskD-G1")
parser.add_argument("--num_steps", type=int, default=500)
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


def main():
    policy_a = os.path.join(_DEMO_DIR, "policy_a.pt")
    bridge = G1VelocityPolicyBridge(policy_path=policy_a, device="cuda")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    scene = env.unwrapped.scene
    robot, box = scene["robot"], scene["box"]
    obs, _ = env.reset()
    bridge.reset()

    box_x0 = float(box.data.root_pos_w[0, 0])
    print(f"[push] box_x0={box_x0:.3f} box_y0={float(box.data.root_pos_w[0,1]):.3f}")

    # Box lane target: get behind box in x (lower x), aligned in y with box center.
    BOX_Y = 1.6
    score = 0.0
    for step in range(args_cli.num_steps):
        rp = robot.data.root_pos_w[0].tolist()
        bp = box.data.root_pos_w[0].tolist()
        yaw = yaw_from_quat(robot.data.root_quat_w[0].tolist())

        # Desired world target depends on alignment.
        aligned_y = abs(rp[1] - BOX_Y) < 0.15
        behind_x = rp[0] < bp[0] - 0.55
        if not aligned_y and not behind_x:
            # move to (bp_x - 0.7, BOX_Y)
            tx, ty = bp[0] - 0.7, BOX_Y
            mode = "approach"
        else:
            # push: drive toward +x through the box
            tx, ty = bp[0] + 1.5, BOX_Y
            mode = "push"

        ex, ey = tx - rp[0], ty - rp[1]
        c, s = math.cos(yaw), math.sin(yaw)
        bex = c * ex + s * ey      # body-frame forward error
        bey = -s * ex + c * ey
        vx = max(-0.5, min(1.0, 1.4 * bex))
        vy = max(-0.6, min(0.6, 1.4 * bey))
        wz = max(-1.0, min(1.0, 2.0 * (0.0 - yaw)))  # keep facing +x
        if mode == "push":
            vx = max(vx, 0.8)  # commit forward

        action = bridge.act(obs["proprio"], (vx, vy, wz))
        action = torch.tensor(action, dtype=torch.float32, device="cuda").view(1, -1)
        obs, reward, terminated, truncated, info = env.step(action)
        sim_dt = info.get("Step_dt", 0.02)
        score += (reward.mean().item() if isinstance(reward, torch.Tensor) else float(reward)) / sim_dt

        if step % 20 == 0:
            print(f"[push] step={step:03d} mode={mode:8s} score={score:5.1f} "
                  f"robot=({rp[0]:.2f},{rp[1]:.2f},{rp[2]:.2f}) yaw={yaw:+.2f} "
                  f"box=({bp[0]:.2f},{bp[1]:.2f},{bp[2]:.2f}) box_dx={bp[0]-box_x0:+.2f} cmd=({vx:+.2f},{vy:+.2f},{wz:+.2f})")
        if terminated.item() or truncated.item():
            print(f"[push] END step={step} term={terminated.item()} trunc={truncated.item()}")
            break

    bp = box.data.root_pos_w[0].tolist()
    rp = robot.data.root_pos_w[0].tolist()
    print(f"[push] FINAL score={score:.1f} box_dx={bp[0]-box_x0:+.3f} box=({bp[0]:.2f},{bp[1]:.2f}) robot=({rp[0]:.2f},{rp[1]:.2f},{rp[2]:.2f})")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
