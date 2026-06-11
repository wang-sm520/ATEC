# Task D: push box to the wall FIRST, then push it laterally ALONG the wall into the pit.
# User's insight: with the box jammed against the platform wall, a lateral (-y) push is
# constrained (no forward slip) -> no lateral slipping, box slides cleanly until its
# leading edge overhangs the pit and it tips in.
#
# Oracle (true-state) to validate the maneuver. Usage:
#   PYTHONPATH=. python scripts/probe_task_d_wallpush.py --task=ATEC-TaskD-G1 \
#       --num_envs=1 --enable_cameras --num_steps 3000

import argparse
import math
import os

from isaaclab.app import AppLauncher

_DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo")

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


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def vel_to(rx, ry, ryaw, tx, ty, tyaw, vx_hi=0.8, vy_abs=0.55):
    ex, ey = tx - rx, ty - ry
    c, s = math.cos(ryaw), math.sin(ryaw)
    return (clamp(1.4 * (c * ex + s * ey), -0.5, vx_hi), clamp(1.4 * (-s * ex + c * ey), -vy_abs, vy_abs),
            clamp(2.2 * wrap(tyaw - ryaw), -1.0, 1.0))


def reached(rx, ry, ryaw, wp, pt=0.22, yt=0.3):
    return math.hypot(wp[0] - rx, wp[1] - ry) < pt and abs(wrap(wp[2] - ryaw)) < yt


def main():
    bridge = G1VelocityPolicyBridge(policy_path=os.path.join(_DEMO_DIR, "policy_a.pt"), device="cuda")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    scene = env.unwrapped.scene
    robot, box = scene["robot"], scene["box"]
    obs, _ = env.reset()
    bridge.reset()

    DOWN = -math.pi / 2
    phase, wpi = "approach_back", 0
    jam_x, jam_step = -1e9, 0
    around_top = None
    around_back2 = None
    x_line = None              # fixed lateral-push line (box x at the wall)
    box_min_z = 1e9

    for step in range(args_cli.num_steps):
        rx, ry, rz = robot.data.root_pos_w[0].tolist()
        bx, by, bz = box.data.root_pos_w[0].tolist()
        yaw = yaw_from_quat(robot.data.root_quat_w[0].tolist())
        box_min_z = min(box_min_z, bz)

        if phase == "approach_back":
            vx, vy, wz = vel_to(rx, ry, yaw, bx - 0.7, by, 0.0)
            if rx < bx - 0.5 and abs(ry - by) < 0.15 and abs(wrap(yaw)) < 0.15:
                phase, jam_x, jam_step = "push_x", rx, step

        elif phase == "push_x":
            vx, vy, wz = 0.9, clamp(1.2 * (by - ry), -0.4, 0.4), clamp(-2.2 * yaw, -0.8, 0.8)
            if rx - jam_x > 0.06:
                jam_x, jam_step = rx, step
            elif step - jam_step > 120:                     # box jammed against wall
                x_line = bx                                  # lock the straight lateral-push line
                around_top = [(bx - 1.5, by, DOWN), (bx - 1.5, by + 1.0, DOWN), (bx, by + 1.0, DOWN)]
                phase, wpi = "around_top", 0

        elif phase == "around_top":
            wp = around_top[wpi]
            vx, vy, wz = vel_to(rx, ry, yaw, *wp)
            if reached(rx, ry, yaw, wp):
                wpi += 1
                if wpi >= len(around_top):
                    phase = "push_y"

        elif phase == "push_y":
            # walk a STRAIGHT line at x=x_line (the wall x), heading -y, so the box goes
            # straight and doesn't get pushed crooked. Strong lateral + heading hold.
            vx = 0.85                                         # body forward -> world -y
            vy = clamp(2.0 * (x_line - rx), -0.45, 0.45)      # hold the line (body lat -> world +x)
            wz = clamp(2.8 * wrap(DOWN - yaw), -0.7, 0.7)     # firm heading lock
            # push far enough that the WHOLE box (1.0 m in y) clears the platform (y<0.5):
            # box +y edge = by+0.5 < 0.5  ->  by < 0; use -0.2 for margin.
            if by <= -0.2:
                around_back2 = [(x_line - 1.5, by, 0.0), (x_line - 0.75, by, 0.0)]
                phase, wpi = "around_back2", 0

        elif phase == "around_back2":
            wp = around_back2[wpi]
            vx, vy, wz = vel_to(rx, ry, yaw, *wp)
            if reached(rx, ry, yaw, wp):
                wpi += 1
                if wpi >= len(around_back2):
                    phase = "push_x_pit"

        elif phase == "push_x_pit":
            # box is in the pit lane now (no wall): push +x until it drops into the pit
            vx, vy, wz = 0.9, clamp(1.2 * (by - ry), -0.4, 0.4), clamp(-2.2 * yaw, -0.8, 0.8)
            if bz < -0.2:
                phase = "done"

        else:  # done
            vx, vy, wz = 0.0, 0.0, clamp(-2.0 * yaw, -0.6, 0.6)

        a = bridge.act(obs["proprio"], (vx, vy, wz))
        obs, reward, term, trunc, info = env.step(torch.tensor(a, dtype=torch.float32, device="cuda").view(1, -1))

        if step % 25 == 0:
            extra = f" wp={wpi}" if phase == "around_top" else ""
            print(f"[wallpush] step={step:04d} ph={phase:12s}{extra} "
                  f"robot=({rx:+.2f},{ry:+.2f},{rz:+.2f}) box=({bx:+.2f},{by:+.2f},{bz:+.2f}) "
                  f"yaw={yaw:+.2f} box_min_z={box_min_z:+.2f}")
        if term.item() or trunc.item():
            print(f"[wallpush] END step={step} term={term.item()} trunc={trunc.item()} box=({bx:+.2f},{by:+.2f},{bz:+.2f})")
            break

    bx, by, bz = box.data.root_pos_w[0].tolist()
    print(f"[wallpush] FINAL phase={phase} box=({bx:.2f},{by:.2f},{bz:.2f}) box_min_z={box_min_z:.2f} "
          f"box_in_pit={box_min_z < -0.2}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
