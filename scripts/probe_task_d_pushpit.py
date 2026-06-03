# Task D: push the box INTO the right-side pit (user's goal; crossing handled later).
#
# Pit (world): x in [-0.55, 0.75], y < 0.5, depth 1.0. Box starts (-3, 1.6) on the
# +y/platform side. Plan: route AROUND the box (don't shove it), push it -y into the
# pit lane, then push it +x until it drops into the pit (box_z < -0.2).
#
# Oracle (true-state) waypoint controller to validate the maneuver before perception.
# Usage:
#   PYTHONPATH=. python scripts/probe_task_d_pushpit.py --task=ATEC-TaskD-G1 \
#       --num_envs=1 --enable_cameras --num_steps 4000

import argparse
import math
import os

from isaaclab.app import AppLauncher

_DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo")

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="ATEC-TaskD-G1")
parser.add_argument("--num_steps", type=int, default=4000)
parser.add_argument("--pit_lane_y", type=float, default=-0.3)
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
    bex, bey = c * ex + s * ey, -s * ex + c * ey
    return (clamp(1.4 * bex, -0.5, vx_hi), clamp(1.4 * bey, -vy_abs, vy_abs),
            clamp(2.2 * wrap(tyaw - ryaw), -1.0, 1.0))


def reached(rx, ry, ryaw, wp, pos_tol=0.22, yaw_tol=0.3):
    return (math.hypot(wp[0] - rx, wp[1] - ry) < pos_tol and abs(wrap(wp[2] - ryaw)) < yaw_tol)


def main():
    bridge = G1VelocityPolicyBridge(policy_path=os.path.join(_DEMO_DIR, "policy_a.pt"), device="cuda")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    scene = env.unwrapped.scene
    robot, box = scene["robot"], scene["box"]
    obs, _ = env.reset()
    bridge.reset()

    PY = args_cli.pit_lane_y
    DOWN = -math.pi / 2          # face -y
    # Phase 1 waypoints: route around the box's -x side, up to its +y side, face -y.
    around_top = [(-4.6, 0.8, DOWN), (-4.6, 2.5, DOWN), (-3.0, 2.5, DOWN)]
    phase, wpi = "around_top", 0
    around_back = None
    score, box_min_z = 0.0, 1e9

    for step in range(args_cli.num_steps):
        rx, ry, rz = robot.data.root_pos_w[0].tolist()
        bx, by, bz = box.data.root_pos_w[0].tolist()
        yaw = yaw_from_quat(robot.data.root_quat_w[0].tolist())
        box_min_z = min(box_min_z, bz)

        if phase == "around_top":
            wp = around_top[wpi]
            vx, vy, wz = vel_to(rx, ry, yaw, *wp)
            if reached(rx, ry, yaw, wp):
                wpi += 1
                if wpi >= len(around_top):
                    phase = "push_y"

        elif phase == "push_y":
            # push box toward -y; keep robot x aligned with box x
            vx, vy, wz = vel_to(rx, ry, yaw, bx, PY - 0.6, DOWN)
            vx = max(vx, 0.55)
            if by <= PY + 0.1:
                # set up around_back waypoints using current box pose
                around_back = [(bx, by + 1.0, 0.0), (bx - 1.7, by + 1.0, 0.0),
                               (bx - 1.7, by, 0.0), (bx - 0.75, by, 0.0)]
                phase, wpi = "around_back", 0

        elif phase == "around_back":
            wp = around_back[wpi]
            vx, vy, wz = vel_to(rx, ry, yaw, *wp)
            if reached(rx, ry, yaw, wp):
                wpi += 1
                if wpi >= len(around_back):
                    phase = "push_x"

        elif phase == "push_x":
            vx = 0.9
            vy = clamp(1.3 * (by - ry), -0.4, 0.4)
            wz = clamp(-2.2 * yaw, -0.8, 0.8)
            if bz < -0.2:
                phase = "done"

        else:  # done — box is in the pit; hold
            vx, vy, wz = 0.0, 0.0, clamp(-2.0 * yaw, -0.6, 0.6)

        a = bridge.act(obs["proprio"], (vx, vy, wz))
        obs, reward, term, trunc, info = env.step(torch.tensor(a, dtype=torch.float32, device="cuda").view(1, -1))
        score += (reward.mean().item() if isinstance(reward, torch.Tensor) else float(reward)) / info.get("Step_dt", 0.02)

        if step % 25 == 0:
            extra = f" wp={wpi}" if phase in ("around_top", "around_back") else ""
            print(f"[pushpit] step={step:04d} ph={phase:11s}{extra} score={score:5.1f} "
                  f"robot=({rx:+.2f},{ry:+.2f},{rz:+.2f}) box=({bx:+.2f},{by:+.2f},{bz:+.2f}) "
                  f"yaw={yaw:+.2f} box_min_z={box_min_z:+.2f}")
        if phase == "done" and step % 25 != 0:
            pass
        if term.item() or trunc.item():
            print(f"[pushpit] END step={step} term={term.item()} trunc={trunc.item()} "
                  f"box=({bx:+.2f},{by:+.2f},{bz:+.2f}) box_min_z={box_min_z:+.2f}")
            break

    bx, by, bz = box.data.root_pos_w[0].tolist()
    print(f"[pushpit] FINAL phase={phase} score={score:.1f} box=({bx:.2f},{by:.2f},{bz:.2f}) "
          f"box_min_z={box_min_z:.2f} box_in_pit={box_min_z < -0.2}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
