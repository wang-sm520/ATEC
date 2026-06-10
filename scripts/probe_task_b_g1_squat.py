"""蹲下时测量 G1 手基座高度与到物体距离，标定扫地几何与 squat 目标高度。

仅本地标定使用（读 env.scene 真值，不进 solution）。

Usage:
  PYTHONPATH=. python scripts/probe_task_b_g1_squat.py \
    --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless --out outputs/task_b_g1_squat
"""

from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe G1 squat geometry for Task B.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-G1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=400)
parser.add_argument("--out", type=str, default="outputs/task_b_g1_squat")
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
from demo.solution_task_b_g1 import AlgSolution  # noqa: E402


def _any_done(value):
    if hasattr(value, "any"):
        try:
            return bool(value.any())
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return any(_any_done(v) for v in value)
    return bool(value)


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    scene = env.unwrapped.scene
    robot = scene["robot"]
    l_idx = int(robot.find_bodies("left_hand_base_link")[0][0])
    r_idx = int(robot.find_bodies("right_hand_base_link")[0][0])
    sol = AlgSolution()
    obs, _ = env.reset()
    sol.reset()

    rows = []
    try:
        for step in range(args_cli.num_steps):
            if not simulation_app.is_running():
                break
            resp = sol.predicts(obs, 0.0)
            actions = torch.tensor(resp["action"], dtype=torch.float32, device=args_cli.device).view(1, -1)
            obs, reward, terminated, truncated, info = env.step(actions)
            lpos = robot.data.body_pos_w[0, l_idx].detach().cpu().tolist()
            rpos = robot.data.body_pos_w[0, r_idx].detach().cpu().tolist()
            nearest = None
            for oi in range(1, 19):
                opos = scene[f"object_{oi}"].data.root_pos_w[0].detach().cpu().tolist()
                dl = math.dist(lpos, opos); dr = math.dist(rpos, opos)
                d = min(dl, dr)
                if nearest is None or d < nearest["dist"]:
                    nearest = {"object": f"object_{oi}", "dist": d, "hand": "L" if dl < dr else "R"}
            phase = getattr(sol.planner, "phase", "?")
            row = {"step": step, "phase": phase, "l_hand_z": lpos[2], "r_hand_z": rpos[2], "nearest": nearest}
            rows.append(row)
            if step % 25 == 0:
                print(f"[squat] step={step} phase={phase} l_z={lpos[2]:.3f} r_z={rpos[2]:.3f} nearest={nearest}")
            if _any_done(terminated) or _any_done(truncated):
                print(f"[squat] terminated at step={step}")
                break

        with open(os.path.join(args_cli.out, "squat_geometry.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"[squat] wrote {len(rows)} rows to {args_cli.out}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
