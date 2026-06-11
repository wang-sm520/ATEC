"""Diagnostic: why does Task B approach never walk? Log per-step the planner's
selected-target ground distance, arrival-gate decision, dead-reckoned vs TRUE base
pose (odometry drift), and detection count. Privileged: reads env.scene truth.

Usage:
  PYTHONPATH=. python scripts/diag_task_b_g1_approach.py \
    --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless --num_steps 1500 \
    --out outputs/diag_approach
"""
from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="ATEC-TaskB-G1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=1500)
parser.add_argument("--out", type=str, default="outputs/diag_approach")
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
from demo.solution import AlgSolution  # noqa: E402


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    scene = env.unwrapped.scene
    robot = scene["robot"]
    sol = AlgSolution()
    obs, _ = env.reset()
    sol.reset()

    rows = []
    last_phase = None
    for step in range(args_cli.num_steps):
        if not simulation_app.is_running():
            break
        resp = sol.predicts(obs, 0.0)
        actions = torch.tensor(resp["action"], dtype=torch.float32, device=args_cli.device).view(1, -1)
        obs, reward, terminated, truncated, info = env.step(actions)

        pl = sol.planner
        pose = sol.odom  # DeadReckoningOdometry holds x,y,yaw
        true_base = robot.data.root_pos_w[0].detach().cpu().tolist()
        ndet = len(sol._cached_detections)
        # selected-target ground distance (odometry frame)
        tgt_dist = None
        if pl.active is not None:
            tx, ty = pl.active[1], pl.active[2]
            tgt_dist = math.hypot(tx - pose.x, ty - pose.y)
        # odometry drift = |dead-reckoned (x,y) - true (x,y)|
        drift = math.hypot(pose.x - true_base[0], pose.y - true_base[1])
        # nearest TRUE object to true base
        nd = None
        for oi in range(1, 19):
            op = scene[f"object_{oi}"].data.root_pos_w[0].detach().cpu().tolist()
            d = math.hypot(op[0] - true_base[0], op[1] - true_base[1])
            if nd is None or d < nd:
                nd = d
        phase = pl.phase
        row = {
            "step": step, "phase": phase, "ndet": ndet,
            "odom_xy": [round(pose.x, 3), round(pose.y, 3)],
            "true_xy": [round(true_base[0], 3), round(true_base[1], 3)],
            "drift": round(drift, 3),
            "tgt_ground_dist": None if tgt_dist is None else round(tgt_dist, 3),
            "nearest_true_obj": round(nd, 3),
        }
        rows.append(row)
        if phase != last_phase or step % 50 == 0:
            print(f"[diag] s={step} ph={phase:11s} ndet={ndet} drift={drift:.2f} "
                  f"tgt_d={row['tgt_ground_dist']} near_true={nd:.2f} "
                  f"odom={row['odom_xy']} true={row['true_xy']}", flush=True)
        last_phase = phase
        done = (bool(terminated.any()) if hasattr(terminated, "any") else bool(terminated)) or \
               (bool(truncated.any()) if hasattr(truncated, "any") else bool(truncated))
        if done:
            print(f"[diag] done/terminated at step={step}", flush=True)
            break

    with open(os.path.join(args_cli.out, "diag.json"), "w") as f:
        json.dump(rows, f)
    print(f"[diag] wrote {len(rows)} rows", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
