"""录制 demo.solution.AlgSolution 在 ATEC-TaskB-G1 上跑一局的视频。

Usage:
  PYTHONPATH=. python scripts/record_task_b_g1.py \
    --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless \
    --video_length 900 --out logs/videos/task_b_g1
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a Task B G1 episode video.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-G1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video_length", type=int, default=900)
parser.add_argument("--out", type=str, default="logs/videos/task_b_g1")
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
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=os.path.abspath(args_cli.out),
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
        name_prefix="task_b_g1",
    )

    obs, _ = env.reset()
    sol = AlgSolution()
    sol.reset()
    total = 0.0
    last_phase = None
    for step in range(args_cli.video_length + 5):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            resp = sol.predicts(obs, total)
        actions = torch.tensor(resp["action"], dtype=torch.float32, device=args_cli.device).view(1, -1)
        obs, reward, terminated, truncated, info = env.step(actions)
        r = reward.mean().item() if isinstance(reward, torch.Tensor) else float(reward)
        sd = info.get("Step_dt") if isinstance(info, dict) else None
        sd = (sd.item() if hasattr(sd, "item") else float(sd)) if sd is not None else 0.02
        total += r / sd if sd else 0.0
        phase = getattr(sol.planner, "phase", "?")
        if phase != last_phase or step % 50 == 0:
            print(f"[rec] step={step} phase={phase} score={total:.2f}")
        last_phase = phase
        done = (bool(terminated.any()) if hasattr(terminated, "any") else bool(terminated)) or \
               (bool(truncated.any()) if hasattr(truncated, "any") else bool(truncated))
        if done:
            print(f"[rec] done at step={step}")
            break
    env.close()
    print(f"[rec] video saved under {os.path.abspath(args_cli.out)} ; final score {total:.2f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
