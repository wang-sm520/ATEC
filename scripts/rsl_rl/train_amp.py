"""Train an RL agent with PPO + AMP (Adversarial Motion Priors).

Mirrors `scripts/rsl_rl/train.py` but swaps the algorithm class for AMPPPO and injects
AMP-specific kwargs (motion files, reward coefficients, etc.) into the agent cfg dict.

Degenerate mode: pass `--amp_motion_files ""` (empty) to run as plain PPO. This is used
to verify the AMPPPO subclass does not break the existing training pipeline.

Example (degenerate test on B2 PPO):
    python scripts/rsl_rl/train_amp.py --task ATEC-Isaac-Velocity-Flat-Unitree-B2-v0 \\
        --num_envs 16 --max_iterations 2 --headless

Example (full G1 AMP — once Stage 2 G1 env exists):
    python scripts/rsl_rl/train_amp.py --task ATEC-Isaac-AMP-Unitree-G1-v0 \\
        --num_envs 4096 --headless \\
        --amp_motion_files motion_data/g1_29dof/walk_forward.json motion_data/g1_29dof/stairs.json
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with PPO + AMP.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--export_io_descriptors", action="store_true", default=False)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# AMP-specific CLI
parser.add_argument(
    "--amp_motion_files",
    type=str,
    nargs="*",
    default=[],
    help="Paths to AMP expert motion JSON files. Empty list disables AMP losses (degenerate-PPO mode).",
)
parser.add_argument("--amp_reward_coef", type=float, default=0.3)              # bxi
parser.add_argument("--amp_task_reward_lerp", type=float, default=0.6)         # bxi
parser.add_argument("--amp_obs_dim", type=int, default=73, help="AMP obs dim; auto-overridden by env's obs['amp'] if present.")
parser.add_argument("--amp_replay_buffer_size", type=int, default=100000)      # bxi
parser.add_argument("--amp_num_preload_transitions", type=int, default=200000) # bxi
parser.add_argument("--amp_loss_coef", type=float, default=1.0)                # bxi (amploss_coef)
parser.add_argument("--amp_grad_pen_lambda", type=float, default=10.0)         # bxi

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    cmd_bin = r".\isaaclab.bat" if platform.system() == "Windows" else "./isaaclab.sh"
    print(
        f"Please install rsl-rl-lib >= {RSL_RL_VERSION}.\nExisting: {installed_version}\n"
        f"Run:\n\t{cmd_bin} -p -m pip install rsl-rl-lib=={RSL_RL_VERSION}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import atec_rl_lab.train  # noqa: F401  # isort: skip

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


AMPPPO_CLASS_PATH = "atec_rl_lab.algorithms.amp.amp_ppo.AMPPPO"


def patch_agent_dict_for_amp(agent_dict: dict, args) -> dict:
    """Swap algorithm class to AMPPPO and inject AMP-specific kwargs."""
    alg = agent_dict["algorithm"]
    alg["class_name"] = AMPPPO_CLASS_PATH
    alg["amp_motion_files"] = list(args.amp_motion_files) if args.amp_motion_files else []
    alg["amp_obs_dim"] = int(args.amp_obs_dim)
    alg["amp_reward_coef"] = float(args.amp_reward_coef)
    alg["amp_task_reward_lerp"] = float(args.amp_task_reward_lerp)
    alg["amp_replay_buffer_size"] = int(args.amp_replay_buffer_size)
    alg["amp_num_preload_transitions"] = int(args.amp_num_preload_transitions)
    alg["amp_loss_coef"] = float(args.amp_loss_coef)
    alg["amp_grad_pen_lambda"] = float(args.amp_grad_pen_lambda)
    # AMPPPO does not currently support RND/symmetry; force-disable
    alg["rnd_cfg"] = None
    alg["symmetry_cfg"] = None
    return agent_dict


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with PPO+AMP via OnPolicyRunner."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # Migrate deprecated `policy` cfg to `actor`/`critic` (rsl-rl >= 4.0.0 expects new schema)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training requires GPU.")

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.join("logs", "rsl_rl_amp", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Convert agent cfg to a dict and patch in AMP fields.
    agent_dict = agent_cfg.to_dict()
    agent_dict = patch_agent_dict_for_amp(agent_dict, args_cli)

    if not agent_dict["algorithm"]["amp_motion_files"]:
        print("[train_amp] amp_motion_files is empty -> running in degenerate PPO mode (AMP losses skipped).")
    else:
        print(f"[train_amp] AMP enabled with {len(agent_dict['algorithm']['amp_motion_files'])} motion file(s).")

    runner = OnPolicyRunner(env, agent_dict, log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
