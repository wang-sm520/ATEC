"""Play a trained AMP-PPO checkpoint.

Mirror of `play.py` but loads checkpoints saved by `train_amp.py` (which use the AMPPPO
algorithm class). Looks under `logs/rsl_rl_amp/<experiment>/...` to find the run.

Examples:
    # Play latest checkpoint of latest run, 64 envs:
    python scripts/rsl_rl/play_amp.py --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0

    # Specific checkpoint, real-time playback, 16 envs:
    python scripts/rsl_rl/play_amp.py --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 \\
        --num_envs=16 --real-time \\
        --checkpoint logs/rsl_rl_amp/g1_amp_rough/2026-05-08_10-00-00/model_5000.pt

    # Record an MP4 of one rollout:
    python scripts/rsl_rl/play_amp.py --task=ATEC-Isaac-AMP-Unitree-G1-TaskA-v0 \\
        --num_envs=16 --video --video_length=600
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a trained AMP-PPO agent.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=600)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--real-time", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy
import importlib.metadata as metadata
import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)


class _ActorInferenceModule(torch.nn.Module):
    """Deterministic-mean inference wrapper for rsl-rl-lib >= 3.0.0 MLPModel actors.

    The IsaacLab exporter assumes the older API where `policy_nn.actor` is a plain MLP
    and the normalizer lives separately on the runner. In rsl-rl 3.0.1 the actor is an
    `MLPModel` with a built-in `obs_normalizer` and a `GaussianDistribution` head; calling
    it stochastically samples actions, which we don't want for jit export.

    This module rebuilds the inference path: `obs_normalizer -> mlp` (mean of Gaussian).
    Output shape: (batch, action_dim).
    """

    def __init__(self, actor: torch.nn.Module):
        super().__init__()
        if not hasattr(actor, "obs_normalizer") or not hasattr(actor, "mlp"):
            raise ValueError("Actor module must have 'obs_normalizer' and 'mlp' attributes.")
        self.obs_normalizer = copy.deepcopy(actor.obs_normalizer)
        self.mlp = copy.deepcopy(actor.mlp)
        self.eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.obs_normalizer(obs))


def _export_actor(actor: torch.nn.Module, obs_dim: int, action_dim: int, path: str) -> None:
    """Save TorchScript .pt and ONNX .onnx of the deterministic actor to `path`."""
    os.makedirs(path, exist_ok=True)
    infer = _ActorInferenceModule(actor).cpu()
    example = torch.zeros(1, obs_dim, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(infer, example, check_trace=False)
    pt_path = os.path.join(path, "policy.pt")
    traced.save(pt_path)
    print(f"[INFO] Saved TorchScript policy: {pt_path}")
    onnx_path = os.path.join(path, "policy.onnx")
    torch.onnx.export(
        infer,
        example,
        onnx_path,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )
    print(f"[INFO] Saved ONNX policy: {onnx_path} (input dim={obs_dim}, action dim={action_dim})")

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import atec_rl_lab.train  # noqa: F401

AMPPPO_CLASS_PATH = "atec_rl_lab.algorithms.amp.amp_ppo.AMPPPO"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Spawn at random terrain levels (not curriculum-driven), trim grid to save memory.
    env_cfg.scene.terrain.max_init_terrain_level = None
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False
    # Disable noise / pushes for clean inspection.
    env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg.events, "randomize_apply_external_force_torque"):
        env_cfg.events.randomize_apply_external_force_torque = None
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.curriculum, "command_levels_lin_vel"):
        env_cfg.curriculum.command_levels_lin_vel = None
    if hasattr(env_cfg.curriculum, "command_levels_ang_vel"):
        env_cfg.curriculum.command_levels_ang_vel = None

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl_amp", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Patch agent dict to use AMPPPO so the discriminator weights load correctly.
    agent_dict = agent_cfg.to_dict()
    agent_dict["algorithm"]["class_name"] = AMPPPO_CLASS_PATH
    agent_dict["algorithm"]["rnd_cfg"] = None
    agent_dict["algorithm"]["symmetry_cfg"] = None
    # AMP fields are not used at inference but must exist for AMPPPO.__init__.
    agent_dict["algorithm"].setdefault("amp_motion_files", [])
    agent_dict["algorithm"].setdefault("amp_obs_dim", 73)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_dict, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    actor_module = runner.alg.actor
    linear_layers = [m for m in actor_module.mlp.modules() if isinstance(m, torch.nn.Linear)]
    if not linear_layers:
        raise RuntimeError("Could not find any Linear layers inside actor.mlp.")
    obs_dim = int(linear_layers[0].in_features)
    action_dim = int(linear_layers[-1].out_features)
    export_model_dir = os.path.join(log_dir, "exported")
    _export_actor(actor_module, obs_dim=obs_dim, action_dim=action_dim, path=export_model_dir)
    policy_nn = actor_module

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
