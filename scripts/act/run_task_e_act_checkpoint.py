"""Run a trained Task-E ACT checkpoint directly.

This is a development runner, not a submission solution. It reuses the
training-time Agent from train_task_e.py so we can first verify that a saved
checkpoint loads and produces actions before packaging it into demo/solution.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T

DEFAULT_CHECKPOINT = Path(
    "runs/act-task-e-rgb-full-order-123-20260624-1925/checkpoints/best_loss.pt"
)
DEFAULT_TASK = "ATEC-TaskE-Piper"
IMAGE_SIZE = 224
ACTION_DIM = 8
NUM_QUERIES = 30  # ACT chunk length; must match train_task_e.Args.num_queries
# JointPositionActionCfg(scale=0.5, use_default_offset=True) in the Task-E env.
ACTION_SCALE = 0.5
DEFAULT_JOINT_POS = (0.0, 1.2, -1.5, 0.0, 1.2, 0.0, 0.035, -0.035)
# Collection home pose: the first recorded demo frame, i.e. where collector.py lands
# after its write(default)+IK warmup+settle. ATEC-TaskE-Piper resets joints to ~zeros
# (reset_robot_joints disabled), so eval MUST drive here before handing over to ACT,
# otherwise the policy starts far outside the training state/image distribution.
HOME_JOINT_POS = (0.0, 0.9245, -1.515, 0.0, 1.22, 0.0, 0.035, -0.035)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_checkpoint(path: str | None) -> Path:
    candidate = Path(path) if path else DEFAULT_CHECKPOINT
    if not candidate.is_absolute():
        candidate = _repo_root() / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"checkpoint not found: {candidate}")
    return candidate


def _patch_backbone_weight_download() -> None:
    """Prevent torchvision from downloading ResNet weights during model construction.

    The ACT checkpoint immediately overwrites model weights, so pretrained ImageNet
    initialization is unnecessary and can fail on offline machines.
    """
    try:
        import atec_rl_lab.train.act.act.detr.backbone as backbone_module
    except Exception:
        return
    backbone_module.is_main_process = lambda: False


def _import_train_components():
    act_dir = Path(__file__).resolve().parent
    if str(act_dir) not in sys.path:
        sys.path.insert(0, str(act_dir))
    _patch_backbone_weight_download()
    from train_task_e import Agent, Args  # noqa: WPS433

    return Agent, Args


def _to_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    return torch.as_tensor(value, device=device)


def absolute_state_from_relative_proprio(proprio_rel: Any, device: torch.device) -> torch.Tensor:
    rel = _to_tensor(proprio_rel, device).to(dtype=torch.float32)
    if rel.ndim == 1:
        rel = rel.unsqueeze(0)
    if rel.shape[-1] < ACTION_DIM:
        raise ValueError(f"expected at least {ACTION_DIM} relative joint values, got {rel.shape[-1]}")
    default = torch.tensor(DEFAULT_JOINT_POS, dtype=torch.float32, device=device).view(1, ACTION_DIM)
    return rel[:, :ACTION_DIM] + default


def prepare_state(proprio: Any, norm_stats: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    state = _to_tensor(proprio, device).to(dtype=torch.float32)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.shape[-1] < ACTION_DIM:
        raise ValueError(f"expected at least {ACTION_DIM} state values, got {state.shape[-1]}")
    state = state[:, :ACTION_DIM]
    mean = norm_stats["state_mean"].to(device=device, dtype=torch.float32)
    std = norm_stats["state_std"].to(device=device, dtype=torch.float32).clamp(min=1e-2)
    return (state - mean) / std


def prepare_rgb(rgb: Any, device: torch.device) -> torch.Tensor:
    x = _to_tensor(rgb, device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"expected RGB tensor with 3 or 4 dims, got {tuple(x.shape)}")

    if x.shape[-1] in (3, 4):  # BHWC / HWC
        x = x[..., :3].permute(0, 3, 1, 2).contiguous()
    elif x.shape[1] in (3, 4):  # BCHW / CHW
        x = x[:, :3].contiguous()
    else:
        raise ValueError(f"cannot identify RGB channel dimension in shape {tuple(x.shape)}")

    x = T.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)(x)
    return x.to(dtype=torch.float32).unsqueeze(1)  # (B, num_cams=1, 3, 224, 224), 0..255 like training data


def denormalize_action_chunk(chunk: torch.Tensor, norm_stats: dict[str, torch.Tensor]) -> torch.Tensor:
    mean = norm_stats["action_mean"].to(device=chunk.device, dtype=torch.float32)
    std = norm_stats["action_std"].to(device=chunk.device, dtype=torch.float32).clamp(min=1e-2)
    return chunk * std + mean


def compute_real_time_sleep(step_dt: float | None, elapsed_wall: float) -> float:
    if step_dt is None:
        return 0.0
    return max(0.0, float(step_dt) - float(elapsed_wall))


def home_warmup_action(default_jpos: torch.Tensor) -> torch.Tensor:
    """Env action that drives the arm to HOME_JOINT_POS under the Task-E action model
    (JointPositionActionCfg, use_default_offset=True, scale=ACTION_SCALE):
        joint_target = default_jpos + ACTION_SCALE * action
    Returned shape: (1, ACTION_DIM)."""
    default = default_jpos[:, :ACTION_DIM].to(dtype=torch.float32)
    home = torch.tensor(HOME_JOINT_POS, dtype=torch.float32, device=default.device).view(1, ACTION_DIM)
    return (home - default) / ACTION_SCALE


def collection_actuator_kwargs() -> dict[str, float]:
    """Actuator override used by scripts/act/task_e/env_setup.build_task_e_env during
    data collection (config.py ACT_*). The default ATEC-TaskE-Piper env uses
    effort_limit/velocity_limit=100, so demos were generated with stronger/faster
    actuators than the default eval env. Use --match_collection_actuators to reproduce
    the collection dynamics at eval time."""
    return {
        "effort_limit": 1000.0,
        "velocity_limit": 1000.0,
        "stiffness": 800.0,
        "damping": 80.0,
    }


def apply_collection_actuators(env_cfg) -> None:
    from isaaclab.actuators import ImplicitActuatorCfg

    env_cfg.scene.robot.actuators["default"] = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        **collection_actuator_kwargs(),
    )


class TemporalEnsembler:
    """ACT temporal ensembling (Zhao et al.).

    A fresh action chunk is predicted every step. At step t, every still-valid chunk
    predicted at step tau (tau in [t-num_queries+1, t]) contributes its prediction for
    time t, i.e. chunk[t - tau]. Those predictions are combined oldest->newest with
    weights exp(-m * i) (i=0 = oldest), matching the reference implementation, then
    normalised. Older predictions get the larger weight, which smooths the trajectory.
    """

    def __init__(self, num_queries: int, action_dim: int, m: float = 0.01,
                 device: torch.device | str = "cpu"):
        self.num_queries = int(num_queries)
        self.action_dim = int(action_dim)
        self.m = float(m)
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self._chunks: list[tuple[int, torch.Tensor]] = []  # (pred_step, chunk)

    def add_and_query(self, chunk: torch.Tensor) -> torch.Tensor:
        chunk = chunk.to(device=self.device, dtype=torch.float32)
        if chunk.ndim != 2 or chunk.shape[0] < 1:
            raise ValueError(f"chunk must be (num_queries, action_dim), got {tuple(chunk.shape)}")
        self._chunks.append((self._step, chunk))
        oldest_valid = self._step - self.num_queries + 1
        self._chunks = [(s, c) for s, c in self._chunks if s >= oldest_valid]

        preds = []
        for pred_step, c in self._chunks:  # ascending pred_step => oldest first
            offset = self._step - pred_step
            if 0 <= offset < c.shape[0]:
                preds.append(c[offset])
        stacked = torch.stack(preds, dim=0)  # (n, action_dim), oldest..newest
        idx = torch.arange(stacked.shape[0], device=self.device, dtype=torch.float32)
        weights = torch.exp(-self.m * idx).unsqueeze(1)
        action = (stacked * weights).sum(dim=0) / weights.sum().clamp(min=1e-8)

        self._step += 1
        return action


def select_action(action_chunk: torch.Tensor, mode: str, chunk_index: int) -> torch.Tensor:
    if action_chunk.ndim != 3 or action_chunk.shape[0] < 1:
        raise ValueError(f"expected action_chunk shape (B, T, A), got {tuple(action_chunk.shape)}")
    if mode == "first":
        index = 0
    elif mode == "chunk":
        index = min(max(int(chunk_index), 0), action_chunk.shape[1] - 1)
    else:
        raise ValueError(f"unsupported action mode: {mode}")
    return action_chunk[0, index]


def load_policy(checkpoint_path: Path, device: torch.device):
    Agent, Args = _import_train_components()
    ckpt = torch.load(checkpoint_path, map_location=device)
    norm_stats = {
        key: value.to(device=device, dtype=torch.float32)
        for key, value in ckpt["norm_stats"].items()
    }
    args = Args(include_rgb=True, cuda=device.type == "cuda")
    state_dim = int(norm_stats["state_mean"].shape[-1])
    act_dim = int(norm_stats["action_mean"].shape[-1])
    agent = Agent(state_dim, act_dim, args).to(device)
    weights = ckpt.get("ema_agent") or ckpt["agent"]
    agent.load_state_dict(weights, strict=True)
    agent.eval()
    return agent, norm_stats, args


def run_fake_smoke(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = resolve_checkpoint(args.checkpoint)
    agent, norm_stats, _ = load_policy(checkpoint, device)

    proprio = torch.zeros((1, ACTION_DIM), dtype=torch.float32, device=device)
    rgb = torch.zeros((1, 480, 640, 3), dtype=torch.uint8, device=device)
    obs = {
        "state": prepare_state(proprio, norm_stats, device),
        "rgb": prepare_rgb(rgb, device),
    }
    with torch.inference_mode():
        action_chunk = denormalize_action_chunk(agent.get_action(obs), norm_stats)
    first_action = action_chunk[0, 0]

    print(f"loaded checkpoint: {checkpoint}")
    print(f"device: {device}")
    print(f"state shape={tuple(obs['state'].shape)}")
    print(f"rgb shape={tuple(obs['rgb'].shape)}")
    print(f"action_chunk shape={tuple(action_chunk.shape)}")
    print(f"first_action shape={tuple(first_action.shape)}")
    print(f"first_action={first_action.detach().cpu().tolist()}")


def _extract_video_rgb(obs: dict[str, Any]) -> Any:
    image = obs.get("image", {})
    if isinstance(image, dict):
        for key in ("video_rgb", "ee_rgb", "head_rgb"):
            if image.get(key) is not None:
                return image[key]
    raise KeyError("obs does not contain image.video_rgb, image.ee_rgb, or image.head_rgb")


def _live_state(env, obs: dict[str, Any], state_source: str, device: torch.device) -> torch.Tensor:
    if state_source == "robot":
        robot = env.unwrapped.scene.articulations["robot"]
        return robot.data.joint_pos[:, :ACTION_DIM].to(device=device, dtype=torch.float32)
    return absolute_state_from_relative_proprio(obs["proprio"], device)


def run_live(args: argparse.Namespace) -> None:
    from isaaclab.app import AppLauncher

    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym  # noqa: WPS433
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: WPS433
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: WPS433
    import atec_rl_lab.tasks  # noqa: F401, WPS433

    try:
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        checkpoint = resolve_checkpoint(args.checkpoint)
        agent, norm_stats, _ = load_policy(checkpoint, device)
        print(f"loaded checkpoint: {checkpoint}")
        print(f"policy device: {device}")

        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
            use_fabric=not args.disable_fabric,
        )
        if args.match_collection_actuators:
            apply_collection_actuators(env_cfg)
            print(f"actuators: collection override {collection_actuator_kwargs()}")
        else:
            print("actuators: default env (effort_limit/velocity_limit=100)")
        env = gym.make(args.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        obs, _ = env.reset()
        total_episode_reward = 0.0
        total_elapsed_time = 0.0
        step_dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
        done = False
        cached_chunk: torch.Tensor | None = None
        chunk_index = 0
        ensembler = TemporalEnsembler(
            num_queries=NUM_QUERIES, action_dim=ACTION_DIM,
            m=args.temporal_m, device=device,
        ) if args.action_mode == "temporal" else None

        # Drive the arm from the env reset pose (~zeros) to the collection home pose so
        # ACT starts inside its training distribution. Deployable: pure joint actions, no
        # teleport — a submission solution can return these same actions for its first
        # warmup_steps predicts() calls.
        robot = env.unwrapped.scene.articulations["robot"]
        warm_action = home_warmup_action(robot.data.default_joint_pos).to(
            device=args.device, dtype=torch.float32
        )
        for _ in range(max(0, args.warmup_steps)):
            if not simulation_app.is_running():
                break
            wall_start = time.time()
            obs, _, _, _, _ = env.step(warm_action)
            if args.real_time:
                sleep_time = compute_real_time_sleep(step_dt, time.time() - wall_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        if args.warmup_steps > 0:
            jp = robot.data.joint_pos[0, :ACTION_DIM].detach().cpu().tolist()
            print(f"warmup done ({args.warmup_steps} steps); joint_pos={[round(v, 3) for v in jp]}")

        def object_z() -> dict[int, float]:
            zs: dict[int, float] = {}
            objs = env.unwrapped.scene.rigid_objects
            for i in (1, 2, 3):
                key = f"object_{i}"
                if key in objs:
                    zs[i] = float(objs[key].data.root_pos_w[0, 2].item())
            return zs

        init_z = object_z()
        max_lift = {i: 0.0 for i in init_z}

        for step in range(args.max_steps):
            if not simulation_app.is_running():
                break
            wall_start = time.time()
            # temporal/first re-predict every step; chunk re-predicts every NUM_QUERIES steps.
            need_predict = (
                args.action_mode in ("first", "temporal")
                or cached_chunk is None
                or chunk_index >= cached_chunk.shape[1]
            )
            if need_predict:
                state_raw = _live_state(env, obs, args.state_source, device)
                rgb_raw = _extract_video_rgb(obs)
                model_obs = {
                    "state": prepare_state(state_raw, norm_stats, device),
                    "rgb": prepare_rgb(rgb_raw, device),
                }
                with torch.inference_mode():
                    cached_chunk = denormalize_action_chunk(agent.get_action(model_obs), norm_stats)
                chunk_index = 0
            if ensembler is not None:
                action = ensembler.add_and_query(cached_chunk[0])
            else:
                action = select_action(cached_chunk, args.action_mode, chunk_index)
            action = action.to(device=args.device, dtype=torch.float32).view(1, -1)
            chunk_index += 1

            obs, reward, terminated, truncated, info = env.step(action)
            sim_dt = info.get("Step_dt", 1.0) if isinstance(info, dict) else 1.0
            if isinstance(reward, torch.Tensor):
                total_episode_reward += reward.mean().item() / sim_dt
            else:
                total_episode_reward += float(reward) / sim_dt
            if isinstance(info, dict) and "Elapsed_Time" in info:
                elapsed = info["Elapsed_Time"]
                total_elapsed_time = elapsed.item() if hasattr(elapsed, "item") else float(elapsed)

            for i, z in object_z().items():
                max_lift[i] = max(max_lift[i], z - init_z.get(i, z))

            done = bool(terminated.item() or truncated.item())
            if step < 5 or done:
                print(
                    f"step={step} action_shape={tuple(action.shape)} "
                    f"reward={float(reward.mean().item() if isinstance(reward, torch.Tensor) else reward):.4f} "
                    f"done={done}"
                )
            if done:
                break
            if args.real_time:
                sleep_time = compute_real_time_sleep(step_dt, time.time() - wall_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        print(f"score: {total_episode_reward:.2f}, elapsed_time: {total_elapsed_time:.2f}, done={done}")
        print("max_object_lift_m: " + ", ".join(f"object_{i}={max_lift[i]:.3f}" for i in sorted(max_lift)))
        if args.keep_open:
            print("Keeping Isaac window open. Close the window or press Ctrl+C in the terminal to exit.")
            try:
                while simulation_app.is_running():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
        env.close()
    finally:
        simulation_app.close()


def _add_common_args(parser: argparse.ArgumentParser, include_device: bool) -> None:
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--smoke-only", action="store_true", default=False)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--state_source", choices=("robot", "obs"), default="robot")
    parser.add_argument("--action_mode", choices=("first", "chunk", "temporal"), default="temporal")
    parser.add_argument("--temporal_m", type=float, default=0.01,
                        help="ACT temporal ensembling decay; larger = favor older predictions less.")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Joint-action steps driving the arm to the collection home pose "
                             "before ACT takes over (matches data-collection start; 0 disables).")
    parser.add_argument("--match_collection_actuators", action="store_true", default=False)
    parser.add_argument("--real_time", action="store_true", default=False)
    parser.add_argument("--keep_open", action="store_true", default=False)
    if include_device:
        parser.add_argument("--device", type=str, default="cuda")


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    _add_common_args(pre_parser, include_device=True)
    prelim, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Run a trained Task-E ACT checkpoint directly.")
    if prelim.smoke_only:
        _add_common_args(parser, include_device=True)
    else:
        _add_common_args(parser, include_device=False)
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_only:
        run_fake_smoke(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
