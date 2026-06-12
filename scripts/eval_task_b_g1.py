#!/usr/bin/env python3
"""Direct evaluator for demo.solution.AlgSolution on ATEC Task B G1."""

from __future__ import annotations

import argparse
import time
from typing import Any

from isaaclab.app import AppLauncher


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate demo.solution on ATEC-TaskB-G1.")
    parser.add_argument("--task", type=str, default="ATEC-TaskB-G1", help="Name of the task to evaluate.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--max_steps", type=int, default=6000, help="Maximum number of env steps to run.")
    parser.add_argument("--debug", action="store_true", default=False, help="Print per-step evaluator diagnostics.")
    parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if int(args.max_steps) <= 0:
        parser.error("--max_steps must be a positive integer")
    if int(args.num_envs) != 1:
        parser.error("--num_envs must be 1 for this single-solution evaluator")


def _is_tensor_like(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "reshape")


def _to_float(value: Any) -> float:
    if _is_tensor_like(value):
        tensor = value.detach().float().reshape(-1)
        if int(tensor.numel()) == 0:
            return 0.0
        return float(tensor.mean().item())
    if isinstance(value, dict):
        return _to_float(list(value.values()))
    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        return sum(_to_float(item) for item in value) / float(len(value))
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def any_done(value: Any) -> bool:
    """Return True when any scalar/list/tensor-like done flag is true."""
    if _is_tensor_like(value):
        flat = value.detach().reshape(-1)
        if int(flat.numel()) == 0:
            return False
        return bool(flat.bool().any().item())
    if isinstance(value, dict):
        return any(any_done(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(any_done(item) for item in value)
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def score_increment(reward: Any, step_dt: Any) -> float:
    dt = _to_float(step_dt)
    if dt <= 0.0:
        raise ValueError(f"Step_dt must be positive, got {dt}")
    return _to_float(reward) / dt


def coerce_action_tensor(action: Any, num_envs: int, device: str):
    import torch

    tensor = torch.as_tensor(action, dtype=torch.float32, device=device)
    expected_envs = int(num_envs)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2 and expected_envs == 1 and tensor.shape[1] == 1:
        tensor = tensor.reshape(1, -1)
    elif tensor.ndim > 2:
        tensor = tensor.reshape(tensor.shape[0], -1)

    if tensor.shape[0] == expected_envs:
        return tensor.contiguous()
    if tensor.shape[0] == 1 and expected_envs > 1:
        return tensor.repeat(expected_envs, 1).contiguous()
    if expected_envs == 1 and tensor.shape[0] > 1:
        return tensor[:1].contiguous()
    raise ValueError(f"action batch dimension {tensor.shape[0]} is incompatible with num_envs={expected_envs}")


def _extract_step_dt(info: Any, fallback_dt: float | None) -> float:
    if isinstance(info, dict) and "Step_dt" in info:
        dt = _to_float(info["Step_dt"])
    elif fallback_dt is not None:
        dt = float(fallback_dt)
    else:
        dt = 1.0
    if dt <= 0.0:
        raise ValueError(f"Step_dt must be positive, got {dt}")
    return dt


def _extract_elapsed_time(info: Any, current_elapsed: float, step_dt: float) -> float:
    if isinstance(info, dict) and "Elapsed_Time" in info:
        return _to_float(info["Elapsed_Time"])
    return current_elapsed + float(step_dt)


def _debug_tracks(solution: Any) -> str:
    planner = getattr(solution, "planner", None)
    perception = getattr(solution, "perception", None)
    touched = sorted(getattr(planner, "touched_track_ids", set())) if planner is not None else []
    placed = sorted(getattr(planner, "placed_track_ids", set())) if planner is not None else []
    active = getattr(getattr(planner, "active_detection", None), "track_id", None) if planner is not None else None
    perception_tracks = sorted(getattr(perception, "tracks", {}).keys()) if perception is not None else []
    return f"active={active} seen={perception_tracks} touched={touched} placed={placed}"


def _debug_phase(solution: Any) -> str:
    planner = getattr(solution, "planner", None)
    return str(getattr(planner, "phase", "unknown"))


def print_final_score(score: float, elapsed_time: float) -> None:
    print(f"score: {score:.2f}, elapsed_time: {elapsed_time:.2f} seconds", flush=True)


def run_evaluation(args: argparse.Namespace, simulation_app: Any) -> tuple[float, float]:
    import gymnasium as gym

    import atec_rl_lab.tasks  # noqa: F401
    from demo.solution import AlgSolution
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_tasks.utils import parse_env_cfg

    env = None
    try:
        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
            use_fabric=not args.disable_fabric,
        )
        env = gym.make(args.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        obs, _ = env.reset()
        solution = AlgSolution()
        solution.reset()

        num_envs = int(getattr(env.unwrapped, "num_envs", args.num_envs))
        fallback_dt = getattr(env.unwrapped, "step_dt", None)
        total_score = 0.0
        elapsed_time = 0.0

        for step in range(int(args.max_steps)):
            if not simulation_app.is_running():
                break
            loop_start = time.time()

            with __import__("torch").inference_mode():
                response = solution.predicts(obs, total_score)

            if bool(response.get("giveup", False)):
                if args.debug:
                    print(
                        f"[eval-b] step={step:04d} giveup=True phase={_debug_phase(solution)} "
                        f"score={total_score:.2f} elapsed_time={elapsed_time:.2f} tracks={_debug_tracks(solution)}",
                        flush=True,
                    )
                break

            actions = coerce_action_tensor(response["action"], num_envs=num_envs, device=args.device)
            obs, reward, terminated, truncated, info = env.step(actions)

            step_dt = _extract_step_dt(info, fallback_dt)
            total_score += score_increment(reward, step_dt)
            elapsed_time = _extract_elapsed_time(info, elapsed_time, step_dt)

            if args.debug:
                print(
                    f"[eval-b] step={step + 1:04d} phase={_debug_phase(solution)} "
                    f"score={total_score:.2f} elapsed_time={elapsed_time:.2f} "
                    f"reward={_to_float(reward):.4f} step_dt={step_dt:.4f} "
                    f"terminated={any_done(terminated)} truncated={any_done(truncated)} "
                    f"tracks={_debug_tracks(solution)}",
                    flush=True,
                )

            if any_done(terminated) or any_done(truncated):
                break

            if args.real_time:
                sleep_time = step_dt - (time.time() - loop_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

        return total_score, elapsed_time
    finally:
        if env is not None:
            env.close()


def main() -> tuple[float, float]:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        score, elapsed_time = run_evaluation(args, simulation_app)
        print_final_score(score, elapsed_time)
        return score, elapsed_time
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
