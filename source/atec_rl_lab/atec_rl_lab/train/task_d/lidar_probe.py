"""Runtime probe for Task D LiDAR perception.

Run with:
PYTHONPATH=source/atec_rl_lab python -m atec_rl_lab.train.task_d.lidar_probe --task ATEC-TaskD-G1 --num_steps 50 --enable_cameras
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from typing import Any

from .lidar_perception import TaskDLidarPerception
from .state_estimator import TaskDStateEstimator


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Task D obs['extero'] and LiDAR perception.")
    parser.add_argument("--task", default="ATEC-TaskD-G1")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--enable_cameras", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[lidar_probe] Ignoring unknown arguments: {' '.join(unknown)}")

    missing = _missing_isaaclab_dependencies()
    if missing:
        print(
            "[lidar_probe] IsaacLab runtime dependencies are unavailable: "
            + ", ".join(missing)
            + ". Run this probe inside an IsaacLab/ATEC environment.",
            file=sys.stderr,
        )
        return 2

    return _run_isaaclab_probe(args)


def _missing_isaaclab_dependencies() -> list[str]:
    required = ("isaaclab", "gymnasium")
    return [module for module in required if importlib.util.find_spec(module) is None]


def _run_isaaclab_probe(args: argparse.Namespace) -> int:
    from isaaclab.app import AppLauncher  # type: ignore

    app_args = argparse.Namespace(
        headless=args.headless,
        enable_cameras=args.enable_cameras,
    )
    app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app

    env = None
    try:
        import gymnasium as gym  # type: ignore
        import atec_rl_lab.tasks.task_d  # noqa: F401

        env = gym.make(args.task)
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        perception = TaskDLidarPerception()
        estimator = TaskDStateEstimator()

        for step in range(args.num_steps):
            extero = _find_extero(obs)
            obstacle, box = perception.measure(extero)
            robot_text = _robot_in_obstacle_text(obs, estimator, obstacle.frame)
            print(
                f"step={step:03d} "
                f"extero_shape={_shape_text(extero)} "
                f"obstacle_frame_valid={obstacle.frame.valid} "
                f"obstacle_confidence={obstacle.frame.confidence:.2f} "
                f"obstacle_source={obstacle.source} "
                f"robot_in_obstacle={robot_text} "
                f"box_lidar_valid={box.valid} "
                f"box_lidar_source={box.source} "
                f"box_confidence={box.confidence:.2f} "
                f"box_debug={perception.box_debug()}"
            )

            action = _zero_action(env.action_space)
            step_result = env.step(action)
            obs = step_result[0]
            done = bool(step_result[2]) if len(step_result) >= 3 else False
            truncated = bool(step_result[3]) if len(step_result) >= 4 else False
            if done or truncated:
                reset_result = env.reset()
                obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        return 0
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def _find_extero(obs: Any) -> Any:
    if isinstance(obs, dict):
        if "extero" in obs:
            return obs["extero"]
        for value in obs.values():
            found = _find_extero(value)
            if found is not None:
                return found
    return None


def _robot_in_obstacle_text(obs: Any, estimator: TaskDStateEstimator, obstacle_frame: Any) -> str:
    proprio = _find_proprio(obs)
    if proprio is None:
        return "unavailable"
    try:
        state = estimator.update(proprio, obstacle_frame=obstacle_frame)
    except (TypeError, ValueError):
        return "unavailable"
    return f"({state.pose_obstacle.x:.3f},{state.pose_obstacle.y:.3f},{state.pose_obstacle.yaw:.3f})"


def _find_proprio(obs: Any) -> Any:
    if isinstance(obs, dict):
        for key in ("proprio", "policy"):
            if key in obs:
                return obs[key]
        for value in obs.values():
            found = _find_proprio(value)
            if found is not None:
                return found
    return None


def _shape_text(value: Any) -> str:
    if value is None:
        return "None"
    shape = getattr(value, "shape", None)
    if shape is not None:
        return str(tuple(int(dim) for dim in shape))
    if isinstance(value, (list, tuple)):
        return str(_nested_shape(value))
    return type(value).__name__


def _nested_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        return (0,)
    return (len(value), *_nested_shape(value[0]))


def _zero_action(space: Any) -> Any:
    sample = space.sample()
    if hasattr(sample, "zero_"):
        return sample.zero_()
    if hasattr(sample, "fill"):
        sample.fill(0)
        return sample
    if isinstance(sample, dict):
        return {key: _zero_like(value) for key, value in sample.items()}
    return _zero_like(sample)


def _zero_like(value: Any) -> Any:
    if hasattr(value, "zero_"):
        return value.zero_()
    if hasattr(value, "fill"):
        value.fill(0)
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(_zero_like(item) for item in value)
    try:
        return type(value)(0)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
