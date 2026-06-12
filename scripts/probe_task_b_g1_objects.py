"""Calibrate Task B G1 RGB-D object detections against local scene truth.

This script intentionally uses privileged env.scene access for local calibration
only. Do not copy scene-truth access into demo/task_b_perception.py.

Usage:
  PYTHONPATH=. python scripts/probe_task_b_g1_objects.py \
    --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless \
    --num_steps 60 --out outputs/task_b_g1_objects
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Probe Task B G1 object detections against scene truth.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-G1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=120)
parser.add_argument("--out", type=str, default="outputs/task_b_g1_objects")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_steps <= 0:
    parser.error("--num_steps must be a positive integer")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401,E402
from demo.task_b_nav import Pose2D  # noqa: E402
from demo.task_b_perception import TaskBRgbdPerception  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


OBJECT_NAMES = tuple(f"object_{idx}" for idx in range(1, 19))


def yaw_from_quat_wxyz(quat: Any) -> float:
    """Return planar yaw from an IsaacLab quaternion ordered as (w, x, y, z)."""
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _row_values(tensor: Any, env_index: int, width: int) -> list[float]:
    row = tensor[env_index, :width]
    if hasattr(row, "detach"):
        row = row.detach().cpu()
    return [float(v) for v in row.tolist()]


def _robot_pose(robot: Any, env_index: int) -> dict[str, float]:
    pos = _row_values(robot.data.root_pos_w, env_index, 3)
    quat = _row_values(robot.data.root_quat_w, env_index, 4)
    return {"x": pos[0], "y": pos[1], "z": pos[2], "yaw": yaw_from_quat_wxyz(quat)}


def _object_kind(index: int) -> str:
    if index <= 6:
        return "sugar"
    if index <= 12:
        return "mustard"
    return "banana"


def _truth_objects(scene: Any, env_index: int) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for index, name in enumerate(OBJECT_NAMES, start=1):
        obj = scene[name]
        x, y, z = _row_values(obj.data.root_pos_w, env_index, 3)
        objects.append({"name": name, "kind": _object_kind(index), "x": x, "y": y, "z": z})
    return objects


def _slice_image_obs(image_obs: Any, env_index: int, num_envs: int) -> dict[str, Any]:
    if not isinstance(image_obs, dict):
        return {}

    sliced: dict[str, Any] = {}
    for key, value in image_obs.items():
        shape = tuple(value.shape) if hasattr(value, "shape") else ()
        if shape and int(shape[0]) == int(num_envs):
            # Keep RGB-D tensors batched when they have a channel dimension; the
            # perception helper already handles a leading batch dimension.
            if len(shape) >= 4:
                sliced[key] = value[env_index : env_index + 1]
            else:
                sliced[key] = value[env_index]
        else:
            sliced[key] = value
    return sliced


def _nearest_object(det: Any, objects: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float | None]:
    if not objects:
        return None, None
    nearest = min(objects, key=lambda obj: math.hypot(float(det.world_x) - obj["x"], float(det.world_y) - obj["y"]))
    error = math.hypot(float(det.world_x) - nearest["x"], float(det.world_y) - nearest["y"])
    return nearest, error


def _serialize_detection(det: Any, objects: list[dict[str, Any]]) -> dict[str, Any]:
    nearest, error = _nearest_object(det, objects)
    return {
        "track_id": int(det.track_id),
        "label": str(det.label),
        "rel_x": float(det.rel_x),
        "rel_y": float(det.rel_y),
        "distance": float(det.distance),
        "confidence": float(det.confidence),
        "world_x": float(det.world_x),
        "world_y": float(det.world_y),
        "bbox": [int(v) for v in det.bbox],
        "nearest_object": None if nearest is None else nearest["name"],
        "nearest_error": None if error is None else float(error),
    }


def _done_flags(value: Any, num_envs: int) -> list[bool]:
    if hasattr(value, "detach"):
        flat = value.detach().cpu().reshape(-1).tolist()
        return [bool(v) for v in flat]
    if isinstance(value, (list, tuple)):
        return [bool(v) for v in value]
    return [bool(value)] * int(num_envs)


def _step_rows(
    *,
    step: int,
    obs: dict[str, Any],
    scene: Any,
    robot: Any,
    perceptions: list[TaskBRgbdPerception],
    terminated: list[bool],
    truncated: list[bool],
    num_envs: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    image_obs = obs.get("image", {}) if isinstance(obs, dict) else {}
    for env_index in range(num_envs):
        robot_pose = _robot_pose(robot, env_index)
        pose_2d = Pose2D(robot_pose["x"], robot_pose["y"], robot_pose["yaw"])
        objects = _truth_objects(scene, env_index)
        detections = perceptions[env_index].update(_slice_image_obs(image_obs, env_index, num_envs), pose_2d)
        serialized = [_serialize_detection(det, objects) for det in detections]
        errors = [det["nearest_error"] for det in serialized if det["nearest_error"] is not None]
        rows.append(
            {
                "step": int(step),
                "env_index": int(env_index),
                "terminated": bool(terminated[env_index]) if env_index < len(terminated) else False,
                "truncated": bool(truncated[env_index]) if env_index < len(truncated) else False,
                "robot_pose": robot_pose,
                "objects": objects,
                "detections": serialized,
                "nearest_error": None if not errors else float(min(errors)),
            }
        )
    return rows


def _format_error(value: float | None) -> str:
    return "none" if value is None else f"{value:.3f}"


def main() -> None:
    os.makedirs(args_cli.out, exist_ok=True)
    env = None
    try:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        env = gym.make(args_cli.task, cfg=env_cfg)
        obs, _ = env.reset()

        scene = env.unwrapped.scene
        robot = scene["robot"]
        num_envs = int(getattr(env.unwrapped, "num_envs", args_cli.num_envs))
        perceptions = [TaskBRgbdPerception() for _ in range(num_envs)]
        records: list[dict[str, Any]] = []

        for step in range(args_cli.num_steps):
            action_dim = int(env.unwrapped.action_space.shape[-1])
            actions = torch.zeros((num_envs, action_dim), dtype=torch.float32, device=args_cli.device)
            obs, _reward, terminated_raw, truncated_raw, _info = env.step(actions)
            terminated = _done_flags(terminated_raw, num_envs)
            truncated = _done_flags(truncated_raw, num_envs)

            step_records = _step_rows(
                step=step,
                obs=obs,
                scene=scene,
                robot=robot,
                perceptions=perceptions,
                terminated=terminated,
                truncated=truncated,
                num_envs=num_envs,
            )
            records.extend(step_records)

            done = any(terminated) or any(truncated)
            if step % 20 == 0:
                detection_count = sum(len(row["detections"]) for row in step_records)
                errors = [row["nearest_error"] for row in step_records if row["nearest_error"] is not None]
                best_error = None if not errors else min(errors)
                print(
                    f"[objects] step={step:04d} rows={len(step_records)} "
                    f"objects_per_row={len(step_records[0]['objects']) if step_records else 0} "
                    f"detections={detection_count} nearest_error={_format_error(best_error)} "
                    f"terminated={any(terminated)} truncated={any(truncated)}",
                    flush=True,
                )
            if done:
                print(
                    f"[objects] episode_end step={step:04d} terminated={terminated} truncated={truncated}",
                    flush=True,
                )
                break

        payload = {
            "task": args_cli.task,
            "num_envs": num_envs,
            "num_steps_requested": int(args_cli.num_steps),
            "rows": records,
        }
        out_path = os.path.join(args_cli.out, "detections_vs_truth.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[objects] wrote {len(records)} rows to {out_path}", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
