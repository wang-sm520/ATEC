"""Sweep per-object grasp offsets for Task E ACT demonstration collection.

Example:
python scripts/act/sweep_task_e_grasp_offsets.py \
    --objects 1 2 3 \
    --offsets 0.070 0.075 0.080 0.085 0.090 0.095 \
    --attempts_per_offset 5 \
    --optimized_grasp_flow \
    --headless \
    --enable_cameras \
    --output datasets/atec_task_e/grasp_offset_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence


def build_sweep_payload(
    objects: Sequence[int],
    offsets: Sequence[float],
    attempts_per_offset: int,
    optimized_grasp_flow: bool,
    tool_center_offset_local: Sequence[float] | None,
    sweep_results: Mapping[int, tuple[list[dict], float]],
) -> dict:
    """Build the serializable sweep JSON payload."""
    payload = {
        "objects": [int(obj_id) for obj_id in objects],
        "offsets": [float(offset) for offset in offsets],
        "attempts_per_offset": int(attempts_per_offset),
        "optimized_grasp_flow": bool(optimized_grasp_flow),
        "tool_center_offset_local": (
            [float(v) for v in tool_center_offset_local]
            if tool_center_offset_local is not None else None
        ),
        "best_offsets": {},
        "results": {},
    }
    for obj_id in objects:
        rows, best = sweep_results[int(obj_id)]
        payload["results"][str(obj_id)] = rows
        payload["best_offsets"][str(obj_id)] = float(best)
    return payload


def _build_parser():
    from cli_args import add_collect_demo_args
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Sweep Task E per-object grasp offsets.")
    parser.add_argument("--objects", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--offsets", type=float, nargs="+",
        default=[0.070, 0.075, 0.080, 0.085, 0.090, 0.095],
    )
    parser.add_argument("--attempts_per_offset", type=int, default=5)
    parser.add_argument("--output", type=str, default="datasets/atec_task_e/grasp_offset_sweep.json")
    add_collect_demo_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _make_controller(env):
    from atec_rl_lab.utils import CartesianController
    from task_e.config import EE_BODY_NAME, ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES

    dev = env.unwrapped.device
    robot = env.unwrapped.scene.articulations["robot"]
    arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
    ik_ctrl = CartesianController(
        robot=robot,
        ee_body_name=EE_BODY_NAME,
        arm_joint_names=ARM_JOINT_NAMES,
        num_envs=1,
        device=dev,
        command_type="pose",
        lambda_val=0.05,
        max_joint_delta=0.2,
    )
    return dev, robot, arm_ids, gripper_ids, ik_ctrl, robot.data.default_joint_pos.clone()


def _sweep_object(obj_id: int, offsets: list[float], attempts_per_offset: int, args_cli) -> tuple[list[dict], float]:
    import numpy as np

    from task_e.config import OBJECT_GRASP_Z_OFFSETS, OPTIMIZED_STEPS
    from task_e.collector import collect_one_demo, get_objects_in_basket
    from task_e.env_setup import build_task_e_env
    from task_e.options import choose_best_offset

    print(f"\n[INFO] Sweeping object_{obj_id}")
    need_camera = bool(args_cli.save_video or args_cli.save_images)
    env = build_task_e_env([obj_id], need_camera=need_camera)
    camera = env.unwrapped.scene["video_cam"] if need_camera else None
    dev, robot, arm_ids, gripper_ids, ik_ctrl, default_jpos = _make_controller(env)
    rng = np.random.default_rng()
    steps = OPTIMIZED_STEPS if args_cli.optimized_grasp_flow else None
    imageio = None
    if args_cli.save_video:
        import imageio as _imageio
        imageio = _imageio
        video_dir = args_cli.video_dir or os.path.join(os.path.dirname(args_cli.output), "sweep_videos")
        os.makedirs(video_dir, exist_ok=True)
    else:
        video_dir = None

    rows: list[dict] = []
    for offset in offsets:
        successes = 0
        early_terminations = 0
        print(f"[INFO] object_{obj_id} offset={offset:.3f}")
        for attempt in range(1, attempts_per_offset + 1):
            offsets_for_attempt = dict(OBJECT_GRASP_Z_OFFSETS)
            offsets_for_attempt[obj_id] = float(offset)
            data = collect_one_demo(
                env, robot, ik_ctrl,
                arm_ids, gripper_ids,
                [obj_id], dev,
                default_jpos=default_jpos,
                rng=rng,
                camera=camera,
                steps=steps,
                grasp_z_offsets=offsets_for_attempt,
                tool_center_offset_local=args_cli.tool_center_offset_local,
                use_basket_drop_height=args_cli.optimized_grasp_flow,
            )
            if data is None:
                early_terminations += 1
                print(f"  attempt {attempt}/{attempts_per_offset}: early termination")
                continue

            success = get_objects_in_basket(env, [obj_id])[obj_id]
            successes += int(success)
            if args_cli.save_video and imageio is not None and "frames" in data:
                status = "success" if success else "fail"
                video_path = os.path.join(
                    video_dir,
                    f"object_{obj_id}_offset_{offset:.3f}_attempt_{attempt:03d}_{status}.mp4",
                )
                imageio.mimwrite(video_path, data["frames"], fps=50, quality=7)
                print(f"  video → {video_path}")
            print(f"  attempt {attempt}/{attempts_per_offset}: success={success}")

        row = {
            "object_id": obj_id,
            "offset": float(offset),
            "attempts": attempts_per_offset,
            "successes": successes,
            "success_rate": successes / attempts_per_offset if attempts_per_offset else 0.0,
            "early_terminations": early_terminations,
        }
        rows.append(row)
        print(
            f"[INFO] object_{obj_id} offset={offset:.3f}: "
            f"{successes}/{attempts_per_offset} "
            f"({row['success_rate'] * 100:.1f}%)"
        )

    best = choose_best_offset(rows)
    print(f"[INFO] object_{obj_id} best offset={best:.3f}")
    env.close()
    return rows, best


def main() -> None:
    from isaaclab.app import AppLauncher
    from task_e.options import resolve_pick_objects

    parser = _build_parser()
    args_cli = parser.parse_args()
    if args_cli.save_video or args_cli.save_images:
        args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        objects = resolve_pick_objects(args_cli.objects, full_order_123=False)
        results = {
            obj_id: _sweep_object(obj_id, [float(v) for v in args_cli.offsets], args_cli.attempts_per_offset, args_cli)
            for obj_id in objects
        }
        payload = build_sweep_payload(
            objects=objects,
            offsets=args_cli.offsets,
            attempts_per_offset=args_cli.attempts_per_offset,
            optimized_grasp_flow=args_cli.optimized_grasp_flow,
            tool_center_offset_local=args_cli.tool_center_offset_local,
            sweep_results=results,
        )

        output = args_cli.output
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

        print("\n[INFO] Sweep complete")
        print(f"[INFO] best_offsets={payload['best_offsets']}")
        print(f"[INFO] saved → {output}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
