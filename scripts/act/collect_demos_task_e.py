"""Scripted oracle data collection for Task E (pick-and-place).

Usage
-----
# NOTE: only for object 3 now, if you want to pick objects 1 and 2, please modify the distance accordingly.
python scripts/act/collect_demos_task_e.py --pick_objects 3 --num_demos 50 --headless

"""

import argparse
import os
import sys

# sys.path.insert(0, os.path.dirname(__file__))

from isaaclab.app import AppLauncher
from cli_args import add_collect_demo_args

parser = argparse.ArgumentParser(description="Collect Task E demonstrations for ACT.")
add_collect_demo_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.save_video or args_cli.save_images:
    args_cli.enable_cameras = True

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py
import json
import numpy as np

from atec_rl_lab.utils import CartesianController

from task_e.config import (
    EE_BODY_NAME, ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES,
    OBJECT_GRASP_Z_OFFSETS, OPTIMIZED_STEPS,
)
from task_e.collector import collect_one_demo, get_objects_in_basket
from task_e.env_setup import build_task_e_env
from task_e.options import resolve_grasp_z_offsets, resolve_pick_objects



def init_output(output_dir: str) -> tuple[str, str]:
    """Create output directory, wipe any existing trajectory.hdf5, write JSON metadata."""
    os.makedirs(output_dir, exist_ok=True)
    traj_path = os.path.join(output_dir, "trajectory.hdf5")
    json_path = os.path.join(output_dir, "trajectory.json")
    with h5py.File(traj_path, "w"):   # truncate / create fresh
        pass
    with open(json_path, "w") as fh:
        json.dump({"env_info": {"env_kwargs": {"control_mode": "pd_joint_pos"}}}, fh)
    return traj_path, json_path


def save_traj(traj_path: str, traj_idx: int, data: dict,
              save_images: bool) -> None:
    """Append one trajectory group to the consolidated HDF5."""
    with h5py.File(traj_path, "a") as f:
        grp = f.create_group(f"traj_{traj_idx}")
        grp.create_dataset("obs",     data=data["qpos"],    compression="gzip")
        grp.create_dataset("actions", data=data["action"],  compression="gzip")
        grp.create_dataset("qvel",    data=data["qvel"],    compression="gzip")
        grp.create_dataset("ee_pos",  data=data["ee_pos"],  compression="gzip")
        grp.create_dataset("ee_quat", data=data["ee_quat"], compression="gzip")
        if save_images and "frames" in data:
            grp.create_group("images").create_dataset(
                "rgb", data=data["frames"], compression="gzip"
            )


def main() -> None:
    try:
        pick_objects = resolve_pick_objects(args_cli.pick_objects, args_cli.full_order_123)
        grasp_z_offsets = resolve_grasp_z_offsets(
            args_cli.grasp_z_offsets,
            args_cli.grasp_offset_json,
            defaults=OBJECT_GRASP_Z_OFFSETS,
        )
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc

    if args_cli.full_order_123 and not args_cli.only_success:
        print("[INFO] --full_order_123 enables --only_success for full-order datasets.")
        args_cli.only_success = True

    steps = OPTIMIZED_STEPS if args_cli.optimized_grasp_flow else None
    need_camera = args_cli.save_video or args_cli.save_images

    print(f"[INFO] pick_objects={pick_objects}")
    print(f"[INFO] grasp_z_offsets={grasp_z_offsets}")
    print(f"[INFO] tool_center_offset_local={args_cli.tool_center_offset_local}")
    print(f"[INFO] optimized_grasp_flow={args_cli.optimized_grasp_flow}")

    env    = build_task_e_env(pick_objects, need_camera)
    dev    = env.unwrapped.device
    camera = env.unwrapped.scene["video_cam"] if need_camera else None

    robot = env.unwrapped.scene.articulations["robot"]
    arm_ids,     _ = robot.find_joints(ARM_JOINT_NAMES)
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
    ik_ctrl = CartesianController(
        robot=robot, ee_body_name=EE_BODY_NAME,
        arm_joint_names=ARM_JOINT_NAMES,
        num_envs=1, device=dev,
        command_type="pose",
        lambda_val=0.05,
        max_joint_delta=0.2,
    )
    default_jpos = robot.data.default_joint_pos.clone()

    video_dir = None
    imageio   = None
    if args_cli.save_video:
        video_dir = args_cli.video_dir or os.path.join(args_cli.output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        import imageio as _io
        imageio = _io

    traj_path, _ = init_output(args_cli.output_dir)
    rng = np.random.default_rng()

    n_ok = 0
    attempt = 0
    while n_ok < args_cli.num_demos:
        if args_cli.max_attempts is not None and attempt >= args_cli.max_attempts:
            print(
                f"[ERROR] Reached --max_attempts={args_cli.max_attempts} "
                f"with {n_ok}/{args_cli.num_demos} successful demos."
            )
            env.close()
            raise SystemExit(1)

        attempt += 1
        print(f"\n[INFO] Demo {n_ok + 1}/{args_cli.num_demos}  (attempt {attempt})")
        print(f"[INFO] order={pick_objects} offsets={grasp_z_offsets}")

        data = collect_one_demo(
            env, robot, ik_ctrl,
            arm_ids, gripper_ids,
            pick_objects, dev,
            default_jpos=default_jpos,
            rng=rng,
            camera=camera,
            steps=steps,
            grasp_z_offsets=grasp_z_offsets,
            tool_center_offset_local=args_cli.tool_center_offset_local,
            use_basket_drop_height=args_cli.optimized_grasp_flow,
        )
        if data is None:
            print("[WARN] Early termination — skipping.")
            continue

        success_map = get_objects_in_basket(env, pick_objects)
        print(f"[INFO] success: {success_map}")
        if args_cli.only_success and not all(success_map.values()):
            print("[WARN] Objects not in basket — skipping (--only_success).")
            continue

        save_traj(traj_path, n_ok, data, args_cli.save_images)

        T     = len(data["qpos"])
        notes = [f"{T} steps"]
        if args_cli.save_video and "frames" in data:
            vp = os.path.join(video_dir, f"demo_{n_ok:04d}.mp4")
            imageio.mimwrite(vp, data["frames"], fps=50, quality=7)
            notes.append(f"video → {vp}")
        if args_cli.save_images and "frames" in data:
            notes.append("images saved")
        print(f"[INFO] traj_{n_ok}: {', '.join(notes)}")
        n_ok += 1

    print(f"\n[INFO] Collected {n_ok} demos → {traj_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
