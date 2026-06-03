# Task D ground-truth probe.
#
# Drives Task D with the current solution while logging PRIVILEGED state that the
# real submission never sees: true robot/box root_pos_w, whether the robot fell
# into the pit, the raw LiDAR ray hits (does MultiMeshRayCaster actually hit the
# Box?), and the height_scan pit/box signature.
#
# Usage:
#   PYTHONPATH=. python scripts/probe_task_d_groundtruth.py \
#       --task=ATEC-TaskD-G1 --num_envs=1 --headless --enable_cameras \
#       --num_steps 600 --out outputs/task_d_probe

import argparse
import math
import os
import json

from isaaclab.app import AppLauncher

from demo.solution import AlgSolution

parser = argparse.ArgumentParser(description="Task D ground-truth probe.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="ATEC-TaskD-G1")
parser.add_argument("--num_steps", type=int, default=600)
parser.add_argument("--log_every", type=int, default=10, help="Print/record every N steps.")
parser.add_argument("--snapshot_every", type=int, default=100, help="Dump full lidar grid every N steps.")
parser.add_argument("--out", type=str, default="outputs/task_d_probe", help="Output directory.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _yaw_from_quat(quat) -> float:
    # IsaacLab quats are (w, x, y, z).
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _grid_signature(flat, channels=16, bins=360):
    """Summarize a height_scan: pit = large positive, box/platform = small/negative."""
    vals = flat.reshape(channels, bins)
    finite = vals[torch.isfinite(vals)]
    if finite.numel() == 0:
        return {}
    median = float(finite.median())
    # Per-azimuth-bin max-over-channels (collapse elevation rings).
    col_max = torch.nan_to_num(vals, nan=-1e9).max(dim=0).values  # deepest hit per azimuth
    col_min = torch.nan_to_num(vals, nan=1e9).min(dim=0).values   # highest hit per azimuth
    pit_bins = torch.nonzero(col_max - median >= 0.4).flatten().tolist()
    elevated_bins = torch.nonzero(median - col_min >= 0.4).flatten().tolist()
    return {
        "median": round(median, 3),
        "min": round(float(finite.min()), 3),
        "max": round(float(finite.max()), 3),
        "n_pit_bins": len(pit_bins),
        "pit_bin_span": [min(pit_bins), max(pit_bins)] if pit_bins else None,
        "n_elevated_bins": len(elevated_bins),
        "elevated_bin_span": [min(elevated_bins), max(elevated_bins)] if elevated_bins else None,
    }


def main() -> None:
    os.makedirs(args_cli.out, exist_ok=True)
    solution = AlgSolution()

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    scene = env.unwrapped.scene
    robot = scene["robot"]
    box = scene["box"]
    lidar = scene.sensors.get("lidar_sensor") if hasattr(scene.sensors, "get") else scene.sensors["lidar_sensor"]
    env_origin = scene.env_origins[0].tolist()

    obs, _ = env.reset()
    if hasattr(solution, "reset"):
        solution.reset()

    print(f"[probe] env_origin={env_origin}")
    print(f"[probe] box_init_world={box.data.root_pos_w[0].tolist()}")
    print(f"[probe] robot_init_world={robot.data.root_pos_w[0].tolist()}")
    if lidar is not None:
        print(f"[probe] lidar.pos_w={lidar.data.pos_w[0].tolist()} ray_hits shape={tuple(lidar.data.ray_hits_w.shape)}")

    records = []
    score = 0.0
    min_robot_z = 1e9
    box_reached_target = False
    robot_max_x = -1e9

    for step in range(args_cli.num_steps):
        resp = solution.predicts(obs, score)
        action = torch.tensor(resp["action"], dtype=torch.float32, device="cuda").view(1, -1)
        obs, reward, terminated, truncated, info = env.step(action)
        sim_dt = info.get("Step_dt", 0.02) if isinstance(info, dict) else 0.02
        r = reward.mean().item() if isinstance(reward, torch.Tensor) else float(reward)
        score += r / sim_dt

        gt_robot = robot.data.root_pos_w[0].tolist()
        gt_box = box.data.root_pos_w[0].tolist()
        yaw = _yaw_from_quat(robot.data.root_quat_w[0].tolist())
        min_robot_z = min(min_robot_z, gt_robot[2])
        robot_max_x = max(robot_max_x, gt_robot[0])
        if -1.4 <= gt_box[0] <= 0.7:
            box_reached_target = True

        # LiDAR ground-truth: do any rays hit the box region (y around box_y, z>0.2)?
        lidar_box_hits = None
        if lidar is not None:
            hits = lidar.data.ray_hits_w[0]  # (num_rays, 3)
            finite = torch.isfinite(hits).all(dim=1)
            elevated = finite & (hits[:, 2] > 0.25)  # above ground, box top ~0.8
            lidar_box_hits = int(elevated.sum())

        if step % args_cli.log_every == 0 or terminated.item() or truncated.item():
            est = getattr(solution.controller, "last_debug", {}) if hasattr(solution, "controller") else {}
            rec = {
                "step": step,
                "score": round(score, 2),
                "robot_w": [round(v, 3) for v in gt_robot],
                "box_w": [round(v, 3) for v in gt_box],
                "robot_yaw": round(yaw, 3),
                "lidar_elevated_hits": lidar_box_hits,
                "est_robot_obs": est.get("robot_in_obstacle"),
                "est_box_obs": est.get("box_in_obstacle"),
                "phase": est.get("phase"),
                "cmd": est.get("cmd"),
            }
            records.append(rec)
            print(f"[probe] step={step:03d} score={score:5.1f} "
                  f"robot_w=({gt_robot[0]:.2f},{gt_robot[1]:.2f},{gt_robot[2]:.2f}) "
                  f"box_w=({gt_box[0]:.2f},{gt_box[1]:.2f},{gt_box[2]:.2f}) "
                  f"lidar_elev_hits={lidar_box_hits} phase={rec['phase']}")

        if step % args_cli.snapshot_every == 0 and "extero" in obs:
            flat = obs["extero"].detach().float().cpu().reshape(-1)
            torch.save(flat, os.path.join(args_cli.out, f"lidar_flat_step{step:04d}.pt"))
            sig = _grid_signature(flat)
            print(f"[probe]   lidar_signature@{step}: {sig}")
            if lidar is not None:
                torch.save(lidar.data.ray_hits_w[0].detach().float().cpu(),
                           os.path.join(args_cli.out, f"ray_hits_step{step:04d}.pt"))

        if terminated.item() or truncated.item():
            print(f"[probe] EPISODE END at step {step}: terminated={terminated.item()} truncated={truncated.item()}")
            break

    summary = {
        "final_score": round(score, 2),
        "robot_max_x_world": round(robot_max_x, 3),
        "min_robot_z_world": round(min_robot_z, 3),
        "fell_in_pit": min_robot_z < -0.3,
        "box_final_w": [round(v, 3) for v in box.data.root_pos_w[0].tolist()],
        "box_reached_target_x": box_reached_target,
        "env_origin": env_origin,
    }
    print(f"[probe] SUMMARY: {json.dumps(summary, indent=2)}")
    with open(os.path.join(args_cli.out, "trace.json"), "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"[probe] wrote {os.path.join(args_cli.out, 'trace.json')}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
