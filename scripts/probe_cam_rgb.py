# Save head_camera RGB + depth as viewable PNGs while the robot turns to face the box,
# so we can SEE what the head camera actually images (is the ground box in view?).

import argparse
import math
import os

from isaaclab.app import AppLauncher

_DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo")

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="ATEC-TaskD-G1")
parser.add_argument("--num_steps", type=int, default=700)
parser.add_argument("--out", type=str, default="outputs/cam_rgb")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from atec_rl_lab.train.task_d.policy_bridge import G1VelocityPolicyBridge  # noqa: E402

try:
    from PIL import Image
except Exception:
    Image = None


def yaw_from_quat(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def save_png(arr, path):
    a = arr.detach().cpu().numpy()
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    bridge = G1VelocityPolicyBridge(policy_path=os.path.join(_DEMO_DIR, "policy_a.pt"), device="cuda")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    scene = env.unwrapped.scene
    robot, box = scene["robot"], scene["box"]
    cam = scene.sensors["head_camera"]
    obs, _ = env.reset()
    bridge.reset()

    saved = 0
    for step in range(args_cli.num_steps):
        rx, ry, rz = robot.data.root_pos_w[0].tolist()
        bx, by, bz = box.data.root_pos_w[0].tolist()
        yaw = yaw_from_quat(robot.data.root_quat_w[0].tolist())
        dir_to_box = math.atan2(by - ry, bx - rx)
        face_err = (dir_to_box - yaw + math.pi) % (2 * math.pi) - math.pi
        # turn base toward box, then creep forward
        wz = max(-0.8, min(0.8, 2.0 * face_err))
        vx = 0.4 if abs(face_err) < 0.25 else 0.0
        a = bridge.act(obs["proprio"], (vx, 0.0, wz))
        obs, *_ = env.step(torch.tensor(a, dtype=torch.float32, device="cuda").view(1, -1))

        dist = math.hypot(bx - rx, by - ry)
        if Image is not None and abs(face_err) < 0.12 and saved < 4 and step % 15 == 0:
            rgb = cam.data.output["rgb"][0][..., :3]
            depth = cam.data.output["depth"][0].squeeze(-1)
            save_png(rgb, os.path.join(args_cli.out, f"rgb_{saved}_d{dist:.1f}.png"))
            dn = depth.clone(); dn[~torch.isfinite(dn)] = 0
            dn = (dn / (dn.max() + 1e-6) * 255.0)
            save_png(dn, os.path.join(args_cli.out, f"depth_{saved}_d{dist:.1f}.png"))
            print(f"[rgb] saved {saved} dist={dist:.2f} base_yaw={yaw:.2f} dir_to_box={dir_to_box:.2f} "
                  f"cam_z={cam.data.pos_w[0][2].item():.2f}")
            saved += 1
        if saved >= 4 and dist < 1.0:
            break
    print(f"[rgb] done, {saved} images -> {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
