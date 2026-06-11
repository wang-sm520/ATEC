# Validate head-depth box perception DURING A REAL PUSH (box close, torso pitched).
# Robot approaches behind the box and pushes +x; every few steps it estimates the box
# (x,y) from head_depth and compares to god-view truth. Compares a fixed camera pitch
# vs a per-frame pitch derived from the proprio gravity vector (torso tilt).
#
# Uses TRUE camera pose for the transform to isolate the depth-detector error from
# odometry drift (which is a separate error source).
#
# Usage:
#   PYTHONPATH=. python scripts/probe_perception_validate.py --task=ATEC-TaskD-G1 \
#       --num_envs=1 --headless --enable_cameras --num_steps 700

import argparse
import math
import os

from isaaclab.app import AppLauncher

_DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo")

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="ATEC-TaskD-G1")
parser.add_argument("--num_steps", type=int, default=700)
parser.add_argument("--phi_deg", type=float, default=40.0)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from atec_rl_lab.train.task_d.policy_bridge import G1VelocityPolicyBridge  # noqa: E402


def yaw_from_quat(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def detect_box(depth, intr, cam_pos, yaw, phi):
    H, W = depth.shape
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    vv, uu = torch.meshgrid(torch.arange(H, device=depth.device, dtype=torch.float32),
                            torch.arange(W, device=depth.device, dtype=torch.float32), indexing="ij")
    d = depth
    valid = torch.isfinite(d) & (d > 0.05) & (d < 12.0)
    xr = ((uu - cx) / fx * d)[valid]
    yd = ((vv - cy) / fy * d)[valid]
    zf = d[valid]
    c, s = math.cos(yaw), math.sin(yaw)
    dev = depth.device
    f = torch.tensor([c, s, 0.0], device=dev); right = torch.tensor([s, -c, 0.0], device=dev)
    u = torch.tensor([0., 0., 1.], device=dev)
    fwd = math.cos(phi) * f - math.sin(phi) * u
    down = math.sin(phi) * f - math.cos(phi) * u
    cp = torch.tensor(cam_pos, device=dev)
    P = cp + xr[:, None] * right + yd[:, None] * down + zf[:, None] * fwd
    z = P[:, 2]
    p = P[(z > 0.06) & (z < 0.55)]
    if p.shape[0] < 50:
        return None
    rel = p[:, :2] - cp[:2]
    fwdc = rel[:, 0] * c + rel[:, 1] * s
    lat = -rel[:, 0] * s + rel[:, 1] * c
    m = (fwdc > 0.1) & (fwdc < 4.0) & (lat.abs() < 1.0)
    p, fwdc = p[m], fwdc[m]
    if p.shape[0] < 50:
        return None
    near = torch.quantile(fwdc, 0.05)
    pc = p[fwdc < near + 0.5]
    return (float(pc[:, 0].mean()) + 0.40 * c, float(pc[:, 1].mean()) + 0.40 * s)


def main():
    bridge = G1VelocityPolicyBridge(policy_path=os.path.join(_DEMO_DIR, "policy_a.pt"), device="cuda")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    scene = env.unwrapped.scene
    robot, box = scene["robot"], scene["box"]
    cam = scene.sensors["head_camera"]
    obs, _ = env.reset()
    bridge.reset()
    d435_idx = robot.data.body_names.index("d435_link")   # live camera link pose
    print(f"[validate] d435_link body idx = {d435_idx}")

    phi0 = math.radians(args_cli.phi_deg)
    phase = "approach"
    errs_fix, errs_grav = [], []
    for step in range(args_cli.num_steps):
        rx, ry, rz = robot.data.root_pos_w[0].tolist()
        bx, by, bz = box.data.root_pos_w[0].tolist()
        yaw = yaw_from_quat(robot.data.root_quat_w[0].tolist())

        # drive: behind box (-x), face +x, then push +x  (god-view driving)
        if phase == "approach":
            tx, ty = bx - 0.7, by
            ex, ey = tx - rx, ty - ry
            c, s = math.cos(yaw), math.sin(yaw)
            vx = clamp(1.3 * (c * ex + s * ey), -0.5, 0.8)
            vy = clamp(1.3 * (-s * ex + c * ey), -0.5, 0.5)
            wz = clamp(-2.0 * yaw, -1.0, 1.0)
            if rx < bx - 0.45 and abs(ry - by) < 0.15 and abs(yaw) < 0.12:
                phase = "push"
        else:
            vx = 0.85
            vy = clamp(1.2 * (by - ry), -0.4, 0.4)
            wz = clamp(-2.0 * yaw, -0.8, 0.8)
        a = bridge.act(obs["proprio"], (vx, vy, wz))
        obs, *_ = env.step(torch.tensor(a, dtype=torch.float32, device="cuda").view(1, -1))

        if phase == "push" and step % 12 == 0:
            depth = cam.data.output["depth"][0].squeeze(-1)
            intr = cam.data.intrinsic_matrices[0]
            cpos = robot.data.body_pos_w[0, d435_idx].tolist()   # LIVE camera position
            tyaw = yaw_from_quat(robot.data.root_quat_w[0].tolist())
            # per-frame pitch: fixed mount + torso pitch from proprio gravity
            pg = obs["proprio"][0, 9:12].tolist()
            torso_pitch = math.asin(clamp(pg[0], -1.0, 1.0))   # +x gravity comp ~ forward tilt
            phi_g = phi0 + torso_pitch
            d = math.hypot(bx - cpos[0], by - cpos[1])
            ef = detect_box(depth, intr, cpos, tyaw, phi0)
            eg = detect_box(depth, intr, cpos, tyaw, phi_g)
            erf = math.hypot(ef[0] - bx, ef[1] - by) if ef else float("nan")
            erg = math.hypot(eg[0] - bx, eg[1] - by) if eg else float("nan")
            if ef:
                errs_fix.append(erf)
            if eg:
                errs_grav.append(erg)
            efs = f"({ef[0]:.2f},{ef[1]:.2f})" if ef else "None"
            print(f"step={step:03d} robot=({rx:.2f},{ry:.2f}) yaw={math.degrees(tyaw):+.0f} cam=({cpos[0]:.2f},{cpos[1]:.2f},{cpos[2]:.2f}) "
                  f"d={d:.2f} true=({bx:.2f},{by:.2f}) est_fix={efs} fix_err={erf:.3f}")

    def stat(e):
        return f"mean={sum(e)/len(e):.3f} max={max(e):.3f} n={len(e)}" if e else "none"
    print(f"\n[fixed phi={args_cli.phi_deg}]  {stat(errs_fix)}")
    print(f"[gravity pitch]    {stat(errs_grav)}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
