"""标定 G1 右臂关节 → 右手相机近物 blob 的运动方向/灵敏度（手眼 Jacobian 符号）。

流程：用 AlgSolution 驱动到稳定深蹲且右手相机看到近物 → 切手动保持 → 逐个右臂关节
施加 ±delta，测 blob (cx,cy) 与 depth 的变化 → 输出每个关节的 dcx/dq, dcy/dq, ddepth/dq。

仅本地标定用。

Usage:
  PYTHONPATH=. python scripts/probe_task_b_g1_jointcal.py \
    --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless
"""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Calibrate G1 right-arm joint -> hand-cam blob motion.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-G1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--warmup_steps", type=int, default=300)
parser.add_argument("--delta", type=float, default=0.5)
parser.add_argument("--settle", type=int, default=18)
parser.add_argument("--out", type=str, default="outputs/task_b_g1_jointcal")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401,E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from demo.solution_task_b_g1 import AlgSolution, SquatCommand  # noqa: E402

# right arm joints to calibrate (action indices)
RIGHT_JOINTS = {22: "r_shoulder_pitch", 23: "r_shoulder_roll", 24: "r_shoulder_yaw", 25: "r_elbow"}
# steady arm baseline (no sweep swing) that keeps the object in the right-hand view
BASELINE_ARM = {15: 0.7, 18: 0.6, 22: 0.7, 25: 0.6, 29: 0.4, 30: 0.4, 31: 0.4, 32: 0.4}
SQUAT_CMD = SquatCommand(height=0.40, pitch=0.25)


def colored_near_blob(rgb, depth, lo=0.08, hi=0.6):
    if rgb is None or depth is None:
        return None
    rgbt = rgb.detach().cpu()
    if rgbt.ndim == 4:
        rgbt = rgbt[0]
    r = rgbt[..., 0].float(); g = rgbt[..., 1].float(); b = rgbt[..., 2].float()
    maxc = torch.maximum(torch.maximum(r, g), b)
    minc = torch.minimum(torch.minimum(r, g), b)
    sat = maxc - minc
    mask = ((r > 120) & (g > 90) & (b < 130)) | ((r > 130) & (g > 45) & (b < 150) & (r > b + 35)) | ((maxc > 110) & (sat > 45))
    d = depth.detach().cpu()
    if d.ndim == 4:
        d = d[0]
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    near = mask & torch.isfinite(d) & (d > lo) & (d < hi)
    n = int(near.sum().item())
    if n < 25:
        return None
    ys, xs = torch.nonzero(near, as_tuple=True)
    return {"n": n, "cx": float(xs.float().mean()), "cy": float(ys.float().mean()), "depth": float(d[near].median())}


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    sol = AlgSolution()
    obs, _ = env.reset()
    sol.reset()
    dev = args_cli.device

    def hand_blob():
        img = obs.get("image", {})
        return colored_near_blob(img.get("ee_dual_rgb"), img.get("ee_dual_depth"))

    def step_action(arm_overrides):
        nonlocal obs
        proprio = obs["proprio"]
        legs = sol.squat_bridge.act(proprio, SQUAT_CMD)
        action = [0.0] * 33
        action[:12] = legs
        for idx, val in BASELINE_ARM.items():
            action[idx] = float(val)
        for idx, val in arm_overrides.items():
            action[idx] = float(val)
        t = torch.tensor(action, dtype=torch.float32, device=dev).view(1, -1)
        obs, _, term, trunc, _ = env.step(t)
        return bool(term.any()) or bool(trunc.any())

    # Phase 1: let AlgSolution navigate into a squat near an object
    got = False
    for step in range(args_cli.warmup_steps):
        if not simulation_app.is_running():
            break
        resp = sol.predicts(obs, 0.0)
        t = torch.tensor(resp["action"], dtype=torch.float32, device=dev).view(1, -1)
        obs, _, term, trunc, _ = env.step(t)
        if getattr(sol.planner, "phase", "") == "squat_sweep" and hand_blob() is not None:
            got = True
            print(f"[jointcal] reached squat with right-hand blob at step {step}")
            break
        if bool(term.any()) or bool(trunc.any()):
            print(f"[jointcal] episode ended early at step {step}")
            break

    result = {"reached_squat_with_blob": got, "delta": args_cli.delta, "joints": {}}
    if got:
        # Phase 2: manual steady hold, settle, measure baseline
        for _ in range(args_cli.settle):
            step_action({})
        base = None
        for _ in range(6):
            b = hand_blob()
            if b is not None:
                base = b
            step_action({})
        result["baseline_blob"] = base
        # save baseline right-hand image so we can SEE if the near-blob is the object or the hand
        try:
            from PIL import Image
            img = obs.get("image", {})
            rgb = img.get("ee_dual_rgb")
            if rgb is not None:
                a = rgb.detach().cpu()
                if a.ndim == 4:
                    a = a[0]
                Image.fromarray(a[..., :3].clamp(0, 255).to(torch.uint8).numpy()).save(
                    os.path.join(args_cli.out, "baseline_R_rgb.png"))
        except Exception as exc:
            print("[jointcal] image save failed:", exc)

        # Phase 3: perturb each right joint +/- delta
        if base is not None:
            for idx, name in RIGHT_JOINTS.items():
                jrec = {}
                for sign in (+1.0, -1.0):
                    for _ in range(args_cli.settle):
                        step_action({idx: BASELINE_ARM.get(idx, 0.0) + sign * args_cli.delta})
                    meas = None
                    for _ in range(5):
                        m = hand_blob()
                        if m is not None:
                            meas = m
                        step_action({idx: BASELINE_ARM.get(idx, 0.0) + sign * args_cli.delta})
                    jrec[f"{'plus' if sign > 0 else 'minus'}"] = meas
                    # relax back to baseline before next perturbation
                    for _ in range(args_cli.settle):
                        step_action({})
                # derivative estimate (per +delta) from plus vs minus
                p, m = jrec.get("plus"), jrec.get("minus")
                if p and m:
                    jrec["dcx_per_unit"] = (p["cx"] - m["cx"]) / (2 * args_cli.delta)
                    jrec["dcy_per_unit"] = (p["cy"] - m["cy"]) / (2 * args_cli.delta)
                    jrec["ddepth_per_unit"] = (p["depth"] - m["depth"]) / (2 * args_cli.delta)
                result["joints"][name] = {"idx": idx, **jrec}
                print(f"[jointcal] {name}(idx{idx}): "
                      f"dcx={jrec.get('dcx_per_unit')}, dcy={jrec.get('dcy_per_unit')}, ddepth={jrec.get('ddepth_per_unit')}")

    with open(os.path.join(args_cli.out, "jointcal.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("[jointcal] result:", json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
