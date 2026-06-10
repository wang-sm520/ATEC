"""Smoke-test the mini whole-body controller (policy18.onnx) inside ATEC-TaskB-G1.

Adapts the mini loco-manip ONNX policy to the ATEC env: builds the 115-d obs,
5-frame history, and 15-d ik_input from ATEC proprio, converts the 29-d policy
action (isaaclab joint order) to the ATEC 33-d action (dex1 order).

Command schedule: stand -> walk forward -> stop + lower base + reach right hand down.
Prints base height, tilt, fall, so we can see if the WBC balances in ATEC.

Usage:
  PYTHONPATH=. python scripts/eval_miniwbc.py \
    --task ATEC-TaskB-G1 --num_envs 1 --enable_cameras --headless --max_steps 1500
"""

from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke-test mini WBC in ATEC.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-G1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=1500)
parser.add_argument("--mini_root", type=str, default="/home/hpf/atec/ATEC2026_Simulation_Challenge/mini")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import onnxruntime as ort  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401,E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# ---- joint orders (from mini/utils.py). mujoco order == ATEC dex1 first-29 order. ----
ISAACLAB_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint",
    "right_hip_roll_joint", "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "waist_pitch_joint", "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
ATEC_JOINTS = [  # == mujoco_joints in mini/utils.py == ATEC dex1 first 29
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
ATEC_TO_POLICY = np.array([ATEC_JOINTS.index(j) for j in ISAACLAB_JOINTS])  # policy[i]=atec[ATEC_TO_POLICY[i]]
POLICY_TO_ATEC = np.array([ISAACLAB_JOINTS.index(j) for j in ATEC_JOINTS])  # atec[i]=policy[POLICY_TO_ATEC[i]]

DEFAULT_ANGLES_ATEC = np.array([  # 29, ATEC order (from mini yaml default_angles, which is ATEC order)
    -0.2, 0.0, 0.0, 0.42, -0.23, 0.0, -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
    0.0, 0.0, 0.0, 0.35, 0.16, 0.0, 0.87, 0.0, 0.0, 0.0, 0.35, -0.16, 0.0, 0.87, 0.0, 0.0, 0.0,
], dtype=np.float32)
ACTION_SCALE = 0.25
ATEC_ACTION_SCALE = 0.5
HIS_LEN, NUM_OBS = 5, 115


class MiniWBC:
    def __init__(self, policy_path):
        self.sess = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        self.default_policy = DEFAULT_ANGLES_ATEC[ATEC_TO_POLICY]  # default in policy order (unused but kept)
        self.reset()

    def reset(self):
        self.obs_history = np.zeros((HIS_LEN, NUM_OBS), dtype=np.float32)
        self.last_action = np.zeros(29, dtype=np.float32)
        self.ik_output = None

    def act(self, proprio_row, vel_cmd, base_height, waist_rpy, left_hand, right_hand, waist_weight=1.0):
        row = np.asarray(proprio_row, dtype=np.float32)
        ang_vel = row[3:6]
        gravity = row[9:12]
        jp_atec = row[12:12 + 29]            # body joint_pos (relative to default), ATEC order
        jv_atec = row[12 + 33:12 + 33 + 29]  # body joint_vel, ATEC order
        dof_pos = jp_atec[ATEC_TO_POLICY]    # -> policy order, dof_pos_scale=1
        dof_vel = jv_atec[ATEC_TO_POLICY]
        full_command = np.concatenate([
            np.asarray(vel_cmd, np.float32), np.asarray([base_height], np.float32),
            np.asarray(waist_rpy, np.float32), np.asarray(left_hand, np.float32),
            np.asarray(right_hand, np.float32), np.asarray([waist_weight], np.float32),
        ])
        obs = np.concatenate([ang_vel, gravity, full_command, dof_pos, dof_vel, self.last_action]).astype(np.float32)
        self.obs_history = np.concatenate([self.obs_history[1:], obs.reshape(1, -1)])
        ik_input = np.concatenate([left_hand, right_hand, [waist_weight]]).astype(np.float32)

        outs = self.sess.run(None, {
            "obs": obs[None], "obs_history": self.obs_history[None], "ik_input": ik_input[None],
        })
        action = outs[0].squeeze(0).astype(np.float32)   # 29, policy order
        self.ik_output = outs[1].squeeze(0).astype(np.float32)  # 17
        self.last_action = action

        # replicate apply_action: work in ATEC (mujoco) order
        dof = action[POLICY_TO_ATEC] * ACTION_SCALE       # -> ATEC order, *0.25
        temp_default = DEFAULT_ANGLES_ATEC.copy()
        if self.ik_output is not None and len(self.ik_output) == 17:
            temp_default[12:29] = self.ik_output
        dof[12:29] *= 0.4
        target_q = dof + temp_default                     # target joint angles (ATEC order)
        # convert to ATEC action (env applies target = default + action*0.5)
        atec_action_body = (target_q - DEFAULT_ANGLES_ATEC) / ATEC_ACTION_SCALE
        action_33 = np.zeros(33, dtype=np.float32)
        action_33[:29] = atec_action_body                 # hands (29:33) stay 0 -> default
        return action_33.tolist()


def main():
    policy_path = os.path.join(args_cli.mini_root, "model/0116/policy18.onnx")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    robot = env.unwrapped.scene["robot"]
    wbc = MiniWBC(policy_path)
    obs, _ = env.reset()
    wbc.reset()

    left_hand = [0.3, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0]
    right_hand = [0.3, -0.2, 0.0, 1.0, 0.0, 0.0, 0.0]

    fell = False
    for step in range(args_cli.max_steps):
        if not simulation_app.is_running():
            break
        t = step * 0.02
        # command schedule
        if t < 3.0:
            vel, height, rh = [0.0, 0.0, 0.0], 0.75, [0.3, -0.2, 0.0, 1, 0, 0, 0]   # stand
        elif t < 8.0:
            vel, height, rh = [0.4, 0.0, 0.0], 0.75, [0.3, -0.2, 0.0, 1, 0, 0, 0]   # walk
        elif t < 12.0:
            vel, height, rh = [0.0, 0.0, 0.0], 0.75, [0.3, -0.2, 0.0, 1, 0, 0, 0]   # stop
        else:
            vel, height, rh = [0.0, 0.0, 0.0], 0.45, [0.45, -0.15, -0.25, 1, 0, 0, 0]  # squat + reach R hand down/forward

        proprio = obs["proprio"]
        row = proprio[0] if hasattr(proprio, "shape") and len(proprio.shape) >= 2 else proprio
        row_np = row.detach().cpu().numpy() if hasattr(row, "detach") else np.asarray(row)
        action = wbc.act(row_np, vel, height, [0, 0, 0], left_hand, rh)
        a = torch.tensor(action, dtype=torch.float32, device=args_cli.device).view(1, -1)
        obs, reward, terminated, truncated, info = env.step(a)

        base_z = float(robot.data.root_pos_w[0, 2])
        g = row_np[9:12]
        tilt = math.hypot(float(g[0]), float(g[1]))
        if step % 25 == 0:
            phase = "stand" if t < 3 else "walk" if t < 8 else "stop" if t < 12 else "squat+reach"
            print(f"[wbc] t={t:5.1f}s {phase:12s} base_z={base_z:.3f} tilt={tilt:.3f} base_x={float(robot.data.root_pos_w[0,0]):.2f}", flush=True)
        done = bool(terminated.any()) or bool(truncated.any())
        if done:
            fell = True
            print(f"[wbc] TERMINATED (fell/illegal contact) at t={t:.1f}s step={step}", flush=True)
            break
    print(f"[wbc] finished. fell={fell}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
