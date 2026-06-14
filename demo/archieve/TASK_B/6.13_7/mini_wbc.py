"""ATEC adapter for the mini whole-body loco-manip ONNX policy (policy18.onnx).

Pure numpy + onnxruntime. Builds the 115-d obs / 5-frame history / 15-d ik_input
from an ATEC proprio row, runs the policy, returns a 33-d ATEC action.

Command: base velocity (3), base height, waist rpy (3), left/right hand pose
(7 each: xyz + wxyz quat, base frame), waist_weight.
"""

from __future__ import annotations

import os

import numpy as np

try:
    import onnxruntime as ort
except ModuleNotFoundError:  # pragma: no cover - eval image provides onnxruntime
    ort = None

_DIR = os.path.dirname(os.path.abspath(__file__))

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
ATEC_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
ATEC_TO_POLICY = np.array([ATEC_JOINTS.index(j) for j in ISAACLAB_JOINTS])
POLICY_TO_ATEC = np.array([ISAACLAB_JOINTS.index(j) for j in ATEC_JOINTS])

DEFAULT_ANGLES_ATEC = np.array([
    -0.2, 0.0, 0.0, 0.42, -0.23, 0.0, -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
    0.0, 0.0, 0.0, 0.35, 0.16, 0.0, 0.87, 0.0, 0.0, 0.0, 0.35, -0.16, 0.0, 0.87, 0.0, 0.0, 0.0,
], dtype=np.float32)
ACTION_SCALE = 0.25
ATEC_ACTION_SCALE = 0.5
HIS_LEN, NUM_OBS = 5, 115

DEFAULT_LEFT_HAND = [0.3, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0]
DEFAULT_RIGHT_HAND = [0.3, -0.2, 0.0, 1.0, 0.0, 0.0, 0.0]


def _find_policy(policy_path):
    if policy_path is not None and os.path.exists(policy_path):
        return policy_path
    for cand in (os.path.join(_DIR, "policy18.onnx"),
                 os.path.join(_DIR, "..", "mini", "model", "0116", "policy18.onnx")):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError("policy18.onnx not found (looked in demo/ and ../mini/model/0116/)")


class MiniWBC:
    def __init__(self, policy_path: str | None = None):
        if ort is None:
            raise RuntimeError("onnxruntime is required for MiniWBC")
        self.sess = ort.InferenceSession(_find_policy(policy_path), providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self) -> None:
        self.obs_history = np.zeros((HIS_LEN, NUM_OBS), dtype=np.float32)
        self.last_action = np.zeros(29, dtype=np.float32)
        self.ik_output = None

    def act(self, proprio_row, vel_cmd, base_height, waist_rpy, left_hand, right_hand,
            waist_weight=1.0, fingers=None):
        row = proprio_row
        if hasattr(row, "detach"):
            row = row.detach().cpu().numpy()
        row = np.asarray(row, dtype=np.float32).reshape(-1)
        ang_vel = row[3:6]
        gravity = row[9:12]
        jp_atec = row[12:12 + 29]
        jv_atec = row[12 + 33:12 + 33 + 29]
        dof_pos = jp_atec[ATEC_TO_POLICY]
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
        action = outs[0].squeeze(0).astype(np.float32)
        self.ik_output = outs[1].squeeze(0).astype(np.float32) if len(outs) > 1 else None
        self.last_action = action

        dof = action[POLICY_TO_ATEC] * ACTION_SCALE
        temp_default = DEFAULT_ANGLES_ATEC.copy()
        if self.ik_output is not None and len(self.ik_output) == 17:
            temp_default[12:29] = self.ik_output
        dof[12:29] *= 0.4
        target_q = dof + temp_default
        atec_action_body = (target_q - DEFAULT_ANGLES_ATEC) / ATEC_ACTION_SCALE
        action_33 = np.zeros(33, dtype=np.float32)
        action_33[:29] = atec_action_body
        if fingers is not None:
            action_33[29:33] = np.asarray(fingers, dtype=np.float32)
        return action_33.tolist()
