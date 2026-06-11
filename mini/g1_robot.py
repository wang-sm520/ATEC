import os
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime
from omegaconf import DictConfig
from rich.table import Table
from scipy.spatial.transform import Rotation as R

from mujoco_video_recorder import VideoRecordingMixin
from utils import get_gravity_orientation, isaaclab2mujoco, mujoco2isaaclab


PROJECT_ROOT = Path(__file__).resolve().parent


class G1Mujoco(VideoRecordingMixin):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / cfg.xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = cfg.simulation_dt

        self.policy_sess = onnxruntime.InferenceSession(
            str(PROJECT_ROOT / cfg.policy_path),
            providers=["CPUExecutionProvider"],
        )
        self.onnx_input_names = [input_info.name for input_info in self.policy_sess.get_inputs()]
        self.onnx_output_names = [output_info.name for output_info in self.policy_sess.get_outputs()]
        self.has_history_input = "obs_history" in self.onnx_input_names
        self.has_ik_input = "ik_input" in self.onnx_input_names
        self.has_ik_output = "ik_output" in self.onnx_output_names
        self.delta_action = True
        self.ik_output = None

        self.kps = np.array(cfg.kps, dtype=np.float32)
        self.kds = np.array(cfg.kds, dtype=np.float32)
        self.trq_limits = np.array(cfg.trq_limits, dtype=np.float32)
        self.default_angles = np.array(cfg.default_angles, dtype=np.float32)

        self.hand_joint_indices = list(range(22, 29)) + list(range(36, 43))
        self.policy_joint_indices = [i for i in range(43) if i not in self.hand_joint_indices]
        self._expand_pd_parameters_if_needed()

        self.cmd_scale = np.array(cfg.cmd_scale, dtype=np.float32)
        self.command = np.array(cfg.cmd_init, dtype=np.float32)
        self.vel_command_b = np.zeros(3, dtype=np.float32)
        self.base_height_command = np.array([0.75], dtype=np.float32)
        self.waist_rpy_command = np.zeros(3, dtype=np.float32)
        self.left_hand_pose_command = np.array([0.3, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.right_hand_pose_command = np.array([0.3, -0.2, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.use_waist_weight = True
        self.waist_weight = np.array([1.0], dtype=np.float32)

        self.obs_history = np.zeros((cfg.his_len, cfg.num_obs), dtype=np.float32)
        self.last_action = np.zeros(cfg.num_actions, dtype=np.float32)
        self.dof_trq = np.zeros(43, dtype=np.float32)
        self.tar_dof_pos = np.zeros(43, dtype=np.float32)
        self.inference_time = 0.0
        self.inference_fps = 0.0
        self.gui = None

        self._setup_video()
        self._setup_hand_targets()
        self.init()

    def _expand_pd_parameters_if_needed(self):
        if len(self.kps) == 43:
            return
        if len(self.kps) != 29:
            raise ValueError(f"Unexpected PD parameter length: {len(self.kps)}, expected 29 or 43")

        full_kps = np.zeros(43, dtype=np.float32)
        full_kds = np.zeros(43, dtype=np.float32)
        full_trq_limits = np.zeros(43, dtype=np.float32)
        full_kps[self.policy_joint_indices] = self.kps
        full_kds[self.policy_joint_indices] = self.kds
        full_trq_limits[self.policy_joint_indices] = self.trq_limits
        full_kps[self.hand_joint_indices] = 10.0
        full_kds[self.hand_joint_indices] = 1.0
        full_trq_limits[self.hand_joint_indices] = 5.0
        self.kps = full_kps
        self.kds = full_kds
        self.trq_limits = full_trq_limits

    def _setup_video(self):
        try:
            track_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        except Exception:
            track_body_id = 1

        self.init_video_recording(fps=50, output_dir="log", resolution=(1920, 1080))
        self.video_recorder.add_camera_config(
            "track",
            {
                "mode": "track",
                "trackbodyid": track_body_id,
                "distance": 3.0,
                "azimuth": -180,
                "elevation": -15,
            },
        )
        self.video_recorder.add_camera_config(
            "fixed",
            {
                "mode": "fixed",
                "lookat": [0.0, 0.0, 0.8],
                "distance": 10.0,
                "azimuth": -180,
                "elevation": -90,
            },
        )
        self.setup_video_recorder(self.model)

    def _setup_hand_targets(self):
        left_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_hand_palm")
        right_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_palm")
        if left_site >= 0 and right_site >= 0:
            self.left_hand_site_id = left_site
            self.right_hand_site_id = right_site
            self.use_palm_sites = True
            return

        self.left_hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
        self.right_hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        self.use_palm_sites = False

    @property
    def full_locomotion_command(self):
        command = np.zeros(7, dtype=np.float32)
        command[:3] = self.vel_command_b
        command[3] = self.base_height_command[0]
        command[4:7] = self.waist_rpy_command
        return command

    @property
    def left_hand_command(self):
        return self.left_hand_pose_command

    @property
    def right_hand_command(self):
        return self.right_hand_pose_command

    def reset_command(self):
        self.vel_command_b[:] = 0.0
        self.base_height_command[0] = 0.75
        self.waist_rpy_command[:] = 0.0
        self.left_hand_pose_command[:] = [0.3, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0]
        self.right_hand_pose_command[:] = [0.3, -0.2, 0.0, 1.0, 0.0, 0.0, 0.0]

    def init(self):
        if len(self.default_angles) == 43:
            self.data.qpos[7:] = self.default_angles
        else:
            full_default_angles = np.zeros(43, dtype=np.float32)
            full_default_angles[self.policy_joint_indices] = self.default_angles
            self.default_angles = full_default_angles
            self.data.qpos[7:] = full_default_angles

        self.data.qpos[7 + 12 :] = 0.0
        self.data.qvel[:] = 0.0
        mujoco.mj_step(self.model, self.data)
        self.update()

    def update(self):
        self.lin_vel = self.data.qvel[0:3]
        base_rotation = R.from_quat(self.data.qpos[3:7], scalar_first=True)
        self.base_lin_vel = base_rotation.inv().apply(self.lin_vel)
        self.ang_vel = self.data.qvel[3:6]
        self.gravity_orientation = get_gravity_orientation(self.data.qpos[3:7])
        self.dof_pos = self.data.qpos[7:]
        self.dof_vel = self.data.qvel[6:]

    def get_obs(self):
        policy_dof_pos = self.dof_pos[self.policy_joint_indices]
        policy_dof_vel = self.dof_vel[self.policy_joint_indices]
        policy_default_angles = self.default_angles[self.policy_joint_indices]

        dof_pos = mujoco2isaaclab(policy_dof_pos - policy_default_angles) * self.cfg.dof_pos_scale
        dof_vel = mujoco2isaaclab(policy_dof_vel) * self.cfg.dof_vel_scale
        full_command = np.concatenate(
            [
                self.full_locomotion_command,
                self.left_hand_command,
                self.right_hand_command,
                self.waist_weight,
            ]
        )

        return np.concatenate(
            [
                self.ang_vel,
                self.gravity_orientation,
                full_command,
                dof_pos,
                dof_vel,
                self.last_action,
            ]
        ).astype(np.float32)

    def get_all_obs(self):
        self.update()
        obs = self.get_obs()
        self.obs_history = np.concatenate([self.obs_history[1:], obs.reshape(1, -1)])
        ik_input = np.concatenate([self.left_hand_command, self.right_hand_command, self.waist_weight])
        return {
            "policy": obs,
            "obs_history": self.obs_history,
            "ik_input": ik_input.astype(np.float32),
        }

    def act(self, obs_dict):
        obs = obs_dict["policy"]
        obs_history = obs_dict.get("obs_history")
        ik_input = obs_dict.get("ik_input")

        if obs.ndim == 1:
            obs = obs[None, :]
        if obs_history is not None and obs_history.ndim == 2:
            obs_history = obs_history[None, :]
        if ik_input is not None and ik_input.ndim == 1:
            ik_input = ik_input[None, :]

        if self.has_history_input and self.has_ik_input:
            ort_inputs = {
                "obs": obs.astype(np.float32),
                "obs_history": obs_history.astype(np.float32),
                "ik_input": ik_input.astype(np.float32),
            }
        elif self.has_history_input:
            ort_inputs = {
                "obs": obs.astype(np.float32),
                "obs_history": obs_history.astype(np.float32),
            }
        else:
            ort_inputs = {self.policy_sess.get_inputs()[0].name: obs.astype(np.float32)}

        ort_outs = self.policy_sess.run(None, ort_inputs)
        action = ort_outs[0].squeeze(0).astype(np.float32)

        if self.has_ik_output:
            self.ik_output = ort_outs[self.onnx_output_names.index("ik_output")].squeeze(0).astype(np.float32)

        return action

    def apply_action(self, action):
        self.last_action = action
        dof_pos = isaaclab2mujoco(action) * self.cfg.action_scale
        dof_pos = np.clip(dof_pos, -10, 10)

        if self.has_ik_output and self.ik_output is not None and self.delta_action:
            temp_default_angles = self.default_angles[self.policy_joint_indices].copy()
            if len(self.ik_output) == 17:
                temp_default_angles[12:29] = self.ik_output
            dof_pos[12:29] *= 0.4
            dof_pos = dof_pos + temp_default_angles
        else:
            dof_pos = dof_pos + self.default_angles[self.policy_joint_indices]

        full_dof_pos = np.zeros(43, dtype=np.float32)
        full_dof_pos[self.policy_joint_indices] = dof_pos
        full_dof_pos[self.hand_joint_indices] = self.default_angles[self.hand_joint_indices]
        self.tar_dof_pos = full_dof_pos

        for _ in range(self.cfg.control_decimation):
            self.update()
            trq = self.kps * (full_dof_pos - self.dof_pos) - self.kds * self.dof_vel
            trq = np.clip(trq, -self.trq_limits, self.trq_limits)
            self.dof_trq = trq
            self.data.ctrl[:] = trq
            mujoco.mj_step(self.model, self.data)

    def step(self, action):
        self.apply_action(action)
        self.update()
        return self.get_all_obs()

    def _hand_positions_in_base(self):
        if self.use_palm_sites:
            left_world = self.data.site_xpos[self.left_hand_site_id].copy()
            right_world = self.data.site_xpos[self.right_hand_site_id].copy()
        else:
            left_world = self.data.xpos[self.left_hand_body_id].copy()
            right_world = self.data.xpos[self.right_hand_body_id].copy()

        base_pos = self.data.qpos[:3]
        base_rot = R.from_quat(self.data.qpos[3:7], scalar_first=True)
        return base_rot.inv().apply(left_world - base_pos), base_rot.inv().apply(right_world - base_pos)

    def create_status_display(self):
        left_pos, right_pos = self._hand_positions_in_base()
        left_target = self.left_hand_pose_command[:3]
        right_target = self.right_hand_pose_command[:3]
        left_error = np.linalg.norm(left_pos - left_target)
        right_error = np.linalg.norm(right_pos - right_target)
        height_error = abs(self.data.qpos[2] - self.base_height_command[0])

        table = Table(title="G1 Humanoid Robot Status (Base Frame)", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan", width=20)
        table.add_column("Left Hand", style="yellow", width=30)
        table.add_column("Right Hand", style="green", width=30)
        table.add_row(
            "Target Pos",
            f"[{left_target[0]:.3f}, {left_target[1]:.3f}, {left_target[2]:.3f}]",
            f"[{right_target[0]:.3f}, {right_target[1]:.3f}, {right_target[2]:.3f}]",
        )
        table.add_row(
            "Current Pos",
            f"[{left_pos[0]:.3f}, {left_pos[1]:.3f}, {left_pos[2]:.3f}]",
            f"[{right_pos[0]:.3f}, {right_pos[1]:.3f}, {right_pos[2]:.3f}]",
        )
        table.add_row("Error (m)", f"{left_error:.4f}", f"{right_error:.4f}")
        table.add_row("Velocity Cmd", f"[{self.vel_command_b[0]:.2f}, {self.vel_command_b[1]:.2f}, {self.vel_command_b[2]:.2f}]", "")
        table.add_row(
            "Base Height",
            f"Target: {self.base_height_command[0]:.3f}m",
            f"Actual: {self.data.qpos[2]:.3f}m (d={height_error:.3f})",
        )
        table.add_row(
            "Recording",
            f"{'Recording' if self.video_recorder.is_recording else 'Stopped'} ({self.video_recorder.frame_count()} frames)",
            "",
        )
        table.add_row(
            "Inference",
            f"Time: {self.inference_time * 1000:.2f} ms",
            f"FPS: {self.inference_fps:.1f}",
        )
        return table
