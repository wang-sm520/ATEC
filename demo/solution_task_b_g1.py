"""ATEC Task B G1 baseline: visual search, contact scoring, conservative push.

This file is intentionally self-contained for submission packaging. It does not
import atec_rl_lab modules at runtime; Isaac-only helpers live in scripts/.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - the eval image provides torch.
    torch = None


_DIR = os.path.dirname(os.path.abspath(__file__))
_POLICY_PATH = None
for _name in ("policy_a.pt", "policy.pt"):
    _candidate = os.path.join(_DIR, _name)
    if os.path.exists(_candidate):
        _POLICY_PATH = _candidate
        break


def _wrap_to_pi(angle: float) -> float:
    wrapped = (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if math.isclose(wrapped, -math.pi, abs_tol=1e-12) else wrapped


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _as_float(value: Any) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    def distance_to(self, xy: tuple[float, float]) -> float:
        return math.hypot(float(xy[0]) - self.x, float(xy[1]) - self.y)

    def bearing_to(self, xy: tuple[float, float]) -> float:
        return math.atan2(float(xy[1]) - self.y, float(xy[0]) - self.x)


@dataclass(frozen=True)
class Detection:
    track_id: int
    label: str
    rel_x: float
    rel_y: float
    distance: float
    confidence: float
    world_x: float
    world_y: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class PlannerOutput:
    phase: str
    command: tuple[float, float, float]
    arm_mode: str
    target_world: tuple[float, float] | None = None


class G1VelocityPolicyBridge:
    BODY_29_IDX = list(range(29))
    ACTION_DIM_BODY = 29

    HISTORY_LEN = 10
    DIM_ANG_VEL = 3
    DIM_CMD = 3
    DIM_GRAVITY = 3
    DIM_JP = 29
    DIM_JV = 29
    DIM_LASTACT = 29
    POLICY_INPUT_DIM = HISTORY_LEN * (
        DIM_ANG_VEL + DIM_CMD + DIM_GRAVITY + DIM_JP + DIM_JV + DIM_LASTACT
    )

    TRAINING_ACTION_SCALE_29 = (
        0.231, 0.231, 0.231, 0.231, 0.213, 0.213,
        0.231, 0.231, 0.231, 0.231, 0.213, 0.213,
        0.154, 0.213, 0.213,
        0.373, 0.373, 0.213, 0.373,
        0.23, 0.23, 0.23,
        0.373, 0.373, 0.213, 0.373,
        0.23, 0.23, 0.23,
    )
    EVAL_ACTION_SCALE = 0.5

    def __init__(self, policy_path: str | None, device: str = "cuda"):
        if torch is None:
            raise RuntimeError("torch is required for G1VelocityPolicyBridge")
        self.policy_path = policy_path
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.policy = None
        self.action_scale_ratio = torch.tensor(
            [scale / self.EVAL_ACTION_SCALE for scale in self.TRAINING_ACTION_SCALE_29],
            device=self.device,
            dtype=torch.float32,
        ).view(1, self.ACTION_DIM_BODY)
        self.reset()

    @property
    def policy_input_dim(self) -> int:
        return self.POLICY_INPUT_DIM

    def _make_buffer(self, dim: int):
        return torch.zeros((self.HISTORY_LEN, dim), device=self.device, dtype=torch.float32)

    def reset(self) -> None:
        self._buf_ang_vel = self._make_buffer(self.DIM_ANG_VEL)
        self._buf_cmd = self._make_buffer(self.DIM_CMD)
        self._buf_gravity = self._make_buffer(self.DIM_GRAVITY)
        self._buf_jp = self._make_buffer(self.DIM_JP)
        self._buf_jv = self._make_buffer(self.DIM_JV)
        self._buf_lastact = self._make_buffer(self.DIM_LASTACT)

    @staticmethod
    def _push(buf, row) -> None:
        buf[:-1] = buf[1:].clone()
        buf[-1] = row.reshape(-1)

    def _load_policy(self):
        if self.policy is None:
            if self.policy_path is None:
                raise FileNotFoundError("policy_a.pt or policy.pt is required next to solution_task_b_g1.py")
            self.policy = torch.jit.load(self.policy_path, map_location=self.device)
            self.policy.eval()
        return self.policy

    def _coerce_command(self, command: Sequence[float]):
        out = torch.as_tensor(list(command), device=self.device, dtype=torch.float32).reshape(-1)
        if out.numel() != self.DIM_CMD:
            raise ValueError(f"velocity command must have 3 values, got {out.numel()}")
        return out

    def _build_policy_input(self, proprio, command: Sequence[float]):
        proprio = proprio.to(device=self.device, dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2 or proprio.shape[0] != 1:
            raise ValueError(f"proprio must have shape (1, N) or (N,), got {tuple(proprio.shape)}")

        full_action_dim = (int(proprio.shape[-1]) - 12) // 3
        if 12 + 3 * full_action_dim != int(proprio.shape[-1]):
            raise ValueError(f"invalid ATEC proprio layout: {tuple(proprio.shape)}")
        if full_action_dim < self.ACTION_DIM_BODY:
            raise ValueError(f"G1 policy requires at least 29 action joints, got {full_action_dim}")

        base_ang_vel = proprio[0, 3:6]
        projected_gravity = proprio[0, 9:12]
        cmd = self._coerce_command(command)

        jp_start = 12
        jv_start = jp_start + full_action_dim
        act_start = jv_start + full_action_dim
        joint_pos_body = proprio[0, jp_start:jp_start + self.ACTION_DIM_BODY]
        joint_vel_body = proprio[0, jv_start:jv_start + self.ACTION_DIM_BODY]
        last_action_body = proprio[0, act_start:act_start + self.ACTION_DIM_BODY] / self.action_scale_ratio.reshape(-1)

        self._push(self._buf_ang_vel, base_ang_vel)
        self._push(self._buf_cmd, cmd)
        self._push(self._buf_gravity, projected_gravity)
        self._push(self._buf_jp, joint_pos_body)
        self._push(self._buf_jv, joint_vel_body)
        self._push(self._buf_lastact, last_action_body)

        policy_input = torch.cat(
            [
                self._buf_ang_vel.reshape(-1),
                self._buf_cmd.reshape(-1),
                self._buf_gravity.reshape(-1),
                self._buf_jp.reshape(-1),
                self._buf_jv.reshape(-1),
                self._buf_lastact.reshape(-1),
            ],
            dim=-1,
        ).unsqueeze(0)
        return policy_input, full_action_dim

    def act(self, proprio, command: Sequence[float]) -> list[float]:
        policy_input, full_action_dim = self._build_policy_input(proprio, command)
        with torch.inference_mode():
            action_body = self._load_policy()(policy_input)
        if not isinstance(action_body, torch.Tensor):
            action_body = torch.as_tensor(action_body, device=self.device, dtype=torch.float32)
        if action_body.ndim == 1:
            action_body = action_body.unsqueeze(0)
        action_body = action_body.to(device=self.device, dtype=torch.float32)[:, : self.ACTION_DIM_BODY]
        if action_body.shape[-1] != self.ACTION_DIM_BODY:
            raise ValueError(f"policy returned {action_body.shape[-1]} actions, expected 29")
        action_body = action_body * self.action_scale_ratio
        action_full = torch.zeros((1, full_action_dim), device=self.device, dtype=torch.float32)
        action_full[:, self.BODY_29_IDX] = action_body
        return action_full[0].detach().cpu().tolist()


class DeadReckoningOdometry:
    def __init__(self, dt: float = 0.02, x0: float = -10.0, y0: float = -10.0, yaw0: float = 0.0):
        self.dt = float(dt)
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.yaw0 = float(yaw0)
        self.reset()

    def reset(self) -> Pose2D:
        self.x = self.x0
        self.y = self.y0
        self.yaw = self.yaw0
        self.vx_b = 0.0
        self.vy_b = 0.0
        return self.pose

    @property
    def pose(self) -> Pose2D:
        return Pose2D(self.x, self.y, self.yaw)

    @staticmethod
    def _normalized(v: list[float]) -> list[float]:
        n = math.sqrt(sum(c * c for c in v))
        return [0.0, 0.0, 1.0] if n <= 1e-8 else [c / n for c in v]

    def update(self, proprio_row: Sequence[float]) -> Pose2D:
        lin = [_as_float(proprio_row[i]) for i in range(0, 3)]
        ang = [_as_float(proprio_row[i]) for i in range(3, 6)]
        grav = [_as_float(proprio_row[i]) for i in range(9, 12)]
        up = self._normalized([-grav[0], -grav[1], -grav[2]])
        yaw_rate = sum(a * u for a, u in zip(ang, up))

        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        self.x += (c * lin[0] - s * lin[1]) * self.dt
        self.y += (s * lin[0] + c * lin[1]) * self.dt
        self.yaw = _wrap_to_pi(self.yaw + yaw_rate * self.dt)
        self.vx_b = lin[0]
        self.vy_b = lin[1]
        return self.pose


class TaskBPlanner:
    TARGET_CENTER = (-3.0, -10.0)
    SEARCH_WAYPOINTS = (
        (-14.0, -14.0, 0.0),
        (-6.0, -14.0, 0.0),
        (-6.0, -12.0, math.pi),
        (-14.0, -12.0, math.pi),
        (-14.0, -10.0, 0.0),
        (-6.0, -10.0, 0.0),
        (-6.0, -8.0, math.pi),
        (-14.0, -8.0, math.pi),
        (-14.0, -6.0, 0.0),
        (-6.0, -6.0, 0.0),
    )

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.phase = "search"
        self.waypoint_idx = 0
        self.active_detection: Detection | None = None
        self.touched_track_ids: set[int] = set()
        self.placed_track_ids: set[int] = set()
        self.prev_score = 0.0
        self.phase_steps = 0

    def step(self, pose: Pose2D, detections: list[Detection], current_score: float) -> PlannerOutput:
        score_delta = float(current_score) - self.prev_score
        self.prev_score = float(current_score)
        if score_delta > 0.0 and self.active_detection is not None:
            if self.phase == "touch_object":
                self.touched_track_ids.add(self.active_detection.track_id)
                self.phase = "verify_or_next"
                self.phase_steps = 0
            elif self.phase == "push_to_goal":
                self.placed_track_ids.add(self.active_detection.track_id)
                self.phase = "verify_or_next"
                self.phase_steps = 0

        if self.phase == "search":
            fresh = self._choose_detection(detections)
            if fresh is not None:
                self.active_detection = fresh
                self.phase = "approach_object"
                self.phase_steps = 0
                return self._approach_output(pose, fresh)
            return self._search_output(pose)

        if self.phase == "approach_object":
            self.phase_steps += 1
            det = self._refresh_active_detection(detections)
            if det is None:
                self.phase = "search"
                self.active_detection = None
                self.phase_steps = 0
                return self._search_output(pose)
            self.active_detection = det
            if det.distance <= 0.55:
                self.phase = "touch_object"
                self.phase_steps = 0
                return PlannerOutput("touch_object", self._face_and_creep(det), "left_touch", (det.world_x, det.world_y))
            return self._approach_output(pose, det)

        if self.phase == "touch_object":
            det = self._refresh_active_detection(detections)
            self.phase_steps += 1
            if det is None or self.phase_steps > 120:
                self.phase = "verify_or_next"
                self.phase_steps = 0
                return PlannerOutput("verify_or_next", (0.0, 0.0, 0.0), "stow", None)
            if self._near_target((det.world_x, det.world_y), max_distance=4.0):
                if self.phase_steps > 40:
                    self.phase = "push_to_goal"
                    self.phase_steps = 0
                    return self._push_output(pose, det)
            return PlannerOutput("touch_object", self._face_and_creep(det), "left_touch", (det.world_x, det.world_y))

        if self.phase == "push_to_goal":
            det = self._refresh_active_detection(detections)
            self.phase_steps += 1
            if det is None or self.phase_steps > 180:
                self.phase = "verify_or_next"
                self.phase_steps = 0
                return PlannerOutput("verify_or_next", (0.0, 0.0, 0.0), "stow", None)
            return self._push_output(pose, det)

        if self.phase == "verify_or_next":
            self.phase_steps += 1
            if self.phase_steps >= 20:
                self.active_detection = None
                self.phase = "search"
                self.phase_steps = 0
                return self._search_output(pose)
            return PlannerOutput("verify_or_next", (0.0, 0.0, 0.0), "stow", None)

        self.phase = "search"
        self.active_detection = None
        self.phase_steps = 0
        return self._search_output(pose)

    def _choose_detection(self, detections: list[Detection]) -> Detection | None:
        completed_track_ids = self.touched_track_ids | self.placed_track_ids
        candidates = [d for d in detections if d.track_id not in completed_track_ids and d.confidence >= 0.2]
        if not candidates:
            return None
        return min(candidates, key=lambda d: (d.distance, -d.confidence))

    def _refresh_active_detection(self, detections: list[Detection]) -> Detection | None:
        if self.active_detection is None:
            return None
        for det in detections:
            if det.track_id == self.active_detection.track_id:
                return det
        if self.phase_steps < 30:
            return self.active_detection
        return None

    def _search_output(self, pose: Pose2D) -> PlannerOutput:
        wp = self.SEARCH_WAYPOINTS[self.waypoint_idx]
        if pose.distance_to((wp[0], wp[1])) < 0.45:
            self.waypoint_idx = (self.waypoint_idx + 1) % len(self.SEARCH_WAYPOINTS)
            wp = self.SEARCH_WAYPOINTS[self.waypoint_idx]
        return PlannerOutput("search", self._drive_to(pose, wp[0], wp[1], wp[2], 0.35), "stow", (wp[0], wp[1]))

    def _approach_output(self, pose: Pose2D, det: Detection) -> PlannerOutput:
        bearing = pose.bearing_to((det.world_x, det.world_y))
        standoff = 0.42
        tx = det.world_x - standoff * math.cos(bearing)
        ty = det.world_y - standoff * math.sin(bearing)
        return PlannerOutput("approach_object", self._drive_to(pose, tx, ty, bearing, 0.28), "stow", (det.world_x, det.world_y))

    def _push_output(self, pose: Pose2D, det: Detection) -> PlannerOutput:
        desired_yaw = math.atan2(self.TARGET_CENTER[1] - det.world_y, self.TARGET_CENTER[0] - det.world_x)
        yaw_err = _wrap_to_pi(desired_yaw - pose.yaw)
        vx = 0.18 if abs(yaw_err) < 0.45 else 0.0
        wz = _clamp(1.8 * yaw_err, -0.7, 0.7)
        return PlannerOutput("push_to_goal", (vx, 0.0, wz), "left_push", (det.world_x, det.world_y))

    @staticmethod
    def _near_target(xy: tuple[float, float], max_distance: float) -> bool:
        return math.hypot(xy[0] - TaskBPlanner.TARGET_CENTER[0], xy[1] - TaskBPlanner.TARGET_CENTER[1]) <= max_distance

    @staticmethod
    def _face_and_creep(det: Detection) -> tuple[float, float, float]:
        yaw_err = math.atan2(det.rel_y, max(det.rel_x, 1e-6))
        vx = 0.10 if abs(yaw_err) < 0.35 else 0.0
        return vx, 0.0, _clamp(2.0 * yaw_err, -0.5, 0.5)

    @staticmethod
    def _drive_to(pose: Pose2D, tx: float, ty: float, tyaw: float, max_vx: float) -> tuple[float, float, float]:
        ex = tx - pose.x
        ey = ty - pose.y
        c = math.cos(pose.yaw)
        s = math.sin(pose.yaw)
        body_x = c * ex + s * ey
        body_y = -s * ex + c * ey
        yaw_err = _wrap_to_pi(tyaw - pose.yaw)
        vx = _clamp(0.9 * body_x, -0.18, max_vx)
        vy = _clamp(0.8 * body_y, -0.22, 0.22)
        wz = _clamp(1.8 * yaw_err, -0.7, 0.7)
        if abs(yaw_err) > 0.9:
            vx = min(vx, 0.05)
            vy = _clamp(vy, -0.08, 0.08)
        return vx, vy, wz


@dataclass(frozen=True)
class SquatCommand:
    height: float = 0.75
    pitch: float = 0.0


class _HeuristicSquatRunner:
    """ONNX 不可用时的脚本蹲姿兜底，输出 OpenWBT 原始 12 动作约定。"""
    def run(self, obs, hidden_state):
        height = float(obs[0, 0]); pitch = float(obs[0, 1])
        drop = float(np.clip(0.75 - height, 0.0, 0.40))
        hip = -0.9 * drop - 0.10 * pitch
        knee = 1.8 * drop
        ankle = -0.9 * drop
        leg = np.array([hip, 0.0, 0.0, knee, ankle, 0.0], dtype=np.float32)
        return np.concatenate([leg, leg]).reshape(1, 12), hidden_state


class _OnnxSquatRunner:
    def __init__(self, policy_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    def run(self, obs, hidden_state):
        if hidden_state is None:
            hidden_state = np.zeros((1, 1, 256), dtype=np.float32)
        out = self.session.run(
            ["action", "output_hidden_states"],
            {"obs": obs.astype(np.float32), "input_hidden_states": hidden_state.astype(np.float32)},
        )
        return np.asarray(out[0], dtype=np.float32), np.asarray(out[1], dtype=np.float32)


class OpenWBTSquatBridge:
    NUM_OBS = 78
    NUM_DOF_OBS = 29
    NUM_ACTIONS = 12
    TASKB_ACTION_SCALE = 0.5
    OPENWBT_ACTION_SCALE = 0.25
    ANG_VEL_SCALE = 0.25
    DOF_VEL_SCALE = 0.05
    CLIP_OBS = 100.0
    CLIP_ACTIONS = 100.0
    ANKLE_SAFETY = 0.7  # ankle pitch kp-ratio safety factor (kp_WBT/kp_ATEC=2.0, scaled down to avoid overshoot)
    KP_RATIO = (
        0.50, 0.67, 0.67, 0.75, 2.0 * ANKLE_SAFETY, 0.0,
        0.50, 0.67, 0.67, 0.75, 2.0 * ANKLE_SAFETY, 0.0,
    )
    OPENWBT_DEFAULT_29 = (
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.3, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, -0.3, 0.0, 1.0, 0.0, 0.0, 0.0,
    )
    TASKB_DEFAULT_29 = (
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
        -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
        0.0, 0.0, 0.0,
        0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
        0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
    )

    def __init__(self, policy_runner=None, policy_path=None):
        self.policy_runner = policy_runner if policy_runner is not None else self._make_default_runner(policy_path)
        self._taskb_default = np.asarray(self.TASKB_DEFAULT_29, dtype=np.float32)
        self._openwbt_default = np.asarray(self.OPENWBT_DEFAULT_29, dtype=np.float32)
        self._kp_ratio = np.asarray(self.KP_RATIO, dtype=np.float32)  # online P-term gain compensation
        self.reset()

    def reset(self):
        self.last_action = np.zeros(self.NUM_ACTIONS, dtype=np.float32)
        self.hidden_state = None
        self._last_q_abs_legs = self._taskb_default[: self.NUM_ACTIONS].copy()

    def _first_row(self, proprio):
        if hasattr(proprio, "detach"):
            proprio = proprio.detach().cpu().numpy()
        arr = np.asarray(proprio, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[0]
        return arr

    def build_observation(self, proprio, command):
        row = self._first_row(proprio)
        n = int(row.shape[0]); full_dim = (n - 12) // 3
        if 12 + 3 * full_dim != n or full_dim < self.NUM_DOF_OBS:
            raise ValueError(f"bad proprio length {n}")
        cmd = self._command_vec(command)
        ang_vel = row[3:6] * self.ANG_VEL_SCALE
        gravity = row[9:12]
        jp_start = 12; jv_start = jp_start + full_dim
        jp_rel_29 = row[jp_start:jp_start + self.NUM_DOF_OBS]
        jv_29 = row[jv_start:jv_start + self.NUM_DOF_OBS] * self.DOF_VEL_SCALE
        jp_openwbt = jp_rel_29 + self._taskb_default - self._openwbt_default
        self._last_q_abs_legs = (jp_rel_29[: self.NUM_ACTIONS] + self._taskb_default[: self.NUM_ACTIONS]).copy()
        obs = np.concatenate([cmd, gravity, ang_vel, jp_openwbt, jv_29, self.last_action], dtype=np.float32)
        return np.clip(obs, -self.CLIP_OBS, self.CLIP_OBS).reshape(1, -1).astype(np.float32)

    def act(self, proprio, command):
        obs = self.build_observation(proprio, command)
        raw, self.hidden_state = self.policy_runner.run(obs, self.hidden_state)
        raw = np.clip(np.asarray(raw, dtype=np.float32).reshape(-1), -self.CLIP_ACTIONS, self.CLIP_ACTIONS)
        if raw.shape[0] != self.NUM_ACTIONS:
            raise ValueError(f"squat runner returned {raw.shape[0]} actions, expected 12")
        raw[[5, 11]] = 0.0
        self.last_action = raw.copy()
        # Task 2: 在线 P 项增益补偿。
        q_abs = self._last_q_abs_legs  # updated by build_observation() above
        target_wbt = raw * self.OPENWBT_ACTION_SCALE + self._openwbt_default[: self.NUM_ACTIONS]
        comp_target = q_abs + self._kp_ratio * (target_wbt - q_abs)
        action = (comp_target - self._taskb_default[: self.NUM_ACTIONS]) / self.TASKB_ACTION_SCALE
        action[[5, 11]] = 0.0
        return np.clip(action, -self.CLIP_ACTIONS, self.CLIP_ACTIONS).astype(np.float32).tolist()

    @staticmethod
    def _command_vec(command):
        if isinstance(command, SquatCommand):
            values = (command.height, command.pitch)
        else:
            values = tuple(float(v) for v in command)
        if len(values) != 2:
            raise ValueError("squat command must have 2 values")
        h = float(np.clip(values[0], 0.35, 0.75))
        p = float(np.clip(values[1], 0.0, 0.5))
        return np.asarray([h, p], dtype=np.float32)

    @classmethod
    def _make_default_runner(cls, policy_path):
        path = policy_path
        if path is None:
            for cand in (os.path.join(_DIR, "squat.onnx"),
                         os.path.join(_DIR, "..", "OpenWBT", "ckpts", "squat.onnx")):
                if os.path.exists(cand):
                    path = cand
                    break
        if path is not None and os.path.exists(path):
            try:
                return _OnnxSquatRunner(path)
            except Exception:
                pass
        return _HeuristicSquatRunner()


class LocalObjectInteraction:
    """Conservative G1 upper-body action override.

    Action indices follow UNITREE_G1_29DOF_DEX1_CFG.joint_names:
    left arm 15..21, right arm 22..28, hands 29..32. The action space uses
    default-offset joint position targets scaled by 0.5, so these are small
    normalized offsets rather than absolute joint angles.
    """

    LEFT_TOUCH = {
        15: 0.28,   # left_shoulder_pitch_joint
        16: 0.18,   # left_shoulder_roll_joint
        17: 0.00,   # left_shoulder_yaw_joint
        18: 0.22,   # left_elbow_joint
        19: 0.00,
        20: -0.08,
        21: 0.00,
        29: 0.20,
        30: 0.20,
    }
    LEFT_PUSH = {
        15: 0.38,
        16: 0.20,
        17: 0.00,
        18: 0.32,
        19: 0.00,
        20: -0.10,
        21: 0.00,
        29: 0.25,
        30: 0.25,
    }

    def apply_arm_override(self, action: Sequence[float], arm_mode: str) -> list[float]:
        out = [float(v) for v in action]
        if arm_mode == "left_touch":
            self._apply(out, self.LEFT_TOUCH)
        elif arm_mode == "left_push":
            self._apply(out, self.LEFT_PUSH)
        return out

    @staticmethod
    def _apply(action: list[float], values: dict[int, float]) -> None:
        for idx, value in values.items():
            if idx < len(action):
                action[idx] = float(value)


_HEAD_CAMERA_HFOV_DEG = math.degrees(2.0 * math.atan(20.955 / (2.0 * 24.0)))


class TaskBRgbdPerception:
    def __init__(
        self,
        hfov_deg: float = _HEAD_CAMERA_HFOV_DEG,
        min_pixels: int = 35,
        max_depth: float = 8.0,
        track_match_dist: float = 0.75,
    ):
        self.hfov = math.radians(float(hfov_deg))
        self.min_pixels = int(min_pixels)
        self.max_depth = float(max_depth)
        self.track_match_dist = float(track_match_dist)
        self.next_track_id = 1
        self.tracks: dict[int, tuple[float, float]] = {}

    def reset(self) -> None:
        self.next_track_id = 1
        self.tracks.clear()

    def update(self, image_obs: dict, pose: Pose2D) -> list[Detection]:
        if torch is None:
            return []
        rgb, depth = self._extract_head_rgbd(image_obs)
        if rgb is None or depth is None:
            return []
        rgb = rgb.detach().float().cpu()
        depth = depth.detach().float().cpu()
        if rgb.ndim == 4:
            rgb = rgb[0]
        if depth.ndim == 4:
            depth = depth[0]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if rgb.ndim != 3 or rgb.shape[-1] < 3 or depth.ndim != 2:
            return []
        if tuple(rgb.shape[:2]) != tuple(depth.shape[:2]):
            return []

        mask = self._colored_object_mask(rgb[..., :3], depth)
        components = self._components(mask)
        detections: list[Detection] = []
        used_track_ids: set[int] = set()
        for pixels in components:
            if len(pixels) < self.min_pixels:
                continue
            ys = [p[0] for p in pixels]
            xs = [p[1] for p in pixels]
            y0, y1 = min(ys), max(ys)
            x0, x1 = min(xs), max(xs)
            cy = int(round(sum(ys) / len(ys)))
            cx = int(round(sum(xs) / len(xs)))
            y_idx = torch.as_tensor(ys, dtype=torch.long)
            x_idx = torch.as_tensor(xs, dtype=torch.long)
            pixel_depth = depth[y_idx, x_idx]
            finite = torch.isfinite(pixel_depth) & (pixel_depth > 0.05) & (pixel_depth < self.max_depth)
            if not bool(finite.any()):
                continue
            dist = float(pixel_depth[finite].median().item())
            rel_x, rel_y = self._pixel_to_robot_xy(cx, rgb.shape[1], dist)
            world_x = pose.x + math.cos(pose.yaw) * rel_x - math.sin(pose.yaw) * rel_y
            world_y = pose.y + math.sin(pose.yaw) * rel_x + math.cos(pose.yaw) * rel_y
            if self._is_large_target_background(rgb[..., :3], y_idx, x_idx, x0, y0, x1, y1, world_x, world_y):
                continue
            track_id = self._assign_track(world_x, world_y, used_track_ids)
            confidence = _clamp(len(pixels) / 250.0, 0.05, 1.0)
            detections.append(
                Detection(
                    track_id=track_id,
                    label="colored_object",
                    rel_x=rel_x,
                    rel_y=rel_y,
                    distance=dist,
                    confidence=confidence,
                    world_x=world_x,
                    world_y=world_y,
                    bbox=(x0, y0, x1, y1),
                )
            )
        detections.sort(key=lambda d: (d.distance, -d.confidence))
        return detections[:5]

    @staticmethod
    def _extract_head_rgbd(image_obs: dict):
        if not isinstance(image_obs, dict):
            return None, None
        rgb = image_obs.get("head_rgb")
        depth = image_obs.get("head_depth")
        return rgb, depth

    def _colored_object_mask(self, rgb, depth):
        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]
        maxc = torch.maximum(torch.maximum(r, g), b)
        minc = torch.minimum(torch.minimum(r, g), b)
        saturation = maxc - minc
        yellow = (r > 120.0) & (g > 90.0) & (b < 130.0)
        red_or_orange = (r > 130.0) & (g > 45.0) & (b < 150.0) & (r > b + 35.0)
        bright_colored = (maxc > 110.0) & (saturation > 45.0)
        depth_ok = torch.isfinite(depth) & (depth > 0.15) & (depth < self.max_depth)
        return (yellow | red_or_orange | bright_colored) & depth_ok

    @staticmethod
    def _is_large_target_background(rgb, ys, xs, x0: int, y0: int, x1: int, y1: int, world_x: float, world_y: float) -> bool:
        height = int(rgb.shape[0])
        width = int(rgb.shape[1])
        bbox_area = max(1, (int(x1) - int(x0) + 1) * (int(y1) - int(y0) + 1))
        frame_area = max(1, width * height)
        if bbox_area < 0.45 * frame_area:
            return False
        r = rgb[ys, xs, 0]
        g = rgb[ys, xs, 1]
        b = rgb[ys, xs, 2]
        orange_red = (r > 150.0) & (g > 40.0) & (g < 160.0) & (b < 120.0) & (r > g + 40.0)
        if float(orange_red.float().mean().item()) < 0.7:
            return False
        return TaskBPlanner._near_target((world_x, world_y), max_distance=1.0)

    @staticmethod
    def _components(mask) -> list[list[tuple[int, int]]]:
        coords_tensor = mask.nonzero(as_tuple=False)
        if coords_tensor.numel() == 0:
            return []
        true_pixels = {(int(y), int(x)) for y, x in coords_tensor.tolist()}
        components: list[list[tuple[int, int]]] = []
        while true_pixels:
            start = true_pixels.pop()
            stack = [start]
            pixels = [start]
            while stack:
                cy, cx = stack.pop()
                for neighbor in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if neighbor in true_pixels:
                        true_pixels.remove(neighbor)
                        stack.append(neighbor)
                        pixels.append(neighbor)
            components.append(pixels)
        return components

    def _pixel_to_robot_xy(self, cx: int, width: int, depth: float) -> tuple[float, float]:
        x_norm = (float(cx) + 0.5) / max(float(width), 1.0) - 0.5
        lateral_angle = x_norm * self.hfov
        rel_x = float(depth)
        rel_y = -math.tan(lateral_angle) * float(depth)
        return rel_x, rel_y

    def _assign_track(self, world_x: float, world_y: float, used_track_ids: set[int] | None = None) -> int:
        if used_track_ids is None:
            used_track_ids = set()
        best_id = None
        best_dist = self.track_match_dist
        for track_id, (tx, ty) in self.tracks.items():
            if track_id in used_track_ids:
                continue
            d = math.hypot(world_x - tx, world_y - ty)
            if d < best_dist:
                best_dist = d
                best_id = track_id
        if best_id is None:
            best_id = self.next_track_id
            while best_id in used_track_ids or best_id in self.tracks:
                best_id += 1
            self.next_track_id = best_id + 1
        self.tracks[best_id] = (world_x, world_y)
        used_track_ids.add(best_id)
        return best_id


class AlgSolution:
    PERCEPTION_INTERVAL = 5

    def __init__(self):
        self.bridge = G1VelocityPolicyBridge(policy_path=_POLICY_PATH)
        self.odom = DeadReckoningOdometry(dt=0.02, x0=-10.0, y0=-10.0)
        self.perception = TaskBRgbdPerception()
        self.planner = TaskBPlanner()
        self.interaction = LocalObjectInteraction()
        self._perception_step = 0
        self._cached_detections: list[Detection] = []

    def reset(self, **kwargs) -> None:
        self.bridge.reset()
        self.odom.reset()
        self.perception.reset()
        self.planner.reset()
        self._perception_step = 0
        self._cached_detections = []

    def predicts(self, obs: dict, current_score: float):
        proprio = obs["proprio"]
        row = proprio[0] if hasattr(proprio, "shape") and len(proprio.shape) >= 2 else proprio
        pose = self.odom.update(row)
        image_obs = obs.get("image", {})
        perception_step = getattr(self, "_perception_step", 0)
        detections = getattr(self, "_cached_detections", [])
        if perception_step % self.PERCEPTION_INTERVAL == 0:
            detections = self.perception.update(image_obs, pose)
            self._cached_detections = detections
        self._perception_step = perception_step + 1
        plan = self.planner.step(pose, detections, current_score)
        action = self.bridge.act(proprio, plan.command)
        action = self.interaction.apply_arm_override(action, plan.arm_mode)
        return {"action": action, "giveup": False}
