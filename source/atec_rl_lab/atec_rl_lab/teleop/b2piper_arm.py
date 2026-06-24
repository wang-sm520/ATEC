from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


ACTION_SCALE = 0.5
LEG_ACTION_DIM = 12
ARM_ACTION_DIM = 8
FULL_ACTION_DIM = LEG_ACTION_DIM + ARM_ACTION_DIM
GRIPPER_OPEN = np.array([0.035, -0.035], dtype=np.float32)
GRIPPER_CLOSE = np.array([-0.015, 0.015], dtype=np.float32)


@dataclass(frozen=True)
class ArmTeleopConfig:
    translation_scale: float = 1.0
    max_target_step_m: float = 0.03
    workspace_lower: np.ndarray = field(
        default_factory=lambda: np.array([0.15, -0.45, -0.30], dtype=np.float32)
    )
    workspace_upper: np.ndarray = field(
        default_factory=lambda: np.array([0.85, 0.45, 0.35], dtype=np.float32)
    )
    axis_map: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))


@dataclass(frozen=True)
class CalibrationState:
    controller_ref_robot: np.ndarray
    ee_ref_pos_b: np.ndarray
    ee_ref_quat_b: np.ndarray


@dataclass(frozen=True)
class EeTarget:
    pos_b: np.ndarray
    quat_b: np.ndarray


def _as_float32_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {arr.shape}")
    return arr


def unity_to_robot_position(pos_xyz: np.ndarray) -> np.ndarray:
    pos = _as_float32_vector(pos_xyz, 3, "pos_xyz")
    return np.array([-pos[0], pos[2], pos[1]], dtype=np.float32)


def lerp_gripper(trigger: float) -> np.ndarray:
    alpha = float(np.clip(trigger, 0.0, 1.0))
    return ((1.0 - alpha) * GRIPPER_OPEN + alpha * GRIPPER_CLOSE).astype(np.float32)


def piper_target_to_action(
    target_arm_q: np.ndarray,
    default_arm_q: np.ndarray,
    action_scale: float = ACTION_SCALE,
) -> np.ndarray:
    target = _as_float32_vector(target_arm_q, ARM_ACTION_DIM, "target_arm_q")
    default = _as_float32_vector(default_arm_q, ARM_ACTION_DIM, "default_arm_q")
    return ((target - default) / float(action_scale)).astype(np.float32)


def pack_b2piper_action(piper_action_8: np.ndarray) -> np.ndarray:
    piper = _as_float32_vector(piper_action_8, ARM_ACTION_DIM, "piper_action_8")
    full = np.zeros(FULL_ACTION_DIM, dtype=np.float32)
    full[LEG_ACTION_DIM:] = piper
    return full


def clamp_workspace(pos: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    p = _as_float32_vector(pos, 3, "pos")
    lo = _as_float32_vector(lower, 3, "lower")
    hi = _as_float32_vector(upper, 3, "upper")
    return np.minimum(np.maximum(p, lo), hi).astype(np.float32)


def limit_step(previous: np.ndarray, desired: np.ndarray, max_step: float) -> np.ndarray:
    prev = _as_float32_vector(previous, 3, "previous")
    des = _as_float32_vector(desired, 3, "desired")
    delta = np.clip(des - prev, -float(max_step), float(max_step))
    return (prev + delta).astype(np.float32)


class ArmTeleopMapper:
    def __init__(self, config: ArmTeleopConfig | None = None):
        self.config = config or ArmTeleopConfig()
        self.calibration: CalibrationState | None = None
        self._last_target: EeTarget | None = None

    @property
    def is_calibrated(self) -> bool:
        return self.calibration is not None

    def calibrate(
        self,
        controller_pos_robot: np.ndarray,
        ee_pos_b: np.ndarray,
        ee_quat_b: np.ndarray,
    ) -> None:
        controller = _as_float32_vector(controller_pos_robot, 3, "controller_pos_robot")
        ee_pos = clamp_workspace(ee_pos_b, self.config.workspace_lower, self.config.workspace_upper)
        ee_quat = _as_float32_vector(ee_quat_b, 4, "ee_quat_b")
        norm = np.linalg.norm(ee_quat)
        if norm <= 0.0:
            raise ValueError("ee_quat_b must be non-zero")
        ee_quat = (ee_quat / norm).astype(np.float32)
        self.calibration = CalibrationState(controller, ee_pos, ee_quat)
        self._last_target = EeTarget(ee_pos, ee_quat)

    def reset(self) -> None:
        self.calibration = None
        self._last_target = None

    def update(self, controller_pos_robot: np.ndarray, enabled: bool) -> EeTarget:
        if self.calibration is None:
            raise RuntimeError("ArmTeleopMapper must be calibrated before update()")
        if not enabled and self._last_target is not None:
            return self._last_target

        controller = _as_float32_vector(controller_pos_robot, 3, "controller_pos_robot")
        delta_robot = controller - self.calibration.controller_ref_robot
        mapped_delta = self.config.axis_map.astype(np.float32) @ delta_robot
        desired = self.calibration.ee_ref_pos_b + mapped_delta * float(self.config.translation_scale)
        desired = clamp_workspace(desired, self.config.workspace_lower, self.config.workspace_upper)

        previous = self._last_target.pos_b if self._last_target is not None else self.calibration.ee_ref_pos_b
        limited = limit_step(previous, desired, self.config.max_target_step_m)
        self._last_target = EeTarget(limited, self.calibration.ee_ref_quat_b)
        return self._last_target
