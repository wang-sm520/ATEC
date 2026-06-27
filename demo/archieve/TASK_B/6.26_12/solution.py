from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np
import torch


def _quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert a scalar-first quaternion to a rotation matrix."""
    quat = quat / torch.linalg.vector_norm(quat).clamp_min(1.0e-8)
    w, x, y, z = quat.unbind()
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        )
    ).reshape(3, 3)


def _transform(position: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    result = torch.eye(4, dtype=position.dtype, device=position.device)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def _rotation_z(angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    result = torch.eye(4, dtype=angle.dtype, device=angle.device)
    result[0, 0] = c
    result[0, 1] = -s
    result[1, 0] = s
    result[1, 1] = c
    return result


class PiperKinematics:
    """Submission-side Piper FK, geometric Jacobian, and DLS IK."""

    JOINT_LOWER = (
        math.radians(-150.00035),
        math.radians(0.0),
        math.radians(-169.99657),
        math.radians(-99.98113),
        math.radians(-69.90085),
        math.radians(-120.00027),
    )
    JOINT_UPPER = (
        math.radians(124.21724),
        math.radians(179.90874),
        math.radians(0.0),
        math.radians(99.98113),
        math.radians(69.90085),
        math.radians(120.00027),
    )

    # Physics joint frames copied from robot/b2/b2_piper.usda.
    _PARENT_POSITIONS = (
        (0.0, 0.0, 0.123),
        (0.0, 0.0, 0.0),
        (0.28502998, 0.0, 0.0),
        (-0.021983974, -0.25075004, 0.0),
        (0.0, 0.0, 0.0),
        (0.00008826461, -0.09100002, 0.0),
    )
    _PARENT_ROTATIONS = (
        (1.0, 0.0, 0.0, 0.0),
        (-0.048008468, 0.048013408, 0.7054761, 0.7054739),
        (-0.62399614, 0.0, 0.0, 0.78142744),
        (0.7071055, 0.707108, 0.0, 0.0),
        (-0.7071055, 0.707108, 0.0, 0.0),
        (0.7071055, 0.707108, 0.0, 0.0),
    )

    def __init__(self, device: torch.device | str, dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.lower = torch.tensor(self.JOINT_LOWER, device=self.device, dtype=dtype)
        self.upper = torch.tensor(self.JOINT_UPPER, device=self.device, dtype=dtype)

        self.base_to_arm = _transform(
            torch.tensor((0.2, 0.0, 0.1), device=self.device, dtype=dtype),
            torch.eye(3, device=self.device, dtype=dtype),
        )
        self.parent_joint_transforms = []
        for pos, quat in zip(self._PARENT_POSITIONS, self._PARENT_ROTATIONS):
            position = torch.tensor(pos, device=self.device, dtype=dtype)
            rotation = _quat_wxyz_to_matrix(torch.tensor(quat, device=self.device, dtype=dtype))
            self.parent_joint_transforms.append(_transform(position, rotation))

        camera_rotation = torch.tensor(
            (
                (0.0, 1.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            device=self.device,
            dtype=dtype,
        )
        self.gripper_to_camera = _transform(
            torch.tensor((-0.05, 0.0, 0.06), device=self.device, dtype=dtype),
            camera_rotation,
        )

        self.top_down_rotation = torch.diag(
            torch.tensor((1.0, -1.0, -1.0), device=self.device, dtype=dtype)
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        transform, _, _ = self.forward_with_jacobian_data(q)
        return transform

    def forward_with_jacobian_data(
        self, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if q.shape != (6,):
            raise ValueError(f"Expected Piper q shape (6,), got {tuple(q.shape)}")

        transform = self.base_to_arm.clone()
        joint_origins = []
        joint_axes = []
        local_axis = torch.tensor((0.0, 0.0, 1.0), device=q.device, dtype=q.dtype)

        for index in range(6):
            joint_transform = transform @ self.parent_joint_transforms[index]
            joint_origins.append(joint_transform[:3, 3])
            # USD PhysicsJoint positive motion rotates body1 opposite to the
            # parent joint-frame axis used in this parent-to-child FK chain.
            joint_axes.append(-(joint_transform[:3, :3] @ local_axis))
            transform = joint_transform @ _rotation_z(-q[index])

        return transform, torch.stack(joint_origins), torch.stack(joint_axes)

    def jacobian(self, q: torch.Tensor) -> torch.Tensor:
        ee_transform, origins, axes = self.forward_with_jacobian_data(q)
        ee_position = ee_transform[:3, 3]
        linear = torch.linalg.cross(axes, ee_position.unsqueeze(0) - origins)
        return torch.cat((linear.T, axes.T), dim=0)

    def camera_transform(self, q: torch.Tensor) -> torch.Tensor:
        return self.forward(q) @ self.gripper_to_camera

    def camera_point_to_base(self, q: torch.Tensor, point_camera: torch.Tensor) -> torch.Tensor:
        point_h = torch.cat(
            (point_camera.to(device=q.device, dtype=q.dtype), torch.ones(1, device=q.device, dtype=q.dtype))
        )
        return (self.camera_transform(q) @ point_h)[:3]

    def dls_step(
        self,
        q: torch.Tensor,
        target_position: torch.Tensor,
        target_rotation: torch.Tensor | None = None,
        target_axis: torch.Tensor | None = None,
        damping: float = 0.1,
        max_joint_delta: float = 0.04,
        max_position_error: float = 0.04,
        orientation_weight: float = 0.25,
    ) -> torch.Tensor:
        if target_rotation is not None and target_axis is not None:
            raise ValueError("Specify either target_rotation or target_axis, not both.")

        current = self.forward(q)
        position_error = target_position.to(q) - current[:3, 3]
        error_norm = torch.linalg.vector_norm(position_error)
        if error_norm > max_position_error:
            position_error = position_error * (max_position_error / error_norm)

        jacobian = self.jacobian(q)
        if target_rotation is None and target_axis is None:
            task_jacobian = jacobian[:3]
            task_error = position_error
        elif target_axis is not None:
            current_axis = current[:3, 2]
            desired_axis = target_axis.to(q)
            desired_axis = desired_axis / torch.linalg.vector_norm(desired_axis).clamp_min(1.0e-8)
            axis_jacobian = torch.stack(
                [
                    torch.linalg.cross(jacobian[3:, index], current_axis)
                    for index in range(jacobian.shape[1])
                ],
                dim=1,
            )
            task_jacobian = torch.cat(
                (jacobian[:3], axis_jacobian * orientation_weight), dim=0
            )
            task_error = torch.cat(
                (position_error, (desired_axis - current_axis) * orientation_weight), dim=0
            )
        else:
            current_rotation = current[:3, :3]
            target_rotation = target_rotation.to(q)
            orientation_error = 0.5 * sum(
                torch.linalg.cross(current_rotation[:, axis], target_rotation[:, axis])
                for axis in range(3)
            )
            task_jacobian = torch.cat(
                (jacobian[:3], jacobian[3:] * orientation_weight), dim=0
            )
            task_error = torch.cat(
                (position_error, orientation_error * orientation_weight), dim=0
            )

        regularizer = (float(damping) ** 2) * torch.eye(
            task_jacobian.shape[0], device=q.device, dtype=q.dtype
        )
        delta = task_jacobian.T @ torch.linalg.solve(
            task_jacobian @ task_jacobian.T + regularizer,
            task_error,
        )
        delta = torch.clamp(delta, -max_joint_delta, max_joint_delta)
        margin = 0.01
        return torch.clamp(q + delta, self.lower + margin, self.upper - margin)


CONTROL_DT = 0.02
LEG_DIM = 12
ARM_DIM = 8
ACTION_DIM = LEG_DIM + ARM_DIM


@dataclass
class ParsedObservation:
    base_lin_vel: torch.Tensor
    base_ang_vel: torch.Tensor
    projected_gravity: torch.Tensor
    joint_pos_rel: torch.Tensor
    joint_vel_rel: torch.Tensor
    last_action: torch.Tensor

    @property
    def arm_q(self) -> torch.Tensor:
        return self.joint_pos_rel[LEG_DIM : LEG_DIM + 6]


class ObservationParser:
    @staticmethod
    def parse(proprio: torch.Tensor) -> ParsedObservation:
        if proprio.ndim == 2:
            if proprio.shape[0] != 1:
                raise ValueError("Task B submission currently supports one environment.")
            proprio = proprio[0]
        if proprio.ndim != 1:
            raise ValueError(f"Expected flat proprio, got shape {tuple(proprio.shape)}")

        action_dim = (int(proprio.numel()) - 12) // 3
        if action_dim != ACTION_DIM or proprio.numel() != 12 + 3 * action_dim:
            raise ValueError(
                f"Expected B2-Piper proprio with 72 values, got {proprio.numel()}."
            )

        return ParsedObservation(
            base_lin_vel=proprio[0:3],
            base_ang_vel=proprio[3:6],
            projected_gravity=proprio[9:12],
            joint_pos_rel=proprio[12 : 12 + action_dim],
            joint_vel_rel=proprio[12 + action_dim : 12 + 2 * action_dim],
            last_action=proprio[12 + 2 * action_dim : 12 + 3 * action_dim],
        )


class B2Locomotion:
    LOW_STANCE_ACTION = (
        0.0,
        0.35,
        -0.55,
        0.0,
        0.35,
        -0.55,
        0.0,
        0.35,
        -0.55,
        0.0,
        0.35,
        -0.55,
    )
    TRAIN_TO_ENV = (
        0.125,
        0.25,
        0.25,
        0.125,
        0.25,
        0.25,
        0.125,
        0.25,
        0.25,
        0.125,
        0.25,
        0.25,
    )
    ENV_TO_TRAIN = (
        8.0,
        4.0,
        4.0,
        8.0,
        4.0,
        4.0,
        8.0,
        4.0,
        4.0,
        8.0,
        4.0,
        4.0,
    )

    GAIT_CYCLE_TIME = 0.5
    USE_GAIT_PHASE = True
    PHASE_INSERT_AFTER_COMMAND = True

    def __init__(self, policy_path: str, device: torch.device):
        self.device = device
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()
        self.train_to_env = torch.tensor(
            self.TRAIN_TO_ENV, dtype=torch.float32, device=device
        )
        self.env_to_train = torch.tensor(
            self.ENV_TO_TRAIN, dtype=torch.float32, device=device
        )
        self.low_stance_action = torch.tensor(
            self.LOW_STANCE_ACTION, dtype=torch.float32, device=device
        )
        self._phase_step = 0

    def reset(self) -> None:
        self._phase_step = 0

    def act(self, parsed: ParsedObservation, command: torch.Tensor) -> torch.Tensor:
        command = torch.as_tensor(command, dtype=torch.float32, device=self.device)
        command = torch.stack(
            (
                command[0].clamp(-0.65, 0.85),
                command[1].clamp(-0.45, 0.45),
                command[2].clamp(-1.05, 1.05),
            )
        )
        ang = parsed.base_ang_vel.to(self.device) * 0.25
        grav = parsed.projected_gravity.to(self.device)
        jpos = parsed.joint_pos_rel[:LEG_DIM].to(self.device)
        jvel = parsed.joint_vel_rel[:LEG_DIM].to(self.device) * 0.05
        lact = parsed.last_action[:LEG_DIM].to(self.device) * self.env_to_train

        if self.USE_GAIT_PHASE:
            phase_val = self._phase_step * CONTROL_DT / self.GAIT_CYCLE_TIME
            two_pi_phase = 2.0 * torch.pi * torch.tensor(
                phase_val, dtype=torch.float32, device=self.device
            )
            phase_obs = torch.stack((torch.sin(two_pi_phase), torch.cos(two_pi_phase)))
            if self.PHASE_INSERT_AFTER_COMMAND:
                parts = (ang, grav, command, phase_obs, jpos, jvel, lact)
            else:
                parts = (ang, grav, command, jpos, jvel, lact, phase_obs)
        else:
            parts = (ang, grav, command, jpos, jvel, lact)
        policy_obs = torch.cat(parts).unsqueeze(0)

        with torch.inference_mode():
            action = self.policy(policy_obs)
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        action = action.reshape(-1, LEG_DIM)[0].to(dtype=torch.float32)
        if self.USE_GAIT_PHASE:
            self._phase_step += 1
        return torch.nan_to_num(action * self.train_to_env).clamp(-5.0, 5.0)


class Odometry:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.x = -10.0
        self.y = -10.0
        self.yaw = 0.0

    def update(self, linear_velocity: torch.Tensor, angular_velocity: torch.Tensor) -> None:
        vx = float(linear_velocity[0])
        vy = float(linear_velocity[1])
        wz = float(angular_velocity[2])
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        self.x += (c * vx - s * vy) * CONTROL_DT
        self.y += (s * vx + c * vy) * CONTROL_DT
        self.yaw = _wrap_angle(self.yaw + wz * CONTROL_DT)

    def body_error(self, world_x: float, world_y: float) -> tuple[float, float]:
        dx = world_x - self.x
        dy = world_y - self.y
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        return c * dx + s * dy, -s * dx + c * dy


@dataclass
class Detection:
    position_base: np.ndarray
    pixel: tuple[float, float]
    area: int
    confidence: float

    @property
    def x(self) -> float:
        return float(self.position_base[0])

    @property
    def y(self) -> float:
        return float(self.position_base[1])

    @property
    def z(self) -> float:
        return float(self.position_base[2])


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _debug_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _debug_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_debug_value(item) for item in value]
    return value


def _connected_components(mask: np.ndarray, min_area: int = 3) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=np.bool_)
    if mask.ndim == 0:
        return []
    if mask.ndim > 2:
        mask = np.squeeze(mask)
        if mask.ndim > 2:
            mask = mask[..., 0]
    if mask.ndim != 2:
        return []

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=np.bool_)
    components: list[np.ndarray] = []
    for row, col in np.argwhere(mask):
        row = int(row)
        col = int(col)
        if visited[row, col]:
            continue
        stack = [(row, col)]
        visited[row, col] = True
        component_mask = np.zeros_like(mask, dtype=np.bool_)
        while stack:
            current_row, current_col = stack.pop()
            component_mask[current_row, current_col] = True
            for next_row, next_col in (
                (current_row - 1, current_col),
                (current_row + 1, current_col),
                (current_row, current_col - 1),
                (current_row, current_col + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_col < width
                    and mask[next_row, next_col]
                    and not visited[next_row, next_col]
                ):
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
        component = np.argwhere(component_mask)
        if component.shape[0] >= min_area:
            components.append(component.astype(np.int32, copy=False))
    return components


class RGBDGeometry:
    HEAD_FOCAL = 24.0 / 20.955 * 640.0
    EE_FOCAL = 15.0 / 20.955 * 640.0
    STRIDE = 4

    def __init__(self):
        rows = np.arange(0, 480, self.STRIDE, dtype=np.float32)
        cols = np.arange(0, 640, self.STRIDE, dtype=np.float32)
        self.grid_v, self.grid_u = np.meshgrid(rows, cols, indexing="ij")

        pitch = math.pi / 6.0
        self.head_rotation = np.array(
            (
                (math.cos(pitch), 0.0, math.sin(pitch)),
                (0.0, 1.0, 0.0),
                (-math.sin(pitch), 0.0, math.cos(pitch)),
            ),
            dtype=np.float32,
        )
        self.head_position = np.array((0.42161, 0.025, 0.061851), dtype=np.float32)

    @staticmethod
    def _image_array(image: torch.Tensor | None) -> np.ndarray | None:
        if image is None:
            return None
        if image.ndim == 4:
            image = image[0]
        return image[:: RGBDGeometry.STRIDE, :: RGBDGeometry.STRIDE].detach().cpu().numpy()

    @staticmethod
    def _rgb_array(image: torch.Tensor | None) -> np.ndarray | None:
        rgb = RGBDGeometry._image_array(image)
        if rgb is None or rgb.ndim < 3 or rgb.shape[-1] < 3:
            return None
        rgb = np.asarray(rgb[..., :3], dtype=np.float32)
        if not np.isfinite(rgb).any():
            return None
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
        if float(np.nanmax(rgb)) <= 1.5:
            rgb = rgb * 255.0
        return np.clip(rgb, 0.0, 255.0)

    @staticmethod
    def _color_mask(rgb: np.ndarray | None) -> np.ndarray | None:
        if rgb is None:
            return None
        channel_max = np.max(rgb, axis=-1)
        channel_min = np.min(rgb, axis=-1)
        chroma = channel_max - channel_min
        saturation = chroma / np.maximum(channel_max, 1.0)
        return (channel_max > 45.0) & (chroma > 24.0) & (saturation > 0.12)

    @staticmethod
    def _component_color_evidence(
        color_mask: np.ndarray | None,
        component: np.ndarray,
        *,
        min_pixels: int,
        min_ratio: float,
    ) -> tuple[bool, float, int]:
        if color_mask is None or component.size == 0:
            return False, 0.0, 0
        height, width = color_mask.shape
        min_row = max(0, int(np.min(component[:, 0])) - 1)
        max_row = min(height - 1, int(np.max(component[:, 0])) + 1)
        min_col = max(0, int(np.min(component[:, 1])) - 1)
        max_col = min(width - 1, int(np.max(component[:, 1])) + 1)
        if min_row > max_row or min_col > max_col:
            return False, 0.0, 0

        sample_mask = np.zeros(
            (max_row - min_row + 1, max_col - min_col + 1), dtype=np.bool_
        )
        rows = component[:, 0] - min_row
        cols = component[:, 1] - min_col
        for row_offset in (-1, 0, 1):
            expanded_rows = rows + row_offset
            row_valid = (expanded_rows >= 0) & (expanded_rows < sample_mask.shape[0])
            if not np.any(row_valid):
                continue
            for col_offset in (-1, 0, 1):
                expanded_cols = cols + col_offset
                valid = row_valid & (expanded_cols >= 0) & (
                    expanded_cols < sample_mask.shape[1]
                )
                if np.any(valid):
                    sample_mask[expanded_rows[valid], expanded_cols[valid]] = True

        sampled_pixels = int(np.count_nonzero(sample_mask))
        if sampled_pixels <= 0:
            return False, 0.0, 0
        color_pixels = int(np.count_nonzero(color_mask[min_row : max_row + 1, min_col : max_col + 1] & sample_mask))
        color_ratio = color_pixels / sampled_pixels
        return (
            color_pixels >= min_pixels and color_ratio >= min_ratio,
            float(color_ratio),
            color_pixels,
        )

    def _optical_points(self, depth: np.ndarray, focal: float) -> np.ndarray:
        if depth.ndim == 3:
            depth = depth[..., 0]
        x = (self.grid_u - 319.5) * depth / focal
        y = (self.grid_v - 239.5) * depth / focal
        return np.stack((x, y, depth), axis=-1)

    def _head_points(self, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        optical = self._optical_points(depth, self.HEAD_FOCAL)
        world_convention = np.stack(
            (optical[..., 2], -optical[..., 0], -optical[..., 1]), axis=-1
        )
        points = world_convention @ self.head_rotation.T + self.head_position
        valid = np.isfinite(depth) & (depth > 0.08) & (depth < 6.0)
        return points, valid

    @staticmethod
    def _component_detection(
        component: np.ndarray,
        points: np.ndarray,
        confidence: float,
    ) -> Detection:
        rows = component[:, 0]
        cols = component[:, 1]
        position = np.median(points[rows, cols], axis=0).astype(np.float32)
        return Detection(
            position_base=position,
            pixel=(
                float(np.mean(cols) * RGBDGeometry.STRIDE),
                float(np.mean(rows) * RGBDGeometry.STRIDE),
            ),
            area=int(component.shape[0]),
            confidence=float(confidence),
        )

    def detect_head_object(
        self,
        depth_image: torch.Tensor | None,
        rgb_image: torch.Tensor | None = None,
        projected_gravity: torch.Tensor | None = None,
    ) -> Detection | None:
        if (
            projected_gravity is None
            and rgb_image is not None
            and getattr(rgb_image, "ndim", 0) <= 2
        ):
            projected_gravity = rgb_image
            rgb_image = None
        depth = self._image_array(depth_image)
        color_mask = self._color_mask(self._rgb_array(rgb_image))
        if depth is None:
            return None
        if depth.ndim == 3:
            depth = depth[..., 0]
        points, valid = self._head_points(depth)
        if projected_gravity is None:
            up_axis = np.array((0.0, 0.0, 1.0), dtype=np.float32)
        else:
            gravity = projected_gravity.detach().float().cpu().numpy()
            gravity_norm = float(np.linalg.norm(gravity))
            up_axis = (
                -gravity / gravity_norm
                if gravity_norm > 1.0e-5
                else np.array((0.0, 0.0, 1.0), dtype=np.float32)
            )
        plane_coordinate = points @ up_axis
        floor_samples = plane_coordinate[
            valid
            & (points[..., 0] > 0.25)
            & (points[..., 0] < 4.5)
            & (np.abs(points[..., 1]) < 2.5)
        ]
        if floor_samples.size < 100:
            return None
        floor_coordinate = float(np.percentile(floor_samples, 35.0))
        height = plane_coordinate - floor_coordinate
        mask = (
            valid
            & (points[..., 0] > 0.22)
            & (points[..., 0] < 3.8)
            & (np.abs(points[..., 1]) < 1.8)
            & (height > 0.013)
            & (height < 0.28)
        )

        candidates: list[Detection] = []
        for component in _connected_components(mask, min_area=3):
            if component.shape[0] > 900:
                continue
            row_span = int(np.ptp(component[:, 0])) + 1
            col_span = int(np.ptp(component[:, 1])) + 1
            if row_span > 70 or col_span > 70:
                continue
            detection = self._component_detection(component, points, confidence=0.0)
            detection_height = float(detection.position_base @ up_axis - floor_coordinate)
            if not (0.011 < detection_height < 0.26):
                continue
            has_color, color_ratio, _ = self._component_color_evidence(
                color_mask, component, min_pixels=3, min_ratio=0.02
            )
            if not has_color:
                continue
            detection.confidence = min(
                1.0, component.shape[0] / 20.0
            ) * (0.85 + min(0.30, 2.0 * color_ratio)) / (
                1.0 + 0.15 * detection.x
            )
            candidates.append(detection)
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.confidence)

    def detect_ee_object(
        self,
        depth_image: torch.Tensor | None,
        rgb_image: torch.Tensor | None,
        camera_transform: torch.Tensor | None = None,
    ) -> Detection | None:
        if camera_transform is None:
            camera_transform = rgb_image
            rgb_image = None
        depth = self._image_array(depth_image)
        color_mask = self._color_mask(self._rgb_array(rgb_image))
        if depth is None:
            return None
        if depth.ndim == 3:
            depth = depth[..., 0]
        optical = self._optical_points(depth, self.EE_FOCAL)
        rotation = camera_transform[:3, :3].detach().cpu().numpy()
        translation = camera_transform[:3, 3].detach().cpu().numpy()
        points = optical @ rotation.T + translation
        valid = np.isfinite(depth) & (depth > 0.04) & (depth < 1.5)
        valid_points = points[valid]
        if valid_points.shape[0] < 150:
            return None

        design = np.column_stack(
            (valid_points[:, 0], valid_points[:, 1], np.ones(valid_points.shape[0]))
        )
        coefficients, _, _, _ = np.linalg.lstsq(
            design.astype(np.float64),
            valid_points[:, 2].astype(np.float64),
            rcond=None,
        )
        plane_z = (
            coefficients[0] * points[..., 0]
            + coefficients[1] * points[..., 1]
            + coefficients[2]
        )
        residual = points[..., 2] - plane_z
        mask = valid & (residual > 0.012) & (residual < 0.24)
        candidates: list[Detection] = []
        for component in _connected_components(mask, min_area=4):
            if component.shape[0] > 1000:
                continue
            has_color, color_ratio, _ = self._component_color_evidence(
                color_mask, component, min_pixels=2, min_ratio=0.015
            )
            if not has_color:
                continue
            detection = self._component_detection(component, points, confidence=0.0)
            center_distance = math.hypot(
                detection.pixel[0] - 319.5, detection.pixel[1] - 239.5
            )
            detection.confidence = (
                min(1.0, component.shape[0] / 15.0)
                * (0.85 + min(0.30, 2.0 * color_ratio))
                * math.exp(-center_distance / 260.0)
            )
            candidates.append(detection)
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.confidence)


class ArmController:
    CARRY_Q = torch.tensor((0.0, 0.5, -1.0, 0.0, 1.0, 0.0))
    PREGRASP_READY_Q = torch.tensor((0.0, 1.55, -2.10, 0.0, -1.10, 0.0))
    KINEMATIC_SIGN = torch.tensor((-1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
    OPEN_GRIPPER = torch.tensor((0.035, -0.035))
    CLOSED_GRIPPER = torch.tensor((0.0, 0.0))
    DOWN_AXIS = torch.tensor((0.0, 0.0, -1.0))

    def __init__(self, device: torch.device):
        self.device = device
        self.kinematics = PiperKinematics(device)
        self.kinematic_sign = self.KINEMATIC_SIGN.to(device)
        env_lower = self.kinematics.lower.clone()
        env_upper = self.kinematics.upper.clone()
        signed_lower = env_lower * self.kinematic_sign
        signed_upper = env_upper * self.kinematic_sign
        self.kinematics.lower = torch.minimum(signed_lower, signed_upper)
        self.kinematics.upper = torch.maximum(signed_lower, signed_upper)
        self.carry_q = self.to_env_q(self.CARRY_Q.to(device))
        self.pregrasp_ready_q = self.to_env_q(self.PREGRASP_READY_Q.to(device))
        self.open_gripper = self.OPEN_GRIPPER.to(device)
        self.closed_gripper = self.CLOSED_GRIPPER.to(device)
        self.down_axis = self.DOWN_AXIS.to(device)

    def to_kin_q(self, env_q: torch.Tensor) -> torch.Tensor:
        return env_q * self.kinematic_sign.to(env_q)

    def to_env_q(self, kin_q: torch.Tensor) -> torch.Tensor:
        return kin_q * self.kinematic_sign.to(kin_q)

    def forward(self, env_q: torch.Tensor) -> torch.Tensor:
        return self.kinematics.forward(self.to_kin_q(env_q))

    @staticmethod
    def _smooth_target(
        current: torch.Tensor, desired: torch.Tensor, max_delta: float
    ) -> torch.Tensor:
        return current + (desired - current).clamp(-max_delta, max_delta)

    def joint_action(
        self,
        current_q: torch.Tensor,
        desired_q: torch.Tensor,
        gripper: torch.Tensor,
        max_delta: float = 0.035,
    ) -> torch.Tensor:
        desired_q = self._smooth_target(current_q, desired_q.to(current_q), max_delta)
        target = torch.cat((desired_q, gripper.to(current_q)))
        return (target / 0.5).clamp(-6.0, 6.0)

    def pose_action(
        self,
        current_q: torch.Tensor,
        target_position: torch.Tensor,
        gripper: torch.Tensor,
        align_down: bool = True,
        max_joint_delta: float = 0.035,
        action_max_delta: float = 0.04,
        yaw_limit: float | None = None,
    ) -> torch.Tensor:
        current_kin_q = self.to_kin_q(current_q)
        desired_kin_q = self.kinematics.dls_step(
            current_kin_q,
            target_position.to(current_q),
            target_axis=self.down_axis if align_down else None,
            damping=0.09,
            max_joint_delta=max_joint_delta,
            max_position_error=0.035,
            orientation_weight=0.18,
        )
        if yaw_limit is not None:
            desired_kin_q[0] = desired_kin_q[0].clamp(-float(yaw_limit), float(yaw_limit))
        desired_q = self.to_env_q(desired_kin_q)
        return self.joint_action(current_q, desired_q, gripper, max_delta=action_max_delta)


class State(Enum):
    ARM_INIT = auto()
    SEARCH = auto()
    APPROACH = auto()
    CREEP = auto()
    SWEEP = auto()
    RECOVER = auto()
    DONE = auto()


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class PlannerDetection:
    """Planner-facing detection mirroring task_b_perception.Detection (G1).

    Built from the dog-side RGBDGeometry head Detection plus the pose: world
    coords come from pose ⊕ base, distance from hypot(fwd, lat).
    """
    track_id: int
    rel_x: float          # forward in base frame
    rel_y: float          # left in base frame
    distance: float       # ground distance hypot(rel_x, rel_y)
    confidence: float
    world_x: float
    world_y: float


class PerceptionAdapter:
    """Adapts the dog RGBDGeometry head detector into planner detections.

    The track-assignment idea is ported from task_b_perception._assign_track:
    nearest-existing-track within MATCH_DIST (0.75 m) keeps its id; otherwise a
    fresh id is allocated. The detector itself (RGBDGeometry.detect_head_object)
    is the dog-calibrated one — DO NOT swap in the G1 intrinsics here.
    """
    MATCH_DIST = 0.75

    def __init__(self, geometry: RGBDGeometry):
        self.geometry = geometry
        self.tracks: dict[int, tuple[float, float]] = {}
        self.next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1

    def update(self, obs: dict[str, Any], projected_gravity: torch.Tensor,
               odom_x: float, odom_y: float, yaw: float) -> list[PlannerDetection]:
        images = obs.get("image", {}) if isinstance(obs, dict) else {}
        head_depth = images.get("head_depth") if isinstance(images, dict) else None
        if head_depth is None:
            return []
        head_rgb = images.get("head_rgb") if isinstance(images, dict) else None
        det = self.geometry.detect_head_object(head_depth, head_rgb, projected_gravity)
        if det is None:
            return []
        rel_x = float(det.position_base[0])
        rel_y = float(det.position_base[1])
        distance = math.hypot(rel_x, rel_y)
        c = math.cos(yaw)
        s = math.sin(yaw)
        world_x = odom_x + c * rel_x - s * rel_y
        world_y = odom_y + s * rel_x + c * rel_y
        track_id = self._assign_track(world_x, world_y)
        return [PlannerDetection(
            track_id=track_id,
            rel_x=rel_x,
            rel_y=rel_y,
            distance=distance,
            confidence=float(det.confidence),
            world_x=world_x,
            world_y=world_y,
        )]

    def _assign_track(self, world_x: float, world_y: float) -> int:
        best_id = None
        best_dist = self.MATCH_DIST
        for tid, (tx, ty) in self.tracks.items():
            d = math.hypot(world_x - tx, world_y - ty)
            if d < best_dist:
                best_dist = d
                best_id = tid
        if best_id is None:
            best_id = self.next_id
            while best_id in self.tracks:
                best_id += 1
            self.next_id = best_id + 1
        self.tracks[best_id] = (world_x, world_y)
        return best_id


class TaskBController:
    """G1 FSM ported verbatim onto the B2Piper dog.

    Phases match task_b_planner.TaskBPlanner: SEARCH → APPROACH → CREEP → SWEEP →
    RECOVER (G1's stand_up). The G1 "squat + two-hand floor brush" is replaced by
    a crouch-and-down arm sweep: the body lowers via the LOW_STANCE_ACTION blend
    (the dog cannot lower its base via a height command), then the Piper arm IK
    drives the EE down (target z ≈ −0.45 in base frame) while oscillating
    forward/lateral over the target. Any score increase while pursuing a target
    claims it (the dog never grasps — touching is the score event).
    """

    # ---- Score / completion -------------------------------------------------
    TOUCH_SCORE_TARGET = 18.0          # 18 colored objects @ +1 each → giveup
    ARM_INIT_STEPS = 60                # warm-up: settle arm to carry pose

    # ---- Velocity calibration (B2 walking policy) ---------------------------
    # The flat policy stalls below ~0.4 m/s forward command (legs freeze in
    # place). Keep G1's FSM logic, but pin a forward floor whenever we want the
    # dog to translate so proportional control cannot decay into the stall zone.
    # Policy clamp is vx∈[-0.65, 0.85], vy∈[-0.45, 0.45], wz∈[-1.05, 1.05].
    FORWARD_FLOOR = 0.45               # m/s — empirical minimum stable forward
    APPROACH_VX_MAX = 0.75
    APPROACH_VY_MAX = 0.40
    APPROACH_WZ_MAX = 1.0
    # Search: scan a bit SLOWER (more perception frames per degree → less likely
    # to spin past an object the head cam only sees up close) but RELOCATE FARTHER
    # so the dog covers the 10×10 arena in fewer spin-stops — addresses the
    # "一直自己转动" (keeps spinning in place) the operator saw. Still a full 360°
    # per stop so the post-spin heading is consistent and relocation goes forward.
    SEARCH_YAW = 0.9                   # spin rate (also used for scan-in-place)
    # --- Systematic waypoint (boustrophedon) coverage --------------------------
    # The old open-loop rotate-then-relocate search had a hard spin-vs-coverage
    # trade-off (tight relocate covers well but spins a lot; loose relocate
    # wanders). Replace with a deterministic snake sweep over the object region.
    # Arena objects span world x∈[-15,-4], y∈[-15,-7]; we cover the interior
    # x∈[-14,-5], y∈[-13,-7] at ~3 m spacing. Snake order so consecutive
    # waypoints are adjacent (efficient travel, no teleporting across the arena).
    SEARCH_WAYPOINTS = (
        (-14.0, -13.0), (-14.0, -10.0), (-14.0, -7.0),
        (-11.0, -7.0),  (-11.0, -10.0), (-11.0, -13.0),
        (-8.0, -13.0),  (-8.0, -10.0),  (-8.0, -7.0),
        (-5.0, -7.0),   (-5.0, -10.0),  (-5.0, -13.0),
    )
    SEARCH_WP_ARRIVE = 0.9             # odometry distance that counts as "at" a wp.
                                       # Loose: odometry drifts over the 1200 s
                                       # episode and the head cam only sees ~0.6-2 m
                                       # ahead, so coverage only needs to get NEAR a
                                       # region, scan, and move on.
    SEARCH_WP_TIMEOUT = 600           # max travel steps to one wp before forcing a
                                       # scan + advance (drift/blocked fallback — see
                                       # note in _step_search; prevents a drifted or
                                       # unreachable wp from stalling coverage).
    # One slow full 360° scan at each waypoint: round(2π / SEARCH_YAW / dt).
    SEARCH_SCAN_STEPS = round(2.0 * math.pi / SEARCH_YAW / 0.02)  # ≈ 349
    CREEP_VEL = 0.45                   # at/above forward floor
    CREEP_STEPS = 10                   # short forward poke before sweeping
    # Commit harder to a SEEN object: drive to its remembered world coords via
    # odometry for longer before giving up + re-searching (was 250 → respun too
    # eagerly when the head cam lost the object up close).
    BLIND_WALK_STEPS = 400             # max approach steps with no fresh sighting
    ARRIVE_DIST = 0.35                 # odometry distance that triggers creep
    APPROACH_STANDOFF = 0.25           # drive-to point sits this far short
    MAX_ATTEMPTS = 2                   # creep/sweep tries per object (was 2 —
                                       # more retries to claw back flat-object
                                       # misses on the coverage pass)

    # ---- Sweep (the manipulation swap) --------------------------------------
    # Offline IK on the Piper bottoms out at base-frame z≈−0.12; ground objects
    # sit at z≈−0.45 at normal stance, so the body must crouch for the EE to
    # reach the ground. The ~6 missed objects are FLAT ones sitting ~0.15 m
    # below the arm even at the symmetric low stance, so the SWEEP crouch is now
    # a controller-owned, DEEPER + NOSE-DOWN posture (see SWEEP_CROUCH_FRONT /
    # SWEEP_CROUCH_REAR) that dips the front-mounted Piper toward the floor.
    SWEEP_CROUCH_BLEND = 1.0           # full crouch during SWEEP
    # Per-leg (hip, thigh, calf) crouch targets used ONLY in SWEEP. Front legs
    # (FR, FL) fold more than rear (RR, RL) so the body pitches nose-down and
    # the front-mounted arm gets lower. Larger thigh + more-negative calf =
    # more folded = lower. Dial DOWN if sim shows the dog tipping over.
    # Conservative nose-down: only modestly deeper than the proven-stable
    # symmetric LOW_STANCE (0.35,−0.55) which got 12/18 without falling. The
    # aggressive (0.75,−1.20) front fold dropped a thigh/trunk onto the floor →
    # illegal_contact termination (B2Piper illegal bodies = base_link, *_hip,
    # *_thigh). Front a touch deeper than rear for a slight nose-down dip.
    SWEEP_CROUCH_FRONT = (0.0, 0.35, -0.55)   # FR, FL (symmetric = proven 12/18 champion)
    SWEEP_CROUCH_REAR = (0.0, 0.35, -0.55)    # RR, RL (nose-down hurt: tilted sweep misaligns EE)
    SWEEP_MAX_STEPS = 220              # first attempt: full sweep (hits land up to
                                       # ~step 120; the tail catches late ones)
    SWEEP_MAX_STEPS_RETRY = 130        # 2nd+ attempt: if a FULL sweep already
                                       # missed this object, a long re-sweep rarely
                                       # helps — give up faster (saves time / less
                                       # repeated poking at flat, unscoreable ones)
    SWEEP_SETTLE_STEPS = 5             # short bottom hold after sweep
    SWEEP_FWD_AMP = 0.14               # widened from G1 0.10 to cover more ground
    SWEEP_LAT_AMP = 0.18               # widened from G1 0.14 to cover more ground
    SWEEP_FWD_PERIOD = 25              # G1 SWEEP_FWD_PERIOD
    SWEEP_LAT_PERIOD = 36              # G1 SWEEP_LAT_PERIOD
    # Target z just below the EE's kinematic floor (≈−0.12): targeting the old
    # −0.45 made DLS extend the arm forward (EE fwd≈0.59 vs object≈0.38, missing
    # it). −0.20 keeps the arm at its lowest while centering the EE over the
    # object so the oscillation actually passes within the 0.12 m scoring sphere.
    SWEEP_TARGET_Z = -0.20
    # Reach clamps so DLS doesn't get pushed past kinematic limits.
    SWEEP_FWD_MIN, SWEEP_FWD_MAX = 0.15, 0.58
    SWEEP_LAT_MIN, SWEEP_LAT_MAX = -0.34, 0.34

    # ---- Recover (stand_up) -------------------------------------------------
    RECOVER_STEPS = 65                 # G1 STAND_RAMP_STEPS

    # ---- Perception ----------------------------------------------------------
    PERCEPTION_INTERVAL = 5            # G1 solution.PERCEPTION_INTERVAL
    CONFIDENCE_FLOOR = 0.2             # G1 CONFIDENCE_FLOOR
    TARGET_MIN_DIST = 0.55             # dog head cam min visible ≈ 0.74 m;
                                       # be permissive on the low end so we
                                       # don't reject true sightings of close
                                       # objects.
    TARGET_MAX_DIST = 1.8              # reverted from 2.2: far objects localize
                                       # imprecisely, so the dog chased then lost
                                       # them → blind-timeouts. With systematic
                                       # waypoint coverage the dog gets close to
                                       # each object, so it doesn't need the far band.
    REFRESH_MIN_DIST = 0.50            # ignore matches closer than this

    # ---- Telemetry ----------------------------------------------------------
    SWEEP_TELEMETRY_INTERVAL = 40      # emit EE/target distance every N steps

    def __init__(
        self,
        policy_path: str | None = None,
        debug: bool | None = None,
        debug_interval: int | None = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if policy_path is None:
            policy_path = os.path.join(os.path.dirname(__file__), "policy.pt")
        self.debug = _env_flag("ATEC_TASK_B_DEBUG") if debug is None else bool(debug)
        self.return_debug = _env_flag("ATEC_TASK_B_RETURN_DEBUG")
        if debug_interval is None:
            debug_interval = int(os.environ.get("ATEC_TASK_B_DEBUG_INTERVAL", "50"))
        self.debug_interval = max(1, int(debug_interval))
        self.debug_max_events = max(
            1, int(os.environ.get("ATEC_TASK_B_DEBUG_MAX_EVENTS", "256"))
        )
        self.locomotion = B2Locomotion(policy_path, self.device)
        self.arm = ArmController(self.device)
        self.geometry = RGBDGeometry()
        self.odometry = Odometry()
        self.perception = PerceptionAdapter(self.geometry)
        # SWEEP crouch posture: deeper + nose-down (front legs folded more than
        # rear). B2 leg order in the 12-vector is FR, FL, RR, RL, each
        # (hip, thigh, calf): FR=0-2, FL=3-5, RR=6-8, RL=9-11.
        self.sweep_crouch_action = torch.tensor(
            list(self.SWEEP_CROUCH_FRONT) + list(self.SWEEP_CROUCH_FRONT)
            + list(self.SWEEP_CROUCH_REAR) + list(self.SWEEP_CROUCH_REAR),
            dtype=torch.float32,
            device=self.device,
        )
        self.reset()

    # ------------------------------------------------------------------- reset
    def reset(self, **_: Any) -> None:
        self.debug_events: list[dict[str, Any]] = []
        self.metrics = {
            "transitions": 0,
            "head_detections": 0,
            "score_changes": 0,
            "sweeps": 0,
            "recoveries": 0,
        }
        self.state = State.ARM_INIT
        self.state_steps = 0
        self.total_steps = 0
        self.last_score = 0.0
        self.leg_posture_blend = 0.0
        self._perception_step = 0
        self._cached_detections: list[PlannerDetection] = []

        # Active target (track_id, world_x, world_y), like the G1 planner.
        self.active: tuple[int, float, float] | None = None
        self.memory: dict[int, tuple[float, float]] = {}
        self.attempts: dict[int, int] = {}
        self.scored: set[int] = set()

        # Per-phase counters.
        self.steps_since_seen = 0
        self.creep_step = 0
        self.sweep_step = 0
        self.settle_step = 0
        self.recover_step = 0
        self.sweep_subphase = "sweep"  # "sweep" | "settle"

        # Systematic-coverage search state. search_waypoint_idx is reset ONLY
        # here (not on every SEARCH re-entry) so coverage resumes where it left
        # off after the dog breaks away to pursue an object.
        self.search_waypoint_idx = 0
        self.search_wp_steps = 0       # steps heading to the current waypoint
        self.search_scan_steps = 0     # steps spent scanning at the waypoint
        self.search_mode = "travel"    # "travel" | "scan"

        self.odometry.reset()
        self.locomotion.reset()
        self.perception.reset()
        self._record_debug("reset")

    # ---------------------------------------------------------------- debug
    def _record_debug(self, event: str, **fields: Any) -> None:
        record = {
            "event": event,
            "step": self.total_steps,
            "state": self.state.name,
            "state_steps": self.state_steps,
        }
        record.update({key: _debug_value(value) for key, value in fields.items()})
        self.debug_events.append(record)
        if len(self.debug_events) > self.debug_max_events:
            del self.debug_events[: len(self.debug_events) - self.debug_max_events]
        if self.debug:
            print(
                "TASK_B_DEBUG " + json.dumps(record, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )

    def _transition(self, state: State, reason: str | None = None, **extra: Any) -> None:
        previous_state = self.state
        self.metrics["transitions"] += 1
        if state == State.SWEEP:
            self.metrics["sweeps"] += 1
        elif state == State.RECOVER:
            self.metrics["recoveries"] += 1
        self._record_debug(
            "transition",
            from_state=previous_state.name,
            to_state=state.name,
            duration_steps=self.state_steps,
            reason=reason,
            **extra,
        )
        self.state = state
        self.state_steps = 0

    def get_debug_summary(self) -> dict[str, Any]:
        return {
            "state": self.state.name,
            "state_steps": self.state_steps,
            "total_steps": self.total_steps,
            "last_score": self.last_score,
            "leg_posture_blend": self.leg_posture_blend,
            "scored": sorted(self.scored),
            "active": None if self.active is None else list(self.active),
            "memory_count": len(self.memory),
            "odometry": {
                "x": self.odometry.x,
                "y": self.odometry.y,
                "yaw": self.odometry.yaw,
            },
            "metrics": dict(self.metrics),
            "last_events": list(self.debug_events[-10:]),
        }

    # -------------------------------------------------------------- perception
    def _refresh_vision(self, obs: dict[str, Any], parsed: ParsedObservation) -> None:
        # G1-style throttling: only run the detector every PERCEPTION_INTERVAL.
        if self._perception_step % self.PERCEPTION_INTERVAL == 0:
            self._cached_detections = self.perception.update(
                obs,
                parsed.projected_gravity,
                self.odometry.x,
                self.odometry.y,
                self.odometry.yaw,
            )
            if self._cached_detections:
                self.metrics["head_detections"] += len(self._cached_detections)
        self._perception_step += 1

    # ------------------------------------------------------ planner helpers
    def _in_sighting_band(self, distance: float) -> bool:
        return self.TARGET_MIN_DIST <= distance <= self.TARGET_MAX_DIST

    def _exhausted(self, track_id: int) -> bool:
        return track_id in self.scored or self.attempts.get(track_id, 0) >= self.MAX_ATTEMPTS

    def _charge_attempt(self, track_id: int) -> None:
        self.attempts[track_id] = self.attempts.get(track_id, 0) + 1

    def _select_target(self, detections: list[PlannerDetection]) -> tuple[int, float, float] | None:
        """G1 _select_target: (a) nearest non-exhausted fresh in-band detection,
        else (b) nearest non-exhausted remembered object, else None."""
        fresh = [
            d for d in detections
            if d.confidence >= self.CONFIDENCE_FLOOR
            and self._in_sighting_band(d.distance)
            and not self._exhausted(d.track_id)
        ]
        if fresh:
            best = min(fresh, key=lambda d: (d.distance, -d.confidence))
            return (best.track_id, best.world_x, best.world_y)
        remembered = [
            (tid, wx, wy) for tid, (wx, wy) in self.memory.items()
            if not self._exhausted(tid)
        ]
        if remembered:
            ox, oy = self.odometry.x, self.odometry.y
            return min(remembered, key=lambda t: math.hypot(t[1] - ox, t[2] - oy))
        return None

    def _refresh_active(self, detections: list[PlannerDetection]) -> bool:
        if self.active is None:
            return False
        tid = self.active[0]
        for d in detections:
            if d.track_id == tid and d.distance >= self.REFRESH_MIN_DIST:
                self.active = (tid, d.world_x, d.world_y)
                self.steps_since_seen = 0
                return True
        return False

    # -------------------------------------------------------- motion helpers
    def _drive_to(self, tx: float, ty: float, tyaw: float) -> tuple[float, float, float]:
        """Pose-aware drive to (tx, ty) facing tyaw. Mirrors G1 _drive_to."""
        ex = tx - self.odometry.x
        ey = ty - self.odometry.y
        c = math.cos(self.odometry.yaw)
        s = math.sin(self.odometry.yaw)
        body_x = c * ex + s * ey
        body_y = -s * ex + c * ey
        yaw_err = _wrap_to_pi(tyaw - self.odometry.yaw)
        vx = _clamp(0.9 * body_x, -0.18, self.APPROACH_VX_MAX)
        vy = _clamp(0.8 * body_y, -0.22, 0.22)
        wz = _clamp(1.8 * yaw_err, -self.APPROACH_WZ_MAX, self.APPROACH_WZ_MAX)
        if abs(yaw_err) > 0.9:
            # Spin-in-place first; only allow trickle forward.
            vx = min(vx, 0.05)
            vy = _clamp(vy, -0.08, 0.08)
        return vx, vy, wz

    def _approach_velocity(self, target: tuple[int, float, float]) -> tuple[float, float, float]:
        """Drive toward a standoff point STANDOFF short of the object, then apply
        the B2 forward floor so the policy doesn't stall while still far away."""
        _, wx, wy = target
        bearing = math.atan2(wy - self.odometry.y, wx - self.odometry.x)
        sx = wx - self.APPROACH_STANDOFF * math.cos(bearing)
        sy = wy - self.APPROACH_STANDOFF * math.sin(bearing)
        vx, vy, wz = self._drive_to(sx, sy, bearing)
        # If we still need to translate forward, hold the floor (yaw-aligned).
        yaw_err = abs(_wrap_to_pi(bearing - self.odometry.yaw))
        ground_dist = math.hypot(wx - self.odometry.x, wy - self.odometry.y)
        if ground_dist > self.ARRIVE_DIST and yaw_err < 0.7:
            vx = max(vx, self.FORWARD_FLOOR)
        vx = _clamp(vx, -0.65, 0.85)
        vy = _clamp(vy, -self.APPROACH_VY_MAX, self.APPROACH_VY_MAX)
        wz = _clamp(wz, -self.APPROACH_WZ_MAX, self.APPROACH_WZ_MAX)
        return vx, vy, wz

    def _creep_velocity(self, target: tuple[int, float, float]) -> tuple[float, float, float]:
        _, wx, wy = target
        bearing = math.atan2(wy - self.odometry.y, wx - self.odometry.x)
        yaw_err = _wrap_to_pi(bearing - self.odometry.yaw)
        wz = _clamp(1.5 * yaw_err, -0.5, 0.5)
        # Above the policy stall floor (G1 used 0.35; the dog needs ≥0.40).
        return (max(self.CREEP_VEL, self.FORWARD_FLOOR), 0.0, wz)

    def _object_base_frame(self, target_world: tuple[float, float]) -> tuple[float, float]:
        wx, wy = target_world
        dx, dy = wx - self.odometry.x, wy - self.odometry.y
        c, s = math.cos(self.odometry.yaw), math.sin(self.odometry.yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def _sweep_target(self, target_world: tuple[float, float], sweep_t: int) -> tuple[float, float, float]:
        """Oscillating base-frame target (forward, lateral, z) for the Piper IK.

        Oscillation amplitudes/periods are the G1 ones; z is pinned LOW so the
        DLS step drives the EE all the way down (the body crouch supplies the
        reach budget that the standalone arm doesn't have).
        """
        fwd0, lat0 = self._object_base_frame(target_world)
        fwd = _clamp(
            fwd0 + self.SWEEP_FWD_AMP * math.sin(2.0 * math.pi * sweep_t / self.SWEEP_FWD_PERIOD),
            self.SWEEP_FWD_MIN, self.SWEEP_FWD_MAX,
        )
        lat = _clamp(
            lat0 + self.SWEEP_LAT_AMP * math.sin(2.0 * math.pi * sweep_t / self.SWEEP_LAT_PERIOD),
            self.SWEEP_LAT_MIN, self.SWEEP_LAT_MAX,
        )
        return fwd, lat, self.SWEEP_TARGET_Z

    # ------------------------------------------------- leg-posture (crouch)
    def _leg_posture_target_blend(self) -> float:
        """Crouch at FULL during SWEEP; un-crouch in RECOVER; otherwise stand."""
        if self.state == State.SWEEP:
            return self.SWEEP_CROUCH_BLEND
        return 0.0

    def _apply_leg_posture(self, leg_action: torch.Tensor) -> torch.Tensor:
        target_blend = self._leg_posture_target_blend()
        if target_blend > self.leg_posture_blend:
            self.leg_posture_blend = min(target_blend, self.leg_posture_blend + 0.025)
        else:
            self.leg_posture_blend = max(target_blend, self.leg_posture_blend - 0.015)
        if self.leg_posture_blend <= 0.0:
            return leg_action
        # During SWEEP, blend toward the deeper nose-down crouch; for any other
        # transient (e.g. the blend ramping back down in RECOVER) use the milder
        # symmetric low stance so we don't hold a nose-down pitch while standing.
        if self.state == State.SWEEP:
            posture = self.sweep_crouch_action.to(leg_action)
        else:
            posture = self.locomotion.low_stance_action.to(leg_action)
        blend = float(self.leg_posture_blend)
        return ((1.0 - blend) * leg_action + blend * posture).clamp(-5.0, 5.0)

    # ----------------------------------------------------------- main step
    def step(self, obs: dict[str, Any], current_score: float) -> dict[str, Any]:
        current_score = float(current_score)
        if current_score >= self.TOUCH_SCORE_TARGET:
            self.state = State.DONE
            self.last_score = current_score
            return {"action": [0.0] * ACTION_DIM, "giveup": True}

        parsed = ObservationParser.parse(obs["proprio"])
        parsed = ParsedObservation(
            *(torch.nan_to_num(value).to(self.device) for value in parsed.__dict__.values())
        )
        self.odometry.update(parsed.base_lin_vel, parsed.base_ang_vel)
        self._refresh_vision(obs, parsed)

        # Score bookkeeping: any positive delta while pursuing claims the target.
        score_delta = current_score - self.last_score
        if abs(score_delta) > 1.0e-6:
            self.metrics["score_changes"] += 1
            self._record_debug(
                "score_changed",
                previous_score=self.last_score,
                current_score=current_score,
                delta=score_delta,
                active=None if self.active is None else list(self.active),
            )

        # Memory: every confident in-band detection's world coords (any phase).
        for d in self._cached_detections:
            if d.confidence >= self.CONFIDENCE_FLOOR and self._in_sighting_band(d.distance):
                self.memory[d.track_id] = (d.world_x, d.world_y)

        # If we are pursuing a target and the score went up, claim it + recover.
        if (
            score_delta > 1.0e-6
            and self.active is not None
            and self.state in (State.APPROACH, State.CREEP, State.SWEEP)
        ):
            self.scored.add(self.active[0])
            self._begin_recover("score_delta")

        # Compute the per-phase velocity command + arm action.
        command, arm_action = self._phase_step(parsed, current_score)

        # Walking policy + crouch blend.
        leg_action = self.locomotion.act(parsed, command)
        leg_action = self._apply_leg_posture(leg_action)
        action = torch.cat((leg_action, arm_action))
        action = torch.nan_to_num(action).clamp(-6.0, 6.0)

        self.last_score = current_score
        self.total_steps += 1
        self.state_steps += 1
        result = {"action": action.detach().cpu().tolist(), "giveup": False}
        if self.return_debug:
            result["debug"] = self.get_debug_summary()
        return result

    # ----------------------------------------------------- phase dispatch
    def _phase_step(
        self, parsed: ParsedObservation, current_score: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.state == State.ARM_INIT:
            return self._step_arm_init(parsed)
        if self.state == State.SEARCH:
            return self._step_search(parsed)
        if self.state == State.APPROACH:
            return self._step_approach(parsed)
        if self.state == State.CREEP:
            return self._step_creep(parsed)
        if self.state == State.SWEEP:
            return self._step_sweep(parsed)
        if self.state == State.RECOVER:
            return self._step_recover(parsed)
        # DONE
        zero_cmd = torch.zeros(3, device=self.device)
        arm_action = self.arm.joint_action(
            parsed.arm_q, self.arm.carry_q, self.arm.open_gripper
        )
        return zero_cmd, arm_action

    # ----------------------------------------------------------- ARM_INIT
    def _step_arm_init(self, parsed: ParsedObservation) -> tuple[torch.Tensor, torch.Tensor]:
        command = torch.zeros(3, device=self.device)
        arm_action = self.arm.joint_action(
            parsed.arm_q, self.arm.carry_q, self.arm.open_gripper
        )
        if (
            torch.linalg.vector_norm(parsed.arm_q - self.arm.carry_q) < 0.12
            and self.state_steps > 20
        ) or self.state_steps > self.ARM_INIT_STEPS:
            self._transition(State.SEARCH, "arm_initialized")
        return command, arm_action

    # ------------------------------------------------------------ SEARCH
    def _step_search(self, parsed: ParsedObservation) -> tuple[torch.Tensor, torch.Tensor]:
        # Detection interrupts coverage from anywhere: a target seen mid-travel
        # or mid-scan immediately breaks away into APPROACH. search_waypoint_idx
        # is NOT reset here, so coverage resumes from this waypoint afterward.
        target = self._select_target(self._cached_detections)
        if target is not None:
            self.active = target
            self.steps_since_seen = 0
            self._transition(State.APPROACH, "target_selected",
                             target=list(target))
            return self._step_approach(parsed)

        arm_action = self.arm.joint_action(
            parsed.arm_q, self.arm.carry_q, self.arm.open_gripper
        )
        wx, wy = self.SEARCH_WAYPOINTS[self.search_waypoint_idx]

        if self.search_mode == "travel":
            self.search_wp_steps += 1
            dist = math.hypot(wx - self.odometry.x, wy - self.odometry.y)
            # Arrived (loose, drift-tolerant) OR timed out (drift/blocked
            # fallback) → stop and scan-in-place around the waypoint.
            if dist <= self.SEARCH_WP_ARRIVE or self.search_wp_steps > self.SEARCH_WP_TIMEOUT:
                self.search_mode = "scan"
                self.search_scan_steps = 0
                self._record_debug(
                    "search_wp_reached",
                    waypoint_idx=self.search_waypoint_idx,
                    waypoint=[wx, wy],
                    distance=dist,
                    travel_steps=self.search_wp_steps,
                    timed_out=bool(self.search_wp_steps > self.SEARCH_WP_TIMEOUT),
                )
                command = torch.zeros(3, device=self.device)
                return command, arm_action
            # Drive toward the waypoint, facing it; hold the B2 forward floor so
            # the policy actually steps while still far (it stalls below ~0.4 m/s).
            bearing = math.atan2(wy - self.odometry.y, wx - self.odometry.x)
            vx, vy, wz = self._drive_to(wx, wy, bearing)
            yaw_err = abs(_wrap_to_pi(bearing - self.odometry.yaw))
            if dist > self.SEARCH_WP_ARRIVE and yaw_err < 0.7:
                vx = max(vx, self.FORWARD_FLOOR)
            vx = _clamp(vx, -0.65, 0.85)
            command = torch.tensor((vx, vy, wz), device=self.device)
            return command, arm_action

        # scan mode: rotate in place to look all around the waypoint.
        self.search_scan_steps += 1
        if self.search_scan_steps >= self.SEARCH_SCAN_STEPS:
            self.search_waypoint_idx = (self.search_waypoint_idx + 1) % len(self.SEARCH_WAYPOINTS)
            self.search_wp_steps = 0
            self.search_mode = "travel"
            self._record_debug(
                "search_wp_advance",
                next_waypoint_idx=self.search_waypoint_idx,
                next_waypoint=list(self.SEARCH_WAYPOINTS[self.search_waypoint_idx]),
            )
        command = torch.tensor((0.0, 0.0, self.SEARCH_YAW), device=self.device)
        return command, arm_action

    # ---------------------------------------------------------- APPROACH
    def _step_approach(self, parsed: ParsedObservation) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.active is not None
        # Refresh active coords from detections if available.
        if not self._refresh_active(self._cached_detections):
            self.steps_since_seen += 1
            if self.steps_since_seen > self.BLIND_WALK_STEPS:
                self._charge_attempt(self.active[0])
                self._record_debug("approach_blind_timeout",
                                   track_id=self.active[0],
                                   steps_since_seen=self.steps_since_seen)
                self.active = None
                self._transition(State.SEARCH, "blind_walk_timeout")
                return self._step_search(parsed)

        tid, wx, wy = self.active
        dx = wx - self.odometry.x
        dy = wy - self.odometry.y
        ground_dist = math.hypot(dx, dy)

        if ground_dist <= self.ARRIVE_DIST:
            self._charge_attempt(tid)
            self.creep_step = 0
            self._transition(State.CREEP, "arrived",
                             ground_distance=ground_dist)
            return self._step_creep(parsed)

        vx, vy, wz = self._approach_velocity(self.active)
        command = torch.tensor((vx, vy, wz), device=self.device)
        arm_action = self.arm.joint_action(
            parsed.arm_q, self.arm.carry_q, self.arm.open_gripper
        )
        return command, arm_action

    # ------------------------------------------------------------- CREEP
    def _step_creep(self, parsed: ParsedObservation) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.active is not None
        self.creep_step += 1
        if self.creep_step > self.CREEP_STEPS:
            self.sweep_step = 0
            self.settle_step = 0
            self.sweep_subphase = "sweep"
            self._transition(State.SWEEP, "creep_done")
            return self._step_sweep(parsed)

        vx, vy, wz = self._creep_velocity(self.active)
        command = torch.tensor((vx, vy, wz), device=self.device)
        arm_action = self.arm.joint_action(
            parsed.arm_q, self.arm.carry_q, self.arm.open_gripper
        )
        return command, arm_action

    # ------------------------------------------------------------- SWEEP
    def _step_sweep(self, parsed: ParsedObservation) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.active is not None
        tid, wx, wy = self.active
        target_world = (wx, wy)

        # Base velocity is zero during the sweep — the dog stands still and the
        # arm/oscillation does the work.
        command = torch.zeros(3, device=self.device)

        if self.sweep_subphase == "sweep":
            self.sweep_step += 1
            fwd, lat, tz = self._sweep_target(target_world, self.sweep_step)
            target_xyz = torch.tensor((fwd, lat, tz), device=self.device,
                                      dtype=torch.float32)
            arm_action = self.arm.pose_action(
                parsed.arm_q,
                target_xyz,
                self.arm.open_gripper,  # gripper STAYS OPEN — proximity scoring
                align_down=True,
                max_joint_delta=0.10,
                action_max_delta=0.5,
                yaw_limit=1.2,
            )

            # Periodic telemetry: EE pos, target pos, 3-D distance, tilt.
            if self.sweep_step % self.SWEEP_TELEMETRY_INTERVAL == 0:
                ee_xyz = self.arm.forward(parsed.arm_q)[:3, 3].detach().cpu().numpy()
                grav = parsed.projected_gravity.detach().cpu().numpy()
                target_np = np.array((fwd, lat, tz), dtype=np.float32)
                dist3d = float(np.linalg.norm(ee_xyz - target_np))
                tilt = float(math.hypot(float(grav[0]), float(grav[1])))
                self._record_debug(
                    "sweep_telemetry",
                    track_id=tid,
                    ee_xyz=ee_xyz.tolist(),
                    target_xyz=[float(fwd), float(lat), float(tz)],
                    ee_target_distance=dist3d,
                    tilt=tilt,
                    crouch_blend=float(self.leg_posture_blend),
                )

            # First attempt sweeps fully; a 2nd+ attempt (this object already
            # missed a full sweep) gets the shorter cap so we don't keep poking
            # an unscoreable flat object.
            sweep_cap = (
                self.SWEEP_MAX_STEPS
                if self.attempts.get(tid, 0) <= 1
                else self.SWEEP_MAX_STEPS_RETRY
            )
            if self.sweep_step > sweep_cap:
                self.sweep_subphase = "settle"
                self.settle_step = 0
            return command, arm_action

        # settle: hold still in the crouch briefly, then RECOVER.
        self.settle_step += 1
        arm_action = self.arm.joint_action(
            parsed.arm_q, self.arm.carry_q, self.arm.open_gripper, max_delta=0.05
        )
        if self.settle_step >= self.SWEEP_SETTLE_STEPS:
            self._begin_recover("sweep_done")
        return command, arm_action

    # -------------------------------------------------- RECOVER (stand_up)
    def _begin_recover(self, reason: str) -> None:
        self.recover_step = 0
        self._transition(State.RECOVER, reason)

    def _step_recover(self, parsed: ParsedObservation) -> tuple[torch.Tensor, torch.Tensor]:
        self.recover_step += 1
        # The crouch blend ramps down via _apply_leg_posture (target=0 in
        # RECOVER). Stand still while the arm tucks back to carry.
        command = torch.zeros(3, device=self.device)
        arm_action = self.arm.joint_action(
            parsed.arm_q, self.arm.carry_q, self.arm.open_gripper
        )
        # Finished standing once the crouch blend is fully off AND we've held a
        # minimum number of steps (so the policy actually re-takes the stand
        # action before we issue a new translation command).
        done_standing = (
            self.leg_posture_blend <= 0.02
            and self.recover_step >= self.RECOVER_STEPS
        )
        if done_standing:
            # Pick next target; G1 plans straight back into APPROACH if memory
            # has one, otherwise SEARCH.
            self.active = None
            target = self._select_target(self._cached_detections)
            if target is not None:
                self.active = target
                self.steps_since_seen = 0
                self._transition(State.APPROACH, "recover_done_next_target",
                                 target=list(target))
            else:
                self._transition(State.SEARCH, "recover_done_no_target")
        return command, arm_action


class AlgSolution:
    """Task B B2-Piper submission entrypoint. Expects policy.pt next to solution.py."""

    def __init__(self):
        policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.pt")
        self.controller = TaskBController(policy_path=policy_path)

    def get_action_spec(self) -> dict[str, dict[str, Any]]:
        return {}

    def reset(self, **kwargs: Any) -> None:
        self.controller.reset(**kwargs)

    def predicts(self, obs: dict[str, Any], current_score: float) -> dict[str, Any]:
        return self.controller.step(obs, current_score)

