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
        0.25,
        0.5,
        0.5,
        0.25,
        0.5,
        0.5,
        0.25,
        0.5,
        0.5,
        0.25,
        0.5,
        0.5,
    )
    ENV_TO_TRAIN = (
        4.0,
        2.0,
        2.0,
        4.0,
        2.0,
        2.0,
        4.0,
        2.0,
        2.0,
        4.0,
        2.0,
        2.0,
    )

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

    def act(self, parsed: ParsedObservation, command: torch.Tensor) -> torch.Tensor:
        command = torch.as_tensor(command, dtype=torch.float32, device=self.device)
        command = torch.stack(
            (
                command[0].clamp(-0.65, 0.85),
                command[1].clamp(-0.45, 0.45),
                command[2].clamp(-1.05, 1.05),
            )
        )
        policy_obs = torch.cat(
            (
                parsed.base_ang_vel.to(self.device) * 0.25,
                parsed.projected_gravity.to(self.device),
                command,
                parsed.joint_pos_rel[:LEG_DIM].to(self.device),
                parsed.joint_vel_rel[:LEG_DIM].to(self.device) * 0.05,
                parsed.last_action[:LEG_DIM].to(self.device) * self.env_to_train,
            )
        ).unsqueeze(0)

        with torch.inference_mode():
            action = self.policy(policy_obs)
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        action = action.reshape(-1, LEG_DIM)[0].to(dtype=torch.float32)
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
            & (height > 0.018)
            & (height < 0.28)
        )

        candidates: list[Detection] = []
        for component in _connected_components(mask, min_area=4):
            if component.shape[0] > 900:
                continue
            row_span = int(np.ptp(component[:, 0])) + 1
            col_span = int(np.ptp(component[:, 1])) + 1
            if row_span > 70 or col_span > 70:
                continue
            detection = self._component_detection(component, points, confidence=0.0)
            detection_height = float(detection.position_base @ up_axis - floor_coordinate)
            if not (0.015 < detection_height < 0.24):
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
    SEARCH_OBJECT = auto()
    APPROACH_OBJECT = auto()
    ALIGN_OBJECT = auto()
    PREGRASP = auto()
    SERVO_GRASP = auto()
    CLOSE_GRIPPER = auto()
    RECOVER = auto()
    DONE = auto()


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class TaskBController:
    TOUCH_SCORE_TARGET = 18.0
    LOW_STANCE_MAX_BLEND = 1.0
    SCORED_OBJECT_REJECT_RADIUS = 0.55
    SCORED_OBJECT_RECENT_REJECT_RADIUS = 1.45
    SCORED_OBJECT_RECENT_STEPS = 900
    SCORED_OBJECT_MERGE_RADIUS = 0.28
    SCORED_OBJECT_MAX_TRACKS = 72
    SEARCH_WAYPOINTS = (
        (-10.0, -10.0),
        (-10.0, -7.0),
        (-7.0, -7.0),
        (-7.0, -10.0),
        (-7.0, -13.0),
        (-10.0, -13.0),
        (-13.0, -13.0),
        (-13.0, -10.0),
        (-13.0, -7.0),
        (-10.0, -7.0),
        (-5.8, -7.0),
        (-5.8, -10.0),
        (-5.8, -13.0),
        (-14.2, -13.0),
        (-14.2, -10.0),
        (-14.2, -7.0),
        (-10.0, -5.8),
        (-7.0, -5.8),
        (-13.0, -5.8),
        (-10.0, -14.2),
        (-7.0, -14.2),
        (-13.0, -14.2),
    )
    SEARCH_WAYPOINT_STOP_DISTANCE = 0.85
    SEARCH_WAYPOINT_TIMEOUT_STEPS = 420
    SEARCH_SCAN_STEPS = 85
    SEARCH_STALL_STEPS = 180
    TOUCH_SWEEP_START_STEPS = 55
    TOUCH_SWEEP_SEGMENT_STEPS = 24
    TOUCH_JOINT_SWEEP_START_STEPS = 1000000
    TOUCH_JOINT_SWEEP_SEGMENT_STEPS = 28
    TOUCH_BASE_SWEEP_START_STEPS = 80
    TOUCH_BASE_SWEEP_SEGMENT_STEPS = 28
    TOUCH_BASE_FORWARD_BIAS = 0.04
    TOUCH_SERVO_TIMEOUT_STEPS = 165
    OBJECT_COLOR_PREGRASP_MAX_AGE_STEPS = 180
    OBJECT_COLOR_GRASP_MAX_AGE_STEPS = 220
    TOUCH_SWEEP_OFFSETS = (
        (0.00, 0.00, 0.00),
        (0.03, 0.00, -0.02),
        (0.06, 0.00, -0.035),
        (0.04, 0.045, -0.03),
        (0.04, -0.045, -0.03),
        (0.06, 0.035, -0.040),
        (0.06, -0.035, -0.040),
        (0.02, 0.00, -0.055),
    )
    TOUCH_JOINT_ACTION_OFFSETS = (
        (0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
        (0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
    )
    TOUCH_BASE_SWEEP_COMMANDS = (
        (0.03, 0.00, 0.00),
        (0.03, 0.035, 0.07),
        (0.03, -0.035, -0.07),
        (0.06, 0.02, 0.04),
        (0.06, -0.02, -0.04),
    )
    RETRY_TARGET_OFFSETS = (
        (0.00, 0.00, 0.00),
        (0.07, 0.070, -0.030),
        (0.07, -0.070, -0.030),
        (0.11, 0.000, -0.050),
        (-0.02, 0.000, -0.030),
    )

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
        self.reset()

    def reset(self, **_: Any) -> None:
        self.debug_events: list[dict[str, Any]] = []
        self.metrics = {
            "transitions": 0,
            "head_detections": 0,
            "ee_detections": 0,
            "score_changes": 0,
            "pick_attempts": 0,
            "pick_retries": 0,
            "touch_sweeps": 0,
            "touch_joint_sweeps": 0,
            "touch_base_sweeps": 0,
            "recoveries": 0,
        }
        self._last_head_debug_step = -10000
        self._last_ee_debug_step = -10000
        self.state = State.ARM_INIT
        self.state_steps = 0
        self.total_steps = 0
        self.last_score = 0.0
        self.score_at_grasp = 0.0
        self.pick_start_score = 0.0
        self.grasp_confirmed = False
        self.grasp_retries = 0
        self.completed_cycles = 0
        self.head_detection: Detection | None = None
        self.head_detection_step = -10000
        self.object_track_world: np.ndarray | None = None
        self.object_track_step = -10000
        self.object_track_confidence = 0.0
        self.object_color_step = -10000
        self.scored_object_tracks: list[np.ndarray] = []
        self.scored_object_track_steps: list[int] = []
        self.target_locked = False
        self.leg_posture_blend = 0.0
        self.search_waypoint_idx = 0
        self.search_waypoint_steps = 0
        self.search_at_waypoint_steps = 0
        self.search_best_distance = float("inf")
        self.search_stalled_steps = 0
        self.ee_detection: Detection | None = None
        self.pregrasp_target: torch.Tensor | None = None
        self.servo_target: torch.Tensor | None = None
        self.last_ee_world: np.ndarray | None = None
        self.touch_sweep_index = -1
        self.touch_joint_sweep_index = -1
        self.touch_base_sweep_index = -1
        self.odometry.reset()
        self._record_debug("reset")

    def _transition(self, state: State, reason: str | None = None) -> None:
        previous_state = self.state
        self.metrics["transitions"] += 1
        if state == State.PREGRASP:
            self.metrics["pick_attempts"] += 1
        elif state == State.RECOVER:
            self.metrics["recoveries"] += 1
        if state == State.SERVO_GRASP:
            self.touch_sweep_index = -1
            self.touch_joint_sweep_index = -1
            self.touch_base_sweep_index = -1
        self._record_debug(
            "transition",
            from_state=previous_state.name,
            to_state=state.name,
            duration_steps=self.state_steps,
            reason=reason,
        )
        self.state = state
        self.state_steps = 0

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

    @staticmethod
    def _detection_summary(detection: Detection | None) -> dict[str, Any] | None:
        if detection is None:
            return None
        return {
            "x": detection.x,
            "y": detection.y,
            "z": detection.z,
            "pixel": [float(detection.pixel[0]), float(detection.pixel[1])],
            "area": int(detection.area),
            "confidence": float(detection.confidence),
        }

    def _record_detection(
        self,
        name: str,
        detection: Detection,
        metric_name: str,
        last_step_attr: str,
    ) -> None:
        self.metrics[metric_name] += 1
        last_step = getattr(self, last_step_attr)
        if self.total_steps - last_step < self.debug_interval:
            return
        setattr(self, last_step_attr, self.total_steps)
        self._record_debug(f"{name}_detected", **self._detection_summary(detection))

    @staticmethod
    def _head_detection_is_good_for_acquisition(detection: Detection) -> bool:
        u, v = detection.pixel
        return (
            detection.confidence >= 0.18
            and 0.55 <= detection.x <= 3.60
            and abs(detection.y) <= 1.40
            and -0.62 <= detection.z <= -0.18
            and 8.0 <= u <= 632.0
            and 5.0 <= v <= 475.0
        )

    @staticmethod
    def _head_detection_is_good_for_search_lock(detection: Detection) -> bool:
        u, v = detection.pixel
        if not TaskBController._head_detection_is_good_for_acquisition(detection):
            return False
        if abs(detection.y) > 0.72:
            return False
        return 32.0 <= u <= 608.0 and 8.0 <= v <= 475.0

    @staticmethod
    def _head_detection_is_good_for_tracking(detection: Detection) -> bool:
        u, v = detection.pixel
        return (
            detection.confidence >= 0.10
            and 0.35 <= detection.x <= 3.8
            and abs(detection.y) <= 1.6
            and -0.62 <= detection.z <= -0.10
            and 8.0 <= u <= 632.0
            and 8.0 <= v <= 472.0
        )

    @staticmethod
    def _ee_detection_is_good_for_servo(detection: Detection) -> bool:
        u, v = detection.pixel
        return (
            detection.confidence >= 0.25
            and 0.20 <= detection.x <= 0.82
            and abs(detection.y) <= 0.20
            and -0.36 <= detection.z <= -0.08
            and 55.0 <= u <= 595.0
            and 45.0 <= v <= 465.0
        )

    def _ee_detection_matches_current_target(self, detection: Detection) -> bool:
        target = self._object_target_detection(max_track_age_steps=320)
        if target is None:
            target_position = self.servo_target if self.servo_target is not None else self.pregrasp_target
            if target_position is None:
                return False
            dx = abs(float(detection.x) - float(target_position[0]))
            dy = abs(float(detection.y) - float(target_position[1]))
            dz = abs(float(detection.z) - float(target_position[2]))
        else:
            dx = abs(float(detection.x) - float(target.x))
            dy = abs(float(detection.y) - float(target.y))
            dz = abs(float(detection.z) - float(target.z))
        return dx < 0.28 and dy < 0.14 and dz < 0.26

    def get_debug_summary(self) -> dict[str, Any]:
        return {
            "state": self.state.name,
            "state_steps": self.state_steps,
            "total_steps": self.total_steps,
            "last_score": self.last_score,
            "grasp_confirmed": self.grasp_confirmed,
            "grasp_retries": self.grasp_retries,
            "completed_cycles": self.completed_cycles,
            "leg_posture_blend": self.leg_posture_blend,
            "scored_object_tracks": len(self.scored_object_tracks),
            "odometry": {
                "x": self.odometry.x,
                "y": self.odometry.y,
                "yaw": self.odometry.yaw,
            },
            "detections": {
                "head": self._detection_summary(self.head_detection),
                "tracked_head": self._detection_summary(
                    self._tracked_head_detection()
                ),
                "ee": self._detection_summary(self.ee_detection),
            },
            "object_color_age_steps": self.total_steps - self.object_color_step,
            "metrics": dict(self.metrics),
            "last_events": list(self.debug_events[-10:]),
        }

    def _refresh_vision(self, obs: dict[str, Any], parsed: ParsedObservation) -> None:
        if self.total_steps % 5 != 0:
            return
        images = obs.get("image", {})
        head_depth = images.get("head_depth")
        self.head_detection = self.geometry.detect_head_object(
            head_depth, images.get("head_rgb"), parsed.projected_gravity
        )
        if self.head_detection is not None:
            self.head_detection_step = self.total_steps
            grasp_tracking_state = self.state in (
                State.PREGRASP,
                State.SERVO_GRASP,
                State.CLOSE_GRIPPER,
            )
            if not self._head_detection_is_good_for_tracking(self.head_detection):
                self._record_debug(
                    "head_detection_quality_rejected",
                    **self._detection_summary(self.head_detection),
                )
                self.head_detection = None
            elif (
                not self.target_locked
                and not self._head_detection_is_good_for_acquisition(self.head_detection)
            ):
                self._record_debug(
                    "head_detection_acquisition_rejected",
                    **self._detection_summary(self.head_detection),
                )
                self.head_detection = None
            elif self._detection_near_scored_object(self.head_detection):
                self._record_debug(
                    "head_detection_scored_rejected",
                    **self._detection_summary(self.head_detection),
                )
                self.head_detection = None
            elif grasp_tracking_state and not (
                self.target_locked and self._detection_matches_track(self.head_detection)
            ):
                self._record_debug(
                    "head_detection_grasp_rejected",
                    **self._detection_summary(self.head_detection),
                )
                self.head_detection = None
            elif self._detection_matches_track(self.head_detection):
                self._update_object_track(self.head_detection)
                self._record_detection(
                    "head",
                    self.head_detection,
                    "head_detections",
                    "_last_head_debug_step",
                )
            else:
                self._record_debug(
                    "head_detection_rejected",
                    **self._detection_summary(self.head_detection),
                )
                self.head_detection = None

        if self.state in (State.PREGRASP, State.SERVO_GRASP, State.CLOSE_GRIPPER):
            camera_transform = self.arm.kinematics.camera_transform(
                self.arm.to_kin_q(parsed.arm_q.to(self.device))
            )
            self.ee_detection = self.geometry.detect_ee_object(
                images.get("ee_depth"), images.get("ee_rgb"), camera_transform
            )
            if self.ee_detection is not None:
                if (
                    self._ee_detection_is_good_for_servo(self.ee_detection)
                    and self._ee_detection_matches_current_target(self.ee_detection)
                ):
                    self.object_color_step = self.total_steps
                    self._record_detection(
                        "ee",
                        self.ee_detection,
                        "ee_detections",
                        "_last_ee_debug_step",
                    )
                else:
                    self._record_debug(
                        "ee_detection_quality_rejected",
                        **self._detection_summary(self.ee_detection),
                    )
                    self.ee_detection = None

    def _base_to_world(self, position_base: np.ndarray) -> np.ndarray:
        c = math.cos(self.odometry.yaw)
        s = math.sin(self.odometry.yaw)
        return np.array(
            (
                self.odometry.x + c * float(position_base[0]) - s * float(position_base[1]),
                self.odometry.y + s * float(position_base[0]) + c * float(position_base[1]),
                float(position_base[2]),
            ),
            dtype=np.float32,
        )

    def _update_object_track(self, detection: Detection) -> None:
        if detection.confidence < 0.10:
            return
        self.object_track_world = self._base_to_world(detection.position_base)
        self.object_track_step = self.total_steps
        self.object_track_confidence = float(detection.confidence)
        self.object_color_step = self.total_steps

    def _target_has_recent_color(self, max_age_steps: int) -> bool:
        age_steps = self.total_steps - self.object_color_step
        return 0 <= age_steps <= max_age_steps

    def _record_stale_color_target(self, reason: str, max_age_steps: int) -> None:
        self._record_debug(
            "target_color_stale",
            reason=reason,
            color_age_steps=self.total_steps - self.object_color_step,
            max_age_steps=max_age_steps,
        )

    def _scored_reject_radius(self, index: int) -> float:
        if index < len(self.scored_object_track_steps):
            age_steps = self.total_steps - self.scored_object_track_steps[index]
            if 0 <= age_steps <= self.SCORED_OBJECT_RECENT_STEPS:
                return self.SCORED_OBJECT_RECENT_REJECT_RADIUS
        return self.SCORED_OBJECT_REJECT_RADIUS

    def _append_scored_object_world(self, world: np.ndarray, reason: str) -> None:
        world = np.asarray(world, dtype=np.float32).reshape(-1)[:3]
        if world.size < 3 or not np.isfinite(world).all():
            return

        for index, scored_world in enumerate(self.scored_object_tracks):
            if np.linalg.norm(world[:2] - scored_world[:2]) < self.SCORED_OBJECT_MERGE_RADIUS:
                self.scored_object_tracks[index] = world.astype(np.float32)
                if index < len(self.scored_object_track_steps):
                    self.scored_object_track_steps[index] = self.total_steps
                self._record_debug(
                    "scored_object_track_updated",
                    reason=reason,
                    index=index,
                    world=world,
                )
                return

        self.scored_object_tracks.append(world.astype(np.float32))
        self.scored_object_track_steps.append(self.total_steps)
        if len(self.scored_object_tracks) > self.SCORED_OBJECT_MAX_TRACKS:
            extra = len(self.scored_object_tracks) - self.SCORED_OBJECT_MAX_TRACKS
            del self.scored_object_tracks[:extra]
            del self.scored_object_track_steps[:extra]
        self._record_debug(
            "scored_object_track_added",
            reason=reason,
            count=len(self.scored_object_tracks),
            world=world,
        )

    def _detection_near_scored_object(self, detection: Detection) -> bool:
        if not self.scored_object_tracks:
            return False
        world = self._base_to_world(detection.position_base)
        for index, scored_world in enumerate(self.scored_object_tracks):
            distance = np.linalg.norm(world[:2] - scored_world[:2])
            if distance < self._scored_reject_radius(index):
                if distance < self.SCORED_OBJECT_MERGE_RADIUS:
                    self._append_scored_object_world(world, "rejected_detection")
                else:
                    self._record_debug(
                        "scored_object_recent_rejected",
                        index=index,
                        distance=float(distance),
                        world=world,
                    )
                return True
        return False

    def _remember_scored_object(self) -> None:
        candidates: list[tuple[str, np.ndarray]] = []
        if self.ee_detection is not None:
            candidates.append(
                ("ee_detection", self._base_to_world(self.ee_detection.position_base))
            )
        if self.last_ee_world is not None:
            candidates.append(("last_ee", self.last_ee_world.copy()))
        if self.object_track_world is not None:
            track_age_steps = self.total_steps - self.object_track_step
            if 0 <= track_age_steps <= 320:
                candidates.append(("object_track", self.object_track_world.copy()))
        if self.head_detection is not None:
            candidates.append(
                ("head_detection", self._base_to_world(self.head_detection.position_base))
            )
        if self.servo_target is not None:
            candidates.append(
                ("servo_target", self._base_to_world(self.servo_target.detach().cpu().numpy()))
            )
        if self.pregrasp_target is not None:
            candidates.append(
                ("pregrasp_target", self._base_to_world(self.pregrasp_target.detach().cpu().numpy()))
            )

        for reason, world in candidates:
            self._append_scored_object_world(world, reason)
            return

    def _detection_matches_track(self, detection: Detection) -> bool:
        if not self.target_locked or self.object_track_world is None:
            return True
        tracked = self._tracked_head_detection(max_age_steps=220)
        if tracked is None:
            return False
        longitudinal_error = abs(detection.x - tracked.x)
        lateral_error = abs(detection.y - tracked.y)
        z_error = abs(detection.z - tracked.z)
        longitudinal_limit = 0.45 + 0.18 * max(0.0, tracked.x)
        lateral_limit = 0.42 + 0.16 * max(0.0, tracked.x)
        return (
            longitudinal_error < longitudinal_limit
            and lateral_error < lateral_limit
            and z_error < 0.24
        )

    def _tracked_head_detection(self, max_age_steps: int = 140) -> Detection | None:
        if self.object_track_world is None:
            return None
        age_steps = self.total_steps - self.object_track_step
        if age_steps < 0 or age_steps > max_age_steps:
            return None

        x, y = self.odometry.body_error(
            float(self.object_track_world[0]), float(self.object_track_world[1])
        )
        if x < 0.18 or x > 4.0 or abs(y) > 2.0:
            return None

        confidence_decay = max(0.20, 1.0 - 0.80 * age_steps / max_age_steps)
        return Detection(
            position_base=np.array(
                (x, y, float(self.object_track_world[2])), dtype=np.float32
            ),
            pixel=(-1.0, -1.0),
            area=0,
            confidence=self.object_track_confidence * confidence_decay,
        )

    def _object_target_detection(self, max_track_age_steps: int = 140) -> Detection | None:
        tracked = self._tracked_head_detection(max_track_age_steps)
        if tracked is not None and self._detection_near_scored_object(tracked):
            tracked = None
        if self.target_locked:
            if (
                self.head_detection is not None
                and self.total_steps - self.head_detection_step <= 15
                and self._detection_matches_track(self.head_detection)
                and not self._detection_near_scored_object(self.head_detection)
            ):
                return self.head_detection
            return tracked
        if (
            self.head_detection is not None
            and self.total_steps - self.head_detection_step <= 15
            and not self._detection_near_scored_object(self.head_detection)
        ):
            return self.head_detection
        return tracked

    def _navigation_command(
        self, world_target: tuple[float, float], stop_distance: float
    ) -> tuple[torch.Tensor, float]:
        forward, lateral = self.odometry.body_error(*world_target)
        distance = math.hypot(forward, lateral)
        heading = math.atan2(lateral, forward)
        if distance <= stop_distance:
            command = torch.tensor((0.0, 0.0, 1.4 * heading), device=self.device)
        else:
            speed = min(0.75, 0.30 + 0.32 * distance)
            command = torch.tensor(
                (
                    speed * max(0.08, math.cos(heading)),
                    float(np.clip(0.55 * lateral, -0.38, 0.38)),
                    float(np.clip(1.7 * heading, -1.0, 1.0)),
                ),
                device=self.device,
            )
        return command, distance

    def _current_search_waypoint(self) -> tuple[float, float]:
        return self.SEARCH_WAYPOINTS[
            self.search_waypoint_idx % len(self.SEARCH_WAYPOINTS)
        ]

    def _advance_search_waypoint(self, reason: str) -> None:
        previous_idx = self.search_waypoint_idx
        self.search_waypoint_idx = (self.search_waypoint_idx + 1) % len(
            self.SEARCH_WAYPOINTS
        )
        self.search_waypoint_steps = 0
        self.search_at_waypoint_steps = 0
        self.search_best_distance = float("inf")
        self.search_stalled_steps = 0
        self._record_debug(
            "search_waypoint_advanced",
            from_idx=previous_idx,
            to_idx=self.search_waypoint_idx,
            target=self._current_search_waypoint(),
            reason=reason,
        )

    def _object_approach_command(self, detection: Detection) -> torch.Tensor:
        heading = math.atan2(detection.y, max(0.05, detection.x))
        forward_error = detection.x - 0.44
        return torch.tensor(
            (
                float(np.clip(1.15 * forward_error, -0.18, 0.58)),
                float(np.clip(0.95 * detection.y, -0.30, 0.30)),
                float(np.clip(1.6 * heading, -0.90, 0.90)),
            ),
            device=self.device,
        )

    def _search_command(self) -> torch.Tensor:
        self.search_waypoint_steps += 1
        command, distance = self._navigation_command(
            self._current_search_waypoint(), self.SEARCH_WAYPOINT_STOP_DISTANCE
        )
        if distance < self.search_best_distance - 0.05:
            self.search_best_distance = distance
            self.search_stalled_steps = 0
        else:
            self.search_stalled_steps += 1

        if distance <= self.SEARCH_WAYPOINT_STOP_DISTANCE:
            self.search_at_waypoint_steps += 1
            spin_direction = 1.0 if self.search_waypoint_idx % 2 == 0 else -1.0
            command = torch.tensor(
                (0.0, 0.0, 0.55 * spin_direction), device=self.device
            )
            if self.search_at_waypoint_steps >= self.SEARCH_SCAN_STEPS:
                self._advance_search_waypoint("waypoint_scan_complete")
            return command

        self.search_at_waypoint_steps = 0
        if self.search_waypoint_steps >= self.SEARCH_WAYPOINT_TIMEOUT_STEPS:
            self._advance_search_waypoint("waypoint_timeout")
        elif (
            self.search_waypoint_steps > 140
            and self.search_stalled_steps >= self.SEARCH_STALL_STEPS
        ):
            self._advance_search_waypoint("waypoint_stalled")
        return command

    def _clear_current_target(self) -> None:
        self.head_detection = None
        self.ee_detection = None
        self.object_track_world = None
        self.object_color_step = -10000
        self.target_locked = False
        self.pregrasp_target = None
        self.servo_target = None
        self.touch_sweep_index = -1
        self.touch_joint_sweep_index = -1
        self.touch_base_sweep_index = -1

    def _arm_target_position(self, detection: Detection) -> torch.Tensor:
        return torch.tensor(
            (
                float(np.clip(detection.x, 0.30, 0.55)),
                float(np.clip(detection.y, -0.22, 0.22)),
                -0.24,
            ),
            device=self.device,
        )

    def _touch_biased_target(self, target: torch.Tensor) -> torch.Tensor:
        biased = target.clone()
        biased[0] = biased[0] + 0.035
        biased[2] = biased[2] - 0.045
        biased[0] = biased[0].clamp(0.30, 0.62)
        biased[1] = biased[1].clamp(-0.24, 0.24)
        biased[2] = biased[2].clamp(-0.34, -0.18)
        return biased

    def _touch_target_position(self, detection: Detection) -> torch.Tensor:
        target = torch.tensor(
            detection.position_base, device=self.device, dtype=torch.float32
        )
        target[0] = target[0].clamp(0.28, 0.64)
        target[1] = target[1].clamp(-0.23, 0.23)
        target[2] = (target[2] + 0.02).clamp(-0.35, -0.18)
        return target

    def _retry_adjusted_target(self, target: torch.Tensor) -> torch.Tensor:
        offset_idx = min(self.grasp_retries, len(self.RETRY_TARGET_OFFSETS) - 1)
        offset = torch.tensor(
            self.RETRY_TARGET_OFFSETS[offset_idx], device=self.device, dtype=target.dtype
        )
        adjusted = target + offset
        adjusted[0] = adjusted[0].clamp(0.34, 0.66)
        adjusted[1] = adjusted[1].clamp(-0.24, 0.24)
        adjusted[2] = adjusted[2].clamp(-0.26, -0.13)
        self._record_debug(
            "retry_target_adjusted",
            retry=self.grasp_retries,
            offset_index=offset_idx,
            offset=offset,
            target=adjusted,
        )
        return adjusted

    def _touch_sweep_target(self, target: torch.Tensor) -> torch.Tensor:
        if self.state_steps < self.TOUCH_SWEEP_START_STEPS:
            return target

        sweep_idx = min(
            (self.state_steps - self.TOUCH_SWEEP_START_STEPS)
            // self.TOUCH_SWEEP_SEGMENT_STEPS,
            len(self.TOUCH_SWEEP_OFFSETS) - 1,
        )
        offset = torch.tensor(
            self.TOUCH_SWEEP_OFFSETS[sweep_idx], device=self.device, dtype=target.dtype
        )
        adjusted = target + offset
        adjusted[0] = adjusted[0].clamp(0.26, 0.64)
        adjusted[1] = adjusted[1].clamp(-0.26, 0.26)
        adjusted[2] = adjusted[2].clamp(-0.36, -0.16)

        if sweep_idx != self.touch_sweep_index:
            self.touch_sweep_index = int(sweep_idx)
            self.metrics["touch_sweeps"] += 1
            self._record_debug(
                "touch_sweep",
                sweep_index=sweep_idx,
                offset=offset,
                target=adjusted,
            )
        return adjusted

    def _touch_joint_exploration_action(self, arm_action: torch.Tensor) -> torch.Tensor:
        if self.state_steps < self.TOUCH_JOINT_SWEEP_START_STEPS:
            return arm_action

        sweep_idx = (
            (self.state_steps - self.TOUCH_JOINT_SWEEP_START_STEPS)
            // self.TOUCH_JOINT_SWEEP_SEGMENT_STEPS
        ) % len(self.TOUCH_JOINT_ACTION_OFFSETS)
        offset = torch.tensor(
            self.TOUCH_JOINT_ACTION_OFFSETS[sweep_idx],
            device=self.device,
            dtype=arm_action.dtype,
        )
        adjusted = arm_action.clone()
        adjusted[:6] = (adjusted[:6] + offset).clamp(-6.0, 6.0)

        if sweep_idx != self.touch_joint_sweep_index:
            self.touch_joint_sweep_index = int(sweep_idx)
            self.metrics["touch_joint_sweeps"] += 1
            self._record_debug(
                "touch_joint_sweep",
                sweep_index=sweep_idx,
                offset=offset,
                action=adjusted[:6],
            )
        return adjusted

    def _touch_base_exploration_command(self, command: torch.Tensor) -> torch.Tensor:
        command = command.clone()
        command[0] = command[0] + self.TOUCH_BASE_FORWARD_BIAS
        if self.state_steps < self.TOUCH_BASE_SWEEP_START_STEPS:
            command[0] = command[0].clamp(-0.12, 0.28)
            command[1] = command[1].clamp(-0.16, 0.16)
            command[2] = command[2].clamp(-0.35, 0.35)
            return command

        sweep_idx = (
            (self.state_steps - self.TOUCH_BASE_SWEEP_START_STEPS)
            // self.TOUCH_BASE_SWEEP_SEGMENT_STEPS
        ) % len(self.TOUCH_BASE_SWEEP_COMMANDS)
        offset = torch.tensor(
            self.TOUCH_BASE_SWEEP_COMMANDS[sweep_idx],
            device=self.device,
            dtype=command.dtype,
        )
        adjusted = torch.stack(
            (
                (command[0] + offset[0]).clamp(-0.14, 0.34),
                (command[1] + offset[1]).clamp(-0.18, 0.18),
                (command[2] + offset[2]).clamp(-0.40, 0.40),
            )
        )

        if sweep_idx != self.touch_base_sweep_index:
            self.touch_base_sweep_index = int(sweep_idx)
            self.metrics["touch_base_sweeps"] += 1
            self._record_debug(
                "touch_base_sweep",
                sweep_index=sweep_idx,
                offset=offset,
                command=adjusted,
            )
        return adjusted

    def _object_align_command(self, detection: Detection) -> torch.Tensor:
        heading = math.atan2(detection.y, max(0.05, detection.x))
        return torch.tensor(
            (
                float(np.clip(0.85 * (detection.x - 0.46), -0.08, 0.34)),
                float(np.clip(0.55 * detection.y, -0.16, 0.16)),
                float(np.clip(1.35 * heading, -0.45, 0.45)),
            ),
            device=self.device,
        )

    @staticmethod
    def _in_approach_window(detection: Detection) -> bool:
        return 0.62 < detection.x < 1.00 and abs(detection.y) < 0.16

    @staticmethod
    def _in_grasp_window(detection: Detection) -> bool:
        return 0.38 < detection.x < 0.66 and abs(detection.y) < 0.14

    def _leg_posture_target_blend(self) -> float:
        if self.state in (
            State.PREGRASP,
            State.SERVO_GRASP,
            State.CLOSE_GRIPPER,
        ):
            return self.LOW_STANCE_MAX_BLEND
        return 0.0

    def _apply_leg_posture(self, leg_action: torch.Tensor) -> torch.Tensor:
        target_blend = self._leg_posture_target_blend()
        if target_blend > self.leg_posture_blend:
            self.leg_posture_blend = min(
                target_blend, self.leg_posture_blend + 0.025
            )
        else:
            self.leg_posture_blend = max(target_blend, self.leg_posture_blend - 0.015)

        if self.leg_posture_blend <= 0.0:
            return leg_action
        low_stance = self.locomotion.low_stance_action.to(leg_action)
        blend = float(self.leg_posture_blend)
        return ((1.0 - blend) * leg_action + blend * low_stance).clamp(-5.0, 5.0)

    def _confirm_grasp_from_score(self, current_score: float, reason: str) -> None:
        if self.grasp_confirmed:
            return
        self.grasp_confirmed = True
        self.score_at_grasp = float(current_score)
        self._remember_scored_object()
        self._record_debug(
            "grasp_confirmed",
            reason=reason,
            pick_start_score=self.pick_start_score,
            current_score=current_score,
        )

    def _finish_touch_score(self, current_score: float, reason: str) -> None:
        self._confirm_grasp_from_score(current_score, reason)
        self.completed_cycles += 1
        self._advance_search_waypoint(reason)
        self._clear_current_target()
        self.grasp_confirmed = False
        if self.completed_cycles >= 18:
            self._transition(State.DONE, reason)
        else:
            self._transition(State.RECOVER, reason)

    def _retry_pick(self, reason: str) -> None:
        self.grasp_retries += 1
        self.metrics["pick_retries"] += 1
        self.grasp_confirmed = False
        object_target = self._object_target_detection(max_track_age_steps=320)
        if (
            object_target is not None
            and self.grasp_retries < 2
            and self._target_has_recent_color(self.OBJECT_COLOR_GRASP_MAX_AGE_STEPS)
        ):
            self.pregrasp_target = self._retry_adjusted_target(
                self._arm_target_position(object_target)
            )
            self.servo_target = None
            self._transition(State.PREGRASP, reason)
            return
        if object_target is not None and not self._target_has_recent_color(
            self.OBJECT_COLOR_GRASP_MAX_AGE_STEPS
        ):
            self._record_stale_color_target(
                "retry_without_recent_color",
                self.OBJECT_COLOR_GRASP_MAX_AGE_STEPS,
            )

        self._clear_current_target()
        self._transition(State.RECOVER, reason)

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
        ee_position_base = self.arm.forward(parsed.arm_q)[:3, 3].detach().cpu().numpy()
        self.last_ee_world = self._base_to_world(ee_position_base)
        self._refresh_vision(obs, parsed)

        score_delta = current_score - self.last_score
        if abs(score_delta) > 1.0e-6:
            self.metrics["score_changes"] += 1
            self._record_debug(
                "score_changed",
                previous_score=self.last_score,
                current_score=current_score,
                delta=score_delta,
            )
        if (
            score_delta > 1.0e-6
            and self.state
            in (
                State.APPROACH_OBJECT,
                State.ALIGN_OBJECT,
                State.PREGRASP,
                State.SERVO_GRASP,
                State.CLOSE_GRIPPER,
            )
            and current_score > self.pick_start_score + 1.0e-6
        ):
            self._confirm_grasp_from_score(current_score, "score_delta")
            self._finish_touch_score(current_score, "touch_score_collected")

        command = torch.zeros(3, device=self.device)
        gripper = self.arm.open_gripper
        arm_action = self.arm.joint_action(
            parsed.arm_q, self.arm.carry_q, gripper
        )

        if self.state == State.ARM_INIT:
            if (
                torch.linalg.vector_norm(parsed.arm_q - self.arm.carry_q) < 0.12
                and self.state_steps > 20
            ) or self.state_steps > 180:
                self._transition(State.SEARCH_OBJECT, "arm_initialized")

        elif self.state == State.SEARCH_OBJECT:
            command = self._search_command()
            if (
                self.head_detection is not None
                and self._head_detection_is_good_for_search_lock(self.head_detection)
            ):
                self.target_locked = True
                self._update_object_track(self.head_detection)
                self._transition(State.APPROACH_OBJECT, "head_object_detected")

        elif self.state == State.APPROACH_OBJECT:
            object_target = self._object_target_detection()
            if object_target is None:
                self.target_locked = False
                self._transition(State.SEARCH_OBJECT, "lost_object_during_approach")
            else:
                command = self._object_approach_command(object_target)
                if self._in_approach_window(object_target):
                    self._transition(State.ALIGN_OBJECT, "object_in_approach_window")

        elif self.state == State.ALIGN_OBJECT:
            object_target = self._object_target_detection(max_track_age_steps=180)
            if object_target is None:
                if self.state_steps > 40:
                    self.target_locked = False
                    self._transition(State.SEARCH_OBJECT, "lost_object_during_align")
            else:
                command = self._object_align_command(object_target)
                stable = (
                    torch.linalg.vector_norm(parsed.base_lin_vel[:2]) < 0.10
                    and abs(float(parsed.base_ang_vel[2])) < 0.12
                    and self._in_grasp_window(object_target)
                )
                ready_by_window = self._in_grasp_window(object_target) and self.state_steps > 45
                if (stable and self.state_steps > 25) or ready_by_window:
                    if self._target_has_recent_color(
                        self.OBJECT_COLOR_PREGRASP_MAX_AGE_STEPS
                    ):
                        self.pick_start_score = float(current_score)
                        self.grasp_confirmed = False
                        self.grasp_retries = 0
                        self.pregrasp_target = self._arm_target_position(object_target)
                        self._transition(State.PREGRASP, "object_aligned_stably")
                    else:
                        self._record_stale_color_target(
                            "align_without_recent_color",
                            self.OBJECT_COLOR_PREGRASP_MAX_AGE_STEPS,
                        )
                        self._clear_current_target()
                        self._transition(State.SEARCH_OBJECT, "stale_color_before_pregrasp")

        elif self.state == State.PREGRASP:
            if not self._target_has_recent_color(
                self.OBJECT_COLOR_GRASP_MAX_AGE_STEPS
            ):
                command = torch.tensor((-0.18, 0.0, 0.0), device=self.device)
                arm_action = self.arm.joint_action(
                    parsed.arm_q, self.arm.carry_q, self.arm.open_gripper
                )
                self._record_stale_color_target(
                    "pregrasp_without_recent_color",
                    self.OBJECT_COLOR_GRASP_MAX_AGE_STEPS,
                )
                self._clear_current_target()
                self._transition(State.RECOVER, "stale_color_during_pregrasp")
                object_target = None
            else:
                object_target = self._object_target_detection(max_track_age_steps=260)
            if object_target is None:
                if self.state_steps > 8:
                    command = torch.tensor((-0.18, 0.0, 0.0), device=self.device)
                    arm_action = self.arm.joint_action(
                        parsed.arm_q, self.arm.carry_q, self.arm.open_gripper
                    )
                    self._clear_current_target()
                    self._transition(State.RECOVER, "lost_object_during_pregrasp")
            else:
                self.pregrasp_target = (
                    0.88 * self.pregrasp_target
                    + 0.12 * self._arm_target_position(object_target)
                )
                command = self._object_align_command(object_target) * torch.tensor(
                    (0.55, 0.45, 0.55), device=self.device
                )
            if self.state == State.PREGRASP and self.state_steps < 35:
                ready_kin_q = self.arm.to_kin_q(self.arm.pregrasp_ready_q).clone()
                ready_kin_q[0] = float(
                    np.clip(
                        -math.atan2(
                            float(self.pregrasp_target[1]),
                            max(0.08, float(self.pregrasp_target[0]) - 0.2),
                        ),
                        float(self.arm.kinematics.lower[0] + 0.05),
                        float(self.arm.kinematics.upper[0] - 0.05),
                    )
                )
                ready_q = self.arm.to_env_q(ready_kin_q)
                arm_action = self.arm.joint_action(
                    parsed.arm_q, ready_q, self.arm.open_gripper, max_delta=0.16
                )
            elif self.state == State.PREGRASP:
                pose_target = self.pregrasp_target
                if self.state_steps > 65:
                    pose_target = self._touch_biased_target(self.pregrasp_target)
                arm_action = self.arm.pose_action(
                    parsed.arm_q,
                    pose_target,
                    self.arm.open_gripper,
                    align_down=True,
                    max_joint_delta=0.08,
                    action_max_delta=0.12,
                    yaw_limit=0.85,
                )
            if self.state == State.PREGRASP:
                position_error = torch.linalg.vector_norm(
                    self.arm.forward(parsed.arm_q)[:3, 3]
                    - self._touch_biased_target(self.pregrasp_target)
                )
                if (
                    position_error < 0.105 and self.state_steps > 75
                ) or self.state_steps > 120:
                    self.servo_target = self._touch_biased_target(self.pregrasp_target)
                    self._transition(State.SERVO_GRASP, "pregrasp_reached")

        elif self.state == State.SERVO_GRASP:
            if not self._target_has_recent_color(
                self.OBJECT_COLOR_GRASP_MAX_AGE_STEPS
            ):
                command = torch.tensor((-0.25, 0.0, 0.0), device=self.device)
                arm_action = self.arm.joint_action(
                    parsed.arm_q, self.arm.carry_q, self.arm.open_gripper
                )
                self._record_stale_color_target(
                    "servo_without_recent_color",
                    self.OBJECT_COLOR_GRASP_MAX_AGE_STEPS,
                )
                self._clear_current_target()
                self._transition(State.RECOVER, "stale_color_during_servo")
                object_target = None
            else:
                object_target = self._object_target_detection(max_track_age_steps=260)
            if object_target is None and self.state_steps > 45:
                command = torch.tensor((-0.25, 0.0, 0.0), device=self.device)
                arm_action = self.arm.joint_action(
                    parsed.arm_q, self.arm.carry_q, self.arm.open_gripper
                )
                self._clear_current_target()
                self._transition(State.RECOVER, "lost_object_during_servo")
            else:
                if object_target is not None:
                    command = self._object_align_command(object_target) * torch.tensor(
                        (0.40, 0.35, 0.45), device=self.device
                    )
                    vision_target = self._touch_target_position(object_target)
                    self.servo_target = 0.78 * self.servo_target + 0.22 * vision_target
                if self.ee_detection is not None:
                    target = self._touch_target_position(self.ee_detection)
                    self.servo_target = 0.86 * self.servo_target + 0.14 * target
                command = self._touch_base_exploration_command(command)
                touch_target = self._touch_sweep_target(self.servo_target)
                arm_action = self.arm.pose_action(
                    parsed.arm_q,
                    touch_target,
                    self.arm.open_gripper,
                    align_down=True,
                    max_joint_delta=0.08,
                    action_max_delta=0.12,
                    yaw_limit=0.85,
                )
                arm_action = self._touch_joint_exploration_action(arm_action)
            if self.state == State.SERVO_GRASP and self.grasp_confirmed:
                gripper = self.arm.closed_gripper
                arm_action = self.arm.joint_action(
                    parsed.arm_q, parsed.arm_q, gripper
                )
                if self.state_steps > 4:
                    self._transition(State.CLOSE_GRIPPER, "grasp_score_confirmed")
            elif self.state == State.SERVO_GRASP and self.state_steps > self.TOUCH_SERVO_TIMEOUT_STEPS:
                self._retry_pick("grasp_score_timeout")

        elif self.state == State.CLOSE_GRIPPER:
            object_target = self._object_target_detection(max_track_age_steps=260)
            if object_target is not None:
                command = self._object_align_command(object_target) * torch.tensor(
                    (0.25, 0.25, 0.30), device=self.device
                )
            gripper = self.arm.closed_gripper
            hold_target = self.servo_target
            if hold_target is None:
                hold_target = self.arm.forward(parsed.arm_q)[:3, 3].detach()
            hold_target = hold_target.clone()
            hold_target[2] = hold_target[2].clamp(-0.18, -0.08)
            arm_action = self.arm.pose_action(
                parsed.arm_q,
                hold_target,
                gripper,
                align_down=True,
                max_joint_delta=0.07,
                action_max_delta=0.10,
                yaw_limit=0.85,
            )
            if self.grasp_confirmed and self.state_steps > 25:
                self._finish_touch_score(current_score, "touch_score_collected")
            elif not self.grasp_confirmed and self.state_steps > 70:
                self._retry_pick("close_without_grasp_score")

        elif self.state == State.RECOVER:
            command = torch.tensor((-0.38, 0.0, 0.0), device=self.device)
            arm_action = self.arm.joint_action(
                parsed.arm_q, self.arm.carry_q, self.arm.open_gripper
            )
            if self.state_steps > 40:
                self.head_detection = None
                self.object_track_world = None
                self.object_color_step = -10000
                self.target_locked = False
                self._transition(State.SEARCH_OBJECT, "recover_finished")

        elif self.state == State.DONE:
            arm_action = self.arm.joint_action(
                parsed.arm_q, self.arm.carry_q, self.arm.open_gripper
            )

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
