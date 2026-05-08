# Reference: https://github.com/fan-ziqi/robot_lab

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def amp_obs_g1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the 73-dim AMP observation for G1: `[jp_rel(29), jv(29), ee_pos_b(15)]`.

    `asset_cfg.joint_ids` selects the 29 body DoF (must match motion-loader joint order).
    `ee_body_cfg.body_ids` selects the 5 end-effector links (left_ankle, right_ankle, waist, left_wrist, right_wrist).
    End-effector positions are expressed in the robot's base frame (translation- and yaw-invariant).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    jp = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    jv = asset.data.joint_vel[:, joint_ids]

    body_ids = ee_body_cfg.body_ids
    ee_pos_w = asset.data.body_pos_w[:, body_ids, :]            # (N, K, 3)
    root_pos_w = asset.data.root_pos_w                           # (N, 3)
    root_quat_w = asset.data.root_quat_w                         # (N, 4) -- (w, x, y, z)
    rel_pos_w = ee_pos_w - root_pos_w.unsqueeze(1)               # (N, K, 3)

    num_envs, num_ee, _ = rel_pos_w.shape
    quat_flat = root_quat_w.unsqueeze(1).expand(-1, num_ee, -1).reshape(-1, 4)
    pos_flat = rel_pos_w.reshape(-1, 3)
    rel_pos_b = math_utils.quat_apply_inverse(quat_flat, pos_flat).reshape(num_envs, num_ee * 3)
    return torch.cat([jp, jv, rel_pos_b], dim=-1)


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor
