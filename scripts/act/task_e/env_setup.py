"""Environment setup helpers for Task E ACT collection scripts."""

from __future__ import annotations

import time

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv

from atec_rl_lab.tasks.task_e.env_cfg import TaskEEnvPiperCfg

from .config import ACT_DAMPING, ACT_EFFORT_LIMIT, ACT_STIFFNESS, ACT_VEL_LIMIT


def build_task_e_env(pick_objects: list[int], need_camera: bool) -> ManagerBasedRLEnv:
    """Build the single-env Task E Piper environment used for ACT collection."""
    cfg = TaskEEnvPiperCfg()
    cfg.seed = int(time.time_ns() % (2**31))
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 40.0 * len(pick_objects) + 10.0
    cfg.scene.robot.actuators["default"] = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        effort_limit=ACT_EFFORT_LIMIT,
        velocity_limit=ACT_VEL_LIMIT,
        stiffness=ACT_STIFFNESS,
        damping=ACT_DAMPING,
    )
    _ = need_camera  # Camera is already configured by TaskEEnvPiperCfg when cameras are enabled.
    return ManagerBasedRLEnv(cfg)
