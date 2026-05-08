"""Unitree G1 (29 body DoF) AMP training task registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="ATEC-Isaac-AMP-Unitree-G1-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1AMPRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:UnitreeG1AMPRoughRunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-AMP-Unitree-G1-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeG1AMPFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:UnitreeG1AMPFlatRunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-AMP-Unitree-G1-TaskA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_a_env_cfg:UnitreeG1AMPTaskAEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:UnitreeG1AMPRoughRunnerCfg",
    },
)
