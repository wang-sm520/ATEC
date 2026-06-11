from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib.util
import sys

import torch


def _install_isaaclab_stubs():
    isaaclab = ModuleType("isaaclab")
    sys.modules.setdefault("isaaclab", isaaclab)

    assets = ModuleType("isaaclab.assets")
    assets.Articulation = object
    assets.RigidObject = object
    sys.modules["isaaclab.assets"] = assets

    envs = ModuleType("isaaclab.envs")
    envs.mdp = SimpleNamespace(joint_deviation_l1=lambda *args, **kwargs: None)
    sys.modules["isaaclab.envs"] = envs

    managers = ModuleType("isaaclab.managers")

    class ManagerTermBase:
        def __init__(self, cfg, env):
            pass

    class SceneEntityCfg:
        def __init__(self, name, body_names=None, body_ids=None, joint_names=None, joint_ids=None):
            self.name = name
            self.body_names = body_names
            self.body_ids = body_ids or []
            self.joint_names = joint_names
            self.joint_ids = joint_ids or []

    class RewardTermCfg:
        pass

    managers.ManagerTermBase = ManagerTermBase
    managers.SceneEntityCfg = SceneEntityCfg
    managers.RewardTermCfg = RewardTermCfg
    sys.modules["isaaclab.managers"] = managers

    sensors = ModuleType("isaaclab.sensors")
    sensors.ContactSensor = object
    sensors.RayCaster = object
    sys.modules["isaaclab.sensors"] = sensors

    math_mod = ModuleType("isaaclab.utils.math")
    math_mod.quat_apply_inverse = lambda quat, vec: vec
    math_mod.quat_apply = lambda quat, vec: vec
    math_mod.quat_conjugate = lambda quat: quat
    math_mod.yaw_quat = lambda quat: quat
    sys.modules["isaaclab.utils.math"] = math_mod

    utils = ModuleType("isaaclab.utils")
    utils.math = math_mod
    sys.modules["isaaclab.utils"] = utils

    return SceneEntityCfg


SceneEntityCfg = _install_isaaclab_stubs()
_REPO = Path(__file__).resolve().parents[1]
_REWARDS_PATH = _REPO / "source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/mdp/rewards.py"
_spec = importlib.util.spec_from_file_location("velocity_rewards_under_test", _REWARDS_PATH)
_rewards = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_rewards)
feet_min_clearance = _rewards.feet_min_clearance


class _Scene(dict):
    @property
    def sensors(self):
        return {"contact_forces": self["contact_forces"]}


class _CommandManager:
    def __init__(self, command):
        self._command = command

    def get_command(self, name):
        assert name == "base_velocity"
        return self._command


def _make_env(foot_z, contact_forces, command):
    scene = _Scene()
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=torch.tensor(foot_z, dtype=torch.float32),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        )
    )
    scene["contact_forces"] = SimpleNamespace(
        data=SimpleNamespace(
            net_forces_w_history=torch.tensor(contact_forces, dtype=torch.float32),
        )
    )
    return SimpleNamespace(
        scene=scene,
        command_manager=_CommandManager(torch.tensor(command, dtype=torch.float32)),
    )


def test_feet_min_clearance_penalizes_only_airborne_feet_below_minimum_height():
    env = _make_env(
        foot_z=[[[0.0, 0.0, 0.20], [0.0, 0.0, 0.40]]],
        contact_forces=[[[[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]]],
        command=[[0.5, 0.0, 0.0]],
    )

    penalty = feet_min_clearance(
        env,
        command_name="base_velocity",
        sensor_cfg=SceneEntityCfg("contact_forces", body_names=["left", "right"], body_ids=[0, 1]),
        asset_cfg=SceneEntityCfg("robot", body_names=["left", "right"], body_ids=[0, 1]),
        min_height=0.30,
        contact_threshold=1.0,
        command_threshold=0.1,
    )

    assert torch.allclose(penalty, torch.tensor([0.01]))


def test_feet_min_clearance_is_zero_when_command_is_small():
    env = _make_env(
        foot_z=[[[0.0, 0.0, 0.20], [0.0, 0.0, 0.20]]],
        contact_forces=[[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]],
        command=[[0.0, 0.0, 0.0]],
    )

    penalty = feet_min_clearance(
        env,
        command_name="base_velocity",
        sensor_cfg=SceneEntityCfg("contact_forces", body_names=["left", "right"], body_ids=[0, 1]),
        asset_cfg=SceneEntityCfg("robot", body_names=["left", "right"], body_ids=[0, 1]),
        min_height=0.30,
        contact_threshold=1.0,
        command_threshold=0.1,
    )

    assert torch.allclose(penalty, torch.tensor([0.0]))
