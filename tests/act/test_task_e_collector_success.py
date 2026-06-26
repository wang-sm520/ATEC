from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib
import sys

import torch

_REPO = Path(__file__).resolve().parents[2]
_ACT_DIR = _REPO / "scripts" / "act"
if str(_ACT_DIR) not in sys.path:
    sys.path.insert(0, str(_ACT_DIR))


def _install_stubs():
    env_cfg = ModuleType("atec_rl_lab.tasks.task_e.env_cfg")
    env_cfg.TABLE_CENTER_X = 0.50
    env_cfg.TABLE_CENTER_Y = 0.00
    env_cfg.TABLE_TOP_Z = 0.20
    env_cfg.TABLE_HALF_X = 0.30
    env_cfg.BASKET_CENTER_X = 0.40
    env_cfg.BASKET_CENTER_Y = 0.00
    env_cfg.BASKET_EXCL_HALF_X = 0.20
    env_cfg.BASKET_EXCL_HALF_Y = 0.11

    sys.modules.setdefault("atec_rl_lab", ModuleType("atec_rl_lab"))
    sys.modules.setdefault("atec_rl_lab.tasks", ModuleType("atec_rl_lab.tasks"))
    sys.modules.setdefault("atec_rl_lab.tasks.task_e", ModuleType("atec_rl_lab.tasks.task_e"))
    sys.modules["atec_rl_lab.tasks.task_e.env_cfg"] = env_cfg

    utils_mod = ModuleType("atec_rl_lab.utils")
    utils_mod.CartesianController = object
    sys.modules["atec_rl_lab.utils"] = utils_mod

    envs_mod = ModuleType("isaaclab.envs")
    envs_mod.ManagerBasedRLEnv = object
    sys.modules.setdefault("isaaclab", ModuleType("isaaclab"))
    sys.modules["isaaclab.envs"] = envs_mod

    math_mod = ModuleType("isaaclab.utils.math")
    math_mod.matrix_from_quat = lambda quat: torch.eye(3).repeat(quat.shape[0], 1, 1)
    math_mod.quat_from_matrix = lambda matrix: torch.tensor(
        [[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32
    ).repeat(matrix.shape[0], 1)
    math_mod.quat_apply = lambda quat, vec: vec
    sys.modules.setdefault("isaaclab.utils", ModuleType("isaaclab.utils"))
    sys.modules["isaaclab.utils.math"] = math_mod


_install_stubs()
collector = importlib.import_module("task_e.collector")


class _FakeRigidObject:
    def __init__(self, pos):
        self.data = SimpleNamespace(root_pos_w=torch.tensor([pos], dtype=torch.float32))


def _fake_env(positions):
    rigid_objects = {
        f"object_{obj_id}": _FakeRigidObject(pos)
        for obj_id, pos in positions.items()
    }
    scene = SimpleNamespace(rigid_objects=rigid_objects)
    return SimpleNamespace(unwrapped=SimpleNamespace(scene=scene))


def test_get_objects_in_basket_reports_each_requested_object():
    env = _fake_env({
        1: [0.40, 0.00, 0.22],
        2: [0.90, 0.00, 0.22],
        3: [0.40, 0.00, 0.50],
    })

    result = collector.get_objects_in_basket(env, [1, 2, 3])

    assert result == {1: True, 2: False, 3: False}


def test_get_objects_in_basket_matches_official_z_bounds():
    table_top = collector.TABLE_TOP_Z
    official_max_z = table_top + 0.15

    env = _fake_env({
        1: [0.40, 0.00, table_top],
        2: [0.40, 0.00, official_max_z],
        3: [0.40, 0.00, official_max_z + 0.001],
    })

    assert collector.get_objects_in_basket(env, [1, 2, 3]) == {
        1: True,
        2: True,
        3: False,
    }

    below_table_env = _fake_env({1: [0.40, 0.00, table_top - 0.001]})

    assert collector.get_objects_in_basket(below_table_env, [1]) == {1: False}


def test_check_objects_in_basket_remains_all_success_wrapper():
    success_env = _fake_env({1: [0.40, 0.00, 0.22], 2: [0.40, 0.00, 0.22]})
    fail_env = _fake_env({1: [0.40, 0.00, 0.22], 2: [0.90, 0.00, 0.22]})

    assert collector.check_objects_in_basket(success_env, [1, 2]) is True
    assert collector.check_objects_in_basket(fail_env, [1, 2]) is False


def _flag(value: bool) -> torch.Tensor:
    return torch.tensor([value])


def test_classify_step_end_returns_failure_for_truncation_before_success_check():
    env = _fake_env({1: [0.40, 0.00, 0.22], 2: [0.40, 0.00, 0.22], 3: [0.40, 0.00, 0.22]})

    result = collector._classify_step_end(env, [1, 2, 3], _flag(False), _flag(True))

    assert result == "failure"


def test_classify_step_end_returns_success_for_basket_success_termination():
    env = _fake_env({1: [0.40, 0.00, 0.22], 2: [0.40, 0.00, 0.22], 3: [0.40, 0.00, 0.22]})

    result = collector._classify_step_end(env, [1, 2, 3], _flag(True), _flag(False))

    assert result == "success"


def test_classify_step_end_returns_failure_for_non_success_termination():
    env = _fake_env({1: [0.40, 0.00, 0.22], 2: [0.90, 0.00, 0.22], 3: [0.40, 0.00, 0.22]})

    result = collector._classify_step_end(env, [1, 2, 3], _flag(True), _flag(False))

    assert result == "failure"


def test_classify_step_end_returns_continue_without_done_flags():
    env = _fake_env({1: [0.40, 0.00, 0.22], 2: [0.40, 0.00, 0.22], 3: [0.40, 0.00, 0.22]})

    result = collector._classify_step_end(env, [1, 2, 3], _flag(False), _flag(False))

    assert result == "continue"
