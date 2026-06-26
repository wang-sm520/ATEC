from pathlib import Path
from types import ModuleType
import importlib
import sys

import torch

_REPO = Path(__file__).resolve().parents[2]
_ACT_DIR = _REPO / "scripts" / "act"
if str(_ACT_DIR) not in sys.path:
    sys.path.insert(0, str(_ACT_DIR))


def _install_stubs():
    env_cfg = ModuleType("atec_rl_lab.tasks.task_e.env_cfg")
    env_cfg.BASKET_CENTER_X = 0.40
    env_cfg.BASKET_CENTER_Y = 0.00
    env_cfg.TABLE_CENTER_X = 0.50
    env_cfg.TABLE_CENTER_Y = 0.00
    env_cfg.TABLE_TOP_Z = 0.20
    env_cfg.TABLE_HALF_X = 0.30
    env_cfg.BASKET_EXCL_HALF_X = 0.20
    env_cfg.BASKET_EXCL_HALF_Y = 0.11

    sys.modules.setdefault("atec_rl_lab", ModuleType("atec_rl_lab"))
    sys.modules.setdefault("atec_rl_lab.tasks", ModuleType("atec_rl_lab.tasks"))
    sys.modules.setdefault("atec_rl_lab.tasks.task_e", ModuleType("atec_rl_lab.tasks.task_e"))
    sys.modules["atec_rl_lab.tasks.task_e.env_cfg"] = env_cfg

    math_mod = ModuleType("isaaclab.utils.math")
    math_mod.matrix_from_quat = lambda quat: torch.eye(3).repeat(quat.shape[0], 1, 1)
    math_mod.quat_from_matrix = lambda matrix: torch.tensor(
        [[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32
    ).repeat(matrix.shape[0], 1)
    math_mod.quat_apply = lambda quat, vec: vec

    sys.modules.setdefault("isaaclab", ModuleType("isaaclab"))
    sys.modules.setdefault("isaaclab.utils", ModuleType("isaaclab.utils"))
    sys.modules["isaaclab.utils.math"] = math_mod


_install_stubs()
state_machine = importlib.import_module("task_e.state_machine")
config = importlib.import_module("task_e.config")
PickPlaceStateMachine = state_machine.PickPlaceStateMachine


def _single_step_table():
    return {state: 1 for state in config.STATE_ORDER}


def test_optimized_steps_use_fast_full_order_timing():
    assert config.OPTIMIZED_STEPS == {
        "INIT": 10,
        "PRE_GRASP": 40,
        "REACH": 50,
        "CLOSE": 20,
        "LIFT": 35,
        "TRANSPORT": 25,
        "PLACE": 25,
        "OPEN": 10,
        "LIFT_RETRACT": 10,
        "RETRACT": 10,
    }


def test_full_order_123_tuned_object_config():
    assert config.OBJECT_STATE_STEPS[1] == {
        "PRE_GRASP": 50,
        "ALIGN_GRIPPER": 40,
        "REACH": 100,
        "CLOSE": 20,
        "LIFT": 50,
        "TRANSPORT": 25,
        "PLACE": 15,
        "OPEN": 10,
        "LIFT_RETRACT": 10,
        "RETRACT": 10,
    }
    assert config.OBJECT_STATE_STEPS[2] == {
        "PRE_GRASP": 50,
        "REACH": 40,
        "CLOSE": 15,
        "LIFT": 70,
    }
    assert config.OBJECT_STATE_STEPS[3] == {
        "INIT": 20,
        "PRE_GRASP": 20,
        "REACH": 50,
        "CLOSE": 10,
        "LIFT": 20,
        "TRANSPORT": 20,
        "PLACE": 15,
        "OPEN": 10,
        "LIFT_RETRACT": 10,
        "RETRACT": 10,
    }
    assert config.OBJECT_GRASP_Z_OFFSETS == {1: 0.08, 2: 0.08, 3: 0.08}
    assert config.OBJECT_TOOL_CENTER_OFFSETS_LOCAL == {
        1: [0.0, -0.05, 0.0],
        2: [-0.04, -0.01, 0.0],
        3: [0.0, 0.0, 0.0],
    }


def test_state_machine_uses_per_object_tool_center_offsets():
    steps = {state: 1 for state in config.STATE_ORDER}
    sm = PickPlaceStateMachine([1, 2, 3], "cpu", steps=steps)

    obj_pos = torch.tensor([1.0, 2.0, 0.30])

    sm.tick(obj_pos)  # object_1 INIT
    object_1_pre_grasp, _, _ = sm.tick(obj_pos)
    while sm.current_object_key == "object_1" and not sm.done:
        sm.tick(obj_pos)

    object_2_pre_grasp, _, _ = sm.tick(obj_pos)
    while sm.current_object_key == "object_2" and not sm.done:
        sm.tick(obj_pos)

    object_3_pre_grasp, _, _ = sm.tick(obj_pos)

    assert torch.allclose(object_1_pre_grasp, torch.tensor([1.0, 2.05, config.CARRY_Z]))
    assert torch.allclose(object_2_pre_grasp, torch.tensor([1.04, 2.01, config.CARRY_Z]))
    assert torch.allclose(object_3_pre_grasp, torch.tensor([1.0, 2.0, config.CARRY_Z]))


def test_state_machine_uses_per_object_grasp_offset_for_reach_and_close():
    sm = PickPlaceStateMachine(
        [1],
        "cpu",
        steps=_single_step_table(),
        grasp_z_offsets={1: 0.123},
        tool_center_offset_local=[0.0, 0.0, 0.0],
        object_state_steps={},
    )

    obj_pos = torch.tensor([1.0, 2.0, 0.30])
    sm.tick(obj_pos)  # INIT
    sm.tick(obj_pos)  # PRE_GRASP caches object position

    reach_pos, _, reach_gripper = sm.tick(torch.tensor([9.0, 9.0, 0.90]))
    close_pos, _, close_gripper = sm.tick(torch.tensor([8.0, 8.0, 0.80]))

    assert torch.allclose(reach_pos, torch.tensor([1.0, 2.0, 0.423]))
    assert reach_gripper == "open"
    assert torch.allclose(close_pos, torch.tensor([1.0, 2.0, 0.423]))
    assert close_gripper == "close"


def test_state_machine_applies_local_tool_center_offset_to_grasp_targets():
    sm = PickPlaceStateMachine(
        [1],
        "cpu",
        steps=_single_step_table(),
        grasp_z_offsets={1: 0.10},
        tool_center_offset_local=[0.0, 0.02, 0.0],
        object_state_steps={},
    )

    obj_pos = torch.tensor([1.0, 2.0, 0.30])
    sm.tick(obj_pos)  # INIT
    pre_grasp_pos, _, _ = sm.tick(obj_pos)  # PRE_GRASP
    reach_pos, _, _ = sm.tick(obj_pos)  # REACH

    assert torch.allclose(pre_grasp_pos, torch.tensor([1.0, 1.98, config.CARRY_Z]))
    assert torch.allclose(reach_pos, torch.tensor([1.0, 1.98, 0.40]))


def test_state_machine_falls_back_to_global_grasp_offset_when_no_object_offset_configured():
    sm = PickPlaceStateMachine(
        [3],
        "cpu",
        steps=_single_step_table(),
        grasp_z_offsets={},
        object_state_steps={},
    )
    sm._grasp_z_offsets = {}

    obj_pos = torch.tensor([1.0, 2.0, 0.30])
    sm.tick(obj_pos)  # INIT
    sm.tick(obj_pos)  # PRE_GRASP
    reach_pos, _, _ = sm.tick(obj_pos)  # REACH

    expected_z = 0.30 + config.GRASP_Z_OFFSET
    assert torch.allclose(reach_pos, torch.tensor([1.0, 2.0, expected_z]))


def test_object_one_aligns_gripper_before_reach():
    steps = {state: 1 for state in [*config.STATE_ORDER, "ALIGN_GRIPPER"]}
    object_state_steps = {1: {"PRE_GRASP": 1, "ALIGN_GRIPPER": 1, "REACH": 1, "CLOSE": 1, "LIFT": 1}}
    sm = PickPlaceStateMachine(
        [1],
        "cpu",
        steps=steps,
        grasp_z_offsets={1: 0.10},
        tool_center_offset_local=[0.0, 0.0, 0.0],
        object_state_steps=object_state_steps,
    )
    obj_pos = torch.tensor([1.0, 2.0, 0.30])
    grasp_quat = torch.tensor([0.5, 0.5, 0.5, 0.5])
    sm._grasp_quat_cache[1] = grasp_quat

    sm.tick(obj_pos)  # INIT
    pre_grasp_pos, pre_grasp_quat, pre_grasp_gripper = sm.tick(obj_pos)
    assert sm.state == "ALIGN_GRIPPER"
    align_pos, align_quat, align_gripper = sm.tick(obj_pos)
    assert sm.state == "REACH"
    reach_pos, reach_quat, reach_gripper = sm.tick(obj_pos)

    assert torch.allclose(pre_grasp_pos, torch.tensor([1.0, 2.0, config.CARRY_Z]))
    assert torch.allclose(pre_grasp_quat, torch.tensor(config.DEFAULT_PLACE_QUAT_W, dtype=torch.float32))
    assert pre_grasp_gripper == "open"
    assert torch.allclose(align_pos, torch.tensor([1.0, 2.0, config.CARRY_Z]))
    assert torch.allclose(align_quat, grasp_quat)
    assert align_gripper == "open"
    assert torch.allclose(reach_pos, torch.tensor([1.0, 2.0, 0.40]))
    assert torch.allclose(reach_quat, grasp_quat)
    assert reach_gripper == "open"


def test_lift_uses_cached_object_position_after_successful_grasp():
    steps = {state: 1 for state in [*config.STATE_ORDER, "ALIGN_GRIPPER"]}
    object_state_steps = {1: {"PRE_GRASP": 1, "ALIGN_GRIPPER": 1, "REACH": 1, "CLOSE": 1, "LIFT": 1}}
    sm = PickPlaceStateMachine(
        [1],
        "cpu",
        steps=steps,
        grasp_z_offsets={1: 0.10},
        tool_center_offset_local=[0.0, 0.0, 0.0],
        object_state_steps=object_state_steps,
    )

    cached_pos = torch.tensor([1.0, 2.0, 0.30])
    moved_pos = torch.tensor([9.0, 9.0, 0.90])
    sm.tick(cached_pos)  # INIT
    sm.tick(cached_pos)  # PRE_GRASP caches object position
    sm.tick(moved_pos)   # ALIGN_GRIPPER must use cache
    sm.tick(moved_pos)   # REACH must use cache
    sm.tick(moved_pos)   # CLOSE must use cache
    lift_pos, _, lift_gripper = sm.tick(moved_pos)

    assert torch.allclose(lift_pos, torch.tensor([1.0, 2.0, config.CARRY_Z]))
    assert lift_gripper == "close"


def test_optimized_place_and_open_use_medium_basket_drop_height():
    steps = {state: 1 for state in config.STATE_ORDER}
    sm = PickPlaceStateMachine([2], "cpu", steps=steps, use_basket_drop_height=True)
    obj_pos = torch.tensor([1.0, 2.0, 0.30])

    while sm.state != "PLACE":
        sm.tick(obj_pos)
    place_pos, _, place_gripper = sm.tick(obj_pos)
    open_pos, _, open_gripper = sm.tick(obj_pos)

    assert torch.allclose(place_pos, torch.tensor([0.40, 0.00, config.BASKET_DROP_Z]))
    assert place_gripper == "close"
    assert torch.allclose(open_pos, torch.tensor([0.40, 0.00, config.BASKET_DROP_Z]))
    assert open_gripper == "open"


def test_baseline_place_and_open_use_original_place_height():
    steps = {state: 1 for state in config.STATE_ORDER}
    sm = PickPlaceStateMachine([2], "cpu", steps=steps, use_basket_drop_height=False)
    obj_pos = torch.tensor([1.0, 2.0, 0.30])

    while sm.state != "PLACE":
        sm.tick(obj_pos)
    place_pos, _, _ = sm.tick(obj_pos)
    open_pos, _, _ = sm.tick(obj_pos)

    assert torch.allclose(place_pos, torch.tensor([0.40, 0.00, config.PLACE_HEIGHT]))
    assert torch.allclose(open_pos, torch.tensor([0.40, 0.00, config.PLACE_HEIGHT]))


def _collect_basket_state_targets(sm: PickPlaceStateMachine, obj_pos: torch.Tensor):
    targets = {}
    while not sm.done:
        state = sm.state
        object_key = sm.current_object_key
        ee_pos, _, _ = sm.tick(obj_pos)
        if state in ("TRANSPORT", "PLACE", "OPEN", "LIFT_RETRACT"):
            targets[(object_key, state)] = ee_pos
    return targets


def _expected_basket_target(obj_idx: int, z: float) -> torch.Tensor:
    dx, dy = config.OBJECT_BASKET_TARGET_OFFSETS[obj_idx]
    return torch.tensor([
        config.BASKET_CENTER_X + dx,
        config.BASKET_CENTER_Y + dy,
        z,
    ])


def test_state_machine_uses_per_object_basket_targets_for_drop_states():
    steps = {state: 1 for state in config.STATE_ORDER}
    sm = PickPlaceStateMachine(
        [1, 2, 3],
        "cpu",
        steps=steps,
        object_state_steps={},
        use_basket_drop_height=True,
    )

    targets = _collect_basket_state_targets(sm, torch.tensor([1.0, 2.0, 0.30]))

    for obj_idx in (1, 2, 3):
        object_key = f"object_{obj_idx}"
        assert torch.allclose(
            targets[(object_key, "TRANSPORT")],
            _expected_basket_target(obj_idx, config.CARRY_Z),
        )
        assert torch.allclose(
            targets[(object_key, "PLACE")],
            _expected_basket_target(obj_idx, config.BASKET_DROP_Z),
        )
        assert torch.allclose(
            targets[(object_key, "OPEN")],
            _expected_basket_target(obj_idx, config.BASKET_DROP_Z),
        )
        assert torch.allclose(
            targets[(object_key, "LIFT_RETRACT")],
            _expected_basket_target(obj_idx, config.CARRY_Z),
        )


def test_basket_target_offsets_match_tuned_drop_points():
    assert config.OBJECT_BASKET_TARGET_OFFSETS == {
        1: (0.2, 0.0),
        2: (0.0, 0.0),
        3: (-0.05, 0.05),
    }
    for dx, dy in config.OBJECT_BASKET_TARGET_OFFSETS.values():
        assert abs(dx) <= config.BASKET_IN_X
        assert abs(dy) <= config.BASKET_IN_Y


def test_state_machine_falls_back_to_basket_center_when_object_has_no_basket_offset():
    steps = {state: 1 for state in config.STATE_ORDER}
    sm = PickPlaceStateMachine(
        [1],
        "cpu",
        steps=steps,
        object_state_steps={},
        use_basket_drop_height=False,
    )
    sm._basket_target_offsets = {}

    while sm.state != "PLACE":
        sm.tick(torch.tensor([1.0, 2.0, 0.30]))
    place_pos, _, place_gripper = sm.tick(torch.tensor([1.0, 2.0, 0.30]))

    assert torch.allclose(
        place_pos,
        torch.tensor([config.BASKET_CENTER_X, config.BASKET_CENTER_Y, config.PLACE_HEIGHT]),
    )
    assert place_gripper == "close"


def test_object_two_uses_normal_top_down_state_sequence():
    side_states = [
        "BOTTLE_FACE_MINUS_X",
        "BOTTLE_DESCEND_HALF",
        "BOTTLE_PUSH_MINUS_X",
        "BOTTLE_CLOSE",
        "BOTTLE_LIFT",
    ]
    steps = {state: 1 for state in [*config.STATE_ORDER, *side_states]}
    object_state_steps = {2: {"PRE_GRASP": 1, "REACH": 1, "CLOSE": 1, "LIFT": 1}}
    sm = PickPlaceStateMachine(
        [2],
        "cpu",
        steps=steps,
        object_state_steps=object_state_steps,
    )
    obj_pos = torch.tensor([1.0, 2.0, 0.30])

    visited = []
    while not sm.done:
        visited.append(sm.state)
        sm.tick(obj_pos)

    assert visited[:6] == ["INIT", "PRE_GRASP", "REACH", "CLOSE", "LIFT", "TRANSPORT"]
    for state in side_states:
        assert state not in visited


def test_object_two_top_down_reach_close_and_lift_targets():
    steps = {state: 1 for state in config.STATE_ORDER}
    object_state_steps = {2: {"PRE_GRASP": 1, "REACH": 1, "CLOSE": 1, "LIFT": 1}}
    sm = PickPlaceStateMachine(
        [2],
        "cpu",
        steps=steps,
        tool_center_offset_local=[0.0, 0.0, 0.0],
        object_state_steps=object_state_steps,
    )
    obj_pos = torch.tensor([1.0, 2.0, 0.30])
    moved_pos = torch.tensor([9.0, 9.0, 0.90])
    default_quat = torch.tensor(config.DEFAULT_PLACE_QUAT_W, dtype=torch.float32)

    sm.tick(obj_pos)  # INIT
    pre_grasp_pos, pre_grasp_quat, pre_grasp_gripper = sm.tick(obj_pos)
    reach_pos, reach_quat, reach_gripper = sm.tick(moved_pos)
    close_pos, close_quat, close_gripper = sm.tick(moved_pos)
    lift_pos, lift_quat, lift_gripper = sm.tick(moved_pos)

    expected_grasp_z = 0.30 + config.OBJECT_GRASP_Z_OFFSETS[2]

    assert torch.allclose(pre_grasp_pos, torch.tensor([1.0, 2.0, config.CARRY_Z]))
    assert torch.allclose(pre_grasp_quat, default_quat)
    assert pre_grasp_gripper == "open"

    assert torch.allclose(reach_pos, torch.tensor([1.0, 2.0, expected_grasp_z]))
    assert torch.allclose(reach_quat, default_quat)
    assert reach_gripper == "open"

    assert torch.allclose(close_pos, torch.tensor([1.0, 2.0, expected_grasp_z]))
    assert torch.allclose(close_quat, default_quat)
    assert close_gripper == "close"

    assert torch.allclose(lift_pos, torch.tensor([1.0, 2.0, config.CARRY_Z]))
    assert torch.allclose(lift_quat, default_quat)
    assert lift_gripper == "close"


def test_state_machine_skips_retract_between_objects_but_retracts_after_last_object():
    bottle_states = [
        "BOTTLE_FACE_MINUS_X",
        "BOTTLE_DESCEND_HALF",
        "BOTTLE_PUSH_MINUS_X",
        "BOTTLE_CLOSE",
        "BOTTLE_LIFT",
    ]
    steps = {state: 1 for state in [*config.STATE_ORDER, "ALIGN_GRIPPER", *bottle_states]}
    object_state_steps = {1: {"PRE_GRASP": 1, "ALIGN_GRIPPER": 1, "REACH": 1, "CLOSE": 1, "LIFT": 1}}
    sm = PickPlaceStateMachine([1, 2, 3], "cpu", steps=steps, object_state_steps=object_state_steps)
    obj_pos = torch.tensor([1.0, 2.0, 0.30])

    visited = []
    while not sm.done:
        visited.append((sm.state, sm.current_object_key))
        sm.tick(obj_pos)

    assert ("LIFT_RETRACT", "object_1") in visited
    assert ("RETRACT", "object_1") not in visited
    assert ("LIFT_RETRACT", "object_2") in visited
    assert ("RETRACT", "object_2") not in visited
    assert ("LIFT_RETRACT", "object_3") in visited
    assert ("RETRACT", "object_3") in visited

    object_1_lift_retract_idx = visited.index(("LIFT_RETRACT", "object_1"))
    assert visited[object_1_lift_retract_idx + 1] == ("PRE_GRASP", "object_2")

    object_2_lift_retract_idx = visited.index(("LIFT_RETRACT", "object_2"))
    assert visited[object_2_lift_retract_idx + 1] == ("PRE_GRASP", "object_3")
