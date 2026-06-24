import numpy as np

from atec_rl_lab.teleop.b2piper_arm import (
    ArmTeleopConfig,
    ArmTeleopMapper,
    clamp_workspace,
    lerp_gripper,
    limit_step,
    pack_b2piper_action,
    piper_target_to_action,
    unity_to_robot_position,
)


def test_pack_b2piper_action_keeps_leg_entries_zero():
    piper = np.arange(8, dtype=np.float32) + 1.0

    full = pack_b2piper_action(piper)

    assert full.shape == (20,)
    np.testing.assert_allclose(full[:12], np.zeros(12, dtype=np.float32))
    np.testing.assert_allclose(full[12:], piper)


def test_piper_target_to_action_uses_default_offset_and_scale():
    default = np.array([0.2, -0.2, 0.0, 0.5, -0.5, 0.1, 0.035, -0.035], dtype=np.float32)
    target = default + np.array([0.1, -0.1, 0.0, 0.25, -0.25, 0.05, -0.05, 0.05], dtype=np.float32)

    action = piper_target_to_action(target, default, action_scale=0.5)

    expected = (target - default) / 0.5
    np.testing.assert_allclose(action, expected.astype(np.float32))


def test_lerp_gripper_clamps_trigger_and_interpolates():
    np.testing.assert_allclose(lerp_gripper(-2.0), np.array([0.035, -0.035], dtype=np.float32))
    np.testing.assert_allclose(lerp_gripper(2.0), np.array([-0.015, 0.015], dtype=np.float32))
    np.testing.assert_allclose(lerp_gripper(0.5), np.array([0.010, -0.010], dtype=np.float32))


def test_workspace_clamp_and_step_limit():
    lower = np.array([0.15, -0.45, -0.30], dtype=np.float32)
    upper = np.array([0.85, 0.45, 0.35], dtype=np.float32)
    pos = np.array([2.0, -2.0, 0.0], dtype=np.float32)

    clamped = clamp_workspace(pos, lower, upper)

    np.testing.assert_allclose(clamped, np.array([0.85, -0.45, 0.0], dtype=np.float32))

    previous = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    desired = np.array([0.10, -0.10, 0.01], dtype=np.float32)
    limited = limit_step(previous, desired, max_step=0.03)
    np.testing.assert_allclose(limited, np.array([0.03, -0.03, 0.01], dtype=np.float32))


def test_unity_to_robot_position_matches_sonic_convention():
    unity = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    robot = unity_to_robot_position(unity)

    np.testing.assert_allclose(robot, np.array([-1.0, 3.0, 2.0], dtype=np.float32))


def test_mapper_calibration_has_no_jump_then_tracks_relative_motion():
    cfg = ArmTeleopConfig(
        translation_scale=1.0,
        max_target_step_m=0.50,
        workspace_lower=np.array([0.15, -0.45, -0.30], dtype=np.float32),
        workspace_upper=np.array([0.85, 0.45, 0.35], dtype=np.float32),
    )
    mapper = ArmTeleopMapper(cfg)
    controller0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ee0 = np.array([0.40, 0.0, 0.10], dtype=np.float32)
    quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    mapper.calibrate(controller0, ee0, quat0)
    target0 = mapper.update(controller0, enabled=True)
    target1 = mapper.update(np.array([0.10, -0.10, 0.05], dtype=np.float32), enabled=True)
    held = mapper.update(np.array([0.40, 0.40, 0.40], dtype=np.float32), enabled=False)

    np.testing.assert_allclose(target0.pos_b, ee0)
    np.testing.assert_allclose(target0.quat_b, quat0)
    np.testing.assert_allclose(target1.pos_b, np.array([0.50, -0.10, 0.15], dtype=np.float32))
    np.testing.assert_allclose(held.pos_b, target1.pos_b)
