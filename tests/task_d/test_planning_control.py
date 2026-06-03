import math
import unittest

from atec_rl_lab.train.task_d.phase_machine import TaskDPhaseMachine
from atec_rl_lab.train.task_d.types import (
    BoxState,
    ControlTarget,
    Pose2D,
    RobotState,
    TaskDPhase,
)
from atec_rl_lab.train.task_d.velocity_controller import TaskDVelocityController
from atec_rl_lab.train.task_d.waypoint_planner import TaskDWaypointPlanner


def robot_pose(x, y, yaw=0.0, step=0):
    return RobotState(
        pose_world=Pose2D(x=x, y=y, yaw=yaw),
        pose_obstacle=Pose2D(x=x, y=y, yaw=yaw),
        body_vx=0.0,
        body_vy=0.0,
        yaw_rate=0.0,
        step_count=step,
    )


def box_pose(x=-3.0, y=1.6):
    return BoxState(
        pose_obstacle=Pose2D(x=x, y=y, yaw=0.0),
        source="test",
        confidence=1.0,
    )


class TaskDPhaseMachineTest(unittest.TestCase):
    def test_phase_sequence_for_representative_task_states(self):
        machine = TaskDPhaseMachine(warmup_steps=2)
        box = box_pose()

        self.assertEqual(machine.update(robot_pose(-4.0, 1.0, step=0), box, 0.0), TaskDPhase.WARMUP)
        self.assertEqual(machine.update(robot_pose(-4.0, 1.0, step=2), box, 0.0), TaskDPhase.MOVE_TO_BOX_LANE)
        self.assertEqual(machine.update(robot_pose(-3.02, 1.36, step=3), box, 0.0), TaskDPhase.ALIGN_BEHIND_BOX)
        self.assertEqual(machine.update(robot_pose(-3.76, 1.6, step=4), box, 0.0), TaskDPhase.PUSH_BOX_TO_BRIDGE)
        self.assertEqual(machine.update(robot_pose(-1.55, 1.6, step=5), box_pose(-0.9, 1.6), 0.0), TaskDPhase.BACK_OFF_AND_CENTER)
        self.assertEqual(machine.update(robot_pose(-1.3, 1.6, step=6), box_pose(-0.9, 1.6), 0.0), TaskDPhase.CROSS_ON_BOX)
        self.assertEqual(machine.update(robot_pose(2.05, 1.6, step=7), box_pose(-0.9, 1.6), 1.0), TaskDPhase.FINISH)

    def test_timeout_advances_deterministically(self):
        machine = TaskDPhaseMachine(warmup_steps=0, phase_timeout_steps=3)
        box = box_pose()

        self.assertEqual(machine.update(robot_pose(-4.0, 0.0, step=0), box, 0.0), TaskDPhase.MOVE_TO_BOX_LANE)
        self.assertEqual(machine.update(robot_pose(-4.0, 0.0, step=3), box, 0.0), TaskDPhase.ALIGN_BEHIND_BOX)

    def test_reset_returns_to_warmup(self):
        machine = TaskDPhaseMachine(warmup_steps=0)
        box = box_pose(-0.9, 1.6)

        self.assertEqual(machine.update(robot_pose(2.1, 1.6, step=10), box, 1.0), TaskDPhase.FINISH)
        machine.reset()
        self.assertEqual(machine.update(robot_pose(-4.0, 1.0, step=0), box_pose(), 0.0), TaskDPhase.MOVE_TO_BOX_LANE)

    def test_partial_score_does_not_skip_push_sequence(self):
        machine = TaskDPhaseMachine(warmup_steps=0)
        box = box_pose()

        phase = machine.update(robot_pose(-2.0, 1.35, step=1), box, current_score=2.0)

        self.assertNotEqual(phase, TaskDPhase.FINISH)

    def test_near_max_score_finishes(self):
        machine = TaskDPhaseMachine(warmup_steps=0, finish_score=35.0)
        box = box_pose()

        phase = machine.update(robot_pose(-2.0, 1.35, step=1), box, current_score=35.0)

        self.assertEqual(phase, TaskDPhase.FINISH)


class TaskDWaypointPlannerTest(unittest.TestCase):
    def test_targets_for_each_phase(self):
        planner = TaskDWaypointPlanner()
        robot = robot_pose(-1.0, 0.3)
        box = box_pose(-2.2, 1.7)

        cases = {
            TaskDPhase.MOVE_TO_BOX_LANE: (-3.0, 1.35, 0.0),
            TaskDPhase.ALIGN_BEHIND_BOX: (-2.95, 1.7, 0.0),
            TaskDPhase.PUSH_BOX_TO_BRIDGE: (-2.85, 1.7, 0.0),
            TaskDPhase.BACK_OFF_AND_CENTER: (-1.4, 1.7, 0.0),
            TaskDPhase.CROSS_ON_BOX: (2.2, 1.7, 0.0),
            TaskDPhase.FINISH: (3.8, 1.2, 0.0),
        }

        for phase, expected in cases.items():
            with self.subTest(phase=phase):
                target = planner.target(phase, robot, box)
                self.assertEqual(target.phase, phase)
                self.assertAlmostEqual(target.pose_obstacle.x, expected[0])
                self.assertAlmostEqual(target.pose_obstacle.y, expected[1])
                self.assertAlmostEqual(target.pose_obstacle.yaw, expected[2])


class TaskDVelocityControllerTest(unittest.TestCase):
    def test_obstacle_error_is_rotated_into_body_frame(self):
        controller = TaskDVelocityController(k_xy=1.0, k_yaw=1.0)
        robot = robot_pose(0.0, 0.0, yaw=math.pi / 2)
        target = ControlTarget(Pose2D(1.0, 0.0, 0.0), TaskDPhase.MOVE_TO_BOX_LANE)

        command = controller.command(robot, target)

        self.assertAlmostEqual(command.vx, 0.0, places=6)
        self.assertAlmostEqual(command.vy, -0.6, places=6)

    def test_global_clamps_apply_to_non_push_commands(self):
        controller = TaskDVelocityController(k_xy=10.0, k_yaw=10.0)
        robot = robot_pose(0.0, 0.0, yaw=0.0)
        target = ControlTarget(Pose2D(10.0, -10.0, math.pi), TaskDPhase.FINISH)

        command = controller.command(robot, target)

        self.assertAlmostEqual(command.vx, 1.2)
        self.assertAlmostEqual(command.vy, -0.6)
        self.assertAlmostEqual(command.wz, 1.57)

    def test_push_phase_uses_forward_and_lateral_limits(self):
        controller = TaskDVelocityController(k_xy=10.0, k_yaw=10.0)
        robot = robot_pose(0.0, 0.0, yaw=0.0)
        target = ControlTarget(Pose2D(-1.0, 1.0, 0.0), TaskDPhase.PUSH_BOX_TO_BRIDGE)

        command = controller.command(robot, target)

        self.assertAlmostEqual(command.vx, 0.35)
        self.assertAlmostEqual(command.vy, 0.25)
        self.assertAlmostEqual(command.wz, 0.0)


if __name__ == "__main__":
    unittest.main()
