import unittest
from contextlib import redirect_stdout
from io import StringIO

try:
    from atec_rl_lab.train.task_d.solution_adapter import AlgSolution
    from atec_rl_lab.train.task_d.types import VelocityCommand
except ModuleNotFoundError:  # pragma: no cover - torch may be unavailable in base Python.
    AlgSolution = None
    VelocityCommand = None


class FakeBridge:
    def __init__(self):
        self.reset_called = False
        self.commands = []

    def reset(self):
        self.reset_called = True

    def act(self, proprio, command):
        self.commands.append(command)
        return [0.0] * 33


class FakeController:
    def __init__(self):
        self.reset_called = False
        self.last_debug = {
            "phase": "MOVE_TO_BOX_LANE",
            "robot_in_obstacle": (-3.0, 0.0, 0.0),
            "box_in_obstacle": (-3.0, 1.6, 0.0),
            "target": (-3.0, 1.35, 0.0),
            "cmd": (0.1, 0.2, 0.3),
            "obstacle_valid": True,
            "box_lidar_valid": False,
            "box_source": "initial",
        }

    def reset(self):
        self.reset_called = True

    def update(self, obs, current_score):
        return VelocityCommand(vx=0.1, vy=0.2, wz=0.3)


@unittest.skipIf(AlgSolution is None, "torch is not installed in this Python environment")
class AlgSolutionAdapterTest(unittest.TestCase):
    def test_predicts_uses_controller_update_result(self):
        bridge = FakeBridge()
        controller = FakeController()
        solution = AlgSolution(
            policy_path="unused.pt",
            device="cpu",
            controller=controller,
            bridge=bridge,
            debug_interval=0,
        )

        result = solution.predicts({"proprio": [0.0] * 111}, current_score=0.0)

        self.assertFalse(result["giveup"])
        self.assertEqual(len(result["action"]), 33)
        self.assertEqual(bridge.commands[-1], VelocityCommand(vx=0.1, vy=0.2, wz=0.3))

    def test_reset_resets_bridge_and_controller(self):
        bridge = FakeBridge()
        controller = FakeController()
        solution = AlgSolution(
            policy_path="unused.pt",
            device="cpu",
            controller=controller,
            bridge=bridge,
            debug_interval=0,
        )

        solution.reset()

        self.assertTrue(bridge.reset_called)
        self.assertTrue(controller.reset_called)

    def test_predicts_prints_controller_coordinates_on_interval(self):
        bridge = FakeBridge()
        controller = FakeController()
        solution = AlgSolution(
            policy_path="unused.pt",
            device="cpu",
            controller=controller,
            bridge=bridge,
            debug_interval=2,
            debug_single_line=False,
        )

        captured = StringIO()
        with redirect_stdout(captured):
            solution.predicts({"proprio": [0.0] * 111}, current_score=0.0)
            solution.predicts({"proprio": [0.0] * 111}, current_score=0.0)
            solution.predicts({"proprio": [0.0] * 111}, current_score=1.0)

        output = captured.getvalue()
        self.assertEqual(output.count("[TaskD]"), 2)
        self.assertIn("robot=(-3.00,0.00,0.00)", output)
        self.assertIn("box=(-3.00,1.60,0.00)", output)
        self.assertIn("box_source=initial", output)

    def test_predicts_can_refresh_debug_on_one_terminal_line(self):
        bridge = FakeBridge()
        controller = FakeController()
        solution = AlgSolution(
            policy_path="unused.pt",
            device="cpu",
            controller=controller,
            bridge=bridge,
            debug_interval=1,
            debug_single_line=True,
        )

        captured = StringIO()
        with redirect_stdout(captured):
            solution.predicts({"proprio": [0.0] * 111}, current_score=0.0)

        output = captured.getvalue()
        self.assertTrue(output.startswith("\r[TaskD]"))
        self.assertNotIn("\n", output)
        self.assertIn("robot=(-3.00,0.00,0.00)", output)

    def test_debug_interval_zero_disables_printing(self):
        bridge = FakeBridge()
        controller = FakeController()
        solution = AlgSolution(
            policy_path="unused.pt",
            device="cpu",
            controller=controller,
            bridge=bridge,
            debug_interval=0,
        )

        captured = StringIO()
        with redirect_stdout(captured):
            solution.predicts({"proprio": [0.0] * 111}, current_score=0.0)

        self.assertEqual(captured.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
