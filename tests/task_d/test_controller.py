import unittest

from atec_rl_lab.train.task_d.controller import TaskDController
from atec_rl_lab.train.task_d.types import TaskDPhase, VelocityCommand


def proprio(vx=0.0, vy=0.0, wz=0.0):
    values = [0.0] * (12 + 3 * 33)
    values[0] = vx
    values[1] = vy
    values[5] = wz
    values[11] = -1.0
    return [values]


class TaskDControllerTest(unittest.TestCase):
    def test_no_extero_still_outputs_velocity_command(self):
        controller = TaskDController()

        command = controller.update({"proprio": proprio(), "extero": None}, current_score=0.0)

        self.assertIsInstance(command, VelocityCommand)
        self.assertTrue(controller.last_debug["obstacle_valid"])
        self.assertEqual(controller.last_debug["box_source"], "initial")

    def test_warmup_then_moves_toward_box_lane(self):
        controller = TaskDController(warmup_steps=1)

        first = controller.update({"proprio": proprio(), "extero": None}, current_score=0.0)
        second = controller.update({"proprio": proprio(), "extero": None}, current_score=0.0)

        self.assertEqual(controller.last_debug["phase"], TaskDPhase.MOVE_TO_BOX_LANE.value)
        self.assertAlmostEqual(first.vx, 0.0)
        self.assertGreater(second.vy, 0.0)

    def test_reset_returns_controller_to_warmup(self):
        controller = TaskDController(warmup_steps=1)
        controller.update({"proprio": proprio(), "extero": None}, current_score=0.0)
        controller.update({"proprio": proprio(), "extero": None}, current_score=0.0)

        controller.reset()
        controller.update({"proprio": proprio(), "extero": None}, current_score=0.0)

        self.assertEqual(controller.last_debug["phase"], TaskDPhase.WARMUP.value)


if __name__ == "__main__":
    unittest.main()
