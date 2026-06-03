import unittest

from atec_rl_lab.train.task_d.debug import classify_failure, summarize_controller_debug


def normal_debug():
    return {
        "phase": "PUSH_BOX_TO_BRIDGE",
        "robot_in_obstacle": (-3.6, 1.58, 0.01),
        "box_in_obstacle": (-2.95, 1.6, 0.0),
        "obstacle_valid": True,
        "obstacle_confidence": 0.95,
        "box_source": "lidar",
        "box_confidence": 0.8,
        "box_lidar_valid": True,
        "target": (-1.0, 1.6, 0.0),
        "target_description": "push to bridge",
        "cmd": (0.8, 0.0, 0.0),
        "lidar_box_debug": "box cluster accepted",
    }


class TaskDDebugTest(unittest.TestCase):
    def test_summary_contains_phase_robot_box_command_and_score(self):
        summary = summarize_controller_debug(normal_debug(), current_score=12.5)

        self.assertIn("phase=PUSH_BOX_TO_BRIDGE", summary)
        self.assertIn("robot=", summary)
        self.assertIn("box=", summary)
        self.assertIn("cmd=", summary)
        self.assertIn("score=12.50", summary)

    def test_classifies_lidar_no_box_when_obstacle_is_valid(self):
        debug = normal_debug()
        debug["box_lidar_valid"] = False
        debug["obstacle_valid"] = True

        self.assertEqual(classify_failure(debug, current_score=0.0), "lidar_no_box")

    def test_classifies_running_for_normal_debug(self):
        self.assertEqual(classify_failure(normal_debug(), current_score=12.5), "running")


if __name__ == "__main__":
    unittest.main()
