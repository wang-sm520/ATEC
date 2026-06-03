import unittest

from atec_rl_lab.train.task_d.lidar_perception import TaskDLidarPerception


class TaskDLidarPerceptionTest(unittest.TestCase):
    def test_parse_16_by_360_flat_scan(self):
        perception = TaskDLidarPerception()
        extero = [float(value) for value in range(16 * 360)]

        scan = perception.parse_scan(extero)

        self.assertEqual(scan.flat_length, 16 * 360)
        self.assertEqual(scan.channels, 16)
        self.assertEqual(scan.bins, 360)
        self.assertEqual(tuple(scan.grid.shape), (16, 360))
        self.assertAlmostEqual(scan.flat[17], 17.0)

    def test_parse_arbitrary_flat_scan_fallback(self):
        perception = TaskDLidarPerception()

        scan = perception.parse_scan([0.1, 0.2, 0.3, 0.4, 0.5])

        self.assertEqual(scan.flat_length, 5)
        self.assertEqual(scan.channels, 1)
        self.assertEqual(scan.bins, 5)
        self.assertEqual(tuple(scan.grid.shape), (1, 5))

    def test_empty_or_none_extero_uses_consistent_fallback_obstacle(self):
        perception = TaskDLidarPerception()

        none_obstacle, none_box = perception.measure(None)
        empty_obstacle, empty_box = perception.measure([])

        self.assertTrue(none_obstacle.frame.valid)
        self.assertTrue(empty_obstacle.frame.valid)
        self.assertEqual(none_obstacle.source, "lidar_fallback:no_extero")
        self.assertEqual(empty_obstacle.source, "lidar_fallback:no_extero")
        self.assertAlmostEqual(none_obstacle.frame.confidence, empty_obstacle.frame.confidence)
        self.assertFalse(none_box.valid)
        self.assertFalse(empty_box.valid)

    def test_box_measurement_invalid_when_no_cluster_evidence_exists(self):
        perception = TaskDLidarPerception()
        flat_ground_scan = [0.0] * (16 * 360)

        obstacle, box = perception.measure(flat_ground_scan)

        self.assertTrue(obstacle.frame.valid)
        self.assertFalse(box.valid)
        self.assertEqual(box.source, "lidar")
        self.assertIn("no plausible non-ground cluster", perception.last_debug["box_reason"])


if __name__ == "__main__":
    unittest.main()
