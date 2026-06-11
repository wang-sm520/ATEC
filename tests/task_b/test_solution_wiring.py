"""Thin-wiring contract for demo/solution.py.

The plumbing test runs WITHOUT onnxruntime (MiniWBC is stubbed). It asserts that
solution.py forwards the planner's WBCCommand verbatim to MiniWBC.act() and that
perception is throttled (every PERCEPTION_INTERVAL calls). The integration smoke
needs onnxruntime and exercises the real MiniWBC.
"""

import importlib
import importlib.util
import unittest

import numpy as np

import demo.solution as solution
from demo.task_b_planner import TaskBPlanner

_HAVE_ORT = importlib.util.find_spec("onnxruntime") is not None


def _zero_proprio_obs():
    row = np.zeros(12 + 3 * 33, dtype=np.float32)
    row[11] = -1.0  # projected gravity z (upright)
    return {"proprio": row.reshape(1, -1), "image": {}}


class _StubWBC:
    """Records the last act() call and returns a 33-float action."""

    instances: list["_StubWBC"] = []

    def __init__(self, *args, **kwargs):
        self.calls = []
        _StubWBC.instances.append(self)

    def reset(self):
        pass

    def act(self, proprio_row, vel_cmd, base_height, waist_rpy, left_hand, right_hand,
            waist_weight=1.0, fingers=None):
        self.calls.append(dict(
            vel=list(vel_cmd), base_height=base_height, waist_rpy=list(waist_rpy),
            left_hand=list(left_hand), right_hand=list(right_hand), fingers=fingers,
        ))
        return [0.0] * 33


class _CountingPerception:
    """Records update() call count; returns no detections."""

    def __init__(self, *args, **kwargs):
        self.update_calls = 0

    def reset(self):
        self.update_calls = 0

    def update(self, image_obs, pose):
        self.update_calls += 1
        return []


class SolutionWiringTest(unittest.TestCase):
    def setUp(self):
        self._orig_wbc = solution.MiniWBC
        self._orig_perc = solution.TaskBRgbdPerception
        _StubWBC.instances = []
        solution.MiniWBC = _StubWBC
        solution.TaskBRgbdPerception = _CountingPerception

    def tearDown(self):
        solution.MiniWBC = self._orig_wbc
        solution.TaskBRgbdPerception = self._orig_perc

    def test_command_forwarded_verbatim_to_wbc(self):
        alg = solution.AlgSolution()
        wbc = _StubWBC.instances[-1]

        # Parallel planner + odometry replays the exact same inputs to compute the
        # WBCCommand solution.py should have forwarded for this step.
        ref_odom = solution.DeadReckoningOdometry(dt=0.02, x0=-10.0, y0=-10.0)
        ref_planner = TaskBPlanner()
        ref_guard = solution.PostureGuard()

        obs = _zero_proprio_obs()
        result = alg.predicts(obs, current_score=0.0)

        row = obs["proprio"][0]
        ref_pose = ref_odom.update(row)
        ref_posture = ref_guard.check(row)
        ref_cmd = ref_planner.step(ref_pose, [], 0.0, posture=ref_posture)

        self.assertEqual(len(result["action"]), 33)
        self.assertIs(result["giveup"], False)

        call = wbc.calls[-1]
        self.assertEqual(call["vel"], list(ref_cmd.vel))
        self.assertEqual(call["base_height"], ref_cmd.base_height)
        self.assertEqual(call["waist_rpy"], list(ref_cmd.waist_rpy))
        self.assertEqual(call["left_hand"], list(ref_cmd.left_hand))
        self.assertEqual(call["right_hand"], list(ref_cmd.right_hand))
        self.assertEqual(call["fingers"], ref_cmd.fingers)

    def test_perception_throttled_every_interval(self):
        alg = solution.AlgSolution()
        perc = alg.perception
        self.assertIsInstance(perc, _CountingPerception)
        for _ in range(10):
            alg.predicts(_zero_proprio_obs(), current_score=0.0)
        # Steps 0 and 5 trigger perception with PERCEPTION_INTERVAL == 5.
        self.assertEqual(solution.AlgSolution.PERCEPTION_INTERVAL, 5)
        self.assertEqual(perc.update_calls, 2)

    def test_reset_reinitializes_components(self):
        alg = solution.AlgSolution()
        for _ in range(7):
            alg.predicts(_zero_proprio_obs(), current_score=0.0)
        alg.reset()
        self.assertEqual(alg._perception_step, 0)
        self.assertEqual(alg._cached_detections, [])
        # One more call works after reset (step 0 -> perception fires).
        alg.predicts(_zero_proprio_obs(), current_score=0.0)
        self.assertEqual(alg.perception.update_calls, 1)


@unittest.skipIf(not _HAVE_ORT, "onnxruntime not available")
class SolutionIntegrationSmokeTest(unittest.TestCase):
    def test_real_alg_runs_and_resets(self):
        # Real MiniWBC / TaskBRgbdPerception (module reloaded clean of any stub).
        importlib.reload(solution)
        alg = solution.AlgSolution()
        for _ in range(20):
            result = alg.predicts(_zero_proprio_obs(), current_score=0.0)
            action = result["action"]
            self.assertEqual(len(action), 33)
            self.assertTrue(all(np.isfinite(v) for v in action))
            self.assertIs(result["giveup"], False)
        alg.reset()
        result = alg.predicts(_zero_proprio_obs(), current_score=0.0)
        self.assertEqual(len(result["action"]), 33)


if __name__ == "__main__":
    unittest.main()
