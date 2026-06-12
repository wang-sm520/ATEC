"""MiniWBC.act() finger passthrough (action indices 29-32)."""

import importlib.util
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - bare shells may lack numpy
    np = None

from demo.task_b_planner import STOW_LEFT_HAND, STOW_RIGHT_HAND

_HAVE_ORT = np is not None and importlib.util.find_spec("onnxruntime") is not None
if _HAVE_ORT:
    from demo.mini_wbc import MiniWBC, DEFAULT_LEFT_HAND, DEFAULT_RIGHT_HAND


def _zero_proprio_row():
    row = np.zeros(12 + 3 * 33, dtype=np.float32)
    row[11] = -1.0  # projected gravity z (upright)
    return row


@unittest.skipIf(not _HAVE_ORT, "onnxruntime not available")
class FingerPassthroughTest(unittest.TestCase):
    def test_fingers_written_to_action_29_32(self):
        wbc = MiniWBC()
        out = wbc.act(_zero_proprio_row(), [0, 0, 0], 0.75, [0, 0, 0],
                      list(DEFAULT_LEFT_HAND), list(DEFAULT_RIGHT_HAND),
                      fingers=[0.3, 0.3, -0.4, -0.4])
        self.assertEqual(len(out), 33)
        for value, expected in zip(out[29:33], [0.3, 0.3, -0.4, -0.4]):
            self.assertAlmostEqual(value, expected, places=5)

    def test_fingers_default_none_keeps_zeros(self):
        wbc = MiniWBC()
        out = wbc.act(_zero_proprio_row(), [0, 0, 0], 0.75, [0, 0, 0],
                      list(DEFAULT_LEFT_HAND), list(DEFAULT_RIGHT_HAND))
        self.assertEqual(out[29:33], [0.0, 0.0, 0.0, 0.0])

    def test_stow_constants_match_planner(self):
        # The planner copies (does not import) the stow hand poses to stay
        # onnx-free; this guards the two copies from drifting apart.
        self.assertEqual(tuple(DEFAULT_LEFT_HAND), STOW_LEFT_HAND)
        self.assertEqual(tuple(DEFAULT_RIGHT_HAND), STOW_RIGHT_HAND)


if __name__ == "__main__":
    unittest.main()
