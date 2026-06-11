"""Task D dual-policy hand-off: demo/solution.py switches walk->climb at the
`forward` phase. Mirrors the conventions in test_policy_bridge.py (unittest +
FakePolicy, no real weights / GPU needed)."""

import os
import sys
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment-specific.
    torch = None

_DEMO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "demo"
)
if torch is not None:
    if _DEMO not in sys.path:
        sys.path.insert(0, _DEMO)
    import solution as sol  # noqa: E402  (demo/solution.py)


class FakePolicy:
    """Stand-in TorchScript module; records calls and returns a zero 29-d action."""

    def __init__(self, tag):
        self.tag = tag
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, policy_input):
        self.calls += 1
        return torch.zeros((policy_input.shape[0], sol._G1VelocityPolicyBridge.ACTION_DIM_BODY))


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class BridgeSelectTest(unittest.TestCase):
    def make_bridge(self, climb_path="climb.pt"):
        bridge = sol._G1VelocityPolicyBridge(
            policy_path="walk.pt", climb_path=climb_path, device="cpu"
        )
        bridge._modules["walk"] = FakePolicy("walk")
        if climb_path is not None:
            bridge._modules["climb"] = FakePolicy("climb")
        return bridge

    def test_starts_on_walk(self):
        self.assertEqual(self.make_bridge()._active, "walk")

    def test_select_climb_switches_active(self):
        bridge = self.make_bridge()
        bridge.select("climb")
        self.assertEqual(bridge._active, "climb")

    def test_select_climb_is_noop_without_climb_path(self):
        bridge = self.make_bridge(climb_path=None)
        bridge.select("climb")
        self.assertEqual(bridge._active, "walk")

    def test_reset_restores_walk_active(self):
        bridge = self.make_bridge()
        bridge.select("climb")
        bridge.reset()
        self.assertEqual(bridge._active, "walk")

    def test_act_uses_active_module(self):
        bridge = self.make_bridge()
        proprio = torch.zeros((1, 12 + 3 * 29), dtype=torch.float32)
        bridge.act(proprio, [0.5, 0.0, 0.0])
        bridge.select("climb")
        bridge.act(proprio, [0.5, 0.0, 0.0])
        self.assertEqual(bridge._modules["walk"].calls, 1)
        self.assertEqual(bridge._modules["climb"].calls, 1)


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class HandoffTest(unittest.TestCase):
    def make_solution(self):
        solution = sol.AlgSolution()
        solution.bridge._modules["walk"] = FakePolicy("walk")
        solution.bridge._modules["climb"] = FakePolicy("climb")
        return solution

    def proprio(self):
        return torch.zeros((1, 12 + 3 * 29), dtype=torch.float32)

    def test_climb_path_is_wired_into_bridge(self):
        solution = self.make_solution()
        self.assertEqual(solution.bridge._paths["climb"], sol._CLIMB_POLICY_PATH)

    def test_stays_on_walk_before_forward(self):
        solution = self.make_solution()
        solution.controller.phase = "push_x"
        solution.predicts({"proprio": self.proprio()}, current_score=0.0)
        self.assertEqual(solution.bridge._active, "walk")

    def test_switches_to_climb_in_forward_phase(self):
        solution = self.make_solution()
        solution.controller.phase = "forward"
        out = solution.predicts({"proprio": self.proprio()}, current_score=0.0)
        self.assertEqual(solution.bridge._active, "climb")
        self.assertEqual(len(out["action"]), 29)

    def test_reset_returns_to_walk(self):
        solution = self.make_solution()
        solution.controller.phase = "forward"
        solution.predicts({"proprio": self.proprio()}, current_score=0.0)
        solution.reset()
        self.assertEqual(solution.bridge._active, "walk")


if __name__ == "__main__":
    unittest.main()
