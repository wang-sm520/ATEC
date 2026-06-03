import math
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - dependency availability is environment-specific.
    torch = None

if torch is not None:
    from atec_rl_lab.train.task_d.types import VelocityCommand
    from atec_rl_lab.train.task_d.policy_bridge import G1VelocityPolicyBridge


class FakePolicy:
    def __init__(self):
        self.inputs = []

    def eval(self):
        return self

    def __call__(self, policy_input):
        self.inputs.append(policy_input.detach().clone())
        return torch.zeros((policy_input.shape[0], G1VelocityPolicyBridge.ACTION_DIM_BODY), device=policy_input.device)


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class G1VelocityPolicyBridgeTest(unittest.TestCase):
    def make_bridge(self):
        bridge = G1VelocityPolicyBridge(policy_path="unused.pt", device="cpu")
        bridge.policy = FakePolicy()
        return bridge

    def make_proprio(self, full_action_dim=33):
        return torch.zeros((1, 12 + 3 * full_action_dim), dtype=torch.float32)

    def test_history_vector_shape_is_960(self):
        bridge = self.make_bridge()

        bridge.act(self.make_proprio(), VelocityCommand(vx=0.4, vy=0.1, wz=-0.2))

        self.assertEqual(tuple(bridge.policy.inputs[-1].shape), (1, 960))


    def test_full_action_dim_is_derived_from_proprio(self):
        bridge = self.make_bridge()

        action = bridge.act(self.make_proprio(full_action_dim=37), [0.2, 0.0, 0.1])

        self.assertEqual(len(action), 37)


    def test_zero_fake_output_returns_finite_action(self):
        bridge = self.make_bridge()

        proprio = self.make_proprio()

        proprio[0, -33:] = 0.1


        action = bridge.act(proprio, torch.tensor([0.0, 0.0, 0.0]))

        self.assertTrue(all(math.isfinite(value) for value in action))


if __name__ == "__main__":
    unittest.main()
