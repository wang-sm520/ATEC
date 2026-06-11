"""Tests for demo/solution_b.py Task B OpenWBT squat adapter.

These tests avoid importing onnxruntime by injecting a fake session factory into the
adapter. They validate path resolution, observation construction, recurrent-state
threading, and conversion from OpenWBT target joint positions to ATEC actions.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment-specific.
    torch = None


_REPO = Path(__file__).resolve().parents[2]
_DEMO = _REPO / "demo"
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))

import solution_b as sol_b  # noqa: E402


class FakeSession:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32).reshape(1, 12)
        self.calls = []

    def run(self, output_names, inputs):
        self.calls.append((tuple(output_names), {k: v.copy() for k, v in inputs.items()}))
        hidden = inputs["input_hidden_states"] + 1.0
        return self.action.copy(), hidden.astype(np.float32)


class SquatPathResolutionTest(unittest.TestCase):
    def test_resolves_openwbt_root_from_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root / "ckpts" / "squat.onnx"
            ckpt.parent.mkdir(parents=True)
            ckpt.write_bytes(b"fake")

            old = os.environ.get("OPENWBT_ROOT")
            os.environ["OPENWBT_ROOT"] = str(root)
            try:
                self.assertEqual(sol_b.resolve_squat_onnx_path(), ckpt)
            finally:
                if old is None:
                    os.environ.pop("OPENWBT_ROOT", None)
                else:
                    os.environ["OPENWBT_ROOT"] = old

    def test_environment_root_takes_precedence_over_repo_openwbt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root / "ckpts" / "squat.onnx"
            ckpt.parent.mkdir(parents=True)
            ckpt.write_bytes(b"fake")

            old = os.environ.get("OPENWBT_ROOT")
            os.environ["OPENWBT_ROOT"] = str(root)
            try:
                adapter = sol_b.OpenWBTSquatAdapter(session_factory=lambda path: FakeSession(np.zeros(12)))
                self.assertEqual(adapter.policy_path, ckpt)
            finally:
                if old is None:
                    os.environ.pop("OPENWBT_ROOT", None)
                else:
                    os.environ["OPENWBT_ROOT"] = old


class SquatAdapterConversionTest(unittest.TestCase):
    def make_adapter(self, action=None):
        if action is None:
            action = np.arange(12, dtype=np.float32) / 10.0
        fake = FakeSession(action)
        adapter = sol_b.OpenWBTSquatAdapter(
            policy_path=Path("/tmp/squat.onnx"),
            session_factory=lambda path: fake,
        )
        return adapter, fake

    def make_proprio(self, full_dim=33):
        row = np.zeros(12 + 3 * full_dim, dtype=np.float32)
        row[3:6] = [1.0, 2.0, 3.0]
        row[9:12] = [0.1, 0.2, -0.9]
        jp_start = 12
        jv_start = jp_start + full_dim
        act_start = jv_start + full_dim
        row[jp_start:jp_start + full_dim] = np.arange(full_dim, dtype=np.float32) * 0.01
        row[jv_start:jv_start + full_dim] = np.arange(full_dim, dtype=np.float32) * -0.02
        row[act_start:act_start + full_dim] = np.arange(full_dim, dtype=np.float32) * 0.03
        return row.reshape(1, -1)

    def test_builds_openwbt_squat_observation_without_onnxruntime(self):
        adapter, fake = self.make_adapter(action=np.zeros(12, dtype=np.float32))
        proprio = self.make_proprio()

        adapter.act(proprio, height_pitch_cmd=(0.75, 0.0))

        obs = fake.calls[0][1]["obs"]
        self.assertEqual(obs.shape, (1, sol_b.OpenWBTSquatAdapter.NUM_OBS))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_allclose(obs[0, 0:2], np.array([0.75, 0.0], dtype=np.float32))
        np.testing.assert_allclose(obs[0, 2:5], np.array([0.1, 0.2, -0.9], dtype=np.float32))
        np.testing.assert_allclose(obs[0, 5:8], np.array([0.25, 0.5, 0.75], dtype=np.float32))
        q_abs = proprio[0, 12:41] + sol_b.ATEC_DEFAULT_ANGLES[:29]
        np.testing.assert_allclose(obs[0, 8:37], q_abs - sol_b.OPENWBT_DEFAULT_ANGLES)
        np.testing.assert_allclose(obs[0, 37:66], proprio[0, 45:74] * 0.05)
        np.testing.assert_allclose(obs[0, 66:78], np.zeros(12, dtype=np.float32))

    def test_threads_recurrent_hidden_state_between_calls_and_reset_clears_it(self):
        adapter, fake = self.make_adapter(action=np.zeros(12, dtype=np.float32))
        proprio = self.make_proprio()

        adapter.act(proprio)
        adapter.act(proprio)
        self.assertEqual(len(fake.calls), 2)
        np.testing.assert_allclose(fake.calls[0][1]["input_hidden_states"], np.zeros((1, 1, 256), dtype=np.float32))
        np.testing.assert_allclose(fake.calls[1][1]["input_hidden_states"], np.ones((1, 1, 256), dtype=np.float32))

        adapter.reset()
        adapter.act(proprio)
        np.testing.assert_allclose(fake.calls[2][1]["input_hidden_states"], np.zeros((1, 1, 256), dtype=np.float32))

    def test_converts_target_positions_to_atec_actions_for_lower_body_only(self):
        raw_action = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        adapter, _ = self.make_adapter(action=raw_action)
        proprio = self.make_proprio(full_dim=33)

        action = np.asarray(adapter.act(proprio), dtype=np.float32)

        expected = np.zeros(33, dtype=np.float32)
        target_lower = sol_b.OPENWBT_DEFAULT_ANGLES[:12] + raw_action * sol_b.OPENWBT_ACTION_SCALE
        expected[:12] = (target_lower - sol_b.ATEC_DEFAULT_ANGLES[:12]) / sol_b.ATEC_ACTION_SCALE
        expected[[5, 11]] = 0.0
        np.testing.assert_allclose(action, expected, rtol=1e-6, atol=1e-6)
        self.assertTrue(np.all(action[12:] == 0.0))

    def test_accepts_torch_proprio_and_returns_full_action_dim(self):
        if torch is None:
            self.skipTest("torch is not installed")
        adapter, _ = self.make_adapter(action=np.zeros(12, dtype=np.float32))
        proprio = torch.as_tensor(self.make_proprio(full_dim=33))

        out = adapter.act(proprio)

        self.assertEqual(len(out), 33)
        self.assertTrue(all(isinstance(v, float) for v in out))


class AlgSolutionTest(unittest.TestCase):
    def test_predicts_returns_action_and_no_giveup(self):
        fake = FakeSession(np.zeros(12, dtype=np.float32))
        solution = sol_b.AlgSolution(
            adapter=sol_b.OpenWBTSquatAdapter(
                policy_path=Path("/tmp/squat.onnx"),
                session_factory=lambda path: fake,
            )
        )
        proprio = SquatAdapterConversionTest().make_proprio(full_dim=33)

        out = solution.predicts({"proprio": proprio}, current_score=0.0)

        self.assertEqual(set(out), {"action", "giveup"})
        self.assertEqual(len(out["action"]), 33)
        self.assertFalse(out["giveup"])


if __name__ == "__main__":
    unittest.main()
