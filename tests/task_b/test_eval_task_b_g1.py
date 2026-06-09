import builtins
import contextlib
import importlib
import io
import os
import sys
import types
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment-specific
    torch = None

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _install_isaaclab_app_stub():
    isaaclab = sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
    app = types.ModuleType("isaaclab.app")

    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--device", default="cpu")
            parser.add_argument("--headless", action="store_true", default=False)
            parser.add_argument("--enable_cameras", action="store_true", default=False)

    app.AppLauncher = AppLauncher
    isaaclab.app = app
    sys.modules["isaaclab.app"] = app


_install_isaaclab_app_stub()
eval_b = importlib.import_module("scripts.eval_task_b_g1")


class EvalTaskBG1HelpersTest(unittest.TestCase):
    def test_rejects_non_positive_max_steps_before_launch(self):
        parser = eval_b.build_arg_parser()
        args = parser.parse_args(["--max_steps", "0"])

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                eval_b.validate_args(parser, args)

    def test_any_done_handles_scalars_sequences_and_tensor_like_values(self):
        self.assertTrue(eval_b.any_done(True))
        self.assertTrue(eval_b.any_done([False, True]))
        self.assertFalse(eval_b.any_done((False, False)))
        if torch is not None:
            self.assertTrue(eval_b.any_done(torch.tensor([False, True])))
            self.assertFalse(eval_b.any_done(torch.tensor([False, False])))

    @unittest.skipIf(torch is None, "torch is not installed in this Python environment")
    def test_action_tensor_repeats_single_solution_action_for_requested_envs(self):
        action = eval_b.coerce_action_tensor([0.1, -0.2, 0.3], num_envs=2, device="cpu")

        self.assertEqual(tuple(action.shape), (2, 3))
        self.assertTrue(torch.allclose(action[0], torch.tensor([0.1, -0.2, 0.3])))
        self.assertTrue(torch.allclose(action[1], torch.tensor([0.1, -0.2, 0.3])))

    @unittest.skipIf(torch is None, "torch is not installed in this Python environment")
    def test_score_increment_uses_mean_reward_divided_by_step_dt(self):
        increment = eval_b.score_increment(torch.tensor([2.0, 4.0]), torch.tensor(0.5))

        self.assertEqual(increment, 6.0)

    def test_print_final_score_flushes_before_closing_simulation_app(self):
        calls = []
        original_print = builtins.print
        try:
            builtins.print = lambda *args, **kwargs: calls.append((args, kwargs))
            eval_b.print_final_score(1.25, 2.5)
        finally:
            builtins.print = original_print

        self.assertEqual(calls, [(('score: 1.25, elapsed_time: 2.50 seconds',), {'flush': True})])


if __name__ == "__main__":
    unittest.main()
