"""Regression tests for the self-contained Task E submission demo/solution.py.

These run without Isaac. The construction tests require demo/policy.pt and are
skipped if it is absent.
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
_SOLUTION = _REPO / "demo" / "solution.py"
_POLICY = _REPO / "demo" / "policy.pt"


def _load_solution_module():
    spec = importlib.util.spec_from_file_location("demo_solution_under_test", _SOLUTION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_solution_is_self_contained_no_repo_imports():
    text = _SOLUTION.read_text()
    for token in ("atec_rl_lab", "scripts", "train_task_e"):
        assert token not in text, f"submission solution.py must not reference {token!r}"


def test_defines_algsolution_with_required_api():
    mod = _load_solution_module()
    assert hasattr(mod, "AlgSolution")
    for method in ("__init__", "reset", "predicts"):
        assert callable(getattr(mod.AlgSolution, method))


@pytest.mark.skipif(not _POLICY.exists(), reason="demo/policy.pt not present")
def test_warmup_then_act_then_reset_and_fallback():
    mod = _load_solution_module()
    agent = mod.AlgSolution()
    obs = {
        "proprio": torch.zeros(1, 24),
        "image": {"video_rgb": torch.zeros(1, 480, 640, 3, dtype=torch.uint8)},
    }

    first = agent.predicts(obs, 0.0)
    assert first["giveup"] is False
    assert len(first["action"]) == 8
    warmup = first["action"]

    # remaining warmup steps return the same constant warmup action
    for _ in range(mod.WARMUP_STEPS - 1):
        assert agent.predicts(obs, 0.0)["action"] == warmup

    # first ACT step differs and is finite
    act_step = agent.predicts(obs, 0.0)["action"]
    assert act_step != warmup
    assert len(act_step) == 8 and all(math.isfinite(x) for x in act_step)

    # reset restarts warmup
    agent.reset()
    assert agent.predicts(obs, 0.0)["action"] == warmup

    # malformed obs (missing image) during ACT phase never raises, returns 8 floats
    agent._step = mod.WARMUP_STEPS  # force ACT phase
    bad = agent.predicts({"proprio": torch.zeros(1, 24)}, 0.0)
    assert bad["giveup"] is False and len(bad["action"]) == 8
