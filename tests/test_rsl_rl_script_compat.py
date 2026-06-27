"""Regression tests for rsl-rl 5.x config compatibility in PPO scripts."""

from pathlib import Path


_REPO = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (_REPO / path).read_text(encoding="utf-8")


def _assert_standard_ppo_script_migrates_deprecated_rsl_rl_cfg(path: str):
    src = _source(path)

    assert "handle_deprecated_rsl_rl_cfg" in src
    assert "metadata.version(\"rsl-rl-lib\")" in src

    update_idx = src.index("cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)")
    migrate_idx = src.index("agent_cfg = handle_deprecated_rsl_rl_cfg(")
    runner_idx = src.index("OnPolicyRunner(env, agent_cfg.to_dict()")

    assert update_idx < migrate_idx < runner_idx


def test_play_script_migrates_deprecated_policy_cfg_before_building_runner():
    _assert_standard_ppo_script_migrates_deprecated_rsl_rl_cfg("scripts/rsl_rl/play.py")


def test_train_script_migrates_deprecated_policy_cfg_before_building_runner():
    _assert_standard_ppo_script_migrates_deprecated_rsl_rl_cfg("scripts/rsl_rl/train.py")
