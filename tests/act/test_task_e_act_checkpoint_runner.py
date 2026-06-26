from pathlib import Path
import sys

import math

import torch
import torchvision.transforms as T

_REPO = Path(__file__).resolve().parents[2]
_ACT_DIR = _REPO / "scripts" / "act"
if str(_ACT_DIR) not in sys.path:
    sys.path.insert(0, str(_ACT_DIR))

import run_task_e_act_checkpoint as runner  # noqa: E402


def test_prepare_state_uses_first_eight_proprio_values_and_checkpoint_stats():
    norm_stats = {
        "state_mean": torch.ones(1, 8),
        "state_std": torch.full((1, 8), 2.0),
    }
    proprio = torch.arange(10, dtype=torch.float32).view(1, 10)

    state = runner.prepare_state(proprio, norm_stats, torch.device("cpu"))

    expected = (torch.arange(8, dtype=torch.float32).view(1, 8) - 1.0) / 2.0
    assert state.shape == (1, 8)
    assert torch.allclose(state, expected)


def test_prepare_rgb_accepts_bhwc_uint8_and_keeps_training_like_0_to_255_scale():
    rgb = torch.zeros((1, 480, 640, 3), dtype=torch.uint8)
    rgb[..., 0] = 255

    prepared = runner.prepare_rgb(rgb, torch.device("cpu"))

    assert prepared.shape == (1, 1, 3, 224, 224)
    assert prepared.dtype == torch.float32
    assert torch.allclose(prepared[0, 0, 0], torch.full((224, 224), 255.0))
    assert torch.allclose(prepared[0, 0, 1], torch.zeros((224, 224)))
    assert torch.allclose(prepared[0, 0, 2], torch.zeros((224, 224)))


def test_prepare_rgb_accepts_bchw_float_and_strips_alpha_without_normalizing():
    rgba = torch.zeros((1, 4, 32, 32), dtype=torch.float32)
    rgba[:, 0] = 255.0
    rgba[:, 3] = 128.0

    prepared = runner.prepare_rgb(rgba, torch.device("cpu"))

    assert prepared.shape == (1, 1, 3, 224, 224)
    assert torch.allclose(prepared[0, 0, 0], torch.full((224, 224), 255.0))
    assert torch.allclose(prepared[0, 0, 1], torch.zeros((224, 224)))


def test_prepare_rgb_matches_training_resize_for_nonuniform_image():
    rgb = torch.arange(1 * 17 * 19 * 3, dtype=torch.uint8).view(1, 17, 19, 3)

    prepared = runner.prepare_rgb(rgb, torch.device("cpu"))

    expected = T.Resize((224, 224), antialias=True)(rgb.permute(0, 3, 1, 2)).float().unsqueeze(1)
    assert torch.allclose(prepared, expected)


def test_absolute_state_from_relative_proprio_adds_default_joint_positions():
    rel = torch.tensor([[0.0, -1.2, 1.5, 0.0, -1.2, 0.0, -0.035, 0.035]])

    absolute = runner.absolute_state_from_relative_proprio(rel, torch.device("cpu"))

    assert torch.allclose(absolute, torch.zeros((1, 8)))


def test_denormalize_action_chunk_uses_checkpoint_action_stats():
    norm_stats = {
        "action_mean": torch.full((1, 8), 10.0),
        "action_std": torch.full((1, 8), 2.0),
    }
    chunk = torch.ones((1, 30, 8), dtype=torch.float32)

    denorm = runner.denormalize_action_chunk(chunk, norm_stats)

    assert denorm.shape == (1, 30, 8)
    assert torch.allclose(denorm, torch.full((1, 30, 8), 12.0))


def test_compute_real_time_sleep_uses_remaining_step_time():
    assert runner.compute_real_time_sleep(step_dt=0.02, elapsed_wall=0.005) == 0.015
    assert runner.compute_real_time_sleep(step_dt=0.02, elapsed_wall=0.03) == 0.0
    assert runner.compute_real_time_sleep(step_dt=None, elapsed_wall=0.005) == 0.0


def test_select_action_uses_chunk_index_in_chunk_mode():
    chunk = torch.arange(1 * 4 * 2, dtype=torch.float32).view(1, 4, 2)

    assert torch.equal(runner.select_action(chunk, mode="first", chunk_index=3), chunk[0, 0])
    assert torch.equal(runner.select_action(chunk, mode="chunk", chunk_index=3), chunk[0, 3])
    assert torch.equal(runner.select_action(chunk, mode="chunk", chunk_index=5), chunk[0, 3])


def test_collection_actuator_kwargs_match_collection_override_values():
    # Must mirror scripts/act/task_e/config.py ACT_* used by env_setup.build_task_e_env.
    assert runner.collection_actuator_kwargs() == {
        "effort_limit": 1000.0,
        "velocity_limit": 1000.0,
        "stiffness": 800.0,
        "damping": 80.0,
    }


def test_temporal_ensembler_single_chunk_returns_first_action():
    ens = runner.TemporalEnsembler(num_queries=3, action_dim=1, m=0.01, device=torch.device("cpu"))
    chunk = torch.tensor([[5.0], [10.0], [20.0]])

    out = ens.add_and_query(chunk)

    assert out.shape == (1,)
    assert torch.allclose(out, torch.tensor([5.0]))


def test_temporal_ensembler_weights_older_predictions_more_like_act_reference():
    # ACT reference: predictions for the current step are gathered oldest->newest and
    # weighted by exp(-m * i) with i=0 the oldest, so the older chunk's prediction for
    # this step gets the larger weight.
    ens = runner.TemporalEnsembler(num_queries=3, action_dim=1, m=0.01, device=torch.device("cpu"))

    a = torch.tensor([[0.0], [10.0], [20.0]])    # predicted at step 0
    ens.add_and_query(a)
    b = torch.tensor([[100.0], [110.0], [120.0]])  # predicted at step 1
    out = ens.add_and_query(b)

    w_old, w_new = 1.0, math.exp(-0.01)
    expected = (10.0 * w_old + 100.0 * w_new) / (w_old + w_new)
    assert torch.allclose(out, torch.tensor([expected]), atol=1e-5)


def test_temporal_ensembler_drops_chunks_older_than_horizon():
    ens = runner.TemporalEnsembler(num_queries=2, action_dim=1, m=0.01, device=torch.device("cpu"))

    ens.add_and_query(torch.tensor([[1.0], [2.0]]))   # step 0, covers steps 0,1
    ens.add_and_query(torch.tensor([[3.0], [4.0]]))   # step 1, covers steps 1,2
    out = ens.add_and_query(torch.tensor([[9.0], [9.0]]))  # step 2: chunk0 expired, only chunk1(pred[1]=4) & chunk2(pred[0]=9)

    w_old, w_new = 1.0, math.exp(-0.01)
    expected = (4.0 * w_old + 9.0 * w_new) / (w_old + w_new)
    assert torch.allclose(out, torch.tensor([expected]), atol=1e-5)


def test_home_warmup_action_round_trips_to_home_pose_via_default_offset():
    # Env action model (JointPositionActionCfg, use_default_offset=True, scale=0.5):
    #   joint_target = default_joint_pos + ACTION_SCALE * action
    # So the warmup action must reconstruct the collection home pose.
    default = torch.tensor([[0.0, 1.2, -1.5, 0.0, 1.2, 0.0, 0.035, -0.035]])

    action = runner.home_warmup_action(default)
    reconstructed_target = default + runner.ACTION_SCALE * action

    expected_home = torch.tensor([[0.0, 0.9245, -1.515, 0.0, 1.22, 0.0, 0.035, -0.035]])
    assert action.shape == (1, 8)
    assert torch.allclose(reconstructed_target, expected_home, atol=1e-5)
