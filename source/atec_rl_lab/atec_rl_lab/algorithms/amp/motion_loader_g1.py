"""Motion expert dataset loader for AMP training of Unitree G1 (29 body DoF).

Ported from `bx_lab_amp/rsl_rl/rsl_rl/utils/motion_loader_forg1.py` with these changes:
- JOINT_POS_SIZE / JOINT_VEL_SIZE: 21 -> 29 (G1 body 29 DoF)
- END_EFFECTOR_POS_SIZE kept at 15 (5 EE x 3: left_ankle, right_ankle, waist, left_wrist, right_wrist)
- Replaced deprecated `np.int` with `np.int64`
- Cleaner public API for use with the new rsl-rl-lib 3.0.1 AMPPPO
"""

from __future__ import annotations

import glob
import json
import os
from typing import Iterator, Sequence

import numpy as np
import torch


class MotionLoaderG1:
    """Load reference motion clips for G1 humanoid AMP training.

    Each motion file is a JSON with keys:
        - "Frames"          : list[list[float]], shape (num_frames, FRAME_DIM)
        - "FrameDuration"   : float, seconds between consecutive frames
        - "MotionWeight"    : float, sampling weight relative to other clips
        - "LoopMode"        : "Wrap" | "Clamp" (currently informational only)

    Frame layout per row (FRAME_DIM = 73):
        [0  : 29) joint positions (29 body dof in atec_rl_lab joint order)
        [29 : 58) joint velocities (29 body dof, same order)
        [58 : 73) end-effector positions (5 EE x 3): left_ankle, right_ankle, waist, left_wrist, right_wrist
    """

    JOINT_POS_SIZE = 29
    JOINT_VEL_SIZE = 29
    END_EFFECTOR_POS_SIZE = 15

    JOINT_POS_START_IDX = 0
    JOINT_POS_END_IDX = JOINT_POS_START_IDX + JOINT_POS_SIZE
    JOINT_VEL_START_IDX = JOINT_POS_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE
    END_POS_START_IDX = JOINT_VEL_END_IDX
    END_POS_END_IDX = END_POS_START_IDX + END_EFFECTOR_POS_SIZE
    FRAME_DIM = END_POS_END_IDX  # 73

    def __init__(
        self,
        device: str | torch.device,
        time_between_frames: float,
        motion_files: Sequence[str] | None = None,
        data_dir: str = "",
        preload_transitions: bool = True,
        num_preload_transitions: int = 100_000,
    ) -> None:
        """
        Args:
            device: torch device for cached tensors.
            time_between_frames: control dt of the env (s); used to step from frame t to t+1.
            motion_files: explicit list of JSON paths. If None and data_dir is set, glob
                         `data_dir/*.json` and `data_dir/*.txt`.
            preload_transitions: if True, sample `num_preload_transitions` (s, s_next) pairs
                                up-front for faster generator iteration.
        """
        self.device = device
        self.time_between_frames = float(time_between_frames)

        if motion_files is None:
            if not data_dir:
                raise ValueError("MotionLoaderG1: provide either motion_files or data_dir.")
            motion_files = sorted(glob.glob(os.path.join(data_dir, "*.json"))) + sorted(
                glob.glob(os.path.join(data_dir, "*.txt"))
            )
        if len(motion_files) == 0:
            raise ValueError("MotionLoaderG1: no motion files provided.")

        self.trajectories: list[torch.Tensor] = []
        self.trajectory_names: list[str] = []
        self.trajectory_lens: list[float] = []  # seconds
        self.trajectory_weights: list[float] = []
        self.trajectory_frame_durations: list[float] = []
        self.trajectory_num_frames: list[int] = []

        for motion_file in motion_files:
            with open(motion_file) as f:
                motion_json = json.load(f)
            motion_data = np.asarray(motion_json["Frames"], dtype=np.float32)
            if motion_data.ndim != 2 or motion_data.shape[1] < self.FRAME_DIM:
                raise ValueError(
                    f"{motion_file}: expected Frames shape (N, >= {self.FRAME_DIM}), got {motion_data.shape}"
                )
            traj = torch.tensor(motion_data[:, : self.FRAME_DIM], dtype=torch.float32, device=device)
            frame_duration = float(motion_json["FrameDuration"])
            num_frames = motion_data.shape[0]
            traj_len = (num_frames - 1) * frame_duration

            self.trajectories.append(traj)
            self.trajectory_names.append(os.path.splitext(os.path.basename(motion_file))[0])
            self.trajectory_lens.append(traj_len)
            self.trajectory_weights.append(float(motion_json.get("MotionWeight", 1.0)))
            self.trajectory_frame_durations.append(frame_duration)
            self.trajectory_num_frames.append(num_frames)
            print(f"[MotionLoaderG1] Loaded {traj_len:.2f}s ({num_frames} frames) from {motion_file}")

        weights = np.asarray(self.trajectory_weights, dtype=np.float64)
        self.trajectory_weights_np = weights / weights.sum()
        self.trajectory_frame_durations_np = np.asarray(self.trajectory_frame_durations, dtype=np.float64)
        self.trajectory_lens_np = np.asarray(self.trajectory_lens, dtype=np.float64)
        self.trajectory_num_frames_np = np.asarray(self.trajectory_num_frames, dtype=np.int64)
        self.trajectory_idxs_np = np.arange(len(self.trajectories), dtype=np.int64)

        self.preload_transitions = preload_transitions
        self.preloaded_s: torch.Tensor | None = None
        self.preloaded_s_next: torch.Tensor | None = None
        if self.preload_transitions:
            print(f"[MotionLoaderG1] Preloading {num_preload_transitions} (s, s_next) transitions...")
            traj_idxs = self._weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self._traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self._get_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self._get_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print("[MotionLoaderG1] Preloading done.")

    # ---- sampling helpers ----

    def _weighted_traj_idx_sample_batch(self, size: int) -> np.ndarray:
        return np.random.choice(self.trajectory_idxs_np, size=size, p=self.trajectory_weights_np, replace=True)

    def _traj_time_sample_batch(self, traj_idxs: np.ndarray) -> np.ndarray:
        # Need at least frame_duration + time_between_frames headroom so s_next exists.
        subst = self.time_between_frames + self.trajectory_frame_durations_np[traj_idxs]
        time_samples = self.trajectory_lens_np[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(0.0, time_samples)

    @staticmethod
    def _slerp(frame1: torch.Tensor, frame2: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return (1.0 - blend) * frame1 + blend * frame2

    def _get_frame_at_time_batch(self, traj_idxs: np.ndarray, times: np.ndarray) -> torch.Tensor:
        """Linearly interpolate frames across multiple trajectories at given times."""
        p = times / self.trajectory_lens_np[traj_idxs]
        n = self.trajectory_num_frames_np[traj_idxs]
        # Clamp into [0, n-1] safely
        idx_low = np.clip(np.floor(p * n).astype(np.int64), 0, n - 1)
        idx_high = np.clip(np.ceil(p * n).astype(np.int64), 0, n - 1)

        out_starts = torch.zeros(len(traj_idxs), self.FRAME_DIM, device=self.device)
        out_ends = torch.zeros(len(traj_idxs), self.FRAME_DIM, device=self.device)
        for traj_idx in np.unique(traj_idxs):
            mask = traj_idxs == traj_idx
            traj = self.trajectories[int(traj_idx)]
            out_starts[mask] = traj[idx_low[mask]]
            out_ends[mask] = traj[idx_high[mask]]

        blend_np = (p * n - idx_low).astype(np.float32)
        blend = torch.as_tensor(blend_np, device=self.device).unsqueeze(-1)
        return self._slerp(out_starts, out_ends, blend)

    # ---- public API used by AMPPPO ----

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield `num_mini_batch` (s, s_next) pairs each shape (mini_batch_size, FRAME_DIM)."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size, replace=True)
                yield self.preloaded_s[idxs], self.preloaded_s_next[idxs]
            else:
                traj_idxs = self._weighted_traj_idx_sample_batch(mini_batch_size)
                times = self._traj_time_sample_batch(traj_idxs)
                s = self._get_frame_at_time_batch(traj_idxs, times)
                s_next = self._get_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
                yield s, s_next

    @property
    def observation_dim(self) -> int:
        """Dimensionality of one AMP frame; must equal env-side amp_obs dim."""
        return self.FRAME_DIM

    @property
    def num_motions(self) -> int:
        return len(self.trajectories)


if __name__ == "__main__":
    # Sanity check: synthesize a tiny motion file and round-trip through the loader.
    import tempfile

    rng = np.random.default_rng(0)
    n_frames = 50
    fake_frames = rng.standard_normal((n_frames, MotionLoaderG1.FRAME_DIM)).astype(np.float32).tolist()
    motion = {"FrameDuration": 0.02, "MotionWeight": 1.0, "LoopMode": "Wrap", "Frames": fake_frames}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(motion, f)
        path = f.name

    loader = MotionLoaderG1(
        device="cpu",
        time_between_frames=0.02,
        motion_files=[path],
        preload_transitions=True,
        num_preload_transitions=1024,
    )
    assert loader.observation_dim == 73, f"got {loader.observation_dim}"
    gen = loader.feed_forward_generator(num_mini_batch=2, mini_batch_size=64)
    for i, (s, s_next) in enumerate(gen):
        assert s.shape == (64, 73)
        assert s_next.shape == (64, 73)
        print(f"batch {i}: s.shape={tuple(s.shape)} s_next.shape={tuple(s_next.shape)} ok")
    os.unlink(path)
    print("MotionLoaderG1 sanity check passed.")
