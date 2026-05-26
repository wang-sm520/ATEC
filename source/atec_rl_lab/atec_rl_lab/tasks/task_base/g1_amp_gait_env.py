"""G1 AMP env subclass that adds bxi-style gait clock state.

State exposed (read by reward functions in `mdp.rewards`):
  - `phase_length_buf`:        (N,) long — step counter since last reset
  - `gait_phase`:               (N, 2) float — left/right phase in [0, 1)
  - `phase_ratio`:              (N, 2) float — swing-air ratio per foot
  - `gait_start`:               (N,) float — per-env phase shift, randomized {0, 0.5} on reset
  - `avg_feet_force_per_step`:  (N, 2) float — current foot vertical contact force magnitude
  - `avg_feet_speed_per_step`:  (N, 2) float — current foot world-frame linear speed

State is refreshed at the END of `step()` (i.e. after super's physics + reward + reset cycle).
Reward computed in the next step's super().step() therefore reads the previous step's gait
state — a one-step lag, equivalent to bxi's design where gait_para is the post-physics value
used by the reward manager.

The class expects the env cfg to expose:
  - `cfg.gait`: a `G1GaitCfg`-like object with `gait_cycle`, `air_ratio_l`, `air_ratio_r`,
                `phase_offset_l`, `phase_offset_r` (only ratios are used; offsets are kept for
                parity with bxi but not consumed here).
  - `cfg.foot_body_pattern`: regex resolving to exactly two foot bodies.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv


class G1AMPGaitEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._init_gait_state()

    # ------------------------------------------------------------------
    # init / reset
    # ------------------------------------------------------------------

    def _init_gait_state(self) -> None:
        gait_cfg = self.cfg.gait
        N = self.num_envs
        device = self.device

        self.phase_length_buf = torch.zeros(N, device=device, dtype=torch.long)
        self.gait_phase = torch.zeros(N, 2, device=device, dtype=torch.float)
        self.gait_start = (torch.randint(0, 2, (N,), device=device).float()) * 0.5

        self.phase_ratio = torch.tensor(
            [float(gait_cfg.air_ratio_l), float(gait_cfg.air_ratio_r)],
            device=device,
            dtype=torch.float,
        ).repeat(N, 1)
        self.gait_cycle = float(gait_cfg.gait_cycle)

        self.avg_feet_force_per_step = torch.zeros(N, 2, device=device, dtype=torch.float)
        self.avg_feet_speed_per_step = torch.zeros(N, 2, device=device, dtype=torch.float)

        # Resolve feet body ids on robot articulation and contact sensor.
        robot = self.scene["robot"]
        pattern = self.cfg.foot_body_pattern
        feet_body_ids, _ = robot.find_bodies(pattern)
        if len(feet_body_ids) != 2:
            raise RuntimeError(
                f"G1AMPGaitEnv: foot_body_pattern={pattern!r} resolved {len(feet_body_ids)} "
                "bodies; expected exactly 2 (left + right ankle roll links)."
            )
        self.feet_body_ids = feet_body_ids

        contact = self.scene["contact_forces"]
        feet_sensor_ids, _ = contact.find_bodies(pattern)
        if len(feet_sensor_ids) != 2:
            raise RuntimeError(
                f"G1AMPGaitEnv: contact_forces.find_bodies({pattern!r}) returned "
                f"{len(feet_sensor_ids)} bodies; expected 2."
            )
        self.feet_sensor_ids = feet_sensor_ids

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        # _reset_idx may run from within super().__init__() before our state exists; guard.
        if not hasattr(self, "phase_length_buf"):
            return
        self.phase_length_buf[env_ids] = 0
        # Re-randomize gait_start (0 or 0.5) so reset envs don't all march in sync.
        n = int(env_ids.numel())
        self.gait_start[env_ids] = (torch.randint(0, 2, (n,), device=self.device).float()) * 0.5

    # ------------------------------------------------------------------
    # step override: refresh gait + feet metrics for next reward call
    # ------------------------------------------------------------------

    def step(self, action: torch.Tensor):
        out = super().step(action)
        self._refresh_gait_state()
        return out

    def _refresh_gait_state(self) -> None:
        # Advance the phase counter for envs that did NOT just reset (resets zeroed in _reset_idx).
        self.phase_length_buf = self.phase_length_buf + 1

        elapsed = self.phase_length_buf.float() * float(self.step_dt)
        phase = (elapsed % self.gait_cycle) / self.gait_cycle
        phase = (phase + self.gait_start) % 1.0
        self.gait_phase[:, 0] = phase
        self.gait_phase[:, 1] = (phase + 0.5) % 1.0

        robot = self.scene["robot"]
        contact = self.scene["contact_forces"]

        # Instantaneous foot vertical/lateral contact force magnitude (norm over xyz).
        # bxi uses an average over physics decimation; instantaneous is close enough and
        # avoids overriding the inner physics loop.
        forces = contact.data.net_forces_w[:, self.feet_sensor_ids, :3]
        self.avg_feet_force_per_step = torch.norm(forces, dim=-1)

        # Foot world-frame linear speed.
        feet_lin_vel = robot.data.body_lin_vel_w[:, self.feet_body_ids, :]
        self.avg_feet_speed_per_step = torch.norm(feet_lin_vel, dim=-1)
