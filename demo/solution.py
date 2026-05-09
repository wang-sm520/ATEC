"""ATEC submission for G1 AMP policy trained on Task A terrain.

Critical layout note:
    The training env uses IsaacLab's `ObservationGroupCfg(history_length=10,
    flatten_history_dim=True)`, which produces a TERM-MAJOR concatenation:
        [ang_vel_t0..t9, cmd_t0..t9, gravity_t0..t9, jp_t0..t9, jv_t0..t9, lastact_t0..t9]
    not frame-major. We must reproduce that exact layout at inference, or the policy's
    first Linear layer sees scrambled input and outputs garbage.

Pipeline:
  proprio (1, 111)
    -> drop base_lin_vel; take 29-body subset of joints; override velocity_commands
    -> push current term values into 6 per-term ring buffers (shape (10, dim_term))
    -> concat each term's flat history -> 960-dim vector matching training term-major layout
    -> JIT actor (obs_normalizer + mlp baked in)
    -> 29-dim body action -> pad to full action_dim (extra slots = 0 = default fingers)
"""

import os
import torch


class AlgSolution:
    BODY_29_IDX = list(range(29))
    ACTION_DIM_BODY = 29

    HISTORY_LEN = 10
    # Per-term dims; matches G1AMPObservationsCfg.PolicyCfg
    DIM_ANG_VEL = 3
    DIM_CMD = 3
    DIM_GRAVITY = 3
    DIM_JP = 29
    DIM_JV = 29
    DIM_LASTACT = 29

    FIXED_VELOCITY_CMD = (0.4, 0.0, 0.0)

    def __init__(self):
        policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.pt")
        self.device = "cuda"

        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        self.fixed_cmd = torch.tensor(
            list(self.FIXED_VELOCITY_CMD), device=self.device, dtype=torch.float32
        ).view(1, 3)

        # Per-term ring buffers, shape (HISTORY_LEN, dim). Index 0 = oldest, -1 = newest.
        # IsaacLab's CircularBuffer.reset() zeroes these out, so we start with zeros too.
        self._buf_ang_vel = self._make_buffer(self.DIM_ANG_VEL)
        self._buf_cmd = self._make_buffer(self.DIM_CMD)
        self._buf_gravity = self._make_buffer(self.DIM_GRAVITY)
        self._buf_jp = self._make_buffer(self.DIM_JP)
        self._buf_jv = self._make_buffer(self.DIM_JV)
        self._buf_lastact = self._make_buffer(self.DIM_LASTACT)

    def _make_buffer(self, dim: int) -> torch.Tensor:
        return torch.zeros((self.HISTORY_LEN, dim), device=self.device, dtype=torch.float32)

    @staticmethod
    def _push(buf: torch.Tensor, new_row: torch.Tensor) -> None:
        """Shift left by one (drop oldest at index 0), append new at index -1, in-place."""
        buf[:-1] = buf[1:].clone()
        buf[-1] = new_row.view(-1)

    def reset(self, **kwargs):
        """Called by server.py /reset; zero all term history buffers."""
        for buf in (
            self._buf_ang_vel,
            self._buf_cmd,
            self._buf_gravity,
            self._buf_jp,
            self._buf_jv,
            self._buf_lastact,
        ):
            buf.zero_()

    def predicts(self, obs, current_score):
        if current_score > 1:
            return {"action": [], "giveup": True}

        proprio = obs["proprio"].to(self.device, dtype=torch.float32)
        full_action_dim = (int(proprio.shape[-1]) - 12) // 3

        # ATEC proprio layout:
        #   [base_lin_vel(3), base_ang_vel(3), velocity_commands(3), projected_gravity(3),
        #    joint_pos(N), joint_vel(N), last_action(N)]
        base_ang_vel = proprio[0, 3:6]
        velocity_commands = self.fixed_cmd.view(-1)
        projected_gravity = proprio[0, 9:12]

        jp_start = 12
        jv_start = jp_start + full_action_dim
        act_start = jv_start + full_action_dim

        joint_pos_body = proprio[0, jp_start:jp_start + self.ACTION_DIM_BODY]
        joint_vel_body = proprio[0, jv_start:jv_start + self.ACTION_DIM_BODY]
        last_action_body = proprio[0, act_start:act_start + self.ACTION_DIM_BODY]

        # Push each term into its dedicated ring buffer (oldest -> newest).
        self._push(self._buf_ang_vel, base_ang_vel)
        self._push(self._buf_cmd, velocity_commands)
        self._push(self._buf_gravity, projected_gravity)
        self._push(self._buf_jp, joint_pos_body)
        self._push(self._buf_jv, joint_vel_body)
        self._push(self._buf_lastact, last_action_body)

        # Term-major concat: each term's full history flattened (oldest..newest), then
        # all terms concatenated. This MUST match IsaacLab's ObsGroup output ordering.
        policy_input = torch.cat(
            [
                self._buf_ang_vel.reshape(-1),    # 30
                self._buf_cmd.reshape(-1),         # 30
                self._buf_gravity.reshape(-1),     # 30
                self._buf_jp.reshape(-1),          # 290
                self._buf_jv.reshape(-1),          # 290
                self._buf_lastact.reshape(-1),     # 290
            ],
            dim=-1,
        ).unsqueeze(0)  # (1, 960)

        with torch.inference_mode():
            action_body = self.policy(policy_input)

        if not isinstance(action_body, torch.Tensor):
            action_body = torch.as_tensor(action_body, device=self.device, dtype=torch.float32)
        if action_body.ndim == 1:
            action_body = action_body.unsqueeze(0)

        action_full = torch.zeros(
            (1, full_action_dim), device=self.device, dtype=torch.float32
        )
        action_full[:, self.BODY_29_IDX] = action_body
        return {"action": action_full[0].cpu().numpy().tolist(), "giveup": False}
