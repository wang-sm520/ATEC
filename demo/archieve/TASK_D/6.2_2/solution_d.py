"""ATEC submission for Task D (Obstacle Traversal) — Route 1 hierarchical controller.

Reuses the SAME low-level policy (`policy.pt`) trained on Task A terrain: it is a
velocity-command-tracking walker. For Task D we keep that low-level walker untouched and
add a thin HIGH-LEVEL controller that synthesizes the velocity command `(vx, vy, wz)` each
frame from a phase state machine, driving the robot down the +x corridor toward the
pit/platform obstacle.

This is the v1 "validate the loop + reach the obstacle" version:
  - localization = dead-reckoning (integrate base_lin_vel into a world-x estimate),
    seeded with the known spawn x and the known course landmarks (from
    `tasks/task_d/env_cfg.py` / `terrain.py`);
  - all phases currently just walk straight forward at tuned speeds (no box-push / no
    pit-clearing behavior yet) — the goal is to confirm obs alignment + scoring loop and
    to OBSERVE how the Task-A walker behaves when it reaches a 0.9-1.0 m pit / platform.

Course landmarks (robot working frame, +x = forward), G1 config:
    spawn x = -3.0 ; +2 line at x=-1.4 ; +20 line at x=2.0 ; target/terminate at x=3.5 ;
    pit ~0.9-1.0 m wide x 1.0 m deep ; platform ~0.9-1.0 m tall on the +y side ;
    box (8 kg, 0.8x1.0x0.6) starts at (x=-3, y=1.6).

Everything below the high-level controller (term-major obs layout, heading P-controller,
per-joint action-scale compensation) is reproduced UNCHANGED from the Task A solution —
it is bound to `policy.pt` and must match training exactly.

Critical layout note:
    The training env uses IsaacLab's `ObservationGroupCfg(history_length=10,
    flatten_history_dim=True)`, which produces a TERM-MAJOR concatenation:
        [ang_vel_t0..t9, cmd_t0..t9, gravity_t0..t9, jp_t0..t9, jv_t0..t9, lastact_t0..t9]
    not frame-major. We must reproduce that exact layout at inference, or the policy's
    first Linear layer sees scrambled input and outputs garbage.

Heading-control note:
    Training uses `heading_command=True, heading_control_stiffness=0.5`, so the env
    feeds `cmd[2] = clip(0.5 * wrap_to_pi(target_h - heading_w), -1.57, 1.57)` to the
    policy each frame. We reproduce that loop here with target_h=0 (= initial +x corridor).
"""

import math
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

    # ================= Task D high-level controller =================================
    # Course landmarks in the robot working frame (+x = forward). Values come from
    # tasks/task_d/env_cfg.py (TaskDEnvG1Cfg: spawn x=-3) and the RewardCrossX /
    # x_reached thresholds.
    X_START = -3.0              # robot spawn x
    X_OBSTACLE_ENTER = -1.4     # RewardCrossX low threshold  -> +2  ("reached")
    X_OBSTACLE_EXIT = 2.0       # RewardCrossX high threshold -> +20 ("traversed")
    X_TARGET = 3.5              # robot_x_greater_than -> x_reached termination

    # Per-phase forward speed command (body-x, m/s). v1: all phases walk straight; the
    # traverse phase slows down so we can watch what happens at the pit edge.
    SPEED_APPROACH = 1.2        # cruise toward the obstacle
    SPEED_TRAVERSE = 0.8        # slow & careful through the pit/platform zone
    SPEED_FINISH = 1.0          # push on to the target line

    # Phase labels
    PHASE_APPROACH = "approach"
    PHASE_TRAVERSE = "traverse"
    PHASE_FINISH = "finish"

    # Score at/above which we stop spending wall-clock (Task D max = 36).
    GIVEUP_SCORE = 35.0

    # Set True for local play/debug to print phase transitions + x estimate.
    VERBOSE = False
    # ================================================================================

    # ---- Heading P-controller (mirrors training-time heading_command=True) -------------
    HEADING_TARGET = 0.0           # target heading in world frame, 0 = +x along corridor
    HEADING_STIFFNESS = 0.5        # = rough_env_cfg.heading_control_stiffness
    ANG_VEL_CMD_CLIP = 1.57        # = training ang_vel_z command bound
    STEP_DT = 0.02                 # = env.step_dt (4 decimation × 0.005 physics)

    # ---- action_scale compensation -----------------------------------------------------
    # Policy was trained with per-joint action_scale (bxi-style, mapped to G1):
    # values below match `G1_PER_JOINT_ACTION_SCALE` in rough_env_cfg.py, expanded in
    # G1_BODY_29_JOINT_NAMES order. ATEC eval env uses uniform scale=0.5 (see
    # `tasks/task_base/envs_base_cfg.py`). To make the joint deltas match training,
    # multiply the policy output by (TRAIN_SCALE / EVAL_SCALE) before returning.
    TRAINING_ACTION_SCALE_29 = (
        0.231, 0.231, 0.231, 0.231, 0.213, 0.213,  # left  leg: hip_p/r/y, knee, ank_p/r
        0.231, 0.231, 0.231, 0.231, 0.213, 0.213,  # right leg: same
        0.154, 0.213, 0.213,                       # waist: yaw, roll, pitch
        0.373, 0.373, 0.213, 0.373,                # left  arm: sh_p/r/y, elbow
        0.23,  0.23,  0.23,                        # left  wrist: roll, pitch, yaw
        0.373, 0.373, 0.213, 0.373,                # right arm: sh_p/r/y, elbow
        0.23,  0.23,  0.23,                        # right wrist: roll, pitch, yaw
    )
    EVAL_ACTION_SCALE = 0.5

    def __init__(self):
        policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.pt")
        self.device = "cuda"

        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        # Precompute per-joint action-scale compensation: divide policy output by this so
        # that, after the eval env multiplies by EVAL_ACTION_SCALE, the joint delta matches
        # what the policy saw during training.
        self.action_scale_ratio = torch.tensor(
            [s / self.EVAL_ACTION_SCALE for s in self.TRAINING_ACTION_SCALE_29],
            device=self.device,
            dtype=torch.float32,
        ).view(1, self.ACTION_DIM_BODY)

        # Per-term ring buffers, shape (HISTORY_LEN, dim). Index 0 = oldest, -1 = newest.
        # IsaacLab's CircularBuffer.reset() zeroes these out, so we start with zeros too.
        self._buf_ang_vel = self._make_buffer(self.DIM_ANG_VEL)
        self._buf_cmd = self._make_buffer(self.DIM_CMD)
        self._buf_gravity = self._make_buffer(self.DIM_GRAVITY)
        self._buf_jp = self._make_buffer(self.DIM_JP)
        self._buf_jv = self._make_buffer(self.DIM_JV)
        self._buf_lastact = self._make_buffer(self.DIM_LASTACT)

        # Integrated yaw estimate (rad) since last reset; driven by base_ang_vel (de-tilted).
        self.heading_est = 0.0
        # Dead-reckoned world-x estimate (m); seeded at the known spawn x.
        self.x_est = self.X_START
        self._phase = self.PHASE_APPROACH

    def _make_buffer(self, dim: int) -> torch.Tensor:
        return torch.zeros((self.HISTORY_LEN, dim), device=self.device, dtype=torch.float32)

    @staticmethod
    def _push(buf: torch.Tensor, new_row: torch.Tensor) -> None:
        """Shift left by one (drop oldest at index 0), append new at index -1, in-place."""
        buf[:-1] = buf[1:].clone()
        buf[-1] = new_row.view(-1)

    def reset(self, **kwargs):
        """Called by server.py /reset; zero all term history buffers + odometry state."""
        for buf in (
            self._buf_ang_vel,
            self._buf_cmd,
            self._buf_gravity,
            self._buf_jp,
            self._buf_jv,
            self._buf_lastact,
        ):
            buf.zero_()
        self.heading_est = 0.0
        self.x_est = self.X_START
        self._phase = self.PHASE_APPROACH

    # ---- Task D high-level policy: x estimate -> phase -> forward speed -----------------
    def _select_phase_speed(self, x_est: float) -> tuple[str, float]:
        if x_est < self.X_OBSTACLE_ENTER:
            return self.PHASE_APPROACH, self.SPEED_APPROACH
        if x_est < self.X_OBSTACLE_EXIT:
            return self.PHASE_TRAVERSE, self.SPEED_TRAVERSE
        return self.PHASE_FINISH, self.SPEED_FINISH

    def predicts(self, obs, current_score):
        if current_score > self.GIVEUP_SCORE:
            return {"action": [], "giveup": True}

        proprio = obs["proprio"].to(self.device, dtype=torch.float32)
        full_action_dim = (int(proprio.shape[-1]) - 12) // 3

        # ATEC proprio layout:
        #   [base_lin_vel(3), base_ang_vel(3), velocity_commands(3), projected_gravity(3),
        #    joint_pos(N), joint_vel(N), last_action(N)]
        base_lin_vel = proprio[0, 0:3]
        base_ang_vel = proprio[0, 3:6]
        projected_gravity = proprio[0, 9:12]

        # Heading P-controller — reproduces training-time `heading_command` mechanism.
        # projected_gravity points to world "down"; negating + normalizing gives the world
        # vertical ("up") axis expressed in the body frame. Projecting body angular velocity
        # onto it yields the true world yaw rate, robust to torso pitch/roll on rough ground.
        up_body = -projected_gravity / (projected_gravity.norm() + 1e-6)
        yaw_rate_world = torch.dot(base_ang_vel, up_body)
        self.heading_est += float(yaw_rate_world.item()) * self.STEP_DT
        yaw_err = self.HEADING_TARGET - self.heading_est
        yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
        ang_vel_z_cmd = max(
            -self.ANG_VEL_CMD_CLIP,
            min(self.ANG_VEL_CMD_CLIP, self.HEADING_STIFFNESS * yaw_err),
        )

        # Dead-reckoning: rotate body linear velocity by the heading estimate to get the
        # world +x component, then integrate. Seeded at spawn x; drifts over time, but good
        # enough to switch phases against the known course landmarks for this v1.
        cos_h = math.cos(self.heading_est)
        sin_h = math.sin(self.heading_est)
        world_vx = float(base_lin_vel[0].item()) * cos_h - float(base_lin_vel[1].item()) * sin_h
        self.x_est += world_vx * self.STEP_DT

        # High-level policy: pick phase + forward speed from the x estimate.
        phase, fwd_speed = self._select_phase_speed(self.x_est)
        if self.VERBOSE and phase != self._phase:
            print(f"[TaskD] phase {self._phase} -> {phase}  x_est={self.x_est:.2f}  score={current_score}")
        self._phase = phase

        velocity_commands = torch.tensor(
            [fwd_speed, 0.0, ang_vel_z_cmd],
            device=self.device,
            dtype=torch.float32,
        )

        jp_start = 12
        jv_start = jp_start + full_action_dim
        act_start = jv_start + full_action_dim

        joint_pos_body = proprio[0, jp_start:jp_start + self.ACTION_DIM_BODY]
        joint_vel_body = proprio[0, jv_start:jv_start + self.ACTION_DIM_BODY]
        # ATEC env stores the ACTION WE SENT (which is `policy_out * action_scale_ratio`).
        # Policy was trained with `last_action == raw policy_out`, so undo the compensation
        # to give the actor obs the same magnitude as during training.
        last_action_body = proprio[0, act_start:act_start + self.ACTION_DIM_BODY] / self.action_scale_ratio.view(-1)

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

        # Compensate per-joint action_scale (training dict vs eval uniform 0.5).
        # See class-level TRAINING_ACTION_SCALE_29 / EVAL_ACTION_SCALE notes.
        action_body = action_body * self.action_scale_ratio

        action_full = torch.zeros(
            (1, full_action_dim), device=self.device, dtype=torch.float32
        )
        action_full[:, self.BODY_29_IDX] = action_body
        return {"action": action_full[0].cpu().numpy().tolist(), "giveup": False}
