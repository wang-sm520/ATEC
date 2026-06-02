"""ATEC submission for Task D (Obstacle Traversal) — Route 1 hierarchical controller.

Reuses the SAME low-level policy (`policy.pt`) trained on Task A terrain: it is a
velocity-command-tracking walker. For Task D we keep that low-level walker untouched and
add a thin HIGH-LEVEL controller that synthesizes the velocity command `(vx, vy, wz)` each
frame from a phase state machine, driving the robot down the +x corridor toward the
pit/platform obstacle.

This is the open-loop "validate physical feasibility" version:
  - localization = 2D dead-reckoning (rotate base_lin_vel by the heading estimate into
    the world frame and integrate into pos_est=(x, y)), seeded at the known spawn pose;
  - the controller runs a hand-authored SEGMENTS list of TURN-then-GO legs: each leg
    first rotates in place to a new heading (relative turn_deg), then walks straight
    forward a target distance. We steer ONLY by heading and walk forward — never strafe.
  - purpose: script a box-push-then-cross maneuver out of turn+straight legs and hand-tune
    each leg against the GUI, to learn whether the frozen Task-A walker can PHYSICALLY do it.

WARNING — out-of-distribution commands:
    The walker was trained with lin_vel_x in (1.0, 2.0) (FORWARD ONLY, never 0 or negative).
    Forward GO legs are in-distribution; turning uses the trained heading mechanism. Turning
    IN PLACE (vx=0) and BACKWARD legs (vx<0, negative distance) are OOD — both worked in sim
    here, but if the spin is unstable set TURN_FWD_SPEED to a small positive value (gentle
    arc), and keep BACK_SPEED modest. Test a short SEGMENTS list first.

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

    # ================= Task D open-loop maneuver script =============================
    # Spawn pose (world frame, num_envs=1): robot (-3, 0, 0.8); box (-3, 1.6, 0.5).
    # RewardCrossX robot-x thresholds: -1.4 -> +2, 2.0 -> +20; x_reached at 3.5.
    # Box-in-pit target x in [-1.4, 0.7] -> +14.
    X_START = -3.0
    Y_START = 0.0

    # Hand-authored maneuver as a list of TURN-then-GO legs, executed in order.
    # Each leg: (label, turn_deg, distance_m)
    #   turn_deg   = relative turn applied at the START of the leg (degrees, + = left/CCW),
    #                added to the running heading target. The robot first rotates in place to
    #                the new heading, THEN walks straight forward.
    #   distance_m = forward walk after the turn completes. 0 = pure turn. NEGATIVE = walk
    #                straight BACKWARD |distance| m (same heading, vx<0, speed BACK_SPEED).
    # We steer ONLY by heading (vy is never commanded) — no strafing. The LAST leg walks
    # forward indefinitely (its distance only caps logging). Tune against the GUI.
    # Heading convention: 0 = +x (forward), +90 = +y (left), 180 = -x (back), -90 = -y (right).
    # Maneuver = back straight up to clear behind the box, sidestep to the box's y via a
    # turn+walk, face +x behind the box, then push it +x into the pit and walk across.
    # Expected pose after each leg is noted; tune turn_deg / dist vs the logged pos / yaw.
    # 【整套动作序列】每段 = (名字, 相对转角°, 直行距离m)。正转角=左转/逆时针；
    #   距离>0=前进，<0=后退，=0=只转不走。每段"先转到位再直行"，所以一次转向+紧跟的
    #   前进合并成一段。最后一段会一直前进（不会自动停）。距离均为占位、待按 GUI 调。
    SEGMENTS = [
        # (名字,           转角°,  距离m)   # 对应步骤 / 转完朝向 / 动作
        ("back",            0.0,  -1.3),   # 步1：不转后退                 朝 +x   后退
        ("left_fwd",       90.0,   1.5),   # 步2：左转90 + 前进            朝 +y   走
        ("to_box_left",   -90.0,   0.1),   # 步3：右转90 + 往前一点到箱左   朝 +x   走一点
        ("push_y",        -90.0,   2.5),   # 步4：右转90 + 前进推箱        朝 -y   推箱
        ("right_fwd",     -90.0,   0.1),   # 步5：右转90 + 前进一点        朝 -x   走一点
        ("to_box_back",    100.0,   0.0),   # 步6：左转90 + 前进到箱子后面   朝 -y   走
        ("push_into_pit",  90.0,   2.5),   # 步7：左转90 + 推箱入沟        朝 +x   推箱（末段，无限）
    ]

    # Forward speed (m/s) while walking a GO leg (in-distribution: trained on 1.0-2.0).
    GO_SPEED = 1.5          # 【前进速度】直行前进段的 vx（distance>0 时用）
    # Backward speed (m/s, magnitude) for legs with a NEGATIVE distance (vx<0, same heading).
    BACK_SPEED = 0.5        # 【后退速度】distance<0 的段用，vx = -BACK_SPEED
    # Forward speed while TURNING. This walker CANNOT spin in place (vx=0 stalls part-way),
    # so we turn on an ARC. Lower = tighter arc but risks stalling; if a turn plateaus before
    # reaching its target heading, raise this back toward 1.0.
    TURN_FWD_SPEED = 0.5    # 【转向时前进速度】调小=弧更紧但可能卡死；调大=弧更大更稳
    # Heading tolerance (deg) for declaring a turn complete and switching to the GO sub-phase.
    YAW_TOL_DEG = 5.0       # 【转向到位判据】yaw 误差小于此角度算转完，切到直行

    # Score at/above which we stop spending wall-clock (Task D max = 36).
    GIVEUP_SCORE = 35.0

    # Set True for local play/debug to print per-leg progress (label, leg dist, pos, z).
    VERBOSE = True
    # ================================================================================

    # ---- Heading P-controller (mirrors training-time heading_command=True) -------------
    HEADING_TARGET = 0.0           # 【初始目标朝向】0 = 朝 +x（走廊前进方向）
    HEADING_STIFFNESS = 1.5        # 【转向快慢/弧大小】越大转得越快、弧越紧（从 0.5 提到 1.5）
    ANG_VEL_CMD_CLIP = 1.57        # 【转向角速度上限】= 训练时 ang_vel_z 命令范围 ±1.57
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
        # 2D dead-reckoned world position (m); seeded at the known spawn pose.
        self.x_est = self.X_START
        self.y_est = self.Y_START
        # Turn-then-go script state: running heading target, leg cursor, per-leg forward
        # distance, and the sub-phase ("turn" -> "go"). _enter_segment seeds leg 0's turn.
        self._heading_target = self.HEADING_TARGET
        self._seg_idx = 0
        self._seg_dist = 0.0
        self._seg_phase = "turn"
        self._enter_segment(0)

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
        self.y_est = self.Y_START
        self._heading_target = self.HEADING_TARGET
        self._seg_idx = 0
        self._seg_dist = 0.0
        self._seg_phase = "turn"
        self._enter_segment(0)

    # ---- Task D turn-then-go script ----------------------------------------------------
    def _enter_segment(self, idx: int) -> None:
        """Begin leg `idx`: apply its relative turn to the running heading target, reset the
        per-leg forward-distance accumulator, and start in the TURN sub-phase."""
        turn_deg = self.SEGMENTS[idx][1]
        self._heading_target += math.radians(turn_deg)   # 【设转向目标】把本段转角累加到目标朝向
        self._seg_dist = 0.0                             # 本段已走距离清零
        self._seg_phase = "turn"                         # 每段先转向，再直行

    def _step_script(self, yaw_err: float, forward_step: float) -> tuple[str, float]:
        """Advance the script one control step. Returns (label, vx_cmd) for the active leg;
        vy is always 0 (we steer purely by heading).

        TURN sub-phase: spin toward the leg's heading until |yaw_err| <= YAW_TOL, then switch
        to GO. GO sub-phase: walk forward, accumulating distance; once it reaches the leg's
        target, advance to the next leg (the last leg keeps walking forever)."""
        idx = min(self._seg_idx, len(self.SEGMENTS) - 1)
        label, _turn_deg, distance = self.SEGMENTS[idx]

        # 【转向子相位】转到目标朝向；vx 用 TURN_FWD_SPEED（边走边转的弧线）
        if self._seg_phase == "turn":
            if abs(yaw_err) <= math.radians(self.YAW_TOL_DEG):
                self._seg_phase = "go"          # 转到位 -> 开始直行
            else:
                return label, self.TURN_FWD_SPEED   # 还没转到位，继续转（vx=转向速度）

        # 【直行子相位】distance>=0 前进(GO_SPEED)，<0 后退(BACK_SPEED)；按 |distance| 累计进度
        direction = 1.0 if distance >= 0.0 else -1.0
        speed = self.GO_SPEED if direction > 0.0 else self.BACK_SPEED   # 选前进/后退速度
        self._seg_dist += direction * forward_step       # 累计本段已走距离
        if self._seg_dist >= abs(distance) and self._seg_idx < len(self.SEGMENTS) - 1:
            self._seg_idx += 1                            # 走够 -> 切下一段
            self._enter_segment(self._seg_idx)
        return label, direction * speed                   # 返回本段 vx（带正负方向）

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

        # Heading P-controller — reproduces training-time `heading_command` mechanism, but
        # tracking the SCRIPT's current target heading (set per leg by _enter_segment).
        # projected_gravity points to world "down"; negating + normalizing gives the world
        # vertical ("up") axis expressed in the body frame. Projecting body angular velocity
        # onto it yields the true world yaw rate, robust to torso pitch/roll on rough ground.
        up_body = -projected_gravity / (projected_gravity.norm() + 1e-6)
        yaw_rate_world = torch.dot(base_ang_vel, up_body)
        self.heading_est += float(yaw_rate_world.item()) * self.STEP_DT
        yaw_err = self._heading_target - self.heading_est   # 【转向误差】当前朝向离目标差多少
        yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
        # 【转向指令 wz】P 控制器：误差×刚度，封顶 ±ANG_VEL_CMD_CLIP
        ang_vel_z_cmd = max(
            -self.ANG_VEL_CMD_CLIP,
            min(self.ANG_VEL_CMD_CLIP, self.HEADING_STIFFNESS * yaw_err),
        )

        # 2D dead-reckoning (logging only). base_lin_vel is body-frame; +x is forward.
        bvx = float(base_lin_vel[0].item())
        bvy = float(base_lin_vel[1].item())
        cos_h = math.cos(self.heading_est)
        sin_h = math.sin(self.heading_est)
        self.x_est += (bvx * cos_h - bvy * sin_h) * self.STEP_DT
        self.y_est += (bvx * sin_h + bvy * cos_h) * self.STEP_DT

        # Turn-then-go script. Forward progress is measured along body +x (the walk axis).
        # Returns the leg's forward speed; vy stays 0 (we steer purely by heading).
        prev_seg_idx = self._seg_idx
        prev_phase = self._seg_phase
        seg_label, seg_vx = self._step_script(yaw_err, bvx * self.STEP_DT)   # 【vx】当前段的前进/后退/转向速度
        seg_vy = 0.0                                                          # 【vy】永远 0：不横移

        if self.VERBOSE:
            self._tick = getattr(self, "_tick", 0) + 1
            if self._seg_idx != prev_seg_idx or self._seg_phase != prev_phase or self._tick % 25 == 0:
                print(
                    f"[TaskD] leg[{prev_seg_idx}] {seg_label:>13s} {self._seg_phase:>4s} "
                    f"yaw={math.degrees(self.heading_est):6.1f}->{math.degrees(self._heading_target):6.1f} "
                    f"seg_d={self._seg_dist:5.2f} pos=({self.x_est:6.2f},{self.y_est:6.2f}) "
                    f"score={current_score:.1f}"
                )

        # 【最终速度指令】喂给 policy.pt 的 [前进/后退, 横移=0, 转向]
        velocity_commands = torch.tensor(
            [seg_vx, seg_vy, ang_vel_z_cmd],
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
