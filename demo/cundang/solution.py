"""ATEC Task D submission — self-contained single-file solution.

Pushes the box into x in [-1.4, 0.7] (+14) and advances the robot past x=-1.4 (+2),
for 16 points, reusing the frozen G1 locomotion policy (policy.pt) as a velocity
tracker. NO atec_rl_lab imports: the eval container only ships this file + policy.pt.

Strategy (validated in sim, true score 16):
  warmup -> approach (get behind the box) -> push (drive +x until the box jams against
  the platform, ~x=-1.07) -> advance (sidestep to clear flat ground, cross x=-1.4).
Robot pose is dead-reckoned from the measured base velocity in obs['proprio'];
the box starts at the deterministic world pose (-3, 1.6).
"""

from __future__ import annotations

import math
import os
from typing import Any, Sequence

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
# The bridge expects the Task A AMP actor (960-dim term-major history). Locally it
# lives as policy_a.pt; in the submission container only policy.pt is shipped, so
# copy policy_a.pt -> policy.pt before building. policy_a.pt is preferred when both
# exist (demo's stock policy.pt is a DIFFERENT 1040-dim model and will not work).
_POLICY_PATH = None
for _name in ("policy_a.pt", "policy.pt"):
    _p = os.path.join(_DIR, _name)
    if os.path.exists(_p):
        _POLICY_PATH = _p
        break

# Stair-climbing actor (same 960-dim obs / 29-dim action signature as the walking
# policy). Engaged once the box is in the pit so the robot can climb over it toward
# the finish. The submission container must ALSO ship policy_climb.pt; if it is
# absent the solution degrades gracefully to walk-only.
_CLIMB_POLICY_PATH = None
for _name in ("policy_climb.pt",):
    _p = os.path.join(_DIR, _name)
    if os.path.exists(_p):
        _CLIMB_POLICY_PATH = _p
        break


# --------------------------------------------------------------------------- #
# Small math helpers
# --------------------------------------------------------------------------- #
def _wrap_to_pi(angle: float) -> float:
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if math.isclose(wrapped, -math.pi, abs_tol=1e-12) else wrapped


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _as_float(value: Any) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


# --------------------------------------------------------------------------- #
# Dead-reckoning odometry from proprioception
# --------------------------------------------------------------------------- #
class _Odometry:
    def __init__(self, dt: float = 0.02, x0: float = -3.0, y0: float = 0.0):
        self.dt, self.x0, self.y0 = dt, x0, y0
        self.reset()

    def reset(self) -> None:
        self.x, self.y, self.yaw = self.x0, self.y0, 0.0
        self.vx_b = self.vy_b = 0.0

    def update(self, proprio_row: Sequence[float]) -> tuple[float, float, float]:
        lin = [_as_float(proprio_row[i]) for i in range(0, 3)]
        ang = [_as_float(proprio_row[i]) for i in range(3, 6)]
        grav = [_as_float(proprio_row[i]) for i in range(9, 12)]
        up = self._normalized([-grav[0], -grav[1], -grav[2]])
        yaw_rate = sum(a * u for a, u in zip(ang, up))
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (c * lin[0] - s * lin[1]) * self.dt
        self.y += (s * lin[0] + c * lin[1]) * self.dt
        self.yaw = _wrap_to_pi(self.yaw + yaw_rate * self.dt)
        self.vx_b, self.vy_b = lin[0], lin[1]
        return self.x, self.y, self.yaw

    @staticmethod
    def _normalized(v):
        n = math.sqrt(sum(c * c for c in v))
        return [0.0, 0.0, 1.0] if n <= 1e-8 else [c / n for c in v]


# --------------------------------------------------------------------------- #
# Velocity-command -> joint action bridge around the frozen G1 locomotion policy
# --------------------------------------------------------------------------- #
class _G1VelocityPolicyBridge:
    ACTION_DIM_BODY = 29
    HISTORY_LEN = 10
    DIMS = (3, 3, 3, 29, 29, 29)  # ang_vel, cmd, gravity, jp, jv, last_act
    TRAINING_ACTION_SCALE_29 = (
        0.231, 0.231, 0.231, 0.231, 0.213, 0.213,
        0.231, 0.231, 0.231, 0.231, 0.213, 0.213,
        0.154, 0.213, 0.213,
        0.373, 0.373, 0.213, 0.373, 0.23, 0.23, 0.23,
        0.373, 0.373, 0.213, 0.373, 0.23, 0.23, 0.23,
    )
    EVAL_ACTION_SCALE = 0.5

    def __init__(self, policy_path: str, climb_path: str | None = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # Two interchangeable actors sharing the same obs/action signature and the
        # single proprio history buffer below: "walk" (push the box) and "climb"
        # (cross the box-filled pit). Switched one-way via select().
        self._paths = {"walk": policy_path, "climb": climb_path}
        self._modules = {"walk": None, "climb": None}
        self._active = "walk"
        self.action_scale_ratio = torch.tensor(
            [s / self.EVAL_ACTION_SCALE for s in self.TRAINING_ACTION_SCALE_29],
            device=self.device, dtype=torch.float32,
        ).view(1, self.ACTION_DIM_BODY)
        self.reset()

    def select(self, name: str) -> None:
        """Switch the active actor. No-op if that policy was not shipped, so the
        solution still runs (walk-only) when policy_climb.pt is absent."""
        if self._paths.get(name) is not None:
            self._active = name

    def reset(self) -> None:
        self._active = "walk"
        self._buf = [torch.zeros((self.HISTORY_LEN, d), device=self.device, dtype=torch.float32)
                     for d in self.DIMS]

    @staticmethod
    def _push(buf: torch.Tensor, row: torch.Tensor) -> None:
        buf[:-1] = buf[1:].clone()
        buf[-1] = row.reshape(-1)

    def _load(self):
        if self._modules[self._active] is None:
            self._modules[self._active] = torch.jit.load(
                self._paths[self._active], map_location=self.device).eval()
        return self._modules[self._active]

    def act(self, proprio: torch.Tensor, cmd: Sequence[float]) -> list[float]:
        proprio = proprio.to(device=self.device, dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        full_dim = (int(proprio.shape[-1]) - 12) // 3
        b = self.ACTION_DIM_BODY
        ang = proprio[0, 3:6]
        grav = proprio[0, 9:12]
        command = torch.tensor(tuple(cmd), device=self.device, dtype=torch.float32)
        jp = proprio[0, 12:12 + b]
        jv = proprio[0, 12 + full_dim:12 + full_dim + b]
        la = proprio[0, 12 + 2 * full_dim:12 + 2 * full_dim + b] / self.action_scale_ratio.reshape(-1)
        for buf, row in zip(self._buf, (ang, command, grav, jp, jv, la)):
            self._push(buf, row)
        policy_input = torch.cat([buf.reshape(-1) for buf in self._buf], dim=-1).unsqueeze(0)
        with torch.inference_mode():
            out = self._load()(policy_input)
        if not isinstance(out, torch.Tensor):
            out = torch.as_tensor(out, device=self.device, dtype=torch.float32)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        action_body = out.to(device=self.device, dtype=torch.float32)[:, :b] * self.action_scale_ratio
        action_full = torch.zeros((1, full_dim), device=self.device, dtype=torch.float32)
        action_full[:, :b] = action_body
        return action_full[0].detach().cpu().tolist()


# --------------------------------------------------------------------------- #
# Task D LiDAR height-scan detector for pit confirmation and climb alignment
# --------------------------------------------------------------------------- #
class _LidarClimbObservation:
    def __init__(
        self,
        valid: bool,
        may_have_bridge: bool,
        box_in_pit: bool,
        alignment_angle: float,
        alignment_vy: float,
        confidence: float,
        reason: str,
    ):
        self.valid = valid
        self.may_have_bridge = may_have_bridge
        self.box_in_pit = box_in_pit
        self.alignment_angle = alignment_angle
        self.alignment_vy = alignment_vy
        self.confidence = confidence
        self.reason = reason

    @classmethod
    def invalid(cls, reason: str) -> "_LidarClimbObservation":
        return cls(False, False, False, 0.0, 0.0, 0.0, reason)


class _TaskDLidarClimbDetector:
    """Detect whether the box is usable as a pit bridge from Task D extero.

    Task D exposes a flattened 16 x 360 height_scan. Values are not raw ranges:
    large positive values indicate lower ray hits such as the pit floor, while
    negative values indicate elevated hits such as box faces or box top.
    """

    CHANNELS = 16
    BINS = 360
    FLAT_LENGTH = CHANNELS * BINS
    FRONT_BIN = BINS // 2

    def __init__(
        self,
        front_half_width_bins: int = 35,
        pit_delta: float = 0.35,
        elevated_delta: float = 0.25,
        min_pit_bins: int = 5,
        min_top_bins: int = 4,
        loose_min_pit_bins: int = 4,
        loose_min_top_bins: int = 8,
    ):
        self.front_half_width_bins = front_half_width_bins
        self.pit_delta = pit_delta
        self.elevated_delta = elevated_delta
        self.min_pit_bins = min_pit_bins
        self.min_top_bins = min_top_bins
        self.loose_min_pit_bins = loose_min_pit_bins
        self.loose_min_top_bins = loose_min_top_bins

    def measure(self, extero: Any) -> _LidarClimbObservation:
        flat = tuple(self._flatten_numbers(extero))
        if not flat:
            return _LidarClimbObservation.invalid("missing extero")
        if len(flat) != self.FLAT_LENGTH:
            return _LidarClimbObservation.invalid(f"expected 5760 lidar values, got {len(flat)}")

        finite = [value for value in flat if math.isfinite(value)]
        if not finite:
            return _LidarClimbObservation.invalid("lidar scan has no finite values")

        rows = [flat[c * self.BINS : (c + 1) * self.BINS] for c in range(self.CHANNELS)]
        median = self._median(finite)
        pit_bins: list[int] = []
        top_bins: list[int] = []
        for bin_index in self._front_bins():
            column = [row[bin_index] for row in rows if math.isfinite(row[bin_index])]
            if not column:
                continue
            col_min = min(column)
            col_max = max(column)
            if col_max - median >= self.pit_delta:
                pit_bins.append(bin_index)
            if median - col_min >= self.elevated_delta:
                top_bins.append(bin_index)

        may_have_bridge = (
            len(pit_bins) >= self.loose_min_pit_bins
            or len(top_bins) >= self.loose_min_top_bins
        )
        box_in_pit = len(pit_bins) >= self.min_pit_bins and len(top_bins) >= self.min_top_bins
        if box_in_pit:
            center_bins = top_bins
            center_offset = sum(self._signed_bin_offset(b) for b in center_bins) / len(center_bins)
            alignment_angle = center_offset * (2.0 * math.pi / self.BINS)
            alignment_vy = _clamp(1.1 * math.sin(alignment_angle), -0.35, 0.35)
            confidence = min(1.0, 0.40 + 0.03 * min(len(pit_bins), 10) + 0.03 * min(len(top_bins), 10))
        else:
            alignment_angle = 0.0
            alignment_vy = 0.0
            confidence = 0.0

        reason = (
            f"median={median:.3f} pit_bins={len(pit_bins)} top_bins={len(top_bins)} "
            f"may_have_bridge={may_have_bridge} angle={alignment_angle:.3f} vy={alignment_vy:.3f}"
        )
        return _LidarClimbObservation(
            True, may_have_bridge, box_in_pit, alignment_angle, alignment_vy, confidence, reason
        )

    def _front_bins(self) -> list[int]:
        return [
            (self.FRONT_BIN + offset) % self.BINS
            for offset in range(-self.front_half_width_bins, self.front_half_width_bins + 1)
        ]

    def _signed_bin_offset(self, bin_index: int) -> int:
        return ((bin_index - self.FRONT_BIN + self.BINS // 2) % self.BINS) - self.BINS // 2

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return 0.5 * (ordered[middle - 1] + ordered[middle])

    @classmethod
    def _flatten_numbers(cls, value: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, dict):
            return cls._flatten_numbers(value.get("extero"))
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "reshape") and hasattr(value, "tolist"):
            try:
                return cls._flatten_numbers(value.reshape(-1).tolist())
            except TypeError:
                return cls._flatten_numbers(value.tolist())
        if hasattr(value, "tolist"):
            return cls._flatten_numbers(value.tolist())
        if isinstance(value, (str, bytes)):
            return []
        if isinstance(value, Sequence):
            flattened: list[float] = []
            for item in value:
                flattened.extend(cls._flatten_numbers(item))
            return flattened
        try:
            number = float(value)
        except (TypeError, ValueError):
            return []
        return [number]


# --------------------------------------------------------------------------- #
# Closed-loop pushing state machine
# --------------------------------------------------------------------------- #
class _WallPushController:
    """Push the box into the right-side pit using the wall as a fixture, then walk to the
    finish. Odometry only (no privileged state); box dead-reckoned from its known start
    via a contact model. Validated as an oracle in scripts/probe_task_d_wallpush.py.

      approach_back : get behind the box (-x), facing +x
      push_x        : push +x until the box jams on the platform wall (lock x_line)
      around_top    : route around the box to its +y side, facing -y (don't shove it)
      push_y        : push -y along a STRAIGHT line x=x_line (wall prevents fwd slip) until
                      the WHOLE box clears the wall (by <= -0.2)
      around_back2  : route behind the box (-x) in the pit lane, facing +x
      push_x_pit    : push +x until the box drops into the pit (robot reaches the pit edge)
      forward       : keep walking +x toward the finish (steps onto the box / pit; never stop)
    +14 (box) and +2 (x>-1.4 line) are collected en route.
    """
    BOX_START = (-3.0, 1.6)
    CONTACT_X = 0.55          # box centre ahead of robot during a +x push
    CONTACT_Y = 0.62          # box centre ahead of robot during a -y push (measured)
    PIT_EDGE_X = -0.55        # robot x at which the box (ahead) is over the pit
    CONFIRM_REQUIRED_FRAMES = 3
    CONFIRM_TIMEOUT_STEPS = 80
    ALIGN_REQUIRED_FRAMES = 10
    ALIGN_TIMEOUT_STEPS = 80
    ALIGN_ANGLE_TOL = 0.08
    MIN_REPUSH_STEPS = 30
    DOWN = -math.pi / 2

    def __init__(self, warmup_steps: int = 20, dt: float = 0.02):
        self.warmup_steps = warmup_steps
        self.odom = _Odometry(dt=dt)
        self.lidar = _TaskDLidarClimbDetector()
        self.reset()

    def reset(self) -> None:
        self.odom.reset()
        self.phase = "warmup"
        self.bx, self.by = self.BOX_START
        self.step = 0
        self.wpi = 0
        self.jam_x, self.jam_step = -1e9, 0
        self.x_line = None
        self.around_top = None
        self.around_back2 = None
        self.confirm_start_step = 0
        self.align_start_step = 0
        self.retry_confirm_after_step = 0
        self._lidar_confirm_count = 0
        self._align_stable_count = 0
        self._last_lidar = _LidarClimbObservation.invalid("lidar has not been measured")
        self.last_debug: dict[str, Any] = {}

    def update(self, proprio_row: Sequence[float], extero: Any = None) -> tuple[float, float, float]:
        rx, ry, ryaw = self.odom.update(proprio_row)
        self._last_lidar = self.lidar.measure(extero)
        self._update_box(rx, ry)
        self._update_phase(rx, ry, ryaw)
        cmd = self._control(rx, ry, ryaw)
        self.last_debug = {
            "phase": self.phase,
            "robot": (round(rx, 3), round(ry, 3), round(ryaw, 3)),
            "box_est": (round(self.bx, 3), round(self.by, 3)),
            "lidar_valid": self._last_lidar.valid,
            "lidar_box_in_pit": self._last_lidar.box_in_pit,
            "lidar_confidence": round(self._last_lidar.confidence, 3),
            "lidar_alignment_angle": round(self._last_lidar.alignment_angle, 3),
            "lidar_alignment_vy": round(self._last_lidar.alignment_vy, 3),
            "lidar_reason": self._last_lidar.reason,
            "cmd": tuple(round(v, 3) for v in cmd),
        }
        self.step += 1
        return cmd

    def _update_box(self, rx, ry):
        if self.phase in ("push_x", "push_x_pit"):
            self.bx = max(self.bx, rx + self.CONTACT_X)     # box pushed +x
        elif self.phase == "push_y":
            self.by = min(self.by, ry - self.CONTACT_Y)     # box pushed -y

    def _return_to_push_x_pit(self) -> None:
        self.phase = "push_x_pit"
        self.retry_confirm_after_step = self.step + self.MIN_REPUSH_STEPS
        self._lidar_confirm_count = 0
        self._align_stable_count = 0

    def _update_phase(self, rx, ry, ryaw):
        if self.phase == "warmup":
            if self.step >= self.warmup_steps:
                self.phase = "approach_back"
        elif self.phase == "approach_back":
            if rx < self.bx - 0.5 and abs(ry - self.by) < 0.13 and abs(_wrap_to_pi(ryaw)) < 0.13:
                self.phase, self.jam_x, self.jam_step = "push_x", rx, self.step
        elif self.phase == "push_x":
            if rx - self.jam_x > 0.06:
                self.jam_x, self.jam_step = rx, self.step
            elif self.step - self.jam_step > 120:           # box jammed on the wall
                self.x_line = self.bx
                self.around_top = [(self.bx - 1.5, self.by, self.DOWN),
                                   (self.bx - 1.5, self.by + 1.0, self.DOWN),
                                   (self.bx, self.by + 1.0, self.DOWN)]
                self.phase, self.wpi = "around_top", 0
        elif self.phase == "around_top":
            if self._reached(rx, ry, ryaw, self.around_top[self.wpi]):
                self.wpi += 1
                if self.wpi >= len(self.around_top):
                    self.phase = "push_y"
        elif self.phase == "push_y":
            if self.by <= -0.2:                             # whole box clears the wall
                self.around_back2 = [(self.x_line - 1.5, self.by, 0.0), (self.x_line - 0.75, self.by, 0.0)]
                self.phase, self.wpi = "around_back2", 0
        elif self.phase == "around_back2":
            if self._reached(rx, ry, ryaw, self.around_back2[self.wpi]):
                self.wpi += 1
                if self.wpi >= len(self.around_back2):
                    self.phase, self.jam_x, self.jam_step = "push_x_pit", rx, self.step
        elif self.phase == "push_x_pit":
            can_retry_confirm = self.step >= self.retry_confirm_after_step
            if rx >= self.PIT_EDGE_X and can_retry_confirm and self._last_lidar.may_have_bridge:
                self.phase = "confirm_pit_box"
                self.confirm_start_step = self.step
                self._lidar_confirm_count = 0
        elif self.phase == "confirm_pit_box":
            if self._last_lidar.box_in_pit and self._last_lidar.confidence >= 0.5:
                self._lidar_confirm_count += 1
            else:
                self._lidar_confirm_count = 0

            if self._lidar_confirm_count >= self.CONFIRM_REQUIRED_FRAMES:
                if abs(self._last_lidar.alignment_angle) > self.ALIGN_ANGLE_TOL:
                    self.phase = "align_climb"
                    self.align_start_step = self.step
                    self._align_stable_count = 0
                else:
                    self.phase = "forward"
            elif self.step - self.confirm_start_step >= self.CONFIRM_TIMEOUT_STEPS:
                self._return_to_push_x_pit()
        elif self.phase == "align_climb":
            if not (self._last_lidar.valid and self._last_lidar.box_in_pit):
                self._return_to_push_x_pit()
            elif abs(self._last_lidar.alignment_angle) <= self.ALIGN_ANGLE_TOL:
                self._align_stable_count += 1
                if self._align_stable_count >= self.ALIGN_REQUIRED_FRAMES:
                    self.phase = "forward"
            elif self.step - self.align_start_step >= self.ALIGN_TIMEOUT_STEPS:
                self._return_to_push_x_pit()
            else:
                self._align_stable_count = 0

    def _control(self, rx, ry, ryaw) -> tuple[float, float, float]:
        if self.phase == "warmup":
            return 0.0, 0.0, 0.0
        if self.phase == "approach_back":
            tx = self.bx - (0.90 if rx > self.bx - 0.55 else 0.70)
            return self._drive(rx, ry, ryaw, tx, self.by, 0.0, -0.5, 0.9, 0.6)
        if self.phase == "push_x":
            return 0.9, _clamp(1.2 * (self.by - ry), -0.4, 0.4), _clamp(-2.2 * ryaw, -0.8, 0.8)
        if self.phase == "around_top":
            return self._drive(rx, ry, ryaw, *self.around_top[self.wpi], -0.5, 0.8, 0.55)
        if self.phase == "push_y":
            # straight line at x=x_line, heading -y, strong lateral + heading hold
            vy = _clamp(2.0 * (self.x_line - rx), -0.45, 0.45)
            wz = _clamp(2.8 * _wrap_to_pi(self.DOWN - ryaw), -0.7, 0.7)
            return 0.85, vy, wz
        if self.phase == "around_back2":
            return self._drive(rx, ry, ryaw, *self.around_back2[self.wpi], -0.5, 0.8, 0.55)
        if self.phase == "push_x_pit":
            return 0.9, _clamp(1.2 * (self.by - ry), -0.4, 0.4), _clamp(-2.2 * ryaw, -0.8, 0.8)
        if self.phase == "confirm_pit_box":
            vy = self._last_lidar.alignment_vy if self._last_lidar.valid else 0.0
            return 0.0, _clamp(vy, -0.35, 0.35), _clamp(-2.2 * ryaw, -0.7, 0.7)
        if self.phase == "align_climb":
            vy = self._last_lidar.alignment_vy if self._last_lidar.valid else 0.0
            return 0.15, _clamp(vy, -0.35, 0.35), _clamp(-2.2 * ryaw, -0.6, 0.6)
        # forward — keep walking +x toward the finish (onto the box / across; never stop)
        return self._drive(rx, ry, ryaw, rx + 3.0, ry, 0.0, 0.3, 0.95, 0.4)

    @staticmethod
    def _reached(rx, ry, ryaw, wp, pos_tol=0.22, yaw_tol=0.3):
        return math.hypot(wp[0] - rx, wp[1] - ry) < pos_tol and abs(_wrap_to_pi(wp[2] - ryaw)) < yaw_tol

    @staticmethod
    def _drive(rx, ry, ryaw, tx, ty, tyaw, vx_lo, vx_hi, vy_abs):
        ex, ey = tx - rx, ty - ry
        c, s = math.cos(ryaw), math.sin(ryaw)
        return (
            _clamp(1.4 * (c * ex + s * ey), vx_lo, vx_hi),
            _clamp(1.4 * (-s * ex + c * ey), -vy_abs, vy_abs),
            _clamp(2.2 * _wrap_to_pi(tyaw - ryaw), -1.0, 1.0),
        )


# --------------------------------------------------------------------------- #
# ATEC entry point
# --------------------------------------------------------------------------- #
class AlgSolution:
    def __init__(self):
        self.bridge = _G1VelocityPolicyBridge(policy_path=_POLICY_PATH, climb_path=_CLIMB_POLICY_PATH)
        self.controller = _WallPushController()

    def reset(self, **kwargs) -> None:
        self.bridge.reset()
        self.controller.reset()

    def predicts(self, obs: dict, current_score: float):
        proprio = obs["proprio"]
        extero = obs.get("extero")
        row = proprio[0] if hasattr(proprio, "shape") and len(proprio.shape) >= 2 else proprio
        cmd = self.controller.update(row, extero)
        if self.controller.phase == "forward":
            self.bridge.select("climb")     # box confirmed/aligned or timeout fallback → climb to finish
        action = self.bridge.act(proprio, cmd)
        return {"action": action, "giveup": False}
