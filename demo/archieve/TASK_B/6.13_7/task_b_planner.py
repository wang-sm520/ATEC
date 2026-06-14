"""Unified Task B G1 planner FSM emitting whole-body-controller commands.

One state machine replaces the old two-layer split (legacy planner in
solution_task_b_g1.py + ad-hoc overrides in solution.py). Each step() returns a
complete WBCCommand (base velocity + height + waist + both hand poses) so the
caller (solution.py) becomes thin wiring.

Pure stdlib (math/dataclasses/typing): NO torch/numpy/onnx. The drive/sweep/creep
math and every constant are ported verbatim from the prior solution; see the
worktree CLAUDE.md and the score roadmap for provenance. The valid WBC command
ranges below are calibrated — do not "tune" them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    from task_b_nav import Pose2D, _clamp, _wrap_to_pi
    from task_b_perception import Detection
except ImportError:  # pragma: no cover - local dev path
    from demo.task_b_nav import Pose2D, _clamp, _wrap_to_pi
    from demo.task_b_perception import Detection


# Stow hand poses (xyz + wxyz quat, base frame). MUST equal mini_wbc.py
# DEFAULT_LEFT_HAND/DEFAULT_RIGHT_HAND (copied, not imported, to avoid onnxruntime).
STOW_LEFT_HAND = (0.3, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0)
STOW_RIGHT_HAND = (0.3, -0.2, 0.0, 1.0, 0.0, 0.0, 0.0)

# WBC valid command ranges (from mini/command_gui.py); commanding outside falls.
STAND_HEIGHT = 0.75
SQUAT_HEIGHT = 0.30        # only at base height 0.3 can hand z=-0.2 reach the floor
HAND_Z = -0.20
REACH_FWD_MIN, REACH_FWD_MAX = -0.10, 0.58
LEFT_LAT_MIN, LEFT_LAT_MAX = -0.10, 0.55
RIGHT_LAT_MIN, RIGHT_LAT_MAX = -0.55, 0.10
CREEP_VEL = 0.35
CREEP_STEPS = 10  # was 22; user-observed ~0.1m overshoot past the object.
                  # 10 steps at 0.35 m/s * 0.02s = 0.07m of creep: the sweep
                  # reaches forward (fwd -0.10..0.58) but NOT backward, so
                  # stopping short is strictly safer than overshooting.
HAND_SPREAD = 0.10
SWEEP_LAT_AMP = 0.14
SWEEP_FWD_AMP = 0.10
SWEEP_LAT_PERIOD = 36
SWEEP_FWD_PERIOD = 25
SEARCH_YAW = 1.3
SEARCH_RELOCATE_FWD = 0.55
# Search coverage geometry (iteration 2). vx=0.55 and wz=1.3 are proven-stable WBC
# velocities and are FIXED; only the durations are strategy. At dt=0.02s each step is
# 0.026 rad of yaw or 0.011m of travel. The head cam sees a 0.7-2.5m ground annulus per
# spin, so consecutive scan stops must be ~1.8m apart for their annuli to abut rather
# than overlap. The old cycle (185 rotate = 276deg, 75 relocate = 0.83m) under-rotated
# (left an unscanned azimuth wedge each stop) AND under-translated (annuli heavily
# overlapped), so the robot churned the spawn bubble and missed objects spread across
# the 10x10m arena. New cycle: a FULL 360deg spin at each stop (242 steps -> no azimuth
# gap) then a 1.8m leg (165 steps -> fresh, adjacent annulus). The forward camera strip
# during the longer relocation also scans the gap between annuli.
SEARCH_ROTATE_STEPS = 242     # 242 * 1.3 * 0.02 = 6.29 rad = 360deg full circle
SEARCH_RELOCATE_STEPS = 165   # 165 * 0.55 * 0.02 = 1.82m leg (~2.2x the old 0.83m)
APPROACH_SPEED_GAIN = 1.9
APPROACH_VX_MAX, APPROACH_VY_MAX, APPROACH_WZ_MAX = 0.6, 0.4, 1.6

# From the legacy planner (solution_task_b_g1.py).
SQUAT_SWEEP_MAX_STEPS = 220
SQUAT_DESCEND_STEPS = 15   # light ramp (~0.3s): soften violent 0.75->0.30 step without delaying reach too much
SQUAT_SETTLE_STEPS = 5     # short bottom hold before standing; damps sweep momentum with low score cost
STAND_RAMP_STEPS = 65      # was 50; modestly slower stand after observed stand-up fall
CONFIDENCE_FLOOR = 0.2

# NEW constants (the roadmap fixes).
ARRIVE_DIST = 0.35         # odometry ground distance that triggers creep
APPROACH_STANDOFF = 0.25   # drive-to point sits this far short of the object
BLIND_WALK_STEPS = 250     # max approach steps with no fresh sighting (5s @50Hz)
MAX_ATTEMPTS = 2           # creep/squat attempts per object before giving up

# Valid sighting band. The head camera is pitched ~47.6° down and is physically
# blind for ground objects closer than ~0.62m (they fall below the frame); the
# pinhole ground-projection is only reliable out to ~3m. A "detection" outside
# this band is noise or the robot's own body, NOT a real object: it must never be
# selected, remembered, or used to refresh the active target's coords (doing so is
# exactly what made the robot lock phantoms and squat on empty floor).
TARGET_MIN_DIST = 0.60     # below the head cam's ground-blind radius -> reject
TARGET_MAX_DIST = 1.6      # empirical visible-band top on isaaclab 0.54.4: the
                           # recalibrated 56.9-deg pitch puts the image top edge
                           # at ~1.48m ground distance (max truth-matched detection
                           # ever observed). Anything claiming to be farther is noise.
REFRESH_MIN_DIST = 0.50    # a <0.5m "match" is noise; ignore it for refresh

_STOW_QUAT = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class WBCCommand:
    vel: tuple[float, float, float]                 # base velocity (vx, vy, wz)
    base_height: float                              # WBC valid range [0.3, 0.9]
    waist_rpy: tuple[float, float, float]           # always (0.0, 0.0, 0.0) for now
    left_hand: tuple[float, float, float, float, float, float, float]   # xyz + wxyz quat
    right_hand: tuple[float, float, float, float, float, float, float]
    fingers: tuple[float, float, float, float] | None = None  # None = hold defaults; reserved for Phase-2 grasping, wired into MiniWBC in Task 3
    phase: str = ""                                 # for logging/tests
    target_world: tuple[float, float] | None = None # for logging/tests


class TaskBPlanner:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.phase = "search"
        self.prev_score = 0.0
        # Active target: (track_id, world_x, world_y) tuple (not a Detection).
        self.active: tuple[int, float, float] | None = None
        self.memory: dict[int, tuple[float, float]] = {}
        self.attempts: dict[int, int] = {}
        # Objects that produced a score increase while we were pursuing them.
        # ("scored", not "touched": Phase 2 adds real touch/grasp semantics.)
        self.scored: set[int] = set()
        # Per-phase counters.
        self.search_step = 0
        self.steps_since_seen = 0
        self.creep_step = 0
        self.squat_step = 0
        self.squat_subphase = "descend"
        self.sweep_step = 0
        self.settle_step = 0
        self.stand_step = 0
        self.stand_from_height = SQUAT_HEIGHT

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _in_sighting_band(distance: float) -> bool:
        """A detection is a real, trustworthy sighting only within the head cam's
        physically-visible ground band; outside it is noise/own-body."""
        return TARGET_MIN_DIST <= distance <= TARGET_MAX_DIST

    def _exhausted(self, track_id: int) -> bool:
        return track_id in self.scored or self.attempts.get(track_id, 0) >= MAX_ATTEMPTS

    def _charge_attempt(self, track_id: int) -> None:
        self.attempts[track_id] = self.attempts.get(track_id, 0) + 1

    def _select_target(self, pose: Pose2D, detections: list[Detection]) -> tuple[int, float, float] | None:
        """(a) nearest non-exhausted fresh detection with confidence>=floor, else
        (b) nearest non-exhausted remembered object, else None."""
        fresh = [
            d for d in detections
            if d.confidence >= CONFIDENCE_FLOOR
            and self._in_sighting_band(d.distance)
            and not self._exhausted(d.track_id)
        ]
        if fresh:
            best = min(fresh, key=lambda d: (d.distance, -d.confidence))
            return (best.track_id, best.world_x, best.world_y)
        remembered = [
            (tid, wx, wy) for tid, (wx, wy) in self.memory.items()
            if not self._exhausted(tid)
        ]
        if remembered:
            return min(remembered, key=lambda t: pose.distance_to((t[1], t[2])))
        return None

    def _enter_approach(self, target: tuple[int, float, float]) -> None:
        self.active = target
        self.phase = "approach"
        self.steps_since_seen = 0

    def _stow_command(self, vel: tuple[float, float, float], height: float, phase: str) -> WBCCommand:
        tw = None if self.active is None else (self.active[1], self.active[2])
        return WBCCommand(
            vel=vel,
            base_height=height,
            waist_rpy=(0.0, 0.0, 0.0),
            left_hand=STOW_LEFT_HAND,
            right_hand=STOW_RIGHT_HAND,
            phase=phase,
            target_world=tw,
        )

    # ------------------------------------------------------------ ported motion
    @staticmethod
    def _search_velocity(search_step: int) -> tuple[float, float, float]:
        cycle = search_step % (SEARCH_ROTATE_STEPS + SEARCH_RELOCATE_STEPS)
        if cycle < SEARCH_ROTATE_STEPS:
            return (0.0, 0.0, SEARCH_YAW)
        return (SEARCH_RELOCATE_FWD, 0.0, 0.0)

    @staticmethod
    def _drive_to(pose: Pose2D, tx: float, ty: float, tyaw: float, max_vx: float) -> tuple[float, float, float]:
        ex = tx - pose.x
        ey = ty - pose.y
        c = math.cos(pose.yaw)
        s = math.sin(pose.yaw)
        body_x = c * ex + s * ey
        body_y = -s * ex + c * ey
        yaw_err = _wrap_to_pi(tyaw - pose.yaw)
        vx = _clamp(0.9 * body_x, -0.18, max_vx)
        vy = _clamp(0.8 * body_y, -0.22, 0.22)
        wz = _clamp(1.8 * yaw_err, -0.7, 0.7)
        if abs(yaw_err) > 0.9:
            vx = min(vx, 0.05)
            vy = _clamp(vy, -0.08, 0.08)
        return vx, vy, wz

    @staticmethod
    def _scaled_approach_velocity(command) -> tuple[float, float, float]:
        vx, vy, wz = command
        return (
            _clamp(vx * APPROACH_SPEED_GAIN, -APPROACH_VX_MAX, APPROACH_VX_MAX),
            _clamp(vy * APPROACH_SPEED_GAIN, -APPROACH_VY_MAX, APPROACH_VY_MAX),
            _clamp(wz * 1.3, -APPROACH_WZ_MAX, APPROACH_WZ_MAX),
        )

    @staticmethod
    def _object_base_frame(pose: Pose2D, target_world: tuple[float, float]) -> tuple[float, float]:
        wx, wy = target_world
        dx, dy = wx - pose.x, wy - pose.y
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        return c * dx + s * dy, -s * dx + c * dy  # forward, left

    def _both_hands_sweep(self, pose: Pose2D, target_world: tuple[float, float], sweep_t: int):
        fwd0, lat0 = self._object_base_frame(pose, target_world)
        fwd = _clamp(fwd0 + SWEEP_FWD_AMP * math.sin(2.0 * math.pi * sweep_t / SWEEP_FWD_PERIOD),
                     REACH_FWD_MIN, REACH_FWD_MAX)
        lat_sweep = SWEEP_LAT_AMP * math.sin(2.0 * math.pi * sweep_t / SWEEP_LAT_PERIOD)
        left_center = _clamp(lat0 + HAND_SPREAD, LEFT_LAT_MIN, LEFT_LAT_MAX)
        right_center = _clamp(lat0 - HAND_SPREAD, RIGHT_LAT_MIN, RIGHT_LAT_MAX)
        left_lat = _clamp(left_center + lat_sweep, LEFT_LAT_MIN, LEFT_LAT_MAX)
        right_lat = _clamp(right_center + lat_sweep, RIGHT_LAT_MIN, RIGHT_LAT_MAX)
        left = (fwd, left_lat, HAND_Z) + _STOW_QUAT
        right = (fwd, right_lat, HAND_Z) + _STOW_QUAT
        return left, right

    @staticmethod
    def _creep_velocity(pose: Pose2D, target_world: tuple[float, float]) -> tuple[float, float, float]:
        wx, wy = target_world
        bearing = math.atan2(wy - pose.y, wx - pose.x)
        yaw_err = (bearing - pose.yaw + math.pi) % (2.0 * math.pi) - math.pi
        return (CREEP_VEL, 0.0, _clamp(1.5 * yaw_err, -0.5, 0.5))

    # ------------------------------------------------------------------- driver
    def step(self, pose: Pose2D, detections: list[Detection], current_score: float,
             posture: str = "ok") -> WBCCommand:
        # 1. score bookkeeping. Any score increase while pursuing a target claims
        # that target (in approach, creep, or squat): the delta is the only signal
        # the object was contacted, so it must not be dropped on the way through.
        score_delta = float(current_score) - self.prev_score
        self.prev_score = float(current_score)

        # 2. memory: record every confident detection's world coords (any phase).
        # Gated by the confidence floor so a low-confidence blip cannot leak into
        # memory and later be pursued via the memory-selection branch (the fresh
        # branch already gates on confidence; the memory branch does not).
        for d in detections:
            if d.confidence >= CONFIDENCE_FLOOR and self._in_sighting_band(d.distance):
                self.memory[d.track_id] = (d.world_x, d.world_y)

        # 3. global safety: a recover while not already standing up -> stand_up
        if posture == "recover" and self.phase != "stand_up":
            self._begin_stand_up(from_squat=(self.phase == "squat_sweep"))

        if self.phase == "search":
            return self._step_search(pose, detections, score_delta)
        if self.phase == "approach":
            return self._step_approach(pose, detections, score_delta)
        if self.phase == "creep":
            return self._step_creep(pose, score_delta)
        if self.phase == "squat_sweep":
            return self._step_squat(pose, score_delta)
        if self.phase == "stand_up":
            return self._step_stand(pose)
        # Defensive guard: an unknown phase is not a legal transition. Clear the
        # stale active target (so its target_world cannot leak into the emitted
        # command) and recover into search.
        self.phase = "search"
        self.active = None
        return self._step_search(pose, detections, score_delta)

    def _refresh_active(self, detections: list[Detection]) -> bool:
        """If a detection matches the active track_id, refresh coords + reset the
        blind counter. Returns True when refreshed. A matching detection closer than
        REFRESH_MIN_DIST is ignored (treated as unseen): the real object is invisible
        that close, so the "match" is noise that would corrupt the remembered coords."""
        if self.active is None:
            return False
        tid = self.active[0]
        for d in detections:
            if d.track_id == tid and d.distance >= REFRESH_MIN_DIST:
                self.active = (tid, d.world_x, d.world_y)
                self.steps_since_seen = 0
                return True
        return False

    # ------------------------------------------------------------- phase bodies
    def _step_search(self, pose: Pose2D, detections: list[Detection], score_delta: float) -> WBCCommand:
        target = self._select_target(pose, detections)
        if target is not None:
            self.search_step = 0
            self._enter_approach(target)
            return self._step_approach(pose, detections, score_delta)
        vel = self._search_velocity(self.search_step)
        self.search_step += 1
        return self._stow_command(vel, STAND_HEIGHT, "search")

    def _step_approach(self, pose: Pose2D, detections: list[Detection], score_delta: float) -> WBCCommand:
        assert self.active is not None
        # Score gained while still approaching means the robot kicked/early-touched
        # the object -> claim it and stand up (entry was not from a squat).
        if score_delta > 0.0:
            self.scored.add(self.active[0])
            self._begin_stand_up(from_squat=False)
            return self._step_stand(pose)
        if not self._refresh_active(detections):
            self.steps_since_seen += 1
            if self.steps_since_seen > BLIND_WALK_STEPS:
                self._charge_attempt(self.active[0])
                self.active = None
                self.phase = "search"
                return self._step_search(pose, detections, score_delta)
        tid, wx, wy = self.active
        target_xy = (wx, wy)
        if pose.distance_to(target_xy) <= ARRIVE_DIST:
            self._charge_attempt(tid)
            self.phase = "creep"
            self.creep_step = 0
            return self._step_creep(pose, score_delta)
        vel = self._approach_velocity(pose, self.active)
        return self._stow_command(vel, STAND_HEIGHT, "approach")

    def _step_creep(self, pose: Pose2D, score_delta: float) -> WBCCommand:
        assert self.active is not None
        if score_delta > 0.0:
            self.scored.add(self.active[0])
            self._begin_stand_up(from_squat=False)
            return self._step_stand(pose)
        self.creep_step += 1
        target_xy = (self.active[1], self.active[2])
        if self.creep_step > CREEP_STEPS:
            self.phase = "squat_sweep"
            self.squat_step = 0
            self.squat_subphase = "descend"
            self.sweep_step = 0
            self.settle_step = 0
            return self._step_squat(pose, score_delta)
        vel = self._creep_velocity(pose, target_xy)
        return self._stow_command(vel, STAND_HEIGHT, "creep")

    def _enter_settle(self) -> None:
        self.squat_subphase = "settle"
        self.settle_step = 0

    def _squat_stow_command(self, target_xy: tuple[float, float], height: float) -> WBCCommand:
        return WBCCommand(
            vel=(0.0, 0.0, 0.0),
            base_height=height,
            waist_rpy=(0.0, 0.0, 0.0),
            left_hand=STOW_LEFT_HAND,
            right_hand=STOW_RIGHT_HAND,
            phase="squat_sweep",
            target_world=target_xy,
        )

    def _step_squat(self, pose: Pose2D, score_delta: float) -> WBCCommand:
        assert self.active is not None
        tid, wx, wy = self.active
        target_xy = (wx, wy)

        if score_delta > 0.0:
            self.scored.add(tid)
            if self.squat_subphase != "settle":
                self._enter_settle()

        if self.squat_subphase == "descend":
            self.squat_step += 1
            progress = min(1.0, self.squat_step / SQUAT_DESCEND_STEPS)
            height = STAND_HEIGHT + progress * (SQUAT_HEIGHT - STAND_HEIGHT)
            if self.squat_step >= SQUAT_DESCEND_STEPS:
                self.squat_subphase = "sweep"
                self.sweep_step = 0
            return self._squat_stow_command(target_xy, height)

        if self.squat_subphase == "sweep":
            self.sweep_step += 1
            if self.sweep_step > SQUAT_SWEEP_MAX_STEPS:
                self._enter_settle()
                return self._squat_stow_command(target_xy, SQUAT_HEIGHT)
            left, right = self._both_hands_sweep(pose, target_xy, self.sweep_step)
            return WBCCommand(
                vel=(0.0, 0.0, 0.0),
                base_height=SQUAT_HEIGHT,
                waist_rpy=(0.0, 0.0, 0.0),
                left_hand=left,
                right_hand=right,
                phase="squat_sweep",
                target_world=target_xy,
            )

        self.settle_step += 1
        if self.settle_step >= SQUAT_SETTLE_STEPS:
            self._begin_stand_up(from_squat=True)
            return self._step_stand(pose)
        return self._squat_stow_command(target_xy, SQUAT_HEIGHT)

    def _begin_stand_up(self, from_squat: bool) -> None:
        self.phase = "stand_up"
        self.stand_step = 0
        self.stand_from_height = SQUAT_HEIGHT if from_squat else STAND_HEIGHT

    def _step_stand(self, pose: Pose2D) -> WBCCommand:
        self.stand_step += 1
        progress = min(1.0, self.stand_step / STAND_RAMP_STEPS)
        height = self.stand_from_height + progress * (STAND_HEIGHT - self.stand_from_height)
        if self.stand_step >= STAND_RAMP_STEPS:
            self.active = None
            target = self._select_target(pose, [])  # memory-only selection
            if target is not None:
                self._enter_approach(target)
                return self._stow_command(
                    self._approach_velocity(pose, target), STAND_HEIGHT, "approach"
                )
            self.phase = "search"
            self.search_step = 0
            return self._stow_command((0.0, 0.0, 0.0), STAND_HEIGHT, "search")
        return self._stow_command((0.0, 0.0, 0.0), height, "stand_up")

    def _approach_velocity(self, pose: Pose2D, target: tuple[int, float, float]) -> tuple[float, float, float]:
        _, wx, wy = target
        bearing = pose.bearing_to((wx, wy))
        sx = wx - APPROACH_STANDOFF * math.cos(bearing)
        sy = wy - APPROACH_STANDOFF * math.sin(bearing)
        return self._scaled_approach_velocity(self._drive_to(pose, sx, sy, bearing, 0.28))
