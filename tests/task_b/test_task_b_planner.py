"""Tests for the unified Task B planner FSM (demo/task_b_planner.py).

Pure stdlib (no torch); runs in the bare shell. Builds Detections via a small
`det()` helper that ground-projects a world target into a Detection relative to a
given pose, exactly as the real perception would for a fresh sighting.
"""

import math
import os
import sys
import unittest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from demo.task_b_nav import Pose2D  # noqa: E402
from demo.task_b_perception import Detection  # noqa: E402
from demo import task_b_planner as planner_mod  # noqa: E402
from demo.task_b_planner import TaskBPlanner, WBCCommand  # noqa: E402

P = planner_mod  # constants namespace


def det(track_id, wx, wy, pose, confidence=1.0):
    """Build a fresh Detection of world point (wx, wy) seen from `pose`."""
    dx, dy = wx - pose.x, wy - pose.y
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    rel_x = c * dx + s * dy
    rel_y = -s * dx + c * dy
    distance = math.hypot(rel_x, rel_y)
    return Detection(
        track_id=track_id,
        label="colored_object",
        rel_x=rel_x,
        rel_y=rel_y,
        distance=distance,
        confidence=confidence,
        world_x=wx,
        world_y=wy,
        bbox=(0, 0, 1, 1),
    )


class SearchPhaseTest(unittest.TestCase):
    def test_search_rotate_then_relocate_cycle(self):
        plr = TaskBPlanner()
        pose = Pose2D(-10.0, -10.0, 0.0)
        # First SEARCH_ROTATE_STEPS commands rotate in place at wz=SEARCH_YAW.
        for _ in range(P.SEARCH_ROTATE_STEPS):
            cmd = plr.step(pose, [], 0.0)
            self.assertEqual(cmd.phase, "search")
            self.assertAlmostEqual(cmd.vel[0], 0.0)
            self.assertAlmostEqual(cmd.vel[1], 0.0)
            self.assertAlmostEqual(cmd.vel[2], P.SEARCH_YAW)
            self.assertAlmostEqual(cmd.base_height, P.STAND_HEIGHT)
            self.assertEqual(cmd.left_hand, P.STOW_LEFT_HAND)
            self.assertEqual(cmd.right_hand, P.STOW_RIGHT_HAND)
        # Next SEARCH_RELOCATE_STEPS commands relocate forward.
        for _ in range(P.SEARCH_RELOCATE_STEPS):
            cmd = plr.step(pose, [], 0.0)
            self.assertAlmostEqual(cmd.vel[0], P.SEARCH_RELOCATE_FWD)
            self.assertAlmostEqual(cmd.vel[1], 0.0)
            self.assertAlmostEqual(cmd.vel[2], 0.0)
        # Cycles back to rotate.
        cmd = plr.step(pose, [], 0.0)
        self.assertAlmostEqual(cmd.vel[2], P.SEARCH_YAW)


class SelectionTest(unittest.TestCase):
    def test_fresh_detection_triggers_approach(self):
        plr = TaskBPlanner()
        pose = Pose2D(-10.0, -10.0, 0.0)
        d = det(1, -8.0, -10.0, pose)  # dead ahead, 2m
        cmd = plr.step(pose, [d], 0.0)
        self.assertEqual(cmd.phase, "approach")
        self.assertEqual(cmd.target_world, (-8.0, -10.0))
        self.assertGreater(cmd.vel[0], 0.0)  # drives forward toward standoff

    def test_low_confidence_never_selected(self):
        plr = TaskBPlanner()
        pose = Pose2D(-10.0, -10.0, 0.0)
        d = det(1, -8.0, -10.0, pose, confidence=0.1)
        cmd = plr.step(pose, [d], 0.0)
        self.assertEqual(cmd.phase, "search")


class BlindWalkTest(unittest.TestCase):
    def test_blind_walk_then_creep(self):
        plr = TaskBPlanner()
        target = (-8.0, -10.0)
        pose = Pose2D(-10.0, -10.0, 0.0)
        # Seen once.
        cmd = plr.step(pose, [det(1, *target, pose)], 0.0)
        self.assertEqual(cmd.phase, "approach")
        # 100 steps without any detection: still approach (legacy died at 30).
        for _ in range(100):
            cmd = plr.step(pose, [], 0.0)
            self.assertEqual(cmd.phase, "approach")
        # Now close in to within ARRIVE_DIST -> creep.
        near = Pose2D(target[0] - 0.3, target[1], 0.0)
        cmd = plr.step(near, [], 0.0)
        self.assertEqual(cmd.phase, "creep")

    def test_blind_give_up_charges_attempt(self):
        plr = TaskBPlanner()
        target = (-8.0, -10.0)
        pose = Pose2D(-10.0, -10.0, 0.0)
        plr.step(pose, [det(1, *target, pose)], 0.0)
        # No sighting and no arrival for > BLIND_WALK_STEPS charges one attempt
        # (the give-up). The same memory entry is then re-selected (attempt 1 < MAX),
        # so the FSM re-enters approach in the same step -- but the charge is the
        # observable give-up signal that bounds the loop.
        for _ in range(P.BLIND_WALK_STEPS + 1):
            plr.step(pose, [], 0.0)
        self.assertEqual(plr.attempts.get(1, 0), 1)

    def test_blind_give_up_eventually_exhausts_and_searches(self):
        plr = TaskBPlanner()
        target = (-8.0, -10.0)
        pose = Pose2D(-10.0, -10.0, 0.0)
        plr.step(pose, [det(1, *target, pose)], 0.0)
        # After MAX_ATTEMPTS blind give-ups the object is exhausted; with memory
        # empty of selectable targets the planner falls through to search.
        cmd = None
        for _ in range(P.MAX_ATTEMPTS * (P.BLIND_WALK_STEPS + 1) + 5):
            cmd = plr.step(pose, [], 0.0)
        self.assertGreaterEqual(plr.attempts.get(1, 0), P.MAX_ATTEMPTS)
        self.assertEqual(cmd.phase, "search")


class CreepTest(unittest.TestCase):
    def _drive_to_creep(self, plr, target):
        pose = Pose2D(target[0] - 0.3, target[1], 0.0)
        plr.step(pose, [det(1, *target, pose)], 0.0)  # approach, already within ARRIVE
        cmd = plr.step(pose, [], 0.0)
        self.assertEqual(cmd.phase, "creep")
        return pose

    def test_creep_lasts_exactly_creep_steps(self):
        plr = TaskBPlanner()
        target = (-8.0, -10.0)
        pose = self._drive_to_creep(plr, target)
        # _drive_to_creep already emitted 2 creep commands; CREEP_STEPS total,
        # then the next step transitions to squat_sweep.
        for _ in range(P.CREEP_STEPS - 2):
            cmd = plr.step(pose, [], 0.0)
            self.assertEqual(cmd.phase, "creep")
            self.assertAlmostEqual(cmd.vel[0], P.CREEP_VEL)
        cmd = plr.step(pose, [], 0.0)
        self.assertEqual(cmd.phase, "squat_sweep")

    def test_creep_aims_at_target(self):
        plr = TaskBPlanner()
        # Target within ARRIVE_DIST but offset to the robot's LEFT (yaw 0, +y is left).
        target = (-9.85, -9.75)  # dist ~0.29 < ARRIVE_DIST, bearing > 0
        pose = Pose2D(-10.0, -10.0, 0.0)
        plr.step(pose, [det(1, *target, pose)], 0.0)  # arrive -> creep
        cmd = plr.step(pose, [], 0.0)
        self.assertEqual(cmd.phase, "creep")
        self.assertAlmostEqual(cmd.vel[0], P.CREEP_VEL)
        # bearing to a left-offset target is positive -> wz aims left (> 0).
        self.assertGreater(cmd.vel[2], 0.0)
        self.assertIsNone(cmd.fingers)
        self.assertEqual(cmd.waist_rpy, (0.0, 0.0, 0.0))


class SquatSweepTest(unittest.TestCase):
    def _enter_squat(self, plr, target, pose):
        """pose MUST be within ARRIVE_DIST of target so approach -> creep -> squat."""
        plr.step(pose, [det(1, *target, pose)], 0.0)  # approach, arrives immediately
        for _ in range(P.CREEP_STEPS + 1):
            cmd = plr.step(pose, [], 0.0)
        self.assertEqual(cmd.phase, "squat_sweep")
        return cmd

    def test_squat_sweep_command_and_ranges(self):
        plr = TaskBPlanner()
        # Pose within ARRIVE_DIST but oriented so the target sits far to one side
        # in the base frame, pushing the lateral sweep against the clamps.
        target = (-9.0, -10.0)
        pose = Pose2D(-9.3, -10.0, math.pi / 2.0)  # target is 0.3m to the robot's right
        cmd = self._enter_squat(plr, target, pose)
        # Property: every squat_sweep command keeps every hand coordinate inside the
        # valid ranges. A single squat runs SQUAT_SWEEP_MAX_STEPS (220) steps, which
        # spans several full Lissajous periods (lat 36, fwd 25).
        checked = 0
        while cmd.phase == "squat_sweep":
            checked += 1
            self.assertAlmostEqual(cmd.base_height, P.SQUAT_HEIGHT)
            lx, ly, lz = cmd.left_hand[0], cmd.left_hand[1], cmd.left_hand[2]
            rx, ry, rz = cmd.right_hand[0], cmd.right_hand[1], cmd.right_hand[2]
            self.assertAlmostEqual(lz, P.HAND_Z)
            self.assertAlmostEqual(rz, P.HAND_Z)
            self.assertGreaterEqual(lx, P.REACH_FWD_MIN - 1e-9)
            self.assertLessEqual(lx, P.REACH_FWD_MAX + 1e-9)
            self.assertGreaterEqual(rx, P.REACH_FWD_MIN - 1e-9)
            self.assertLessEqual(rx, P.REACH_FWD_MAX + 1e-9)
            self.assertGreaterEqual(ly, P.LEFT_LAT_MIN - 1e-9)
            self.assertLessEqual(ly, P.LEFT_LAT_MAX + 1e-9)
            self.assertGreaterEqual(ry, P.RIGHT_LAT_MIN - 1e-9)
            self.assertLessEqual(ry, P.RIGHT_LAT_MAX + 1e-9)
            cmd = plr.step(pose, [], 0.0)
        self.assertGreaterEqual(checked, P.SQUAT_SWEEP_MAX_STEPS - 1)

    def test_both_hands_sweep_property_extreme_targets(self):
        """_both_hands_sweep keeps every coordinate inside the valid ranges and the
        quats fixed at identity, for 300 consecutive t values across extreme target
        geometries (far left/right/behind/ahead in the base frame)."""
        plr = TaskBPlanner()
        pose = Pose2D(-10.0, -10.0, 0.0)
        extremes = [
            (-10.0, -2.0),   # far to the robot's left (large +lat)
            (-10.0, -18.0),  # far to the robot's right (large -lat)
            (-18.0, -10.0),  # behind the robot (negative forward)
            (-2.0, -10.0),   # far ahead (large +forward)
            (-8.0, -10.0),   # dead ahead, modest
        ]
        for target in extremes:
            for t in range(1, 301):
                left, right = plr._both_hands_sweep(pose, target, t)
                lx, ly, lz = left[0], left[1], left[2]
                rx, ry, rz = right[0], right[1], right[2]
                self.assertGreaterEqual(lx, P.REACH_FWD_MIN - 1e-9, target)
                self.assertLessEqual(lx, P.REACH_FWD_MAX + 1e-9, target)
                self.assertGreaterEqual(rx, P.REACH_FWD_MIN - 1e-9, target)
                self.assertLessEqual(rx, P.REACH_FWD_MAX + 1e-9, target)
                self.assertGreaterEqual(ly, P.LEFT_LAT_MIN - 1e-9, target)
                self.assertLessEqual(ly, P.LEFT_LAT_MAX + 1e-9, target)
                self.assertGreaterEqual(ry, P.RIGHT_LAT_MIN - 1e-9, target)
                self.assertLessEqual(ry, P.RIGHT_LAT_MAX + 1e-9, target)
                self.assertAlmostEqual(lz, P.HAND_Z)
                self.assertAlmostEqual(rz, P.HAND_Z)
                self.assertEqual(left[3:], (1.0, 0.0, 0.0, 0.0))
                self.assertEqual(right[3:], (1.0, 0.0, 0.0, 0.0))

    def test_squat_sweep_straddle(self):
        plr = TaskBPlanner()
        target = (-8.0, -10.0)  # dead ahead, lat0 == 0
        pose = Pose2D(-8.3, -10.0, 0.0)  # within ARRIVE_DIST, target straight ahead
        cmd = self._enter_squat(plr, target, pose)
        # Step until the lateral dither is exactly zero (sweep_t a multiple of
        # SWEEP_LAT_PERIOD): then the hands straddle the target +/- HAND_SPREAD.
        while plr.sweep_step % P.SWEEP_LAT_PERIOD != 0:
            cmd = plr.step(pose, [], 0.0)
        # target lat0 == 0 (pose dead behind target), so left == +HAND_SPREAD.
        self.assertAlmostEqual(cmd.left_hand[1], P.HAND_SPREAD, delta=1e-6)
        self.assertAlmostEqual(cmd.right_hand[1], -P.HAND_SPREAD, delta=1e-6)

    def test_score_during_squat_marks_scored(self):
        plr = TaskBPlanner()
        target = (-8.0, -10.0)
        pose = Pose2D(-8.3, -10.0, 0.0)
        self._enter_squat(plr, target, pose)
        cmd = plr.step(pose, [], 1.0)  # score jumped
        self.assertEqual(cmd.phase, "stand_up")
        self.assertIn(1, plr.scored)
        # Scored object never re-selected, even from a fresh detection.
        for _ in range(P.STAND_RAMP_STEPS + 1):
            cmd = plr.step(pose, [det(1, *target, pose)], 1.0)
        self.assertNotEqual(cmd.target_world, target)
        self.assertEqual(cmd.phase, "search")

    def test_sweep_timeout_to_stand_up(self):
        plr = TaskBPlanner()
        target = (-8.0, -10.0)
        pose = Pose2D(-8.3, -10.0, 0.0)
        self._enter_squat(plr, target, pose)
        cmd = None
        for _ in range(P.SQUAT_SWEEP_MAX_STEPS + 1):
            cmd = plr.step(pose, [], 0.0)
        self.assertEqual(cmd.phase, "stand_up")


class StandUpTest(unittest.TestCase):
    def test_stand_ramp_then_search_when_memory_empty(self):
        plr = TaskBPlanner()
        target = (-8.0, -10.0)
        pose = Pose2D(-8.3, -10.0, 0.0)
        plr.step(pose, [det(1, *target, pose)], 0.0)
        for _ in range(P.CREEP_STEPS + 1):
            plr.step(pose, [], 0.0)
        # score to leave squat
        cmd = plr.step(pose, [], 1.0)
        self.assertEqual(cmd.phase, "stand_up")
        heights = [cmd.base_height]
        for _ in range(P.STAND_RAMP_STEPS):
            cmd = plr.step(pose, [], 1.0)
            heights.append(cmd.base_height)
        # Monotonic non-decreasing from SQUAT_HEIGHT toward STAND_HEIGHT.
        self.assertAlmostEqual(heights[0], P.SQUAT_HEIGHT, delta=0.05)
        for a, b in zip(heights, heights[1:]):
            self.assertGreaterEqual(b + 1e-9, a)
        self.assertAlmostEqual(cmd.base_height, P.STAND_HEIGHT, delta=1e-6)
        self.assertEqual(cmd.phase, "search")  # scored object only -> search

    def test_stand_then_approach_remembered(self):
        plr = TaskBPlanner()
        t1 = (-8.0, -10.0)
        t2 = (-9.0, -11.0)
        pose = Pose2D(-8.3, -10.0, 0.0)  # within ARRIVE_DIST of t1, far from t2
        # See both; approach t1 (nearer).
        plr.step(pose, [det(1, *t1, pose), det(2, *t2, pose)], 0.0)
        for _ in range(P.CREEP_STEPS + 1):
            plr.step(pose, [], 0.0)
        plr.step(pose, [], 1.0)  # touch t1, stand_up
        cmd = None
        for _ in range(P.STAND_RAMP_STEPS + 1):
            cmd = plr.step(pose, [], 1.0)
        # t1 touched, t2 still remembered -> approach t2 directly.
        self.assertEqual(cmd.phase, "approach")
        self.assertEqual(cmd.target_world, t2)


class MaxAttemptsTest(unittest.TestCase):
    def test_two_failed_sweeps_exhaust_object(self):
        plr = TaskBPlanner()
        target = (-8.0, -10.0)
        pose = Pose2D(target[0] - 0.3, target[1], 0.0)
        for attempt in range(P.MAX_ATTEMPTS):
            plr.step(pose, [det(1, *target, pose)], 0.0)  # approach within arrive
            # creep + squat to timeout, no score -> stand_up
            for _ in range(P.CREEP_STEPS + P.SQUAT_SWEEP_MAX_STEPS + 2):
                plr.step(pose, [], 0.0)
            for _ in range(P.STAND_RAMP_STEPS + 1):
                plr.step(pose, [], 0.0)
        self.assertGreaterEqual(plr.attempts.get(1, 0), P.MAX_ATTEMPTS)
        # Now exhausted: a fresh detection must not be selected.
        cmd = plr.step(pose, [det(1, *target, pose)], 0.0)
        self.assertEqual(cmd.phase, "search")


class ScoreDuringPursuitTest(unittest.TestCase):
    def test_score_during_approach_claims_object(self):
        plr = TaskBPlanner()
        target = (-8.0, -10.0)
        pose = Pose2D(-10.0, -10.0, 0.0)  # 2m away: genuinely mid-approach, not arrived
        cmd = plr.step(pose, [det(1, *target, pose)], 0.0)
        self.assertEqual(cmd.phase, "approach")
        # Robot kicks/early-touches the object while still walking up: score jumps.
        cmd = plr.step(pose, [], 1.0)
        self.assertEqual(cmd.phase, "stand_up")
        self.assertIn(1, plr.scored)
        # Entry was not from a squat, so the ramp holds near STAND_HEIGHT.
        self.assertAlmostEqual(cmd.base_height, P.STAND_HEIGHT, delta=1e-6)
        # Finish stand_up: scored object is never re-selected (detection nor memory).
        for _ in range(P.STAND_RAMP_STEPS + 1):
            cmd = plr.step(pose, [det(1, *target, pose)], 1.0)
        self.assertEqual(cmd.phase, "search")
        self.assertNotEqual(cmd.target_world, target)

    def test_score_on_arrival_transition_step_not_lost(self):
        plr = TaskBPlanner()
        target = (-8.0, -10.0)
        pose = Pose2D(-8.3, -10.0, 0.0)  # within ARRIVE_DIST: this step hands off
        # The first step both selects+arrives (approach -> creep handoff) AND carries
        # a score increase. The delta must not be dropped on the transition: the
        # approach-level score check claims the object before reaching creep.
        cmd = plr.step(pose, [det(1, *target, pose)], 1.0)
        self.assertEqual(cmd.phase, "stand_up")
        self.assertIn(1, plr.scored)
        # And the claimed object is never re-selected afterward.
        for _ in range(P.STAND_RAMP_STEPS + 1):
            cmd = plr.step(pose, [det(1, *target, pose)], 1.0)
        self.assertNotEqual(cmd.target_world, target)


class RecoverTest(unittest.TestCase):
    def test_recover_during_approach(self):
        plr = TaskBPlanner()
        target = (-8.0, -10.0)
        pose = Pose2D(-10.0, -10.0, 0.0)
        plr.step(pose, [det(1, *target, pose)], 0.0)
        cmd = plr.step(pose, [], 0.0, posture="recover")
        self.assertEqual(cmd.phase, "stand_up")

    def test_recover_during_creep(self):
        plr = TaskBPlanner()
        target = (-8.0, -10.0)
        pose = Pose2D(-8.3, -10.0, 0.0)  # within ARRIVE_DIST -> approach->creep
        plr.step(pose, [det(1, *target, pose)], 0.0)  # arrive -> creep
        cmd = plr.step(pose, [], 0.0)
        self.assertEqual(cmd.phase, "creep")  # genuinely in creep between steps
        cmd = plr.step(pose, [], 0.0, posture="recover")
        self.assertEqual(cmd.phase, "stand_up")
        # Entry was NOT from a squat, so the ramp holds near STAND_HEIGHT (0.75),
        # not the floor squat height.
        self.assertAlmostEqual(cmd.base_height, P.STAND_HEIGHT, delta=1e-6)

    def test_recover_during_squat(self):
        plr = TaskBPlanner()
        target = (-8.0, -10.0)
        pose = Pose2D(-8.3, -10.0, 0.0)  # within ARRIVE_DIST so we actually reach squat
        plr.step(pose, [det(1, *target, pose)], 0.0)  # arrive -> creep
        cmd = None
        for _ in range(P.CREEP_STEPS + 1):
            cmd = plr.step(pose, [], 0.0)
        self.assertEqual(cmd.phase, "squat_sweep")  # confirm we are genuinely squatting
        cmd = plr.step(pose, [], 0.0, posture="recover")
        self.assertEqual(cmd.phase, "stand_up")
        # Ramp must START at SQUAT_HEIGHT (0.30) + one increment, NOT jump to 0.75.
        first_inc = P.SQUAT_HEIGHT + (P.STAND_HEIGHT - P.SQUAT_HEIGHT) / P.STAND_RAMP_STEPS
        self.assertAlmostEqual(cmd.base_height, first_inc, delta=1e-6)
        self.assertLess(cmd.base_height, 0.35)  # nowhere near standing

    def test_recover_during_stand_up_stays(self):
        plr = TaskBPlanner()
        target = (-8.0, -10.0)
        pose = Pose2D(-10.0, -10.0, 0.0)
        plr.step(pose, [det(1, *target, pose)], 0.0)
        plr.step(pose, [], 0.0, posture="recover")  # -> stand_up
        cmd = plr.step(pose, [], 0.0, posture="recover")
        self.assertEqual(cmd.phase, "stand_up")


class MemoryTest(unittest.TestCase):
    def test_second_object_seen_mid_approach_later_approached(self):
        plr = TaskBPlanner()
        t1 = (-8.0, -10.0)
        t2 = (-9.0, -11.0)
        pose = Pose2D(-10.0, -10.0, 0.0)
        # Approach t1.
        plr.step(pose, [det(1, *t1, pose)], 0.0)
        # Mid-approach, t2 spotted (recorded in memory even though we keep approaching t1).
        plr.step(pose, [det(1, *t1, pose), det(2, *t2, pose)], 0.0)
        self.assertIn(2, plr.memory)
        self.assertEqual(plr.memory[2], t2)


class WBCCommandDefaultsTest(unittest.TestCase):
    def test_defaults_in_every_phase(self):
        plr = TaskBPlanner()
        target = (-8.0, -10.0)
        pose = Pose2D(-10.0, -10.0, 0.0)
        seen_phases = set()
        # search
        c = plr.step(pose, [], 0.0)
        seen_phases.add(c.phase)
        self.assertIsNone(c.fingers)
        self.assertEqual(c.waist_rpy, (0.0, 0.0, 0.0))
        # approach
        c = plr.step(pose, [det(1, *target, pose)], 0.0)
        seen_phases.add(c.phase)
        self.assertIsNone(c.fingers)
        self.assertEqual(c.waist_rpy, (0.0, 0.0, 0.0))
        # near to creep
        near = Pose2D(target[0] - 0.3, target[1], 0.0)
        c = plr.step(near, [], 0.0)
        seen_phases.add(c.phase)
        self.assertEqual(c.phase, "creep")
        self.assertIsNone(c.fingers)
        self.assertEqual(c.waist_rpy, (0.0, 0.0, 0.0))
        # squat
        for _ in range(P.CREEP_STEPS + 1):
            c = plr.step(near, [], 0.0)
        seen_phases.add(c.phase)
        self.assertIsNone(c.fingers)
        self.assertEqual(c.waist_rpy, (0.0, 0.0, 0.0))
        # stand_up (score during squat -> stand_up)
        c = plr.step(near, [], 1.0)
        seen_phases.add(c.phase)
        self.assertEqual(c.phase, "stand_up")
        self.assertIsNone(c.fingers)
        self.assertEqual(c.waist_rpy, (0.0, 0.0, 0.0))
        self.assertIn("squat_sweep", seen_phases)
        self.assertIn("creep", seen_phases)
        self.assertIn("stand_up", seen_phases)


class ResetTest(unittest.TestCase):
    def test_reset_fresh_planner(self):
        plr = TaskBPlanner()
        target = (-8.0, -10.0)
        pose = Pose2D(-10.0, -10.0, 0.0)
        plr.step(pose, [det(1, *target, pose)], 0.0)
        self.assertTrue(plr.memory)
        plr.reset()
        self.assertEqual(plr.memory, {})
        self.assertEqual(plr.attempts, {})
        self.assertEqual(plr.scored, set())
        cmd = plr.step(pose, [], 0.0)
        self.assertEqual(cmd.phase, "search")


class IntegrationWalkTest(unittest.TestCase):
    def test_full_fsm_sequence(self):
        plr = TaskBPlanner()
        t1 = (-8.0, -10.0)   # 2.0m from far -> selected first
        t2 = (-7.0, -12.0)   # ~3.6m from far -> approached later from memory
        phases = []

        def record(pose, dets, score, posture="ok"):
            cmd = plr.step(pose, dets, score, posture=posture)
            if not phases or phases[-1] != cmd.phase:
                phases.append(cmd.phase)
            return cmd

        far = Pose2D(-10.0, -10.0, 0.0)
        near = Pose2D(t1[0] - 0.3, t1[1], 0.0)
        # search (no dets first)
        record(far, [], 0.0)
        # see both objects -> approach t1
        record(far, [det(1, *t1, far), det(2, *t2, far)], 0.0)
        # arrive near t1 -> creep
        record(near, [], 0.0)
        # creep out -> squat
        for _ in range(P.CREEP_STEPS + 1):
            record(near, [], 0.0)
        # score during squat -> stand_up
        record(near, [], 1.0)
        # finish stand ramp -> approach t2 from memory
        for _ in range(P.STAND_RAMP_STEPS + 1):
            record(near, [], 1.0)
        self.assertEqual(
            phases,
            ["search", "approach", "creep", "squat_sweep", "stand_up", "approach"],
        )
        # final target is the remembered second object
        last = plr.step(near, [], 1.0)
        self.assertEqual(last.target_world, t2)


if __name__ == "__main__":
    unittest.main()
