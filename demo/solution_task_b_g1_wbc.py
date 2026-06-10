"""Task B G1 solution using the mini whole-body controller (policy18.onnx).

Reuses the perception / dead-reckoning / planner from solution_task_b_g1, but
replaces the locomotion+squat+sweep stack with a single whole-body controller
(MiniWBC) that accepts base-velocity, base-height, and hand-pose commands.

Control mapping per planner phase:
  search / approach_object -> base velocity command (drive toward target), stand height
  squat_sweep              -> stop, lower base height, command the near hand to the
                             object's base-frame position so the WBC reaches it
  stand_up                 -> stop, raise base height, hands at default
"""

from __future__ import annotations

import math

from demo.mini_wbc import MiniWBC, DEFAULT_LEFT_HAND, DEFAULT_RIGHT_HAND
from demo.solution_task_b_g1 import (
    DeadReckoningOdometry,
    Detection,
    PostureGuard,
    TaskBPlanner,
    TaskBRgbdPerception,
    _clamp,
)

# Sequence: detect -> approach -> creep forward a bit -> squat to 0.3 -> both hands flat-sweep.
# Command ranges are the WBC's valid (trained) ranges from mini/command_gui.py:
#   base height [0.3, 0.9];  hand x [-0.2, 0.6];  hand z [-0.2, 0.65];
#   left hand y [-0.1, 0.6];  right hand y [-0.6, 0.1].
# At base height 0.3, hand z=-0.2 puts the palm at the floor (pelvis ~0.32 -> world ~0.12).
SQUAT_HEIGHT = 0.30           # squat base height while sweeping (= controller minimum)
HAND_Z = -0.20                # hand z rel pelvis (= controller minimum) -> floor at base 0.30
REACH_FWD_MIN, REACH_FWD_MAX = -0.10, 0.58
LEFT_LAT_MIN, LEFT_LAT_MAX = -0.10, 0.55
RIGHT_LAT_MIN, RIGHT_LAT_MAX = -0.55, 0.10
# creep forward a bit after arriving, before squatting
CREEP_VEL = 0.35
CREEP_STEPS = 22
# both-hands flat (horizontal) sweep across the front ground, straddling the object
HAND_SPREAD = 0.10            # left/right hand offset to either side of the object
SWEEP_LAT_AMP = 0.14
SWEEP_FWD_AMP = 0.10
SWEEP_LAT_PERIOD = 36         # steps
SWEEP_FWD_PERIOD = 25         # steps (coprime-ish -> dense coverage)

# Search: rotate in place to scan; if a full sweep finds nothing, relocate to new ground.
SEARCH_YAW = 1.3             # rad/s scan rotation (faster)
SEARCH_RELOCATE_FWD = 0.55
SEARCH_ROTATE_STEPS = 185    # ~one scan rotation
SEARCH_RELOCATE_STEPS = 75   # then move to a fresh area and scan again
# Faster approach: scale the planner's velocity command (clamped to WBC's valid range).
APPROACH_SPEED_GAIN = 1.9
APPROACH_VX_MAX, APPROACH_VY_MAX, APPROACH_WZ_MAX = 0.6, 0.4, 1.6


class AlgSolution:
    PERCEPTION_INTERVAL = 5

    def __init__(self):
        self.wbc = MiniWBC()
        self.odom = DeadReckoningOdometry(dt=0.02, x0=-10.0, y0=-10.0)
        self.perception = TaskBRgbdPerception()
        self.planner = TaskBPlanner()
        self.guard = PostureGuard()
        self._perception_step = 0
        self._cached_detections: list[Detection] = []
        self._reach_step = 0
        self._search_step = 0

    def reset(self, **kwargs) -> None:
        self.wbc.reset()
        self.odom.reset()
        self.perception.reset()
        self.planner.reset()
        self.guard.reset()
        self._perception_step = 0
        self._cached_detections = []
        self._reach_step = 0
        self._search_step = 0

    def _search_velocity(self):
        """Rotate in place to scan; periodically relocate to fresh ground so the robot
        does not spin forever in one spot (head camera only sees ~0.7-2.5m)."""
        self._search_step += 1
        cycle = self._search_step % (SEARCH_ROTATE_STEPS + SEARCH_RELOCATE_STEPS)
        if cycle < SEARCH_ROTATE_STEPS:
            return [0.0, 0.0, SEARCH_YAW]
        return [SEARCH_RELOCATE_FWD, 0.0, 0.0]

    @staticmethod
    def _scaled_approach_velocity(command):
        vx, vy, wz = command
        return [
            _clamp(vx * APPROACH_SPEED_GAIN, -APPROACH_VX_MAX, APPROACH_VX_MAX),
            _clamp(vy * APPROACH_SPEED_GAIN, -APPROACH_VY_MAX, APPROACH_VY_MAX),
            _clamp(wz * 1.3, -APPROACH_WZ_MAX, APPROACH_WZ_MAX),
        ]

    def _object_base_frame(self, pose, target_world):
        wx, wy = target_world
        dx, dy = wx - pose.x, wy - pose.y
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        return c * dx + s * dy, -s * dx + c * dy  # forward, left

    def _both_hands_sweep(self, pose, target_world, sweep_t):
        """Both hands brush the front ground horizontally, straddling the object's
        lateral position, with a slow forward dither. Hands stay at floor level."""
        fwd0, lat0 = self._object_base_frame(pose, target_world)
        fwd = _clamp(fwd0 + SWEEP_FWD_AMP * math.sin(2.0 * math.pi * sweep_t / SWEEP_FWD_PERIOD),
                     REACH_FWD_MIN, REACH_FWD_MAX)
        lat_sweep = SWEEP_LAT_AMP * math.sin(2.0 * math.pi * sweep_t / SWEEP_LAT_PERIOD)
        left_center = _clamp(lat0 + HAND_SPREAD, LEFT_LAT_MIN, LEFT_LAT_MAX)
        right_center = _clamp(lat0 - HAND_SPREAD, RIGHT_LAT_MIN, RIGHT_LAT_MAX)
        left_lat = _clamp(left_center + lat_sweep, LEFT_LAT_MIN, LEFT_LAT_MAX)
        right_lat = _clamp(right_center + lat_sweep, RIGHT_LAT_MIN, RIGHT_LAT_MAX)
        left = [fwd, left_lat, HAND_Z, 1.0, 0.0, 0.0, 0.0]
        right = [fwd, right_lat, HAND_Z, 1.0, 0.0, 0.0, 0.0]
        return left, right

    def _creep_velocity(self, pose, target_world):
        """Forward velocity that keeps the robot aimed at the object while creeping in."""
        wx, wy = target_world
        bearing = math.atan2(wy - pose.y, wx - pose.x)
        yaw_err = (bearing - pose.yaw + math.pi) % (2.0 * math.pi) - math.pi
        return [CREEP_VEL, 0.0, _clamp(1.5 * yaw_err, -0.5, 0.5)]

    def predicts(self, obs: dict, current_score: float):
        proprio = obs["proprio"]
        row = proprio[0] if hasattr(proprio, "shape") and len(proprio.shape) >= 2 else proprio
        pose = self.odom.update(row)
        posture = self.guard.check(row)
        image_obs = obs.get("image", {})
        perception_step = getattr(self, "_perception_step", 0)
        detections = getattr(self, "_cached_detections", [])
        if perception_step % self.PERCEPTION_INTERVAL == 0:
            detections = self.perception.update(image_obs, pose)
            self._cached_detections = detections
        self._perception_step = perception_step + 1
        plan = self.planner.step(pose, detections, current_score, posture=posture)

        if plan.phase == "squat_sweep" and plan.target_world is not None:
            self._reach_step += 1
            if self._reach_step <= CREEP_STEPS:
                # walk forward a bit (still standing) to close the last distance
                vel = self._creep_velocity(pose, plan.target_world)
                action = self.wbc.act(proprio, vel, 0.75, [0.0, 0.0, 0.0],
                                      list(DEFAULT_LEFT_HAND), list(DEFAULT_RIGHT_HAND))
            else:
                # squat to 0.3 and brush the front ground with both hands
                left_hand, right_hand = self._both_hands_sweep(pose, plan.target_world,
                                                               self._reach_step - CREEP_STEPS)
                action = self.wbc.act(proprio, [0.0, 0.0, 0.0], SQUAT_HEIGHT, [0.0, 0.0, 0.0],
                                      left_hand, right_hand)
            return {"action": action, "giveup": False}

        self._reach_step = 0
        if plan.phase == "stand_up":
            self._search_step = 0
            height = plan.squat_command.height if plan.squat_command is not None else 0.75
            action = self.wbc.act(proprio, [0.0, 0.0, 0.0], height, [0.0, 0.0, 0.0],
                                  list(DEFAULT_LEFT_HAND), list(DEFAULT_RIGHT_HAND))
            return {"action": action, "giveup": False}

        if plan.phase == "search":
            # no object in view -> rotate to scan (and relocate periodically)
            vel = self._search_velocity()
        else:  # approach_object -> drive toward the object, faster
            self._search_step = 0
            vel = self._scaled_approach_velocity(plan.command)
        action = self.wbc.act(proprio, vel, 0.75, [0.0, 0.0, 0.0],
                              list(DEFAULT_LEFT_HAND), list(DEFAULT_RIGHT_HAND))
        return {"action": action, "giveup": False}
