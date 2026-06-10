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

# object resting height (world z), measured ~0.09-0.14m in ATEC-TaskB-G1
OBJECT_Z_WORLD = 0.12
REACH_BASE_HEIGHT = 0.45      # base height while reaching for a floor object
REACH_FWD_MIN, REACH_FWD_MAX = 0.15, 0.62
REACH_LAT_MAX = 0.45


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

    def reset(self, **kwargs) -> None:
        self.wbc.reset()
        self.odom.reset()
        self.perception.reset()
        self.planner.reset()
        self.guard.reset()
        self._perception_step = 0
        self._cached_detections = []

    def _hand_command_for_object(self, pose, target_world, height):
        """Object world xy -> base-frame hand pose; pick the hand on the object's side."""
        wx, wy = target_world
        dx, dy = wx - pose.x, wy - pose.y
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        fwd = c * dx + s * dy
        left = -s * dx + c * dy
        fwd = _clamp(fwd, REACH_FWD_MIN, REACH_FWD_MAX)
        z = OBJECT_Z_WORLD - float(height)  # hand z relative to (pelvis ~ base height)
        if left >= 0.0:  # object to the robot's left -> use left hand
            lat = _clamp(left, -0.05, REACH_LAT_MAX)
            return [fwd, lat, z, 1.0, 0.0, 0.0, 0.0], list(DEFAULT_RIGHT_HAND)
        lat = _clamp(left, -REACH_LAT_MAX, 0.05)  # object to the right -> use right hand
        return list(DEFAULT_LEFT_HAND), [fwd, lat, z, 1.0, 0.0, 0.0, 0.0]

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

        if plan.phase in ("squat_sweep", "stand_up"):
            height = plan.squat_command.height if plan.squat_command is not None else 0.75
            if plan.phase == "squat_sweep" and plan.target_world is not None:
                height = min(height, REACH_BASE_HEIGHT)
                left_hand, right_hand = self._hand_command_for_object(pose, plan.target_world, height)
            else:
                left_hand, right_hand = list(DEFAULT_LEFT_HAND), list(DEFAULT_RIGHT_HAND)
            action = self.wbc.act(proprio, [0.0, 0.0, 0.0], height, [0.0, 0.0, 0.0], left_hand, right_hand)
            return {"action": action, "giveup": False}

        action = self.wbc.act(proprio, list(plan.command), 0.75, [0.0, 0.0, 0.0],
                              list(DEFAULT_LEFT_HAND), list(DEFAULT_RIGHT_HAND))
        return {"action": action, "giveup": False}
