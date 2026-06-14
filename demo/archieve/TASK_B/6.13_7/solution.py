"""ATEC Task B (G1) submission entry point — thin wiring onto the unified planner.

Per-``predicts()`` pipeline (all logic lives in the imported modules):
  dead-reckon pose (task_b_nav.DeadReckoningOdometry)
  -> posture guard (task_b_nav.PostureGuard)
  -> RGB-D perception, throttled every PERCEPTION_INTERVAL calls
     (task_b_perception.TaskBRgbdPerception)
  -> unified FSM emitting one whole-body command (task_b_planner.TaskBPlanner)
  -> whole-body controller (mini_wbc.MiniWBC, policy18.onnx)

This file holds NO planner/motion logic, constants, or state machine — only the
glue (the perception throttle interval and the odometry spawn pose live here).

Submission file set (upload flat, as the import root):
  solution.py, mini_wbc.py, task_b_nav.py, task_b_perception.py,
  task_b_planner.py, policy18.onnx, requirements.txt
Do NOT upload run.sh / server.py — the platform injects those. A staged copy of
the upload set lives at demo/task_b/.
"""

from __future__ import annotations

# Dual-context import: top-level when the platform imports `solution` from the
# submission dir; `demo.`-prefixed when imported as `demo.solution` in local dev.
try:
    from mini_wbc import MiniWBC
    from task_b_nav import DeadReckoningOdometry, PostureGuard
    from task_b_perception import Detection, TaskBRgbdPerception
    from task_b_planner import TaskBPlanner
except ImportError:  # pragma: no cover - local dev path
    from demo.mini_wbc import MiniWBC
    from demo.task_b_nav import DeadReckoningOdometry, PostureGuard
    from demo.task_b_perception import Detection, TaskBRgbdPerception
    from demo.task_b_planner import TaskBPlanner


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

    def predicts(self, obs: dict, current_score: float):
        proprio = obs["proprio"]
        row = proprio[0] if hasattr(proprio, "shape") and len(proprio.shape) >= 2 else proprio
        pose = self.odom.update(row)
        posture = self.guard.check(row)
        if self._perception_step % self.PERCEPTION_INTERVAL == 0:
            self._cached_detections = self.perception.update(obs.get("image", {}), pose)
        self._perception_step += 1
        cmd = self.planner.step(pose, self._cached_detections, current_score, posture=posture)
        action = self.wbc.act(
            proprio, list(cmd.vel), cmd.base_height, list(cmd.waist_rpy),
            list(cmd.left_hand), list(cmd.right_hand), fingers=cmd.fingers,
        )
        return {"action": action, "giveup": False}
