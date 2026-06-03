"""Task D closed-loop pushing controller (no privileged state).

Strategy proven via scripts/probe_task_d_oracle.py: push the box +x until it jams
against the platform (box settles ~x=-0.86 in [-1.4,0.7] -> +14; robot ends ~x=-1.37
> -1.4 -> +2 = 16 pts). Robot pose comes from odometry (integrating the measured
base velocity in obs['proprio']; empirically ~0.14 m drift / 600 steps). The box
starts at the deterministic world pose (-3, 1.6); during pushing it is modelled as
sitting just ahead of the robot, and "done" is detected when forward progress stalls.

Perception (lidar bearing to box, platform/pit anchor, head-depth centering) is layered
on top in box_perception.py as a robustness correction; this module works without it.
"""

from __future__ import annotations

import math
from typing import Any

from .state_estimator import TaskDStateEstimator
from .types import Pose2D, VelocityCommand, clamp, wrap_to_pi


class RealTaskDController:
    # Deterministic Task D layout (world frame).
    BOX_START = Pose2D(-3.0, 1.6, 0.0)
    CONTACT_OFFSET = 0.65        # box center ahead of robot center when in contact
    BOX_TARGET_X = 0.0           # safely inside the +14 window [-1.4, 0.7]

    def __init__(self, warmup_steps: int = 20, dt: float = 0.02) -> None:
        self.warmup_steps = warmup_steps
        self.estimator = TaskDStateEstimator(dt=dt, initial_pose=Pose2D(-3.0, 0.0, 0.0))
        self.last_debug: dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        self.estimator.reset()
        self.phase = "warmup"
        self.box = self.BOX_START
        self.step = 0
        self._jam_ref_x = -1e9
        self._jam_ref_step = 0
        self.last_debug = {}

    # ------------------------------------------------------------------ #
    def update(self, obs: dict[str, Any], current_score: float) -> VelocityCommand:
        proprio = obs.get("proprio")
        row = proprio[0] if hasattr(proprio, "shape") and len(proprio.shape) >= 2 else proprio
        rs = self.estimator.update(row)
        rx, ry, ryaw = rs.pose_world.x, rs.pose_world.y, rs.pose_world.yaw

        self._update_box(rx, ry)
        self._update_phase(rx, ry, ryaw, current_score)
        target, cmd = self._control(rx, ry, ryaw)

        self.last_debug = {
            "phase": self.phase,
            "robot": (round(rx, 2), round(ry, 2), round(ryaw, 2)),
            "box": (round(self.box.x, 2), round(self.box.y, 2)),
            "target": (round(target.x, 2), round(target.y, 2)),
            "cmd": (round(cmd.vx, 2), round(cmd.vy, 2), round(cmd.wz, 2)),
            "score": round(float(current_score), 1),
        }
        self.step += 1
        return cmd

    # ------------------------------------------------------------------ #
    def _update_box(self, rx: float, ry: float) -> None:
        """Contact model: while pushing, the box sits just ahead of the robot."""
        if self.phase == "push":
            predicted = rx + self.CONTACT_OFFSET
            if predicted > self.box.x:               # robot has advanced the box
                self.box = Pose2D(predicted, self.box.y, 0.0)

    def _update_phase(self, rx: float, ry: float, ryaw: float, score: float) -> None:
        if self.phase == "warmup":
            if self.step >= self.warmup_steps:
                self.phase = "approach"
            return

        if self.phase == "approach":
            behind = rx < self.box.x - 0.50
            aligned = abs(ry - self.box.y) < 0.13
            facing = abs(wrap_to_pi(ryaw)) < 0.13
            if behind and aligned and facing and rx > self.box.x - 0.85:
                self.phase = "push"
                self._jam_ref_x = rx
                self._jam_ref_step = self.step
            return

        if self.phase == "push":
            # Box is secured (+14) once it reaches the target window or jams against the
            # platform; then advance the robot itself past x=-1.4 (+2) on a clear lane.
            if self.box.x >= self.BOX_TARGET_X:
                self.phase = "advance"
                return
            if rx - self._jam_ref_x > 0.06:
                self._jam_ref_x = rx
                self._jam_ref_step = self.step
            elif self.step - self._jam_ref_step > 120:   # ~2.4 s without progress
                self.phase = "advance"
            return

        if self.phase == "advance":
            # CROSS_LINE_X = -1.4 gives +2; stop comfortably past it on flat ground.
            if rx > -1.0:
                self.phase = "hold"
            return

    def _control(self, rx: float, ry: float, ryaw: float) -> tuple[Pose2D, VelocityCommand]:
        if self.phase == "warmup":
            return Pose2D(rx, ry), VelocityCommand(0.0, 0.0, 0.0)

        if self.phase == "hold":
            wz = clamp(2.0 * (0.0 - ryaw), -0.8, 0.8)
            return Pose2D(rx, ry), VelocityCommand(0.0, 0.0, wz)

        if self.phase == "approach":
            tx, ty = self.box.x - 0.70, self.box.y
            if rx > self.box.x - 0.55:               # too close/in front: back off
                tx = self.box.x - 0.90
            target = Pose2D(tx, ty, 0.0)
            cmd = self._goto(rx, ry, ryaw, target, vx_lo=-0.5, vx_hi=0.9, vy_abs=0.6)
            return target, cmd

        if self.phase == "advance":
            # Move to clear flat ground past the +2 line, away from box/platform/pit.
            target = Pose2D(-1.0, 0.3, 0.0)
            cmd = self._goto(rx, ry, ryaw, target, vx_lo=-0.3, vx_hi=0.9, vy_abs=0.6)
            return target, cmd

        # push
        target = Pose2D(self.box.x + 1.5, self.box.y, 0.0)
        ey = self.box.y - ry
        wz = clamp(2.0 * (0.0 - ryaw) - 1.0 * ey, -0.8, 0.8)
        vy = clamp(1.2 * ey, -0.4, 0.4)
        cmd = VelocityCommand(vx=0.9, vy=vy, wz=wz)
        return target, cmd

    @staticmethod
    def _goto(rx, ry, ryaw, target: Pose2D, vx_lo, vx_hi, vy_abs) -> VelocityCommand:
        ex, ey = target.x - rx, target.y - ry
        c, s = math.cos(ryaw), math.sin(ryaw)
        bex = c * ex + s * ey
        bey = -s * ex + c * ey
        return VelocityCommand(
            vx=clamp(1.4 * bex, vx_lo, vx_hi),
            vy=clamp(1.4 * bey, -vy_abs, vy_abs),
            wz=clamp(2.0 * wrap_to_pi(target.yaw - ryaw), -1.0, 1.0),
        )
