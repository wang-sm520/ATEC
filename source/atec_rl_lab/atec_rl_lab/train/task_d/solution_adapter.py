"""Task D solution adapter around the G1 velocity policy bridge."""

from __future__ import annotations

from collections.abc import Callable
import os
from typing import Any

import torch

from atec_rl_lab.train.task_d.controller import TaskDController
from atec_rl_lab.train.task_d.debug import summarize_controller_debug
from atec_rl_lab.train.task_d.policy_bridge import G1VelocityPolicyBridge
from atec_rl_lab.train.task_d.types import VelocityCommand


class AlgSolution:
    """ATEC server adapter with an injectable velocity-command controller."""

    def __init__(
        self,
        policy_path: str,
        device: str = "cuda",
        controller: Callable[[dict, float], VelocityCommand | torch.Tensor | list[float] | tuple[float, ...]]
        | None = None,
        bridge: Any | None = None,
        debug_interval: int = 25,
        debug_single_line: bool | None = None,
    ):
        self.bridge = bridge or G1VelocityPolicyBridge(policy_path=policy_path, device=device)
        self.controller = controller or TaskDController()
        self.debug_interval = debug_interval
        self.debug_single_line = _env_bool("TASKD_DEBUG_SINGLE_LINE", True) if debug_single_line is None else debug_single_line
        self._step_count = 0

    def reset(self, **kwargs) -> None:
        self.bridge.reset()
        reset = getattr(self.controller, "reset", None)
        if callable(reset):
            reset()
        self._step_count = 0

    def _velocity_command(self, obs: dict, current_score: float):
        update = getattr(self.controller, "update", None)
        if callable(update):
            return update(obs, current_score)
        return self.controller(obs, current_score)

    def predicts(self, obs: dict, current_score: float):
        proprio = obs["proprio"]
        command = self._velocity_command(obs, current_score)
        self._print_debug(current_score)
        self._step_count += 1
        return {"action": self.bridge.act(proprio, command), "giveup": False}

    def _print_debug(self, current_score: float) -> None:
        if self.debug_interval <= 0:
            return
        if self._step_count % self.debug_interval != 0:
            return
        debug = getattr(self.controller, "last_debug", None)
        if not isinstance(debug, dict) or not debug:
            return
        text = f"[TaskD] {summarize_controller_debug(debug, current_score)}"
        if self.debug_single_line:
            print("\r" + text, end="", flush=True)
        else:
            print(text, flush=True)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}
