"""CLI probe for the Task D G1 velocity policy bridge."""

from __future__ import annotations

import argparse

import torch

from atec_rl_lab.train.task_d.policy_bridge import G1VelocityPolicyBridge
from atec_rl_lab.train.task_d.types import VelocityCommand


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Task D G1 velocity policy bridge dimensions.")
    parser.add_argument("--policy", required=True, help="Path to a torch.jit policy, e.g. demo/policy_a.pt")
    parser.add_argument("--device", default="cuda", help="Torch device for policy loading and inference.")
    parser.add_argument("--full-action-dim", type=int, default=33, help="Dummy full G1 action dimension.")
    args = parser.parse_args()

    bridge = G1VelocityPolicyBridge(policy_path=args.policy, device=args.device)
    proprio = torch.zeros((1, 12 + 3 * args.full_action_dim), device=bridge.device, dtype=torch.float32)

    policy_input, full_action_dim = bridge._build_policy_input(
        proprio,
        VelocityCommand(vx=0.0, vy=0.0, wz=0.0),
    )

    print(f"policy input dim: {policy_input.shape[-1]}")
    print(f"body action dim: {bridge.ACTION_DIM_BODY}")
    print(f"full action dim: {full_action_dim}")


if __name__ == "__main__":
    main()
