"""Unitree G1 AMP training env (rough terrain).

This cfg mirrors bxi's `bx_29_cfg.BXDof29WalkFlatEnvCfg` / `BXDof29RewardCfg` as closely as
possible, with the exception of:
  - WAQ/VAE (not used — pure PPO + AMP)
  - bxi's stair-specific motion data (`USE_CLIMB_DATA`/`USE_DOWN_DATA`)
  - Joint names: G1 uses `*_hip_pitch/roll/yaw` etc. (bxi uses `*_hip_y/x/z`); regex patterns
    are translated accordingly when constructing reward terms.

Observation groups:
  - `policy`: 960-dim (96/frame × 10 history), bxi-compatible noise.
  - `critic`: 100-dim single frame (adds base_lin_vel + feet_contact bool over policy).
  - `amp`:    79-dim = [root_lin_vel_b(3), root_ang_vel_b(3), jp(29), jv(29), ee_pos_b(15)],
              consumed by the AMP discriminator. RAW joint_pos (NO default-subtract).

Gait clock state lives on the env (`G1AMPGaitEnv`); reward functions read it directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import atec_rl_lab.train.locomotion.velocity.mdp as mdp
from atec_rl_lab.assets.robots import UNITREE_G1_29DOF_DEX1_CFG
from atec_rl_lab.train.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg as BaseRewardsCfg,
)


# ---- 29 body joints (subset of g1_29dof_dex1.joint_names that excludes the 4 finger joints) ----
G1_BODY_29_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
assert len(G1_BODY_29_JOINT_NAMES) == 29

# ---- 5 end-effector bodies for the AMP discriminator's relative-position channel ----
G1_AMP_EE_BODIES: list[str] = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

# ---- bxi -> G1 regex translation for reward term joint subsets ----
# bxi `.*hip_z.*` -> G1 hip_yaw (vertical axis); `.*hip_y.*` -> hip_pitch; `.*hip_x.*` -> hip_roll.
# bxi `.*waist_x.*` -> waist_roll (we keep yaw + pitch always-on; bxi only penalizes roll here).
# bxi `.*shoulder_y.*` -> shoulder_pitch; `.*shoulder_x.*` -> shoulder_roll; `.*shoulder_z.*` -> shoulder_yaw.
# bxi `.*elbow_y.*` -> elbow; `.*knee_y.*` -> knee; `.*_ankle*` -> ankle_*.
G1_HIP_YAW_ROLL = [".*_hip_yaw_joint", ".*_hip_roll_joint"]                # bxi: hip_z + hip_x
G1_HIP_YAW_ROLL_ELBOW_SH_PITCH = G1_HIP_YAW_ROLL + [".*_shoulder_pitch_joint", ".*_elbow_joint"]
G1_WAIST_ROLL = [".*waist_roll_joint"]                                     # bxi: waist_x
G1_ARMS_DEVIATION = [
    ".*waist.*",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_.*_joint",
]
G1_LEGS_DEVIATION = [".*_hip_pitch_joint", ".*_knee_joint", ".*_ankle_.*_joint"]
G1_LEG_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
G1_NON_FOOT_BODIES_REGEX = r"^(?!.*ankle_roll).*"  # everything except the two ankle roll links
G1_ARM_TORSO_CONTACT_BODIES = [
    ".*_knee_link", ".*_hip_yaw_link", ".*_hip_pitch_link",
    ".*_shoulder_pitch_link", ".*_shoulder_yaw_link",
    ".*_wrist.*_link", ".*_elbow_link",
    "waist_yaw_link", "torso_link",
]


# ---- Gait clock cfg (used by G1AMPGaitEnv) ----

@configclass
class G1GaitCfg:
    """Gait clock parameters (matches bxi `GaitCfg`)."""
    gait_cycle: float = 1.0
    air_ratio_l: float = 0.5
    air_ratio_r: float = 0.5
    phase_offset_l: float = 0.38
    phase_offset_r: float = 0.88


# ---- Per-joint action scale matching bxi force/torque ratios (mapped to G1 joint order) ----
# bxi's per-joint scale ranges 0.154 (waist_y) to 0.373 (shoulder_yaw / elbow / wrist roll).
# Translation to G1 (29 joints, in G1_BODY_29_JOINT_NAMES order):
#   leg (hip_pitch/roll/yaw/knee): 0.231; ankle_pitch/roll: 0.213; waist_yaw/roll/pitch: 0.154/0.213/0.213;
#   shoulder_pitch/roll: 0.373; shoulder_yaw: 0.213; elbow: 0.373; wrist_roll/pitch/yaw: 0.23.
G1_PER_JOINT_ACTION_SCALE: dict[str, float] = {
    ".*_hip_pitch_joint": 0.231,
    ".*_hip_roll_joint": 0.231,
    ".*_hip_yaw_joint": 0.231,
    ".*_knee_joint": 0.231,
    ".*_ankle_pitch_joint": 0.213,
    ".*_ankle_roll_joint": 0.213,
    "waist_yaw_joint": 0.154,
    "waist_roll_joint": 0.213,
    "waist_pitch_joint": 0.213,
    ".*_shoulder_pitch_joint": 0.373,
    ".*_shoulder_roll_joint": 0.373,
    ".*_shoulder_yaw_joint": 0.213,
    ".*_elbow_joint": 0.373,
    ".*_wrist_roll_joint": 0.23,
    ".*_wrist_pitch_joint": 0.23,
    ".*_wrist_yaw_joint": 0.23,
}


# ---- Three-group observation cfg: policy / critic / amp ----

@configclass
class G1AMPObservationsCfg:
    """Observation groups for G1 AMP training.

    `policy`: actor-visible obs (96-dim/frame × 10 history = 960). Matches bxi actor.
    `critic`: full proprio + base_lin_vel + feet_contact (100-dim single frame).
    `amp`:    79-dim AMP obs = root_lin_vel_b(3)+root_ang_vel_b(3)+jp_29+jv_29+ee_pos_b_15.
              jp is RAW joint_pos (matches retargeted motion JSON; NO default-subtract).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # base_ang_vel = ObsTerm(
        #     func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-100.0, 100.0), scale=1.0
        # )
        # velocity_commands = ObsTerm(
        #     func=mdp.generated_commands, params={"command_name": "base_velocity"}, clip=(-100.0, 100.0), scale=1.0
        # )
        # projected_gravity = ObsTerm(
        #     func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), clip=(-100.0, 100.0), scale=1.0
        # )
        # joint_pos = ObsTerm(
        #     func=mdp.joint_pos_rel,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_BODY_29_JOINT_NAMES, preserve_order=True)},
        #     noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-100.0, 100.0), scale=1.0,
        # )
        # joint_vel = ObsTerm(
        #     func=mdp.joint_vel_rel,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_BODY_29_JOINT_NAMES, preserve_order=True)},
        #     noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-100.0, 100.0), scale=1.0,
        # )
        # actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=0.0, n_max=0.0), clip=(-100.0, 100.0), scale=1.0
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}, clip=(-100.0, 100.0), scale=1.0
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=0.0, n_max=0.0), clip=(-100.0, 100.0), scale=1.0
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_BODY_29_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=0.0, n_max=0.0), clip=(-100.0, 100.0), scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_BODY_29_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=0.0, n_max=0.0), clip=(-100.0, 100.0), scale=1.0,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}, clip=(-100.0, 100.0), scale=1.0
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_BODY_29_JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0), scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_BODY_29_JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0), scale=1.0,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AMPCfg(ObsGroup):
        amp_obs = ObsTerm(
            func=mdp.amp_obs_g1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=G1_BODY_29_JOINT_NAMES, preserve_order=True
                ),
                "ee_body_cfg": SceneEntityCfg(
                    "robot", body_names=G1_AMP_EE_BODIES, preserve_order=True
                ),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    amp: AMPCfg = AMPCfg()


# ---- bxi-aligned reward set (subclass of BaseRewardsCfg with additional terms) ----

@configclass
class G1AMPRewardsCfg(BaseRewardsCfg):
    """Extends the base RewardsCfg with bxi-specific terms.

    All extra terms default to weight 0 here; the env cfg's __post_init__ sets bxi weights
    on terms that should be active, and the base class's `disable_zero_weight_rewards()`
    drops the rest.
    """
    # bxi: lin_vel_z_l2 (already in base, weight set in __post_init__)
    # NEW: lateral (y-axis) base lin vel penalty — used by Task A to keep robot tracking straight
    lin_vel_y_l2 = RewTerm(func=mdp.lin_vel_y_l2, weight=0.0)
    # bxi: ang_vel_xy_l2_body (penalize torso/waist twisting)
    ang_vel_xy_l2_body = RewTerm(
        func=mdp.ang_vel_xy_l2_body, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="")},
    )
    energy = RewTerm(
        func=mdp.energy, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    action_smoothness = RewTerm(func=mdp.action_smoothness, weight=0.0)
    undesired_contacts_arm = RewTerm(
        func=mdp.undesired_contacts, weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 3.0},
    )
    fly = RewTerm(
        func=mdp.fly, weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="")},
    )
    body_orientation_euler = RewTerm(
        func=mdp.body_orientation_euler, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="")},
    )
    feet_force = RewTerm(
        func=mdp.body_force, weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 400.0, "max_reward": 400.0,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=""), "threshold": 0.2},
    )
    feet_y_distance_bxi = RewTerm(
        func=mdp.feet_y_distance, weight=0.0,
        params={
            "command_name": "base_velocity",
            "target_distance": 0.272,
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "cmd_y_threshold": 0.1,
        },
    )
    feet_orientation_l2_body = RewTerm(
        func=mdp.feet_orientation_l2_body, weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    feet_orientation_euler = RewTerm(
        func=mdp.feet_orientation_euler, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="")},
    )

    # Joint deviation family (bxi has 6 separate terms)
    joint_dev_hip = RewTerm(
        func=mdp.joint_deviation_l1_zero_cmd, weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[]),
            "cmd_threshold": 0.1,
        },
    )
    joint_dev_hip_walk = RewTerm(
        func=mdp.joint_deviation_l2_cmd, weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[]),
            "lin_cmd_threshold": 0.1,
            "ang_cmd_threshold": 0.05,
        },
    )
    joint_dev_hip_walk_1 = RewTerm(
        func=mdp.joint_deviation_l2_cmd, weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[]),
            "lin_cmd_threshold": 0.1,
            "ang_cmd_threshold": 0.05,
        },
    )
    joint_dev_waist = RewTerm(
        func=mdp.joint_deviation_l1_always, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[])},
    )
    joint_dev_arms = RewTerm(
        func=mdp.joint_deviation_l1_always, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[])},
    )
    joint_dev_legs = RewTerm(
        func=mdp.joint_deviation_l1_always, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[])},
    )

    # Stand-still / idle
    stand_still_bxi = RewTerm(
        func=mdp.stand_still_joint_exp, weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[]),
        },
    )
    idle_penalty = RewTerm(
        func=mdp.idle_when_commanded, weight=0.0,
        params={"command_name": "base_velocity", "cmd_threshold": 0.2, "vel_threshold": 0.1},
    )

    # Gait clock periodic
    gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=0.0, params={"delta_t": 0.015})
    gait_feet_frc_perio_penalize = RewTerm(
        func=mdp.gait_feet_frc_perio_penalize, weight=0.0, params={"delta_t": 0.015}
    )
    gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=0.0, params={"delta_t": 0.015})


@configclass
class UnitreeG1AMPRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Rough-terrain G1 humanoid AMP training environment (bxi-aligned, no WAQ)."""

    base_link_name = "torso_link"
    foot_link_name = ".*_ankle_roll_link"
    foot_body_pattern = ".*_ankle_roll_link"  # consumed by G1AMPGaitEnv
    joint_names = G1_BODY_29_JOINT_NAMES
    gait: G1GaitCfg = G1GaitCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # ---- Scene ----
        self.scene.robot = UNITREE_G1_29DOF_DEX1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ---- Observations ----
        self.observations = G1AMPObservationsCfg()

        # ---- Actions: per-joint action_scale (bxi-style) ----
        self.actions.joint_pos.scale = G1_PER_JOINT_ACTION_SCALE
        self.actions.joint_pos.use_default_offset = True
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.joint_names = G1_BODY_29_JOINT_NAMES
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        # ---- Events (bxi-aligned DR) ----
        # friction: static (0.3, 1.0), dynamic (0.2, 0.8), restitution (0.0, 0.005)
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.2, 0.8)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.005)
        # base mass: add ±5 kg to torso_link
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-5.0, 5.0)
        self.events.randomize_rigid_body_mass_base.params["operation"] = "add"
        # bxi does not randomize other body masses or COM; zero those out by name-empty regex
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (1.0, 1.0)
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
        }
        # No actuator gain randomization in bxi
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (1.0, 1.0)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (1.0, 1.0)
        # external force/torque: bxi has none on reset; keep small for stability
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (0.0, 0.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (0.0, 0.0)
        # Reset base: x/y (-0.5, 0.5), yaw (-π, π), velocity ±0.5 (bxi-aligned)
        self.events.randomize_reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        }
        # Reset joints: scale ×(0.5, 1.5) (bxi-aligned; was (1.0, 1.0))
        self.events.randomize_reset_joints.params["position_range"] = (0.5, 1.5)
        self.events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)
        # Push robot: disabled (zero range; EventTerm still fires every 10-15s but is a no-op)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-0.1, 0.3), "y": (-0.1, 0.1)}

        # ---- Rewards: replace cfg with extended G1AMPRewardsCfg, then set bxi weights ----
        self.rewards = G1AMPRewardsCfg()

        # General
        self.rewards.is_terminated.weight = -200.0           # bxi termination_penalty

        # Root / orientation
        self.rewards.lin_vel_z_l2.weight = -1.0              # bxi
        self.rewards.ang_vel_xy_l2.weight = -0.05            # bxi
        self.rewards.ang_vel_xy_l2_body.weight = -0.05
        self.rewards.ang_vel_xy_l2_body.params["asset_cfg"].body_names = ["waist_yaw_link"]
        self.rewards.flat_orientation_l2.weight = -1.0       # bxi
        self.rewards.body_orientation_l2.weight = -2.0       # bxi (on torso)
        self.rewards.body_orientation_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_orientation_euler.weight = 1.0     # bxi (positive)
        self.rewards.body_orientation_euler.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_acc_l2.weight = -2.5e-7           # bxi dof_acc_l2
        self.rewards.energy.weight = -1e-3                   # bxi (replaces joint_power)
        self.rewards.joint_pos_limits.weight = -2.0          # bxi dof_pos_limits

        # Joint deviation (6 bxi terms, mapped to G1 joint names)
        # 1) hip/elbow/shoulder-pitch when zero cmd, L1
        self.rewards.joint_dev_hip.weight = -0.15
        self.rewards.joint_dev_hip.params["asset_cfg"].joint_names = G1_HIP_YAW_ROLL_ELBOW_SH_PITCH
        # 2) same joints, walking L2
        self.rewards.joint_dev_hip_walk.weight = -1.0
        self.rewards.joint_dev_hip_walk.params["asset_cfg"].joint_names = G1_HIP_YAW_ROLL_ELBOW_SH_PITCH
        # 3) hip yaw+roll only, walking L2 (stronger)
        self.rewards.joint_dev_hip_walk_1.weight = -2.0
        self.rewards.joint_dev_hip_walk_1.params["asset_cfg"].joint_names = G1_HIP_YAW_ROLL
        # 4) waist roll always-on
        self.rewards.joint_dev_waist.weight = -0.15
        self.rewards.joint_dev_waist.params["asset_cfg"].joint_names = G1_WAIST_ROLL
        # 5) arms always-on
        self.rewards.joint_dev_arms.weight = -0.2
        self.rewards.joint_dev_arms.params["asset_cfg"].joint_names = G1_ARMS_DEVIATION
        # 6) legs always-on (very light)
        self.rewards.joint_dev_legs.weight = -0.02
        self.rewards.joint_dev_legs.params["asset_cfg"].joint_names = G1_LEGS_DEVIATION

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01           # bxi
        self.rewards.action_smoothness.weight = -0.003       # bxi action_rate_smooth

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0        # bxi (non-foot bodies, threshold 1.0)
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [G1_NON_FOOT_BODIES_REGEX]
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        self.rewards.undesired_contacts_arm.weight = -1.0    # bxi (arms+torso, threshold 3.0)
        self.rewards.undesired_contacts_arm.params["sensor_cfg"].body_names = G1_ARM_TORSO_CONTACT_BODIES

        # Tracking — tighter std + heavier yaw weight to push policy to track straight (cmd_y=0,
        # cmd_yaw=0 at Task A eval); also a direct lateral-velocity penalty.
        self.rewards.track_lin_vel_xy_exp.weight = 4.0       # bxi
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25
        self.rewards.track_ang_vel_z_exp.weight = 6.0
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25
        self.rewards.lin_vel_y_l2.weight = -0.5

        # Feet
        self.rewards.fly.weight = -2.0                       # bxi
        self.rewards.fly.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.25               # bxi
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -2.0              # bxi
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_force.weight = -3e-2               # bxi (>400N excess)
        self.rewards.feet_force.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_too_near.weight = -2.0
        self.rewards.feet_too_near.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_y_distance_bxi.weight = -2.0
        self.rewards.feet_y_distance_bxi.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_orientation_l2_body.weight = -1.0
        self.rewards.feet_orientation_l2_body.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_orientation_euler.weight = 0.25
        self.rewards.feet_orientation_euler.params["asset_cfg"].body_names = [self.foot_link_name]

        # Gait clock periodic
        self.rewards.gait_feet_frc_perio.weight = 1.0
        self.rewards.gait_feet_frc_perio_penalize.weight = -1.0
        self.rewards.gait_feet_spd_perio.weight = 1.0

        # Stand-still / idle
        self.rewards.stand_still_bxi.weight = -1.0
        self.rewards.stand_still_bxi.params["asset_cfg"].joint_names = G1_LEG_JOINT_NAMES
        self.rewards.idle_penalty.weight = -2.0

        # ---- Terminations ----
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # ---- Curriculums (no command curriculum; bxi uses fixed ranges) ----
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # ---- Commands: bxi BXDof29WalkFlatEnvCfg ranges ----
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.57, 1.57)
        # Narrow heading sampling so env mostly trains small-angle corrections (Task A eval cmd≈0).
        self.commands.base_velocity.ranges.heading = (-0.3, 0.3)
        self.commands.base_velocity.rel_standing_envs = 0.1  # bxi
        self.commands.base_velocity.rel_heading_envs = 1.0   # bxi
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 0.5

        # Disable zero-weight rewards on the leaf class only (matches B2's pattern)
        if self.__class__.__name__ == "UnitreeG1AMPRoughEnvCfg":
            self.disable_zero_weight_rewards()
