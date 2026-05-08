"""Unitree G1 AMP training env (rough terrain).

Design notes:
- Uses ATEC submission's exact proprio observation order so that `demo/solution.py` can
  forward the obs dict with minimal post-processing (see CANONICAL_PROPRIO_ORDER below).
- 29-body-DoF action subset (excludes 4 finger joints; fingers stay at default).
- Adds an `amp` observation group (73-dim) consumed by the AMP discriminator.
- Asymmetric actor/critic: actor obs drops `base_lin_vel`; critic obs keeps it.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import atec_rl_lab.train.locomotion.velocity.mdp as mdp
from atec_rl_lab.assets.robots import UNITREE_G1_29DOF_DEX1_CFG
from atec_rl_lab.train.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg


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


# ---- Custom three-group observation cfg: policy / critic / amp ----

@configclass
class G1AMPObservationsCfg:
    """Observation groups for G1 AMP training.

    `policy`: actor-visible obs, ATEC submission proprio order minus `base_lin_vel`,
              **stacked over 10 history frames** (matches bxi `actor_obs_history_length=10`).
              Per-frame: [ang_vel(3), cmd(3), gravity(3), jp_29(29), jv_29(29), last_act_29(29)] = 96 dim
              Final actor input: 96 * 10 = 960 dim (auto-flattened by ObsGroup.flatten_history_dim=True)
    `critic`: full proprio with base_lin_vel for asymmetric value learning, **single frame**
              (matches bxi `critic_obs_history_length=1`).
              Order: [lin_vel(3), ang_vel(3), cmd(3), gravity(3), jp_29(29), jv_29(29), last_act_29(29)] = 99 dim
    `amp`:    73-dim AMP obs = jp_29 + jv_29 + ee_pos_b_15 (no noise, used only by discriminator).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-100.0, 100.0), scale=1.0
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}, clip=(-100.0, 100.0), scale=1.0
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), clip=(-100.0, 100.0), scale=1.0
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_BODY_29_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-100.0, 100.0), scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_BODY_29_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-100.0, 100.0), scale=1.0,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10        # bxi actor_obs_history_length = 10
            self.flatten_history_dim = True # 96 * 10 = 960 dim flat

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


@configclass
class UnitreeG1AMPRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Rough-terrain G1 humanoid AMP training environment."""

    base_link_name = "torso_link"
    foot_link_name = ".*_ankle_roll_link"
    joint_names = G1_BODY_29_JOINT_NAMES

    def __post_init__(self) -> None:
        super().__post_init__()

        # ---- Scene ----
        self.scene.robot = UNITREE_G1_29DOF_DEX1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ---- Observations: replace wholesale with the three-group cfg ----
        self.observations = G1AMPObservationsCfg()

        # ---- Actions: 29 body joints, ATEC-aligned (scale=0.5, use_default_offset=True) ----
        self.actions.joint_pos.scale = 0.5
        self.actions.joint_pos.use_default_offset = True
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.joint_names = G1_BODY_29_JOINT_NAMES
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        # ---- Events: humanoid-friendly reset ranges ----
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (0.0, 0.05),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-15.0, 15.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-5.0, 5.0)
        # Match bxi push range: x/y ∈ (-1.0, 1.0)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}

        # ---- Rewards: aligned with bxi (bx_29_cfg.py BXDof29RewardCfg) ----
        # bxi termination_penalty = -200.0
        self.rewards.is_terminated.weight = -200.0

        # Root penalties (bxi values)
        self.rewards.lin_vel_z_l2.weight = -1.0          # bxi: -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.05        # bxi: -0.05
        self.rewards.flat_orientation_l2.weight = -1.0   # bxi: -1.0
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.base_height_l2.params["target_height"] = 0.74
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0.0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties (bxi values)
        self.rewards.joint_torques_l2.weight = 0.0       # bxi has no joint_torques_l2 (uses energy instead)
        self.rewards.joint_acc_l2.weight = -2.5e-7       # bxi dof_acc_l2: -2.5e-7
        self.rewards.joint_vel_l2.weight = 0.0
        self.rewards.joint_pos_limits.weight = -2.0      # bxi dof_pos_limits: -2.0
        self.rewards.joint_vel_limits.weight = 0.0
        self.rewards.joint_power.weight = -1e-3          # bxi energy: -1e-3
        self.rewards.stand_still.weight = -0.5
        self.rewards.joint_pos_penalty.weight = -0.5
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["left_hip.*", "right_hip.*"],
            ["left_shoulder.*", "right_shoulder.*"],
        ]

        # Action penalties (bxi values)
        self.rewards.action_rate_l2.weight = -0.01       # bxi: -0.01

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0    # bxi: -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            f"^(?!.*{self.foot_link_name}).*"
        ]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity tracking (bxi: weight=4.0, std=0.5 for both)
        self.rewards.track_lin_vel_xy_exp.weight = 4.0   # bxi: 4.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        self.rewards.track_ang_vel_z_exp.weight = 4.0    # bxi: 4.0
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5

        # Feet-related (bxi values where applicable)
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0.0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -2.0          # bxi feet_stumble: -2.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.25           # bxi feet_slide: -0.25
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -2.0      # bxi body_orientation_l2 / fly: -2.0
        self.rewards.feet_height_body.params["target_height"] = -0.55
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0.0
        self.rewards.upward.weight = 1.0

        # ---- Terminations: only fall on torso/non-foot illegal contact ----
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # ---- Curriculums ----
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # ---- Commands: aligned with bxi BXDof29WalkFlatEnvCfg ranges ----
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.57, 1.57)

        # Disable zero-weight rewards on the leaf class only (matches B2's pattern)
        if self.__class__.__name__ == "UnitreeG1AMPRoughEnvCfg":
            self.disable_zero_weight_rewards()
