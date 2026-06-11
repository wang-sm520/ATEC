"""Task A-style training env for G1 AMP: same sub-terrain mix as the Task A evaluation
course, tiled across the training arena.

Task A's evaluation terrain (`tasks/task_a/terrain.py`) is a single 300 m corridor of 15
sequential 20 × 20 m tiles in this order:

    flat × 2 → random_rough × 4 → slope/inv_slope × 4 → stairs/inv_stairs × 4 → flat

For RL training we keep the same six sub-terrain types and proportions, but drop the
forced linear sequence so the BetterTerrainGenerator places tiles randomly across a
`num_rows × num_cols` grid. IsaacLab's standard terrain-level curriculum then ramps
difficulty (slope angle, step height, rough noise) as the policy improves.

Defaults: 10 rows × 20 cols × 20 m = 200 m × 400 m total terrain, plenty of room for
4096 envs.
"""

from __future__ import annotations

import copy

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from atec_rl_lab.tasks.task_base import (
    BetterTerrainGenerator,
    BetterTerrainGeneratorCfg,
    BetterTerrainImporter,
)

from .rough_env_cfg import UnitreeG1AMPRoughEnvCfg


@configclass
class UnitreeG1AMPTaskAEnvCfg(UnitreeG1AMPRoughEnvCfg):
    """G1 AMP training env whose terrain matches Task A's mix (flat/rough/slope/stairs)."""

    task_a_num_rows: int = 10
    task_a_num_cols: int = 20

    def __post_init__(self) -> None:
        super().__post_init__()

        physics_material = copy.deepcopy(self.scene.terrain.physics_material)
        visual_material = copy.deepcopy(self.scene.terrain.visual_material)

        self.scene.terrain = TerrainImporterCfg(
            class_type=BetterTerrainImporter,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=BetterTerrainGeneratorCfg(
                class_type=BetterTerrainGenerator,
                seed=0,
                size=(20.0, 20.0),  # match Task A per-tile
                border_width=0.0,
                num_rows=self.task_a_num_rows,
                num_cols=self.task_a_num_cols,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                # `curriculum=True` => steeper slopes / taller stairs as policy progresses
                curriculum=True,
                # `terrain_sequence=None` => BetterTerrainGenerator falls back to
                # proportion-based random tile selection (suitable for parallel training).
                terrain_sequence=None,
                # === 旧地形配置 (flat=30 / rough=30 / stairs=20 / stairs_inv=20) — 暂时注释保留 ===
                # sub_terrains={
                #     # User-requested proportions (slopes removed): flat=30, rough=30,
                #     # stairs (up)=20, stairs (inv/down)=20. Sum = 100 -> no normalization quirk.
                #     "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.30),
                #     "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                #         proportion=0.30,
                #         noise_range=(0.02, 0.10),
                #         noise_step=0.02,
                #         border_width=0.25,
                #     ),
                #     "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                #         proportion=0.20,
                #         step_height_range=(0.05, 0.20),
                #         step_width=0.3,
                #         platform_width=3.0,
                #         border_width=1.0,
                #         holes=False,
                #     ),
                #     "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                #         proportion=0.20,
                #         step_height_range=(0.05, 0.20),
                #         step_width=0.3,
                #         platform_width=3.0,
                #         border_width=1.0,
                #         holes=False,
                #     ),
                # },
                # === 地形配置 v2 (flat=30 / rough=50 / slope=20, 台阶去除) — 暂时注释保留 ===
                # 坡 20% 拆成上坡 10% + 下坡 10%, 类型/参数对齐 Task A 真评测
                # (tasks/task_a/terrain.py: HfPyramidSlopedTerrainCfg, slope≈0.4)。 Sum = 100。
                # sub_terrains={
                #     "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.30),
                #     "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                #         proportion=0.50,
                #         noise_range=(0.02, 0.10),
                #         noise_step=0.02,
                #         border_width=0.25,
                #     ),
                #     "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                #         proportion=0.10,
                #         slope_range=(0.39, 0.40),
                #         platform_width=2.5,
                #         border_width=0.25,
                #     ),
                #     "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                #         proportion=0.10,
                #         slope_range=(0.39, 0.40),
                #         platform_width=2.5,
                #         border_width=0.25,
                #     ),
                # },
                # === 地形配置 v3 (flat=40 / rough=60, 坡/台阶去除) — 暂时注释保留 ===  Sum = 100。
                # sub_terrains={
                #     "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.40),
                #     "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                #         proportion=0.60,
                #         noise_range=(0.02, 0.10),
                #         noise_step=0.02,
                #         border_width=0.25,
                #     ),
                # },
                # === 地形配置 v4 (flat=45 / rough=20 / stairs_inv=35, 坡 & 下楼梯去除) ===
                # 去掉正金字塔（机器人会重生在顶端、只练下楼梯）；只保留倒金字塔（底部重生→
                # 上楼梯）+ 平地/糙地，专注「上楼梯 + 平地走」。台阶高 30-50cm / 踏面 0.5m，
                # 为 Task D 过沟（坑深 1.0m / 箱高 0.6m / 平台 1.0-1.5m）训练 climb 策略。 Sum = 100。
                sub_terrains={
                    "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.45),
                    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                        proportion=0.20,
                        noise_range=(0.02, 0.10),
                        noise_step=0.02,
                        border_width=0.25,
                    ),
                    "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                        proportion=0.35,
                        # 原配置: step_height_range=(0.30, 0.50)
                        step_height_range=(0.20, 0.40),
                        step_width=0.5,
                        platform_width=3.0,
                        border_width=1.0,
                        holes=False,
                    ),
                },
            ),
            max_init_terrain_level=0,
            collision_group=-1,
            physics_material=physics_material,
            visual_material=visual_material,
            debug_vis=False,
        )

        # Re-bind sensor prim paths (defensive; parent already does this).
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        if self.scene.height_scanner_base is not None:
            self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        if self.__class__.__name__ == "UnitreeG1AMPTaskAEnvCfg":
            self.disable_zero_weight_rewards()
