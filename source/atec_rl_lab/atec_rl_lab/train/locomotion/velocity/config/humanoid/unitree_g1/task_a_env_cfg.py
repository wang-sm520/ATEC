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
                sub_terrains={
                    # User-requested proportions (slopes removed): flat=30, rough=30,
                    # stairs (up)=20, stairs (inv/down)=20. Sum = 100 -> no normalization quirk.
                    "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.30),
                    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                        proportion=0.30,
                        noise_range=(0.02, 0.10),
                        noise_step=0.02,
                        border_width=0.25,
                    ),
                    "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                        proportion=0.20,
                        step_height_range=(0.05, 0.20),
                        step_width=0.3,
                        platform_width=3.0,
                        border_width=1.0,
                        holes=False,
                    ),
                    "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                        proportion=0.20,
                        step_height_range=(0.05, 0.20),
                        step_width=0.3,
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
