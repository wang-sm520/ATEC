"""Flat-terrain variant of the G1 AMP env (used for fast smoke tests / debugging)."""

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeG1AMPRoughEnvCfg


@configclass
class UnitreeG1AMPFlatEnvCfg(UnitreeG1AMPRoughEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None

        self.curriculum.terrain_levels = None

        # base_height term references the height_scanner; remove its sensor_cfg under flat
        if hasattr(self.rewards, "base_height_l2") and self.rewards.base_height_l2 is not None:
            if "sensor_cfg" in self.rewards.base_height_l2.params:
                self.rewards.base_height_l2.params["sensor_cfg"] = None

        if self.__class__.__name__ == "UnitreeG1AMPFlatEnvCfg":
            self.disable_zero_weight_rewards()
