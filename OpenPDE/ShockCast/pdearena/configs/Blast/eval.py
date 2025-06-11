from pdearena.configs.config_blast import BlastVelocityDensityTemperatureTimeIntegratorConfig
from pdearena.configs.config import PretrainedCFDConfig

class BlastVelocityDensityTemperaturePretrainedCFDConfig(
    BlastVelocityDensityTemperatureTimeIntegratorConfig,
    PretrainedCFDConfig, 
    ):
    model: str = "Unetmod-cond2-v2-64_4_1"  # placeholder
    lr: float = -1
    weight_decay: float = -1
    devices: int = 1
    plmodel: str = "shock_cast"

config = BlastVelocityDensityTemperaturePretrainedCFDConfig(
    cfl_ckpt = "last-v1.ckpt",
)

if __name__ == "__main__":
    from pprint import pp
    pp(config.model_dump())