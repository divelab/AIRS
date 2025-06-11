from pdearena.configs.config_coal import CoalVelocityVfracTempTimeIntegratorConfig
from pdearena.configs.config import PretrainedCFDConfig

class CoalVelocityVfracTempPretrainedCFDConfig(
    CoalVelocityVfracTempTimeIntegratorConfig,
    PretrainedCFDConfig
    ):
    model: str = "Unetmod-cond2-v2-64_4_1"  # placeholder
    lr: float = -1
    weight_decay: float = -1
    devices: int = 1
    plmodel: str = "shock_cast"


config = CoalVelocityVfracTempPretrainedCFDConfig(
    cfl_ckpt = "ckpts/last-v2.ckpt"
)

if __name__ == "__main__":
    from pprint import pp
    pp(config.model_dump())