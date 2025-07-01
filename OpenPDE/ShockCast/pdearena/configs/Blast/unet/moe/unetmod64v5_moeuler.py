from pdearena.configs.config_blast import BlastVelocityDensityTemperatureTimeIntegratorConfig
from pdearena.utils.constants import SEED0

config = BlastVelocityDensityTemperatureTimeIntegratorConfig(
    model="Unetmod-cond1-v5-MOEx4-euler-32_4_1",
    lr=2e-4,
    weight_decay=0.0,
    devices=1,
    seed=SEED0
)
