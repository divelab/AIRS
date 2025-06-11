from pdearena.configs.config_blast import BlastVelocityDensityTemperatureTimeIntegratorConfig
from pdearena.utils.constants import SEED0

config = BlastVelocityDensityTemperatureTimeIntegratorConfig(
    model="Unetmod-cond1-v2-64_4_1",
    lr=2e-4,
    weight_decay=0.0,
    devices=1,
    seed=SEED0
)
