from pdearena.configs.config_blast import BlastVelocityDensityTemperatureTimeIntegratorConfig
from pdearena.utils.constants import SEED0

config = BlastVelocityDensityTemperatureTimeIntegratorConfig(
    model="CNO-v2-L-cond1-moeulerx4",
    lr=2e-4,
    weight_decay=0.0,
    devices=8,
    seed=SEED0
)
