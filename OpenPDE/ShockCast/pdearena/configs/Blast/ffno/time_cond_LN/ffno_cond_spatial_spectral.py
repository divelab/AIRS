from pdearena.configs.config_blast import BlastVelocityDensityTemperatureTimeIntegratorConfig
from pdearena.utils.constants import SEED0

config = BlastVelocityDensityTemperatureTimeIntegratorConfig(
    model="FFNO_cond1_spatial_spectral",
    lr=1e-3,
    weight_decay=0.0,
    devices=4,
    xpad=2,
    ypad=2,
    seed=SEED0
)
