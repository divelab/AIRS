from pdearena.configs.config_coal import CoalVelocityVfracTempTimeIntegratorConfig
from pdearena.utils.constants import SEED0

config = CoalVelocityVfracTempTimeIntegratorConfig(
    model="FFNO_cond2_spatial_spectral",
    lr=1e-3,
    weight_decay=0.0,
    devices=2,
    xpad=2,
    ypad=2,
    seed=SEED0
)
