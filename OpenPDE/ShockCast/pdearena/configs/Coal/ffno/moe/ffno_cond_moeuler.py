from pdearena.configs.config_coal import CoalVelocityVfracTempTimeIntegratorConfig
from pdearena.utils.constants import SEED0

config = CoalVelocityVfracTempTimeIntegratorConfig(
    model="FFNO_cond2_moeulerx4",
    lr=1e-3,
    weight_decay=0.0,
    devices=4, 
    xpad=2,
    ypad=2,
    seed=SEED0
)
