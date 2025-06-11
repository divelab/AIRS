from pdearena.configs.config_coal import CoalVelocityVfracTempTimeIntegratorConfig
from pdearena.utils.constants import SEED0

config = CoalVelocityVfracTempTimeIntegratorConfig(
    model="CNO-v2-L-cond2",
    lr=2e-4,
    weight_decay=0.0,
    devices=2,
    seed=SEED0
)
