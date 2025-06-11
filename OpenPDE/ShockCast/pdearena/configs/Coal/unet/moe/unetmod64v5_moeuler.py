from pdearena.configs.config_coal import CoalVelocityVfracTempTimeIntegratorConfig
from pdearena.utils.constants import SEED0

config = CoalVelocityVfracTempTimeIntegratorConfig(
    model="Unetmod-cond2-v5-MOEx4-euler-32_4_1",
    lr=2e-4,
    weight_decay=0.0,
    devices=1,
    seed=SEED0
)
