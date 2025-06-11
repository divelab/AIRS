from pdearena.configs.config_coal import CoalVelocityVfracTempTimeIntegratorConfig
from pdearena.utils.constants import SEED0

config = CoalVelocityVfracTempTimeIntegratorConfig(
    model="Transolver_cond2_euler",
    lr=6e-4,
    weight_decay=0.0,
    devices=4,
    seed=SEED0
)
