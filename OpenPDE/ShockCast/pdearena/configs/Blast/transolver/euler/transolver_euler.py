from pdearena.configs.config_blast import BlastVelocityDensityTemperatureTimeIntegratorConfig
from pdearena.utils.constants import SEED0

config = BlastVelocityDensityTemperatureTimeIntegratorConfig(
    model="Transolver_cond1_euler",
    lr=6e-4,
    weight_decay=0.0,
    devices=1,
    seed=SEED0
)
