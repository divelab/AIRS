from pdearena.configs.config_blast import BlastVelocityDensityTemperatureTimeIntegratorConfig
from pdearena.utils.constants import SEED0

config = BlastVelocityDensityTemperatureTimeIntegratorConfig(
    model="Transolver_cond1_moeulerx4",
    lr=6e-4,
    weight_decay=0.0,
    devices=2,
    seed=SEED0
)
