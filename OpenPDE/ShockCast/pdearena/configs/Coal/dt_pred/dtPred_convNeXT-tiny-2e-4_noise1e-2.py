from pdearena.configs.config_coal import CoalVelocityVfracTempDeltaTPredConfig
from pdearena.utils.constants import Paths, CoalConstants

config = CoalVelocityVfracTempDeltaTPredConfig(
    model="convNeXT-tiny",
    noise_level=1e-2,
    lr=2e-4,
    weight_decay=0.0,
    loss="mae",
    devices=1
)
