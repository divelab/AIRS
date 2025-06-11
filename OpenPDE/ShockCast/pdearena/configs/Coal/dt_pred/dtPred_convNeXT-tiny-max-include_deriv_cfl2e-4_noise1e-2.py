from pdearena.configs.config_coal import CoalVelocityVfracTempDeltaTPredConfig
from pdearena.utils.constants import Paths, CoalConstants

config = CoalVelocityVfracTempDeltaTPredConfig(
    model="convNeXT-tiny-max",
    noise_level=1e-2,
    include_derivatives=True,
    include_cfl_features=True,
    lr=2e-4,
    weight_decay=0.0,
    loss="mae",
    devices=1
)
