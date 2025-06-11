from pdearena.configs.config_blast import BlastVelocityDensityTemperatureDeltaTPredConfig
from pdearena.utils.constants import Paths, BlastConstants

config = BlastVelocityDensityTemperatureDeltaTPredConfig(
    model="convNeXT-tiny",
    noise_level=1e-2,
    include_derivatives=True,
    include_cfl_features=True,
    lr=2e-4,
    weight_decay=0.0,
    loss="mae",
    devices=2
)
