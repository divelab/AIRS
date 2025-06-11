from typing import List, Any, Optional
from pdearena.configs.config import CFDConfig
from pdearena.utils.constants import (
    BlastConstants,
    Paths
)

class BlastConfig(CFDConfig):

    use_ratio: bool = True

    nx: int = 128
    ny: int = 128
    xspacing: float = BlastConstants.xspacing
    yspacing: float = BlastConstants.yspacing

    trajlen: int = BlastConstants.trajlen
    max_num_steps: int = BlastConstants.trajlen - 1

    data_dir: str = Paths.blast

    log_losses_dt: float = 0.0005
    sim_time: float = BlastConstants.sim_time

    task: str = BlastConstants.task


class BlastVelocityDensityTemperatureConfig(BlastConfig):
    time_coarsening: int = 1
    fields: List[str] = [BlastConstants.xVel, BlastConstants.yVel, BlastConstants.density, BlastConstants.temperature]
    n_input_fields: int = len(fields)
    n_output_fields: int = len(fields)


class BlastVelocityDensityTemperatureTimeIntegratorConfig(BlastVelocityDensityTemperatureConfig):

    plmodel: str = "neural_solver"

    onestep_batch_size: int = 32
    rollout_batch_size: int = 1

    epochs: int = 400
    check_val_every_n_epoch: int = 20
class BlastVelocityDensityTemperatureDeltaTPredConfig(BlastVelocityDensityTemperatureConfig):
 
    plmodel: str = "neural_cfl"

    onestep_batch_size: int = 320
    rollout_batch_size: int = 1

    epochs: int = 800
    check_val_every_n_epoch: int = 20

    checkpoint_monitor: str = "valid/loss_mean"
