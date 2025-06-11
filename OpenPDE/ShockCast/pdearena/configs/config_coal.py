from typing import List
from pdearena.configs.config import CFDConfig
from pdearena.utils.constants import (
    CoalConstants,
    Paths
)

class CoalConfig(CFDConfig):

    start_time: float = 1e-4 / 4

    use_mach: bool = True
    use_diameter: bool = True

    nx: int = 104
    ny: int = 104
    xspacing: float = CoalConstants.xspacing
    yspacing: float = CoalConstants.yspacing

    trajlen: int = CoalConstants.trajlen
    max_num_steps: int = CoalConstants.trajlen - 1

    data_dir: str = Paths.coal

    log_losses_dt: float = 0.0001
    sim_time: float = CoalConstants.sim_time

    task: str = CoalConstants.task

class CoalVelocityVfracTempConfig(CoalConfig):
    time_coarsening: int = 5
    fields: List[str] = [CoalConstants.xVel_gas, CoalConstants.yVel_gas, CoalConstants.volume_fraction_coal, CoalConstants.temperature_gas]
    n_input_fields: int = len(fields)
    n_output_fields: int = len(fields)

class CoalVelocityVfracTempTimeIntegratorConfig(CoalVelocityVfracTempConfig):

    plmodel: str = "neural_solver"

    onestep_batch_size: int = 32
    rollout_batch_size: int = 1

    epochs: int = 400
    check_val_every_n_epoch: int = 20

class CoalVelocityVfracTempDeltaTPredConfig(CoalVelocityVfracTempConfig):
 
    plmodel: str = "neural_cfl"

    onestep_batch_size: int = 320
    rollout_batch_size: int = 1

    epochs: int = 800
    check_val_every_n_epoch: int = 20

    checkpoint_monitor: str = "valid/loss_mean"
