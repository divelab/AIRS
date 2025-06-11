from pdearena.data.twod.datapipes.coal2d import (
    onestep_test_datapipe_coalSquare_grid,
    onestep_valid_datapipe_coalSquare_grid,
    onestep_train_datapipe_coalSquare_grid,
    trajectory_test_datapipe_coalSquare_grid,
    trajectory_valid_datapipe_coalSquare_grid,
    trajectory_train_datapipe_coalSquare_grid
)

from pdearena.data.twod.datapipes.blast2d import (
    onestep_test_datapipe_blast_grid,
    onestep_valid_datapipe_blast_grid,
    onestep_train_datapipe_blast_grid,
    trajectory_test_datapipe_blast_grid,
    trajectory_valid_datapipe_blast_grid,
    trajectory_train_datapipe_blast_grid
)

from pdearena.utils.constants import (
    CoalConstants,
    BlastConstants,
)

DATAPIPE_REGISTRY = dict()

DATAPIPE_REGISTRY[CoalConstants.task] = dict()
DATAPIPE_REGISTRY[CoalConstants.task]["train"] = [onestep_train_datapipe_coalSquare_grid, trajectory_train_datapipe_coalSquare_grid]
DATAPIPE_REGISTRY[CoalConstants.task]["valid"] = [onestep_valid_datapipe_coalSquare_grid, trajectory_valid_datapipe_coalSquare_grid]
DATAPIPE_REGISTRY[CoalConstants.task]["test"] = [onestep_test_datapipe_coalSquare_grid, trajectory_test_datapipe_coalSquare_grid]

DATAPIPE_REGISTRY[BlastConstants.task] = dict()
DATAPIPE_REGISTRY[BlastConstants.task]["train"] = [onestep_train_datapipe_blast_grid, trajectory_train_datapipe_blast_grid]
DATAPIPE_REGISTRY[BlastConstants.task]["valid"] = [onestep_valid_datapipe_blast_grid, trajectory_valid_datapipe_blast_grid]
DATAPIPE_REGISTRY[BlastConstants.task]["test"] = [onestep_test_datapipe_blast_grid, trajectory_test_datapipe_blast_grid]