# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Tuple, List

from pdearena.configs.config import Config
from pdearena.utils.constants import (
    CoalConstants
)
from torch_geometric.data import Data, Batch
import torch

def get_nt(u):
    if isinstance(u, torch.Tensor):
        return u.shape[1]
    elif isinstance(u, list):
        return len(u)
    else:
        raise NotImplementedError

def time_slice(
        u: torch.Tensor, 
        start: int, 
        end: int
    ) -> torch.Tensor:
    if isinstance(u, torch.Tensor):
        return u[:, start:end]
    elif isinstance(u, list):
        assert end - start == 1
        return u[start]
    else:
        raise NotImplementedError

def check_steps(u, nt):
    if isinstance(u, torch.Tensor):
        if u.shape[1] != nt:
            raise ValueError(f"Incorrect number of time steps: {u.shape[1]}; should be {nt}")


def create_timestep_data(
    args: Config,
    sol: Data,
    start: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create training, test, valid data for one step prediction out of trajectory."""
    assert args.time_history > 0

    # Different starting points of one batch according to field_x(t_0), field_y(t_0), ...
    end_time = start + args.time_history
    target_start_time = end_time + args.time_gap
    target_end_time = target_start_time + args.time_future
    
    sol.x = time_slice(sol.u, start, end_time)
    sol.y = time_slice(sol.u, target_start_time, target_end_time)

    if hasattr(sol, "dt"):
        sol.dt = sol.dt[start]
    
    del sol.u

    check_steps(sol.x, args.time_history)
    check_steps(sol.y, args.time_future)
    return sol

COLLATE_REGISTRY = {}
