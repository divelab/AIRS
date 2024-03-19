# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class PDEDataConfig:
    n_scalar_components: int
    n_vector_components: int
    trajlen: int
    n_spatial_dim: int


def create_data2D(
    n_input_scalar_components: int,
    n_input_vector_components: int,
    n_output_scalar_components: int,
    n_output_vector_components: int,
    scalar_fields: torch.Tensor,
    vector_fields: torch.Tensor,
    grid: Optional[torch.Tensor],
    start: int,
    time_history: int,
    time_future: int,
    time_gap: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create 2D training, test, valid data for one step prediction out of trajectory.

    Args:
        scalar_fields (torch.Tensor): input data of the shape [t * pde.n_scalar_components, x, y]
        vector_fields (torch.Tensor): input data of the shape [2 * t * pde.n_vector_components, x, y]
        start (int): starting point within one trajectory

    Returns:
        (Tuple[torch.Tensor, torch.Tensor]): input trajectories, label trajectories
    """
    assert n_input_scalar_components > 0 or n_input_vector_components > 0
    assert n_output_scalar_components > 0 or n_output_vector_components > 0
    assert time_history > 0

    # Different starting points of one batch according to field_x(t_0), field_y(t_0), ...
    end_time = start + time_history
    target_start_time = end_time + time_gap
    target_end_time = target_start_time + time_future
    data_scalar = torch.Tensor()
    labels_scalar = torch.Tensor()
    if n_input_scalar_components > 0:
        data_scalar = scalar_fields[start:end_time, :n_input_scalar_components]
    if n_output_scalar_components > 0:
        labels_scalar = scalar_fields[target_start_time:target_end_time, :n_output_scalar_components]

    if n_input_vector_components > 0:
        data_vector = vector_fields[start:end_time, : n_input_vector_components * 2]
        data = torch.cat((data_scalar, data_vector), dim=1).unsqueeze(0)
    if n_output_vector_components > 0:
        labels_vector = vector_fields[target_start_time:target_end_time, : n_output_vector_components * 2]
        targets = torch.cat((labels_scalar, labels_vector), dim=1).unsqueeze(0)
    else:
        data = data_scalar.unsqueeze(0)
        targets = labels_scalar.unsqueeze(0)

    if grid is not None:
        raise NotImplementedError("Adding Spatial Grid is not implemented yet.")
    #     data = torch.cat((data, grid), dim=1)

    if targets.size(1) == 0:
        raise ValueError("No targets")
    return data, targets


def create_maxwell_data(
    d_field: torch.Tensor,
    h_field: torch.Tensor,
    start: int,
    time_history: int,
    time_future: int,
    time_gap: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create 3D training, test, valid data for one step prediction out of trajectory.

    Args:
        d_field (torch.Tensor): input data of the shape `[t, 3, x, y, z]`
        h_field (torch.Tensor): output data of the shape `[t, 3, x, y, z]`
        start (int): start (int): starting point within one trajectory
        time_history (int): number of history time steps
        time_future (int): number of future time steps
        time_gap (int): time gap between input and target

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: input, target tensore of shape `[1, t, 6, x, y, z]`
    """   
    # Different starting points of one batch
    end_time = start + time_history
    target_start_time = end_time + time_gap
    target_end_time = target_start_time + time_future

    data_d_field = d_field[start:end_time]
    labels_d_field = d_field[target_start_time:target_end_time]
    data_h_field = h_field[start:end_time]
    labels_h_field = h_field[target_start_time:target_end_time]

    data = torch.cat((data_d_field, data_h_field), dim=1).unsqueeze(0)
    labels = torch.cat((labels_d_field, labels_h_field), dim=1).unsqueeze(0)

    return data, labels


def create_time_conditioned_data(
    n_input_scalar_components: int,
    n_input_vector_components: int,
    n_output_scalar_components: int,
    n_output_vector_components: int,
    scalar_fields,
    vector_fields,
    grid,
    start_time: int,
    end_time: int,
    delta_t,
):
    assert n_input_scalar_components > 0 or n_input_vector_components > 0
    assert n_output_scalar_components > 0 or n_output_vector_components > 0
    if n_input_scalar_components > 0:
        data_scalar = scalar_fields[start_time : start_time + 1]
    if n_output_scalar_components > 0:
        target_scalar = scalar_fields[end_time : end_time + 1]

    if n_input_vector_components > 0:
        data_vector = vector_fields[start_time : start_time + 1]
        data = torch.cat((data_scalar, data_vector), dim=1).unsqueeze(0)
    if n_input_vector_components > 0:
        target_vector = vector_fields[end_time : end_time + 1]
        targets = torch.cat((target_scalar, target_vector), dim=1).unsqueeze(0)
    if grid is not None:
        data = torch.cat((data, grid), dim=1)

    return data, targets, delta_t
