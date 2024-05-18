# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import random
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
# import torchdata.datapipes as dp

import pdearena.data.utils as datautils
from pdearena.data.utils import PDEDataConfig


def build_datapipes(
    pde: PDEDataConfig,
    data_path,
    limit_trajectories,
    usegrid: bool,
    dataset_opener, # : Callable[..., dp.iter.IterDataPipe],
    lister, # : Callable[..., dp.iter.IterDataPipe],
    sharder, # : Callable[..., dp.iter.IterDataPipe],
    filter_fn, # : Callable[..., dp.iter.IterDataPipe],
    mode: str,
    time_history=1,
    time_future=1,
    time_gap=0,
    onestep=False,
    conditioned=False,
    delta_t: Optional[int] = None,
    conditioned_reweigh: bool = True,
):
    """Build datapipes for training and evaluation.

    Args:
        pde (PDEDataConfig): PDE configuration.
        data_path (str): Path to the data.
        limit_trajectories (int): Number of trajectories to use.
        usegrid (bool): Whether to use spatial grid as input.
        dataset_opener (Callable[..., dp.iter.IterDataPipe]): Dataset opener.
        lister (Callable[..., dp.iter.IterDataPipe]): List files.
        sharder (Callable[..., dp.iter.IterDataPipe]): Shard files.
        filter_fn (Callable[..., dp.iter.IterDataPipe]): Filter files.
        mode (str): Mode of the data. ["train", "valid", "test"]
        time_history (int, optional): Number of time steps in the past. Defaults to 1.
        time_future (int, optional): Number of time steps in the future. Defaults to 1.
        time_gap (int, optional): Number of time steps between the past and the future to be skipped. Defaults to 0.
        onestep (bool, optional): Whether to use one-step prediction. Defaults to False.
        conditioned (bool, optional): Whether to use conditioned data. Defaults to False.
        delta_t (Optional[int], optional): Time step size. Defaults to None. Only used for conditioned data.
        conditioned_reweigh (bool, optional): Whether to reweight conditioned data. Defaults to True.

    Returns:
        dpipe (IterDataPipe): IterDataPipe for training and evaluation.
    """
    # dpipe = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_path)) for f in fn if filter_fn(os.path.join(dp, f))]
    dpipe = [os.path.join(data_path, f) for f in os.listdir(data_path) if filter_fn(f)]
    # dpipe = lister(
    #     data_path,
    # ).filter(filter_fn=filter_fn)
    # if mode == "train":
    #     dpipe = dpipe.shuffle()

    dpipe = dataset_opener(
        dpipe, # sharder(dpipe),
        mode=mode,
        limit_trajectories=limit_trajectories,
        usegrid=usegrid,
    )
    # if mode == "train":
    #     # Make sure that in expectation we have seen all the data despite randomization
    #     dpipe = dpipe.cycle(pde.trajlen)

    if mode == "train":
        # Training data is randomized
        if conditioned:
            dpipe = RandomTimeStepConditionedPDETrainData(
                dpipe,
                pde.n_scalar_components,
                pde.n_vector_components,
                pde.n_scalar_components,
                pde.n_vector_components,
                pde.trajlen,
                conditioned_reweigh,
            )
        else:
            dpipe = RandomizedPDETrainData(
                dpipe,
                pde.n_scalar_components,
                pde.n_vector_components,
                pde.n_scalar_components,
                pde.n_vector_components,
                pde.trajlen,
                time_history,
                time_future,
                time_gap,
            )
    else:
        # Evaluation data is not randomized.
        if conditioned and onestep:
            assert delta_t is not None
            dpipe = TimestepConditionedPDEEvalData(dpipe, pde.trajlen, delta_t)
        elif onestep:
            dpipe = PDEEvalTimeStepData(
                dpipe,
                pde.n_scalar_components,
                pde.n_vector_components,
                pde.n_scalar_components,
                pde.n_vector_components,
                pde.trajlen,
                time_history,
                time_future,
                time_gap,
            )

    return dpipe


class ZarrLister():# dp.iter.IterDataPipe):
    """Customized lister for zarr files.

    Args:
        root (Union[str, Sequence[str], dp.iter.IterDataPipe], optional): Root directory. Defaults to ".".

    Yields:
        (str): Path to the zarr file.
    """

    def __init__(
        self,
        root, # : Union[str, Sequence[str], dp.iter.IterDataPipe] = ".",
    ) -> None:
        super().__init__()

        if isinstance(root, str):
            root = [root]
        # if not isinstance(root, dp.iter.IterDataPipe):
        #     root = dp.iter.IterableWrapper(root)

        # self.datapipe: dp.iter.IterDataPipe = root

    def __iter__(self):
        for path in self.datapipe:
            for dirname in os.listdir(path):
                if dirname.endswith(".zarr"):
                    return os.path.join(path, dirname)


class RandomTimeStepConditionedPDETrainData(): #dp.iter.IterDataPipe):
    """Randomized data for training conditioned PDEs.

    Args:
        dp (IterDataPipe): Data pipe that returns individual PDE trajectories.
        n_input_scalar_components (int): Number of input scalar components.
        n_input_vector_components (int): Number of input vector components.
        n_output_scalar_components (int): Number of output scalar components.
        n_output_vector_components (int): Number of output vector components.
        trajlen (int): Length of a trajectory in the dataset.
        reweigh (bool, optional): Whether to rebalance the dataset so that longer horizon predictions get equal weightage despite there being fewer actual such datapoints in a trajectory. Defaults to True.
    """

    def __init__(
        self,
        dp,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        trajlen: int,
        reweigh=True,
    ) -> None:
        super().__init__()
        self.dp = dp
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components

        self.trajlen = trajlen
        self.reweigh = reweigh
    def __len__(self):
        return self.trajlen * len(self.dp)
 
    def __getitem__(self, idx):
        dp_idx = idx % len(self.dp)
        # Length of trajectory
        time_resolution = self.trajlen

        u, v, cond, grid = self.dp[dp_idx]
        if self.reweigh:
            end_time = random.choices(range(1, time_resolution), k=1)[0]
            start_time = random.choices(range(0, end_time), weights=1 / np.arange(1, end_time + 1), k=1)[0]
        else:
            end_time = torch.randint(low=1, high=time_resolution, size=(1,), dtype=torch.long).item()
            start_time = torch.randint(low=0, high=end_time.item(), size=(1,), dtype=torch.long).item()

        delta_t = end_time - start_time
        return (*datautils.create_time_conditioned_data(
                self.n_input_scalar_components,
                self.n_input_vector_components,
                self.n_output_scalar_components,
                self.n_output_vector_components,
                u,
                v,
                grid,
                start_time,
                end_time,
                torch.tensor([delta_t]), ), cond)


class TimestepConditionedPDEEvalData(): # dp.iter.IterDataPipe):
    """Data for evaluation of time conditioned PDEs

    Args:
        dp (torchdata.datapipes.iter.IterDataPipe): Data pipe that returns individual PDE trajectories.
        trajlen (int): Length of a trajectory in the dataset.
        delta_t (int): Evaluates predictions conditioned at that delta_t.

    Tip:
        Make sure `delta_t` is less than half of `trajlen`.
    """

    def __init__(self, dp, # : dp.iter.IterDataPipe, 
                 trajlen: int, delta_t: int) -> None:
        super().__init__()
        self.dp = dp
        self.trajlen = trajlen
        if 2 * delta_t >= self.trajlen:
            raise ValueError("delta_t should be less than half the trajectory length")

        self.delta_t = delta_t

        self.time_history = self.time_future = 1

        time_resolution = self.trajlen
        # Max number of previous points solver can eat
        reduced_time_resolution = time_resolution - self.time_history
        # Number of future points to predict
        max_start_time = reduced_time_resolution - delta_t # self.time_future - self.time_gap
        # We ignore these timesteps in the testing
        self.start_time = [t for t in range(0, max_start_time + 1)]

    def __len__(self):
        return len(self.start_time) * len(self.dp) 

    def __getitem__(self, idx):
        pde_idx = idx // len(self.start_time)
        time_idx = idx % len(self.start_time)
        start = self.start_time[time_idx]
        u, v, cond, grid = self.dp[pde_idx]
        
        end_time = start + self.time_history
        target_start_time = start + self.delta_t # end_time + self.time_gap
        target_end_time = target_start_time + self.time_future
        data_scalar = torch.Tensor()
        data_vector = torch.Tensor()
        labels_scalar = torch.Tensor()
        labels_vector = torch.Tensor()
        data_scalar = u[
            start:end_time,
            ...,
        ]
        labels_scalar = u[
            target_start_time:target_end_time,
            ...,
        ]
        data_vector = v[
            start:end_time,
            ...,
        ]
        labels_vector = v[
            target_start_time:target_end_time,
            ...,
        ]

        # add batch dim
        data = torch.cat((data_scalar, data_vector), dim=1).unsqueeze(0)

        # add batch dim
        labels = torch.cat((labels_scalar, labels_vector), dim=1).unsqueeze(0)
        return data, labels, torch.tensor([self.delta_t]), cond



class RandomizedPDETrainData(torch.utils.data.Dataset):
    """Randomized data for training PDEs.

    Args:
        dp (IterDataPipe): Data pipe that returns individual PDE trajectories.
        n_input_scalar_components (int): Number of input scalar components.
        n_input_vector_components (int): Number of input vector components.
        n_output_scalar_components (int): Number of output scalar components.
        n_output_vector_components (int): Number of output vector components.
        trajlen (int): Length of a trajectory in the dataset.
        time_history (int): Number of time steps of inputs.
        time_future (int): Number of time steps of outputs.
        time_gap (int): Number of time steps between inputs and outputs.
    """

    def __init__(
        self,
        dp,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        trajlen: int,
        time_history: int,
        time_future: int,
        time_gap: int,
    ) -> None:
        super().__init__()
        self.dp = dp
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.trajlen = trajlen
        self.time_history = time_history
        self.time_future = time_future
        self.time_gap = time_gap

    def __len__(self):
        return self.trajlen * len(self.dp)

    def __getitem__(self, idx):
        dp_idx = idx % len(self.dp)
        # Length of trajectory
        time_resolution = self.trajlen
        # Max number of previous points solver can eat
        reduced_time_resolution = time_resolution - self.time_history
        # Number of future points to predict
        max_start_time = reduced_time_resolution - self.time_future - self.time_gap

        batch = self.dp[dp_idx]
        if len(batch) == 3:
            (u, v, grid) = batch
            cond = None
        elif len(batch) == 4:
            (u, v, cond, grid) = batch
        else:
            raise ValueError(f"Unknown batch length of {len(batch)}.")

        # Choose initial random time point at the PDE solution manifold
        start_time = random.choices([t for t in range(max_start_time + 1)], k=1)
        return datautils.create_data2D(
            self.n_input_scalar_components,
            self.n_input_vector_components,
            self.n_output_scalar_components,
            self.n_output_vector_components,
            u,
            v,
            grid,
            start_time[0],
            self.time_history,
            self.time_future,
            self.time_gap,
        )


class PDEEvalTimeStepData(torch.utils.data.Dataset):
    def __init__(
        self,
        dp,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        trajlen: int,
        time_history: int,
        time_future: int,
        time_gap: int,
    ) -> None:
        super().__init__()
        self.dp = dp
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.trajlen = trajlen
        self.time_history = time_history
        self.time_future = time_future
        self.time_gap = time_gap

        # Length of trajectory
        time_resolution = self.trajlen
        # Max number of previous points solver can eat
        reduced_time_resolution = time_resolution - self.time_history
        # Number of future points to predict
        max_start_time = reduced_time_resolution - self.time_future - self.time_gap
        # We ignore these timesteps in the testing
        self.start_time = [t for t in range(0, max_start_time + 1, self.time_gap + self.time_future)]

    def __len__(self):
        return len(self.start_time) * len(self.dp)

    def __getitem__(self, idx):
        pde_idx = idx // len(self.start_time)
        time_idx = idx % len(self.start_time)
        start = self.start_time[time_idx]
        u, v, cond, grid = self.dp[pde_idx]
        
        end_time = start + self.time_history
        target_start_time = end_time + self.time_gap
        target_end_time = target_start_time + self.time_future
        data_scalar = torch.Tensor()
        data_vector = torch.Tensor()
        labels_scalar = torch.Tensor()
        labels_vector = torch.Tensor()
        if self.n_input_scalar_components > 0:
            data_scalar = u[
                start:end_time,
                ...,
            ]
        if self.n_output_scalar_components > 0:
            labels_scalar = u[
                target_start_time:target_end_time,
                ...,
            ]
        if self.n_input_vector_components > 0:
            data_vector = v[
                start:end_time,
                ...,
            ]
        if self.n_output_vector_components > 0:
            labels_vector = v[
                target_start_time:target_end_time,
                ...,
            ]

        # add batch dim
        data = torch.cat((data_scalar, data_vector), dim=1).unsqueeze(0)

        # add batch dim
        labels = torch.cat((labels_scalar, labels_vector), dim=1).unsqueeze(0)
        return data, labels
