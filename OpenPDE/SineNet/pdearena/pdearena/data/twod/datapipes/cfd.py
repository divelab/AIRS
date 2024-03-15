# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import functools
from typing import Optional

import h5py
import torch
# import torchdata.datapipes as dp
import os

from .common import build_datapipes


class CFDDatasetOpener(torch.utils.data.Dataset):
    """DataPipe to load Navier-Stokes dataset.

    Args:
        dp (dp.iter.IterDataPipe): List of `hdf5` files containing Navier-Stokes data.
        mode (str): Mode to load data from. Can be one of `train`, `val`, `test`.
        limit_trajectories (int, optional): Limit the number of trajectories to load from individual `hdf5` file. Defaults to None.
        usegrid (bool, optional): Whether to output spatial grid or not. Defaults to False.

    Yields:
        (Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]): Tuple containing particle scalar field, velocity vector field, and optionally buoyancy force parameter value  and spatial grid.
    """

    def __init__(self, dp, mode: str, limit_trajectories: Optional[int] = None, usegrid: bool = False) -> None:
        super().__init__()
        assert len(dp) == 1
        self.dp = dp[0] # [f for f in dp]
        self.mode = mode
        self.limit_trajectories = limit_trajectories
        self.usegrid = usegrid
        with h5py.File(self.dp) as f:
            data = f[self.mode]
            self.seeds = list(data.keys())
            if self.limit_trajectories is None or self.limit_trajectories == -1:
                self.num = data[self.seeds[0]]['D'].shape[0]
            else:
                self.num = self.limit_trajectories

    def __len__(self):
        return len(self.seeds) * self.num

    def __getitem__(self, idx):
        file_idx = idx // self.num
        pde_idx = idx % self.num

        path = self.seeds[file_idx]

        with h5py.File(self.dp, "r") as f:
            data = f[self.mode][path]
            d = torch.tensor(data['D'][pde_idx])
            p = torch.tensor(data['P'][pde_idx])
            vx = torch.tensor(data['Vx'][pde_idx])
            vy = torch.tensor(data['Vy'][pde_idx])

        u = torch.cat((d, p), dim=-1).permute(0, 3, 1, 2)
        v = torch.cat((vx, vy), dim=-1).permute(0, 3, 1, 2)

        return u.float(), v.float(), None, None


def _train_filter(fname):
    return "train" in fname and "h5" in fname


def _valid_filter(fname):
    return "valid" in fname and "h5" in fname


def _test_filter(fname):
    return "test" in fname and "h5" in fname


train_datapipe_cfd = functools.partial(
    build_datapipes,
    dataset_opener=CFDDatasetOpener,
    filter_fn=_train_filter,
    lister=None, # dp.iter.FileLister,
    sharder=None, # dp.iter.ShardingFilter,
    mode="train",
)
onestep_valid_datapipe_cfd = functools.partial(
    build_datapipes,
    dataset_opener=CFDDatasetOpener,
    filter_fn=_valid_filter,
    lister=None, # dp.iter.FileLister,
    sharder=None, # dp.iter.ShardingFilter,
    mode="valid",
    onestep=True,
)
trajectory_valid_datapipe_cfd = functools.partial(
    build_datapipes,
    dataset_opener=CFDDatasetOpener,
    filter_fn=_valid_filter,
    lister=None, # dp.iter.FileLister,
    sharder=None, # dp.iter.ShardingFilter,
    mode="valid",
    onestep=False,
)

onestep_test_datapipe_cfd = functools.partial(
    build_datapipes,
    dataset_opener=CFDDatasetOpener,
    filter_fn=_test_filter,
    lister=None, # dp.iter.FileLister,
    sharder=None, # dp.iter.ShardingFilter,
    mode="test",
    onestep=True,
)

trajectory_test_datapipe_cfd = functools.partial(
    build_datapipes,
    dataset_opener=CFDDatasetOpener,
    filter_fn=_test_filter,
    lister=None, # dp.iter.FileLister,
    sharder=None, # dp.iter.ShardingFilter,
    mode="test",
    onestep=False,
)