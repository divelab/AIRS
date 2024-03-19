# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
import math
import os
from typing import Optional

import torch
import torch.distributed as dist
# import torchdata.datapipes as dp
import h5py

from .common import ZarrLister, build_datapipes


class ShallowWaterDatasetOpener(): # dp.iter.IterDataPipe):
    """DataPipe for loading the shallow water dataset

    Args:
        dp: datapipe with paths to load the dataset from.
        mode (str): "train" or "valid" or "test"
        limit_trajectories: number of trajectories to load from the dataset
        usevort (bool): whether to use vorticity or velocity. If False, velocity is returned.
        usegrid (bool): whether to use grid or not. If False, no grid is returned.
        sample_rate: sample rate for the data. Default is 1, which means no sub-sampling.

    Note:
        We manually manage the data distribution across workers and processes. So make sure not to use `torchdata`'s [dp.iter.Sharder][torchdata.datapipes.iter.ShardingFilter] with this data pipe.
    """

    def __init__(
        self,
        dp, # : dp.iter.IterDataPipe,
        mode: str,
        limit_trajectories: Optional[int] = None,
        usevort: bool = False,
        usegrid: bool = False,
        sample_rate: int = 1,
    ) -> None:
        super().__init__()
        dp = [f for f in dp]
        assert len(dp) == 1
        self.dp = [os.path.join(dp[0], f) for f in os.listdir(dp[0])]
        self.mode = mode
        self.limit_trajectories = limit_trajectories
        self.usevort = usevort
        self.usegrid = usegrid
        self.sample_rate = sample_rate        
        with h5py.File(self.dp[0]) as f:
            data = f[self.mode]
            if self.limit_trajectories is None or self.limit_trajectories == -1:
                self.num = data["u"].shape[0]
            else:
                self.num = self.limit_trajectories
        
        self.normstat = torch.load(os.path.join(os.path.dirname(self.dp[0]), "..", "normstats.pt"))
    
    def __len__(self):
        return len(self.dp) * self.num

    def __getitem__(self, idx):
        file_idx = idx // self.num
        pde_idx = idx % self.num

        path = self.dp[file_idx]

        with h5py.File(path, "r") as f:
            data = f[self.mode]

            if self.usevort:
                vort = torch.tensor(data["vor"][pde_idx].to_numpy())
                vort = (vort - self.normstat["vor"]["mean"]) / self.normstat["vor"]["std"]
            else:
                u = torch.tensor(data["u"][pde_idx])
                v = torch.tensor(data["v"][pde_idx])
                vecf = torch.cat((u, v), dim=1)

            pres = torch.tensor(data["pres"][pde_idx])

        pres = (pres - self.normstat["pres"]["mean"]) / self.normstat["pres"]["std"]
        pres = pres.unsqueeze(1)

        if self.sample_rate > 1:
            # TODO: hardocded skip_nt=4
            pres = pres[4 :: self.sample_rate]
            if self.usevort:
                vort = vort[4 :: self.sample_rate]
            else:
                vecf = vecf[4 :: self.sample_rate]
        if self.usegrid:
            raise NotImplementedError("Grid not implemented for weather data")
        else:
            if self.usevort:
                return torch.cat((pres, vort), dim=1).float(), None, None, None
            else:
                return pres.float(), vecf.float(), None, None


class VortShallowWaterDatasetOpener(ShallowWaterDatasetOpener):
    def __init__(self, dp, mode: str, limit_trajectories: Optional[int] = None, usegrid: bool = False) -> None:
        super().__init__(dp, mode, limit_trajectories, usevort=True, usegrid=usegrid)


class ShallowWaterDatasetOpener2Day(ShallowWaterDatasetOpener):
    def __init__(
        self,
        dp,
        mode: str,
        limit_trajectories: Optional[int] = None,
        usegrid: bool = False,
    ) -> None:
        super().__init__(dp, mode, limit_trajectories, usevort=False, usegrid=usegrid, sample_rate=8)


class VortShallowWaterDatasetOpener2Day(ShallowWaterDatasetOpener):
    def __init__(
        self,
        dp,
        mode: str,
        limit_trajectories: Optional[int] = None,
        usegrid: bool = False,
    ) -> None:
        super().__init__(dp, mode, limit_trajectories, usevort=True, usegrid=usegrid, sample_rate=8)


class ShallowWaterDatasetOpener1Day(ShallowWaterDatasetOpener):
    def __init__(
        self,
        dp,
        mode: str,
        limit_trajectories: Optional[int] = None,
        usegrid: bool = False,
    ) -> None:
        super().__init__(dp, mode, limit_trajectories, usevort=False, usegrid=usegrid, sample_rate=4)


class VortShallowWaterDatasetOpener1Day(ShallowWaterDatasetOpener):
    def __init__(
        self,
        dp,
        mode: str,
        limit_trajectories: Optional[int] = None,
        usegrid: bool = False,
    ) -> None:
        super().__init__(dp, mode, limit_trajectories, usevort=True, usegrid=usegrid, sample_rate=4)


def _sharder(x):
    return x

def _weathertrain_filter(fname):
    return fname == "train"

def _weathervalid_filter(fname):
    return fname == "valid"

def _weathertest_filter(fname):
    return fname == "test"


train_datapipe_2day_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathertrain_filter,
    sharder=_sharder,
    mode="train",
)
onestep_valid_datapipe_2day_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathervalid_filter,
    sharder=_sharder,
    mode="valid",
    onestep=True,
)
trajectory_valid_datapipe_2day_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathervalid_filter,
    sharder=_sharder,
    mode="valid",
    onestep=False,
)
onestep_test_datapipe_2day_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathertest_filter,
    sharder=_sharder,
    mode="test",
    onestep=True,
)
trajectory_test_datapipe_2day_vel = functools.partial(
    build_datapipes,
    dataset_opener=ShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathertest_filter,
    sharder=_sharder,
    mode="test",
    onestep=False,
)

train_datapipe_2day_vort = functools.partial(
    build_datapipes,
    dataset_opener=VortShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathertrain_filter,
    sharder=_sharder,
    mode="train",
)
onestep_valid_datapipe_2day_vort = functools.partial(
    build_datapipes,
    dataset_opener=VortShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathervalid_filter,
    sharder=_sharder,
    mode="valid",
    onestep=True,
)
trajectory_valid_datapipe_2day_vort = functools.partial(
    build_datapipes,
    dataset_opener=VortShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathervalid_filter,
    sharder=_sharder,
    mode="valid",
    onestep=False,
)
onestep_test_datapipe_2day_vort = functools.partial(
    build_datapipes,
    dataset_opener=VortShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathertest_filter,
    sharder=_sharder,
    mode="test",
    onestep=True,
)
trajectory_test_datapipe_2day_vort = functools.partial(
    build_datapipes,
    dataset_opener=VortShallowWaterDatasetOpener2Day,
    lister=ZarrLister,
    filter_fn=_weathertest_filter,
    sharder=_sharder,
    mode="test",
    onestep=False,
)
