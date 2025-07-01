# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from typing import Callable
import numpy as np

import torchdata.datapipes as dp
from torchdata.datapipes.iter import IterableWrapper

import pdearena.data.utils as datautils
from pdearena.configs.config import Config
from pdearena.utils import utils
logger = utils.get_logger(__name__)

def build_datapipes(
    args: Config,
    dataset_opener: Callable[..., dp.iter.IterDataPipe],
    filter_fn: Callable[..., dp.iter.IterDataPipe],
    lister: Callable[..., dp.iter.IterDataPipe],
    sharder: Callable[..., dp.iter.IterDataPipe],
    mode: str,
    onestep=False,
):
    """Build datapipes for training and evaluation."""
    dpipe = lister(
        args.data_dir,
    ).filter(filter_fn=filter_fn)
    onestep_train = onestep and mode == "train"
    if onestep_train:
        dpipe_list = list(dpipe)
        nfiles = len(dpipe_list)
        devices = args.devices
        if nfiles % devices != 0:
            repeats = int(np.ceil(nfiles / devices) * devices) - nfiles
            logger.warning(
                f"Number of files {nfiles} is not divisible by number of devices {devices}. Repeating first {repeats} files."
            )
            dpipe_list.extend(dpipe_list[:repeats])
            dpipe = IterableWrapper(dpipe_list)
        dpipe = dpipe.shuffle()

    match mode:
        case "train" | "train_rollout":
            limit_trajectories = args.train_limit_trajectories
        case "valid":
            limit_trajectories = args.valid_limit_trajectories
        case "test":
            limit_trajectories = args.test_limit_trajectories
        case _:
            raise ValueError(f"Invalid mode: {mode}")

    dpipe = dataset_opener(
        dp=sharder(dpipe),
        mode=mode,
        limit_trajectories=limit_trajectories,
        args=args
    )
    if args.time_dependent:
        if onestep_train:
            # Length of trajectory
            time_resolution = args.trajlen
            # Max number of previous points solver can eat
            reduced_time_resolution = time_resolution - args.time_history
            # Number of future points to predict
            max_start_time = reduced_time_resolution - args.time_future - args.time_gap
            # We ignore these timesteps in the testing
            start_time = range(0, max_start_time + 1, args.time_gap + args.time_future)
            # Make sure that in expectation we have seen all the data despite randomization
            dpipe = dpipe.cycle(len(start_time))

        if onestep_train:
            # Training data is randomized
            dpipe = RandomizedPDETrainData(dpipe, args=args)
        elif onestep:
            # Evaluation data is not randomized.
            dpipe = PDEEvalTimeStepData(dpipe, args=args)
            # For multi-step prediction, the original data pipe can be used without change.
    return dpipe


class RandomizedPDETrainData(dp.iter.IterDataPipe):
    """Randomized data for training PDEs."""
    def __init__(
        self,
        dp,
        args: Config
    ) -> None:
        super().__init__()
        self.dp = dp
        self.args = args
        self.sys_rand = random.SystemRandom()

        # Length of trajectory
        time_resolution = self.args.trajlen
        # Max number of previous points solver can eat
        reduced_time_resolution = time_resolution - self.args.time_history
        # Number of future points to predict
        max_start_time = reduced_time_resolution - self.args.time_future - self.args.time_gap
        # We ignore these timesteps in the testing
        start_time = range(0, max_start_time + 1, self.args.time_gap + self.args.time_future)
        self.num_cycles = len(start_time)

    def len(self):
        # UserWarning: Your `IterableDataset` has `__len__` defined. In combination with multi-
        # process data loading (when num_workers > 1), `__len__` could be inaccurate if each worker is not configured independently to avoid having duplicate data.
        return self.dp.source_datapipe.len() * self.num_cycles

    def __iter__(self):
        for sol in self.dp:

            # Length of trajectory
            nt = datautils.get_nt(sol.u)
            time_resolution = min(nt, self.args.trajlen)
            # Max number of previous points solver can eat
            reduced_time_resolution = time_resolution - self.args.time_history
            # Number of future points to predict
            max_start_time = reduced_time_resolution - self.args.time_future - self.args.time_gap

            # Choose initial random time point at the PDE solution manifold
            start_time = self.sys_rand.randint(0, max_start_time)
            sol = datautils.create_timestep_data(
                args=self.args,
                sol=sol,
                start=start_time,
            )
            yield sol


class PDEEvalTimeStepData(dp.iter.IterDataPipe):
    def __init__(
        self,
        dp,
        args: Config
    ) -> None:
        super().__init__()
        self.dp = dp
        self.args = args

    def __iter__(self):
        for sol in self.dp:
            # Length of trajectory
            nt = datautils.get_nt(sol.u)
            time_resolution = min(self.args.trajlen, nt)
            # Max number of previous points solver can eat
            reduced_time_resolution = time_resolution - self.args.time_history
            # Number of future points to predict
            max_start_time = reduced_time_resolution - self.args.time_future - self.args.time_gap
            # We ignore these timesteps in the testing
            start_time = range(0, max_start_time + 1, self.args.time_gap + self.args.time_future)
            for start in start_time:
                step_sol = sol.clone()
                step_sol = datautils.create_timestep_data(
                    args=self.args,
                    sol=step_sol,
                    start=start,
                )
                yield step_sol