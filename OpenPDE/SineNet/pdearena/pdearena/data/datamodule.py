# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .registry import DATAPIPE_REGISTRY
from .utils import PDEDataConfig


def collate_fn_cat(batch):
    # Assuming pairs
    b1 = torch.cat([b[0] for b in batch], dim=0)
    b2 = torch.cat([b[1] for b in batch], dim=0)
    return b1, b2


def collate_fn_stack(batch):
    # Assuming pairs
    b1 = torch.stack([b[0] for b in batch], dim=0)
    if len(batch[0]) > 1:
        if batch[0][1] is not None:
            b2 = torch.stack([b[1] for b in batch], dim=0)
        else:
            b2 = None
        b3 = None
    if len(batch[0]) > 2:
        if batch[0][2] is not None:
            b3 = torch.cat([b[2] for b in batch], dim=0)
        else:
            b3 = None
    if len(batch[0]) > 3:
        if batch[0][3] is not None:
            b4 = torch.cat([b[3] for b in batch], dim=0)
        else:
            b4 = None
        return b1, b2, b3, b4

    return b1, b2, b3


class PDEDataModule(LightningDataModule):
    """Defines the standard dataloading process for PDE data.

    Does not support generalization to different parameterizations or time.
    Consider using [pdearena.data.cond_datamodule.CondPDEDataModule][] for that.

    Args:
        task (str): The task to be solved.
        data_dir (str): The path to the data directory.
        time_history (int): The number of time steps in the past.
        time_future (int): The number of time steps in the future.
        time_gap (int): The number of time steps between the past and the future to be skipped.
        pde (dict): The PDE to be solved.
        batch_size (int): The batch size.
        pin_memory (bool): Whether to pin memory.
        num_workers (int): The number of workers. Make sure when using values greater than 1 on multi-GPU systems, the number of shards is divisible by the number of workers times number of GPUs.
        train_limit_trajectories (int): The number of trajectories to be used for training. This is from each shard.
        valid_limit_trajectories (int): The number of trajectories to be used for validation. This is from each shard.
        test_limit_trajectories (int): The number of trajectories to be used for testing. This is from each shard.
        usegrid (bool, optional): Whether to use a grid. Defaults to False.
    """

    def __init__(
        self,
        task: str,
        data_dir: str,
        time_history: int,
        time_future: int,
        time_gap: int,
        pde: PDEDataConfig,
        batch_size: int,
        pin_memory: bool,
        num_workers: int,
        train_limit_trajectories: int,
        valid_limit_trajectories: int,
        test_limit_trajectories: int,
        usegrid: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.pde = pde

        self.save_hyperparameters(ignore="pde", logger=False)

    def setup(self, stage: Optional[str] = None):
        dps = DATAPIPE_REGISTRY[self.hparams.task]
        self.train_dp = dps["train"](
            pde=self.pde,
            data_path=self.data_dir,
            limit_trajectories=self.hparams.train_limit_trajectories,
            usegrid=self.hparams.usegrid,
            time_history=self.hparams.time_history,
            time_future=self.hparams.time_future,
            time_gap=self.hparams.time_gap,
        )
        self.valid_dp1 = dps["valid"][0](
            pde=self.pde,
            data_path=self.data_dir,
            limit_trajectories=self.hparams.valid_limit_trajectories,
            usegrid=self.hparams.usegrid,
            time_history=self.hparams.time_history,
            time_future=self.hparams.time_future,
            time_gap=self.hparams.time_gap,
        )
        self.valid_dp2 = dps["valid"][1](
            pde=self.pde,
            data_path=self.data_dir,
            limit_trajectories=self.hparams.valid_limit_trajectories,
            usegrid=self.hparams.usegrid,
            time_history=self.hparams.time_history,
            time_future=self.hparams.time_future,
            time_gap=self.hparams.time_gap,
        )
        self.test_dp_onestep = dps["test"][0](
            pde=self.pde,
            data_path=self.data_dir,
            limit_trajectories=self.hparams.test_limit_trajectories,
            usegrid=self.hparams.usegrid,
            time_history=self.hparams.time_history,
            time_future=self.hparams.time_future,
            time_gap=self.hparams.time_gap,
        )
        self.test_dp = dps["test"][1](
            pde=self.pde,
            data_path=self.data_dir,
            limit_trajectories=self.hparams.test_limit_trajectories,
            usegrid=self.hparams.usegrid,
            time_history=self.hparams.time_history,
            time_future=self.hparams.time_future,
            time_gap=self.hparams.time_gap,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dp,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn_cat,
        )

    def val_dataloader(self):
        timestep_loader = DataLoader(
            dataset=self.valid_dp1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn_cat,
        )
        rollout_loader = DataLoader(
            dataset=self.valid_dp2,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            batch_size=self.hparams.batch_size,  # TODO: might need to reduce this
            shuffle=False,
            collate_fn=collate_fn_stack,
        )
        return [timestep_loader, rollout_loader]

    def test_dataloader(self):
        rollout_loader = DataLoader(
            dataset=self.test_dp,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn_stack,
        )
        timestep_loader = DataLoader(
            dataset=self.test_dp_onestep,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn_cat,
        )
        return [timestep_loader, rollout_loader]
