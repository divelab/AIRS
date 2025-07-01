# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader as pyg_DataLoader
from torch.utils.data import DataLoader

from pdearena.data.registry import DATAPIPE_REGISTRY
from pdearena.data.utils import COLLATE_REGISTRY
from pdearena.configs.config import Config


class PDEDataModule(LightningDataModule):
    """Defines the standard dataloading process for PDE data."""
    def __init__(
        self,
        args: Config
    ):
        super().__init__()
        self.args = args
        self.time_dependent = args.time_dependent
        self.collate_fn_onestep, self.collate_fn_rollout = COLLATE_REGISTRY.get(self.args.task, [None, None])
        self.onestep_workers = self.rollout_workers = self.args.num_workers
        if self.collate_fn_onestep is None:
            self.onestep_dataloader_fn = pyg_DataLoader
        else:
            self.onestep_dataloader_fn = DataLoader
        if self.collate_fn_rollout is None:
            self.rollout_dataloader_fn = pyg_DataLoader
        else:
            self.rollout_dataloader_fn = DataLoader

    def setup(self, stage: Optional[str] = None):
        dps = DATAPIPE_REGISTRY[self.args.task]
        self.train_dp1 = dps["train"][0](args=self.args)
        self.valid_dp1 = dps["valid"][0](args=self.args)
        self.test_dp1 = dps["test"][0](args=self.args)
        if self.time_dependent:
            self.train_dp2 = dps["train"][1](args=self.args)
            self.valid_dp2 = dps["valid"][1](args=self.args)
            self.test_dp2 = dps["test"][1](args=self.args)

    def train_dataloader(self):
        return self.onestep_dataloader_fn(
            dataset=self.train_dp1,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            batch_size=self.args.onestep_batch_size,
            shuffle=True,
            drop_last=True,
            persistent_workers=self.args.num_workers > 0,
            collate_fn=self.collate_fn_onestep
        )

    def val_dataloader(self):
        timestep_loader = self.onestep_dataloader_fn(
            dataset=self.valid_dp1,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            batch_size=self.args.onestep_batch_size,
            shuffle=False,
            persistent_workers=self.args.num_workers > 0,
            collate_fn=self.collate_fn_onestep
        )
        if not self.time_dependent:
            return timestep_loader
        rollout_loader = self.rollout_dataloader_fn(
            dataset=self.valid_dp2,
            num_workers=self.rollout_workers,
            pin_memory=self.args.pin_memory,
            batch_size=self.args.rollout_batch_size,
            shuffle=False,
            persistent_workers=self.args.num_workers > 0,
            collate_fn=self.collate_fn_rollout
        )
        return [timestep_loader, rollout_loader]

    def test_dataloader(self):
        timestep_loader = self.onestep_dataloader_fn(
            dataset=self.test_dp1,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            batch_size=self.args.onestep_batch_size,
            shuffle=False,
            persistent_workers=self.args.num_workers > 0,
            collate_fn=self.collate_fn_onestep
        )
        if not self.time_dependent:
            return timestep_loader
        rollout_loader = self.rollout_dataloader_fn(
            dataset=self.test_dp2,
            num_workers=self.rollout_workers,
            pin_memory=self.args.pin_memory,
            batch_size=self.args.rollout_batch_size,
            shuffle=False,
            persistent_workers=self.args.num_workers > 0,
            collate_fn=self.collate_fn_rollout
        )
        return [timestep_loader, rollout_loader]


    def predict_dataloader(self):
        train_rollout_loader = self.rollout_dataloader_fn(
            dataset=self.train_dp2,
            num_workers=self.rollout_workers,
            pin_memory=self.args.pin_memory,
            batch_size=self.args.rollout_batch_size,
            shuffle=False,
            persistent_workers=self.args.num_workers > 0,
            collate_fn=self.collate_fn_rollout
        )
        test_rollout_loader = self.rollout_dataloader_fn(
            dataset=self.test_dp2,
            num_workers=self.rollout_workers,
            pin_memory=self.args.pin_memory,
            batch_size=self.args.rollout_batch_size,
            shuffle=False,
            persistent_workers=self.args.num_workers > 0,
            collate_fn=self.collate_fn_rollout
        )
        return [train_rollout_loader, test_rollout_loader]