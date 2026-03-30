import os
import torch
from torch.utils.data import DataLoader
from orbevo.datasets.pyg_dataset import TDDFTv2_pyg, collate_fn, keep_qm9_good_after_split

from torch.utils.data.distributed import DistributedSampler
import numpy as np


class DataModule:
    def __init__(self, cfg, world_size, time_start=0, time_cond=1, time_future=1, time_gap=1, T=100):
        self.cfg = cfg
        self.world_size = world_size
        self.time_start = time_start
        self.time_cond = time_cond
        self.time_future = time_future
        self.time_gap = time_gap
        assert cfg.batch_size % world_size == 0
        self.device_batch_size = cfg.batch_size // world_size
        assert cfg.num_workers % world_size == 0
        self.device_num_workers = cfg.num_workers // world_size
        self.train_sampler = None  # call set_epoch to get different orders in each epoch in distributed training

        if self.cfg.dataset.name == "MDA":
            perm_inds= np.random.default_rng(seed=34).permutation(np.arange(1000))
            train_inds = perm_inds[:800]
            valid_inds = perm_inds[800:]
        elif self.cfg.dataset.name == "QM9":
            perm_inds= np.random.default_rng(seed=34).permutation(np.arange(5000))
            train_inds = perm_inds[:4000]
            valid_inds = perm_inds[4000 : 4500]
        elif self.cfg.dataset.name == "QM9_ood":
            # Get the directory of the current file
            current_file_path = os.path.abspath(__file__)
            current_file_dir = os.path.dirname(current_file_path)
            split_inds_path = os.path.join(current_file_dir, 'datasets', 'qm9_ood_split_inds.pt')
            all_inds = torch.load(split_inds_path)
            print(f'Loaded split inds from {split_inds_path}')
            train_inds = all_inds['train']
            valid_inds = all_inds['val']
        else:
            raise ValueError('Unknown dataset.')

        train_inds = keep_qm9_good_after_split(self.cfg.dataset.name, train_inds)
        valid_inds = keep_qm9_good_after_split(self.cfg.dataset.name, valid_inds)

        if cfg.debug:
            train_inds = train_inds[:5]
            valid_inds = valid_inds[:5]

        # Number of data pairs per trajectory
        # the last conditioning step is the 40th step, total 100, e.g., num_targets = 1 if time_future * time_gap == 60
        self.num_targets = T + 1 - self.time_future * self.time_gap
        assert self.num_targets > 0

        train_inds = [i * self.num_targets + torch.arange(self.num_targets) for i in train_inds]
        valid_inds = [i * self.num_targets + torch.arange(self.num_targets) for i in valid_inds]
        train_inds = torch.cat(train_inds)
        valid_inds = torch.cat(valid_inds)

        if self.cfg.dataset.name == "MDA_larger":
            # MDA_larger's valid is in another file
            self.train_ds = TDDFTv2_pyg(
                root=cfg.dataset.train_root, 
                time_start=self.time_start,
                time_cond=self.time_cond, 
                time_future=self.time_future, 
                time_gap=self.time_gap, 
                T=T,
                rms_0=cfg.dataset.rms_0,
                rms_t=cfg.dataset.rms_t,
            )

            self.train_ds = torch.utils.data.Subset(self.train_ds, train_inds)

            self.valid_ds = TDDFTv2_pyg(
                root=cfg.dataset.val_test_root, 
                time_start=self.time_start,
                time_cond=self.time_cond, 
                time_future=self.time_future, 
                time_gap=self.time_gap, 
                T=T,
                rms_0=cfg.dataset.rms_0,
                rms_t=cfg.dataset.rms_t,
            )

            self.valid_ds = torch.utils.data.Subset(self.valid_ds, valid_inds)

        else:
            self.dataset = TDDFTv2_pyg(
                root=cfg.dataset.root, 
                time_start=self.time_start,
                time_cond=self.time_cond, 
                time_future=self.time_future, 
                time_gap=self.time_gap, 
                T=T,
                rms_0=cfg.dataset.rms_0,
                rms_t=cfg.dataset.rms_t,
            )

            self.train_ds = torch.utils.data.Subset(self.dataset, train_inds)
            self.valid_ds = torch.utils.data.Subset(self.dataset, valid_inds)


    def train_dataloader(self, rank=0):
        if self.cfg.use_ddp:  
            self.train_sampler = DistributedSampler(self.train_ds, num_replicas=self.world_size, rank=rank, shuffle=True)
        else:
            self.train_sampler = None

        return DataLoader(
            dataset=self.train_ds, 
            batch_size=self.device_batch_size, 
            num_workers=self.device_num_workers,
            pin_memory=self.cfg.pin_memory,
            sampler=self.train_sampler,
            collate_fn=collate_fn,
            shuffle=True if not self.cfg.use_ddp else None)
    

    def valid_dataloader(self, rank=0):
        if self.cfg.use_ddp:  
            sampler = DistributedSampler(self.valid_ds, num_replicas=self.world_size, rank=rank, shuffle=False)
        else:
            sampler = None

        return DataLoader(
            dataset=self.valid_ds, 
            batch_size=self.device_batch_size, 
            num_workers=self.device_num_workers,
            pin_memory=self.cfg.pin_memory,
            sampler=sampler,
            collate_fn=collate_fn,
            shuffle=False if not self.cfg.use_ddp else None)
