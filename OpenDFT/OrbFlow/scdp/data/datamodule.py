import json
import random
from typing import Optional, Dict
from pathlib import Path
from tqdm import tqdm

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, Subset

from scdp.common.pyg import DataLoader
from scdp.data.dataloader import ProbeDataLoader
from scdp.data.gt_coeff_exclusions import (
    GT_COEFF_EXCLUDED_FILENAME,
    filter_split_indices,
    load_gt_coeff_excluded_indices,
)


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)

class DataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        dataset: DictConfig,
        split_file: Optional[Path],
        num_workers: DictConfig,
        batch_size: DictConfig,
        gt_coeff_excluded_file: Optional[Path] = None,
    ):
        super().__init__()
        self.dataset = dataset
        with open(split_file, "r") as fp:
            self.splits = json.load(fp)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.gt_coeff_excluded_file = gt_coeff_excluded_file
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.metadata: Optional[Dict] = None
        
    def setup(self, stage: Optional[str] = None):
        """
        construct datasets and assign data scalers.
        """
        self.dataset = hydra.utils.instantiate(self.dataset)
        gt_path = getattr(self.dataset, "gt_coeffs_path", None)
        lmdb_in = Path(self.dataset.path)
        excluded = load_gt_coeff_excluded_indices(
            Path(gt_path) if gt_path is not None else None,
            excluded_file=self.gt_coeff_excluded_file,
            lmdb_in_path=lmdb_in,
        )
        if excluded:
            n_before = sum(len(self.splits[k]) for k in self.splits)
            self.splits = {
                k: filter_split_indices(self.splits[k], excluded)
                for k in self.splits
            }
            n_after = sum(len(self.splits[k]) for k in self.splits)
            src = (
                self.gt_coeff_excluded_file
                if self.gt_coeff_excluded_file is not None
                else Path(gt_path) / GT_COEFF_EXCLUDED_FILENAME
            )
            print(
                f"gt_coeff_excluded: dropped {n_before - n_after} indices "
                f"({len(excluded)} listed in {src})"
            )
        self.train_dataset = Subset(self.dataset, self.splits["train"])
        self.val_dataset = Subset(self.dataset, self.splits["validation"])
        self.test_dataset = Subset(self.dataset, self.splits["test"])
        if (Path(self.dataset.path) / 'metadata.json').exists():
            with open(self.dataset.path / 'metadata.json', 'r') as fp:
                self.metadata = json.load(fp)
        else:
            self.metadata = self.get_metadata()

    def get_metadata(self):
        x_sum = 0
        x_2 = 0
        unique_atom_types = set()
        avg_num_neighbors = 0
        print('get metadata.')
        progress = tqdm(total=len(self.train_dataset))
        for data in self.train_dataset:
            x_sum += data.chg_labels.mean()
            x_2 += (data.chg_labels ** 2).mean()
            unique_atom_types.update(data.atom_types.numpy().tolist())
            avg_num_neighbors += data.edge_index.shape[1] / len(data.atom_types) / 2
            progress.update(1)
        x_mean = x_sum / len(self.train_dataset)
        x_var = x_2 / len(self.train_dataset) - x_mean ** 2
        avg_num_neighbors = int(avg_num_neighbors / len(self.train_dataset))
        # this is the avg num neighbors without the probes
        metadata = {
            'target_mean': x_mean.item(), 
            'target_var': x_var.item(), 
            'avg_num_neighbors': avg_num_neighbors,
            'unique_atom_types': list(unique_atom_types)
        }
        with open(self.dataset.path / 'metadata.json', 'w') as fp:
            json.dump(metadata, fp)
        return metadata

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            worker_init_fn=worker_init_fn,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            worker_init_fn=worker_init_fn,
        )
        
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.dataset=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )


class ProbeDataModule(DataModule):
    def __init__(
        self,
        dataset: DictConfig,
        split_file: Optional[Path],
        num_workers: DictConfig,
        batch_size: DictConfig,
        n_probe: DictConfig,
        gt_coeff_excluded_file: Optional[Path] = None,
    ):
        super().__init__(
            dataset,
            split_file,
            num_workers,
            batch_size,
            gt_coeff_excluded_file=gt_coeff_excluded_file,
        )
        self.n_probe = n_probe
        
    def train_dataloader(self, shuffle=True):
        if self.n_probe.train > 0:
            return ProbeDataLoader(
                self.train_dataset,
                shuffle=shuffle,
                batch_size=self.batch_size.train,
                num_workers=self.num_workers.train,
                n_probe=self.n_probe.train,
                worker_init_fn=worker_init_fn,
            )
        else:
            return DataLoader(
                self.train_dataset,
                shuffle=shuffle,
                batch_size=self.batch_size.train,
                num_workers=self.num_workers.train,
                worker_init_fn=worker_init_fn,
            )

    def val_dataloader(self):
        if self.n_probe.val > 0:
            return ProbeDataLoader(
                self.val_dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                n_probe=self.n_probe.val,
                worker_init_fn=worker_init_fn,
            )
        else:
            return DataLoader(
                self.val_dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
            )
    
    def test_dataloader(self):
        if self.n_probe.test > 0:
            return ProbeDataLoader(
                self.test_dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                n_probe=self.n_probe.test,
                worker_init_fn=worker_init_fn,
            )
        else:
            return DataLoader(
                self.test_dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.dataset=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
            f"{self.n_probe=})"
        )