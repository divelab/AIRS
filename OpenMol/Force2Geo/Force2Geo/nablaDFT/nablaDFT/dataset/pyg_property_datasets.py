"""Module describes PyTorch Geometric interfaces for nablaDFT datasets"""

import logging
import os
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch
from ase.db import connect
from torch_geometric.data import Data, Dataset, InMemoryDataset
from tqdm import tqdm

from nablaDFT.dataset.registry import dataset_registry
from nablaDFT.utils import download_file

from .hamiltonian_dataset import HamiltonianDatabase

import pandas as pd

logger = logging.getLogger(__name__)


class PyGNablaDFTProperty(InMemoryDataset):
    """Pytorch Geometric interface for nablaDFT datasets.

    Based on `MD17 implementation <https://github.com/atomicarchitects/equiformer/blob/master/datasets/pyg/md17.py>`_.

    .. code-block:: python
        from nablaDFT.dataset import PyGNablaDFT

        dataset = PyGNablaDFT(
            datapath="./datasets/",
            dataset_name="dataset_train_tiny",
            split="train",
        )
        sample = dataset[0]

    .. note::
        If split parameter is 'train' or 'test' and dataset name are ones from nablaDFT splits
        (see nablaDFT/links/energy_databases.json), dataset will be downloaded automatically.

    Args:
        datapath (str): path to existing dataset directory or location for download.
        dataset_name (str): split name from links .json or filename of existing file from datapath directory.
        split (str): type of split, must be one of ['train', 'test', 'predict'].
        transform (Callable): callable data transform, called on every access to element.
        pre_transform (Callable): callable data transform, called on every element during process.
    """

    db_suffix = ".db"

    @property
    def raw_file_names(self) -> List[str]:
        return [(self.dataset_name + self.db_suffix)]

    @property
    def processed_file_names(self) -> str:
        return f"{self.dataset_name}_property_{self.split}.pt"

    def __init__(
        self,
        datapath: str = "database",
        dataset_name: str = "dataset_train_tiny",
        split: str = "train",
        transform: Callable = None,
        pre_transform: Callable = None,
    ):
        self.dataset_name = dataset_name
        self.datapath = datapath
        self.split = split
        self.data_all, self.slices_all = [], []
        self.offsets = [0]
        super(PyGNablaDFTProperty, self).__init__(datapath, transform, pre_transform)

        for path in self.processed_paths:
            data, slices = torch.load(path)
            self.data_all.append(data)
            self.slices_all.append(slices)
            self.offsets.append(len(slices[list(slices.keys())[0]]) - 1 + self.offsets[-1])

    def len(self) -> int:
        return sum(len(slices[list(slices.keys())[0]]) - 1 for slices in self.slices_all)

    def get(self, idx):
        data_idx = 0
        while data_idx < len(self.data_all) - 1 and idx >= self.offsets[data_idx + 1]:
            data_idx += 1
        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        return super(PyGNablaDFTProperty, self).get(idx - self.offsets[data_idx])

    def download(self) -> None:
        url = dataset_registry.get_dataset_url("energy", self.dataset_name)
        dataset_etag = dataset_registry.get_dataset_etag("energy", self.dataset_name)
        download_file(
            url,
            Path(self.raw_paths[0]),
            dataset_etag,
            desc=f"Downloading split: {self.dataset_name}",
        )

    def process(self) -> None:
        # import pdb; pdb.set_trace()
        print("loading summary.csv ...")
        df = pd.read_csv("<path>/summary.csv")
        print("Finish loading summary.csv ...")
        
        db = connect(self.raw_paths[0])
        samples = []
        count = 0
        for db_row in tqdm(db.select(), total=len(db)):
            z = torch.from_numpy(db_row.numbers.copy()).long()
            positions = torch.from_numpy(db_row.positions.copy()).float()
            y = torch.from_numpy(np.array(db_row.data["energy"])).float()
            forces = torch.from_numpy(np.array(db_row.data["forces"])).float()

            # extract some properties
            df_single = df[(df['MOSES id'] == db_row.moses_id) & (df['CONFORMER id'] == db_row.conformation_id)]
            dipole_x = torch.from_numpy(np.array(df_single['DFT DIPOLE X'].item())).float()
            dipole_y = torch.from_numpy(np.array(df_single['DFT DIPOLE Y'].item())).float()
            dipole_z = torch.from_numpy(np.array(df_single['DFT DIPOLE Z'].item())).float()
            homo = torch.from_numpy(np.array(df_single['DFT HOMO'].item())).float()
            lumo = torch.from_numpy(np.array(df_single['DFT LUMO'].item())).float()
            homo_lumo = torch.from_numpy(np.array(df_single['DFT HOMO-LUMO GAP'].item())).float()
            # import pdb;pdb.set_trace()
            y = torch.cat([
                y,
                dipole_x.unsqueeze(0),
                dipole_y.unsqueeze(0),
                dipole_z.unsqueeze(0),
                homo.unsqueeze(0),
                lumo.unsqueeze(0),
                homo_lumo.unsqueeze(0)
            ]).unsqueeze(0)
            
            assert y.ndim == 2
            assert y.shape[0] == 1
            
            samples.append(Data(z=z, pos=positions, y=y, forces=forces))

            count += 1
            if self.split == 'test':
                if count == 10000:
                    break

        if self.pre_filter is not None:
            samples = [data for data in samples if self.pre_filter(data)]

        if self.pre_transform is not None:
            samples = [self.pre_transform(data) for data in samples]

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"Saved processed dataset: {self.processed_paths[0]}")


class PyGHamiltonianNablaDFT(Dataset):
    """Pytorch Geometric interface for nablaDFT Hamiltonian datasets.

    .. code-block:: python
        from nablaDFT.dataset import (
            PyGHamiltonianNablaDFT,
        )

        dataset = PyGHamiltonianNablaDFT(
            datapath="./datasets/",
            dataset_name="dataset_train_tiny",
            split="train",
        )
        sample = dataset[0]

    .. note::
        If split parameter is 'train' or 'test' and dataset name are ones from nablaDFT splits
        (see nablaDFT/links/hamiltonian_databases.json), dataset will be downloaded automatically.

    .. note::
        Hamiltonian matrix for each molecule has different shape. PyTorch Geometric tries to concatenate
        each torch.Tensor in batch, so in order to make batch from data we leave all hamiltonian matrices
        in numpy array form. During train, these matrices will be yield as List[np.array].

    Args:
        datapath (str): path to existing dataset directory or location for download.
        dataset_name (str): split name from links .json or filename of existing file from datapath directory.
        split (str): type of split, must be one of ['train', 'test', 'predict'].
        include_hamiltonian (bool): if True, retrieves full Hamiltonian matrices from database.
        include_overlap (bool): if True, retrieves overlap matrices from database.
        include_core (bool): if True, retrieves core Hamiltonian matrices from database.
        dtype (torch.dtype): defines torch.dtype for energy, positions and forces tensors.
        transform (Callable): callable data transform, called on every access to element.
        pre_transform (Callable): callable data transform, called on every element during process.
    """

    db_suffix = ".db"

    @property
    def raw_file_names(self) -> List[str]:
        return [(self.dataset_name + self.db_suffix)]

    @property
    def processed_file_names(self) -> str:
        return f"{self.dataset_name}_{self.split}.pt"

    def __init__(
        self,
        datapath: str = "database",
        dataset_name: str = "dataset_train_tiny",
        split: str = "train",
        include_hamiltonian: bool = True,
        include_overlap: bool = False,
        include_core: bool = False,
        dtype=torch.float32,
        transform: Callable = None,
        pre_transform: Callable = None,
    ):
        self.dataset_name = dataset_name
        self.datapath = datapath
        self.split = split
        self.data_all, self.slices_all = [], []
        self.offsets = [0]
        self.dtype = dtype
        self.include_hamiltonian = include_hamiltonian
        self.include_overlap = include_overlap
        self.include_core = include_core

        super(PyGHamiltonianNablaDFT, self).__init__(datapath, transform, pre_transform)

        self.max_orbitals = self._get_max_orbitals(datapath, dataset_name)
        self.db = HamiltonianDatabase(self.raw_paths[0])

    def len(self) -> int:
        return len(self.db)

    def get(self, idx):
        data = self.db[idx]
        z = torch.tensor(data[0].copy()).long()
        positions = torch.tensor(data[1].copy()).to(self.dtype)
        # see notes
        hamiltonian = data[4].copy()
        if self.include_overlap:
            overlap = data[5].copy()
        else:
            overlap = None
        if self.include_core:
            core = data[6].copy()
        else:
            core = None
        y = torch.from_numpy(data[2].copy()).to(self.dtype)
        forces = torch.from_numpy(data[3].copy()).to(self.dtype)
        data = Data(
            z=z,
            pos=positions,
            y=y,
            forces=forces,
            hamiltonian=hamiltonian,
            overlap=overlap,
            core=core,
        )
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        return data

    def download(self) -> None:
        url = dataset_registry.get_dataset_url("hamiltonian", self.dataset_name)
        dataset_etag = dataset_registry.get_dataset_etag("hamiltonian", self.dataset_name)
        download_file(
            url,
            Path(self.raw_paths[0]),
            dataset_etag,
            desc=f"Downloading split: {self.dataset_name}",
        )

    def _get_max_orbitals(self, datapath, dataset_name):
        db_path = os.path.join(datapath, "raw/" + dataset_name + self.db_suffix)
        if not os.path.exists(db_path):
            self.download()
        database = HamiltonianDatabase(db_path)
        max_orbitals = []
        for z in database.Z:
            max_orbitals.append(tuple((int(z), int(orb_num)) for orb_num in database.get_orbitals(z)))
        max_orbitals = tuple(max_orbitals)
        return max_orbitals
