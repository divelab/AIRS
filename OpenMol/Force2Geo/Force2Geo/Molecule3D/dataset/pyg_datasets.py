from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data

from sklearn.utils import shuffle

from .utils import mol2graph
from .torsion_utils import *
# from utils import TransformOptim3D, TransformRDKit3D
import pandas as pd
import os.path as osp
import json
from itertools import repeat
from rdkit import Chem
from tqdm import tqdm
import pickle
import os


def split_data(dataset, seed=0, split_size=[0.8, 0.1, 0.1], from_json=False, sample_ratio=[100, 150, 150]):
    if not from_json:
        print("-----Random splitting-----")
        assert sum(split_size) == 1
        train_size = int(split_size[0] * len(dataset))
        train_val_size = int((split_size[0] + split_size[1]) * len(dataset))
        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size:train_val_size]
        test_dataset = dataset[train_val_size:]

    else:
        with open('../data/raw/random_split_inds.json') as f:
            inds = json.load(f)
        
        train_dataset, val_dataset, test_dataset = dataset[inds['train']], dataset[inds['valid']], \
                                                   dataset[inds['test']]
        # sample smaller subset
        if sample_ratio is not None:
            print("-----Random sampling-----")
            idx_train = shuffle(list(range(sample_ratio[0])), random_state=seed)
            idx_val = shuffle(list(range(sample_ratio[1])), random_state=seed)
            idx_test = shuffle(list(range(sample_ratio[2])), random_state=seed)
            train_dataset = train_dataset[idx_train]
            val_dataset = val_dataset[idx_val]
            test_dataset = test_dataset[idx_test]


    return train_dataset, val_dataset, test_dataset

class PyGDataModule(LightningDataModule):

    def __init__(
        self,
        root: str,
        dataset_name: str,
        train_size: float = 0.9,
        val_size: float = 0.1,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.dataset_predict = None

        self.root = root
        self.dataset_name = dataset_name
        self.seed = seed
        self.sizes = [train_size, val_size]
        dataloader_keys = [
            "batch_size",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "drop_last",
        ]
        self.dataloader_kwargs = {}
        for key in dataloader_keys:
            val = kwargs.get(key, None)
            self.dataloader_kwargs[key] = val
            if val is not None:
                del kwargs[key]
        self.kwargs = kwargs

    def dataloader(self, dataset, **kwargs):
        return DataLoader(dataset, **kwargs)

    def setup(self, stage: str) -> None:
        raise NotImplementedError

    def train_dataloader(self):
        return self.dataloader(self.dataset_train, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self):
        return self.dataloader(self.dataset_val, shuffle=False, **self.dataloader_kwargs)

    def test_dataloader(self):
        return self.dataloader(self.dataset_test, shuffle=False, **self.dataloader_kwargs)

    def predict_dataloader(self):
        return self.dataloader(self.dataset_predict, shuffle=False, **self.dataloader_kwargs)
    

class PyGMolecule3DDataModule(PyGDataModule):

    def __init__(
        self,
        root: str,
        dataset_name: str,
        train_size: float = None,
        val_size: float = None,
        **kwargs,
    ) -> None:
        super().__init__(root, dataset_name, train_size, val_size, **kwargs)
        

    def setup(self, stage: str) -> None:
        self.transform = self.kwargs.get('transform', None)
        self.pre_transform = self.kwargs.get('pre_transform', None)
        self.allowed_cids_path = self.kwargs.get('allowed_cids_path', None)
        self.processed_folder = self.kwargs.get('processed_folder', 'processed')
        self.split_mode = self.kwargs.get('split_mode', 'random')
        self.sample_relaxed_traj = self.kwargs.get('sample_relaxed_traj', False)
        self.position_noise_scale = self.kwargs.get('position_noise_scale', 0.04)

        # import pdb; pdb.set_trace()
        
        self.dataset_train = Molecule3D(
            root=self.root,
            name='data',
            processed_folder=self.processed_folder,
            split='train',
            split_mode=self.split_mode,
            transform=self.transform,
            pre_transform=self.pre_transform,
            pre_filter=None,
            allowed_cids_path=self.allowed_cids_path,
            sample_relaxed_traj=self.sample_relaxed_traj,
            position_noise_scale=self.position_noise_scale
        )
        
        self.dataset_val = Molecule3D(
            root=self.root,
            name='data',
            processed_folder=self.processed_folder,
            split='val',
            split_mode=self.split_mode,
            transform=self.transform,
            pre_transform=self.pre_transform,
            pre_filter=None,
            allowed_cids_path=self.allowed_cids_path,
            sample_relaxed_traj=self.sample_relaxed_traj,
            position_noise_scale=self.position_noise_scale
        )

        self.dataset_test = Molecule3D(
            root=self.root,
            name='data',
            processed_folder=self.processed_folder,
            split='test',
            split_mode=self.split_mode,
            transform=self.transform,
            pre_transform=self.pre_transform,
            pre_filter=None,
            allowed_cids_path=self.allowed_cids_path,
            sample_relaxed_traj=self.sample_relaxed_traj,
            position_noise_scale=self.position_noise_scale,
        )
        # import pdb; pdb.set_trace()
        # self.dataset_train, self.dataset_val, self.dataset_test = split_data(dataset, from_json=True, sample_ratio=None)

        print(len(self.dataset_train), ' mols in trainset')
        print(len(self.dataset_val), ' mols in valset')
        print(len(self.dataset_test), ' mols in testset')
    


class Molecule3D(InMemoryDataset):
    """
        A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for 
        datasets used in molecule generation.
        
        .. note::
            Some datasets may not come with any node labels, like :obj:`moses`. 
            Since they don't have any properties in the original data file. The process of the
            dataset can only save the current input property and will load the same  property 
            label when the processed dataset is used. You can change the augment :obj:`processed_filename` 
            to re-process the dataset with intended property.
        
        Args:
            root (string, optional): Root directory where the dataset should be saved. (default: :obj:`./`)
            split (string, optional): If :obj:`"train"`, loads the training dataset.
                If :obj:`"val"`, loads the validation dataset.
                If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
            split_mode (string, optional): Mode of split chosen from :obj:`"random"` and :obj:`"scaffold"`.
                (default: :obj:`penalized_logp`)
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)
        """
    
    def __init__(self,
                 root,
                 name='data',
                 processed_folder='processed',
                 split='train',
                 split_mode='random',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 allowed_cids_path=None,
                 sample_relaxed_traj=False,
                 position_noise_scale=0.04,
                 ):
        
        assert split in ['train', 'val', 'test']
        assert split_mode in ['random', 'scaffold']
        self.split_mode = split_mode
        self.root = root
        self.name = name
        self.processed_folder = processed_folder
        self.sample_relaxed_traj = sample_relaxed_traj
        self.position_noise_scale = position_noise_scale
        self.target_df = pd.read_csv(osp.join(self.raw_dir, 'properties.csv'))
        # if not osp.exists(self.raw_paths[0]):
        #     self.download()
        
        if allowed_cids_path is not None:
            with open(allowed_cids_path, "r") as file:
                allowed_cids = [int(line.strip().lstrip('0')) for line in file]
            self.allowed_cids = set(allowed_cids)
        
        # import pdb; pdb.set_trace()
        
        def filter_by_cid(data):
            return data.cid in self.allowed_cids
        
        # if split == 'test':
        #     pre_filter = filter_by_cid
        # else:
        #     pre_filter = None
        pre_filter = filter_by_cid
        
        # print(self.processed_dir)
        
        super(Molecule3D, self).__init__(root, transform, pre_transform, pre_filter)
        
        print("Loading {} {} split... from {}".format(split_mode, split, self.processed_dir))
        self.data, self.slices = torch.load(osp.join(self.processed_dir, '{}_{}.pt'.format(split_mode, split)))
    
    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0    
    
    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, self.processed_folder)

    @property
    def raw_file_names(self):
        name = self.name + '.csv'
        return name


    @property
    def processed_file_names(self):
        return ['random_train.pt', 'random_val.pt', 'random_test.pt']
    
    
    def download(self):
        # print('making raw files:', self.raw_dir)
        # if not osp.exists(self.raw_dir):
        #     os.makedirs(self.raw_dir)
        # url = self.url
        # path = download_url(url, self.raw_dir)
        pass
    

    def pre_process(self):
        data_list = []
        sdf_paths = [osp.join(self.raw_dir, 'combined_mols_0_to_1000000.sdf'),
                     osp.join(self.raw_dir, 'combined_mols_1000000_to_2000000.sdf'),
                     osp.join(self.raw_dir, 'combined_mols_2000000_to_3000000.sdf'),
                     osp.join(self.raw_dir, 'combined_mols_3000000_to_3899647.sdf')]
        suppl_list = [Chem.SDMolSupplier(p, removeHs=False, sanitize=True) for p in sdf_paths]
        
        abs_idx = -1
        for i, suppl in enumerate(suppl_list):
            for j in tqdm(range(len(suppl)), desc=f'{i+1}/{len(sdf_paths)}'):
                
                # if j > 100:
                #     break
                
                abs_idx += 1
                mol = suppl[j]
                try:
                    smiles = Chem.MolToSmiles(mol)
                except:
                    data_list.append(None)
                    continue
                coords = mol.GetConformer().GetPositions()
                z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

                graph = mol2graph(mol)
                data = Data()
                data.__num_nodes__ = int(graph['num_nodes'])
                data.smiles = smiles
                data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                data.xyz = torch.tensor(coords, dtype=torch.float32)
                data.z = torch.tensor(z, dtype=torch.int64)
                
                
                data.props = torch.FloatTensor(self.target_df.iloc[abs_idx,1:].values)
                data.cid = self.target_df.iloc[abs_idx,0]
                
                data_list.append(data)
                
        return data_list
    
    
    def process(self):
        r"""Processes the dataset from raw data file to the :obj:`self.processed_dir` folder.
        
            If one-hot format is required, the processed data type will include an extra dimension 
            of virtual node and edge feature.
        """
        print(f"split_mode: {self.split_mode}")
        full_list = self.pre_process()
       
        print('making processed files:', self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
        # for m, split_mode in enumerate(['random', 'scaffold']):
        for m, split_mode in enumerate([self.split_mode]):
            ind_path = osp.join(self.raw_dir, '{}_split_inds.json').format(split_mode)
            with open(ind_path, 'r') as f:
                 inds = json.load(f)
            
            for s, split in enumerate(['train', 'valid', 'test']):
                print('Post-processing {} split...'.format(split))
                # data_list = [self.get_data_prop(full_list, idx, split) for idx in inds[split]]
                data_list = []
                data_cid = []
                for idx in inds[split]:
                    data = self.get_data_prop(full_list, idx, split)
                    if data is not None:
                        data_list.append(data)
                        data_cid.append(data.cid)
                if split == 'test':
                    with open(os.path.join(self.processed_dir, f'{split_mode}_data_cid_test.pkl'), "wb") as f:
                        pickle.dump(data_cid, f)
                    # import pdb; pdb.set_trace()
                if self.pre_filter is not None and split == 'test':
                    print('Pre-filtering test split...')
                    data_list = [data for data in tqdm(data_list, desc="Pre-filter {}".format(split)) if self.pre_filter(data)]
                if self.pre_transform is not None:
                    data_list = [self.pre_transform(data) for data in data_list]

                torch.save(self.collate(data_list), self.processed_paths[s+3*m])

    def get_data_prop(self, full_list, abs_idx, split):
        data = full_list[abs_idx]
        if data is None:
            return None
        # if split == 'test':
        data.props = torch.FloatTensor(self.target_df.iloc[abs_idx,1:].values).unsqueeze(0)
        assert data.props.size(0) == 1
        assert data.props.size(1) == 7
        data.cid = self.target_df.iloc[abs_idx,0]
        return data
        
    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
    
    
    def transform_noise(self, data, position_noise_scale):
        noise = torch.randn_like(torch.tensor(data)) * position_noise_scale
        data_noise = data + noise.numpy()
        return data_noise
    
    def extract_mol(self, nested):
        while isinstance(nested, list):
            nested = nested[0]
        return nested
    
    
    def get(self, idx):
        r"""Gets the data object at index :idx:.
        
        Args:
            idx: The index of the data that you want to reach.
        :rtype: A data object corresponding to the input index :obj:`idx` .
        """
        data = self.data.__class__()

        # import pdb; pdb.set_trace()
        # if hasattr(self.data, '__num_nodes__'):
        #     data.num_nodes = self.data.__num_nodes__[idx]
        pre_key = ['x', 'xyz']
        for key in pre_key:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        
        data.num_nodes = data.x.shape[0]

        for key in self.data.keys():
            if key in pre_key:
                continue
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            
            if self.sample_relaxed_traj and key == 'relaxed_traj_xyz':
                num_node = data.num_nodes
                tmp = item[s]
                relaxed_traj = tmp.view(-1, num_node, 3)
                sampled_idx = torch.randint(relaxed_traj.size(0), (1,))
                sampled_relaxed_xyz = relaxed_traj[sampled_idx]
                if sampled_relaxed_xyz.dim() == 3:
                    sampled_relaxed_xyz = sampled_relaxed_xyz.squeeze(0)
                # data[key] = sampled_relaxed_xyz
                data['relaxed_xyz'] = sampled_relaxed_xyz
                if 'relaxed_traj_force' in self.data.keys():
                    tmp = self.data['relaxed_traj_force'][s]
                    relaxed_traj_force = tmp.view(-1, num_node, 3)
                    sampled_relaxed_force = relaxed_traj_force[sampled_idx]
                    if sampled_relaxed_force.dim() == 3:
                        sampled_relaxed_force = sampled_relaxed_force.squeeze(0)
                    data['relaxed_force'] = sampled_relaxed_force
            elif self.sample_relaxed_traj and (key == 'relaxed_xyz' or key == 'relaxed_traj_force'):
                continue
            else:
                data[key] = item[s]
            
            
        # del data['GT_mol']
        return data
    
if __name__ == '__main__':
    from omegaconf import OmegaConf
    import hydra

    
    datamodule = PyGMolecule3DDataModule(
        root='../',
        seed=42,
        dataset_name="Molecule3D",
        split_mode="random",
        allowed_cids_path="../data/raw/allowed_test_cids_scaffold.txt",
        processed_folder="processed",
        sample_relaxed_traj=True,
        position_noise_scale=0.02
    )
    
    datamodule.setup("fit")
    
    # trainset = datamodule.dataset_train
    valset = datamodule.dataset_val
    data = valset[0]