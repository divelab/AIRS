# More ordering studies
import pickle
from multiprocessing import Pool
from sklearn.utils import shuffle
import torch
import os, json
import os.path as osp
from itertools import repeat
import numpy as np
from rdkit import Chem
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
import re
import msgpack
import periodictable
from space_filling_curve_sort import *


def rbf(D, max, num_rbf):
    D_min, D_max, D_count = 0., max, num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count)
    D_mu = D_mu.view([1,1,1,-1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF

def split_data(dataset, val_proportion=0.1, test_proportion=0.1, from_perm_file=True):
    if from_perm_file:
        raw_dir = 'data/geom/'
        perm = np.load(raw_dir + 'geom_permutation.npy').astype(int)
        assert len(perm) == len(dataset)
        num_mol = len(dataset)
        val_index = int(num_mol * val_proportion)
        test_index = val_index + int(num_mol * test_proportion)

        train_dataset, val_dataset, test_dataset = dataset[perm[test_index:]], dataset[perm[:val_index]], dataset[perm[val_index:test_index]]
    else:
        print('not supported')
    return train_dataset, val_dataset, test_dataset


class DrugDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 processed_filename='data.pt',
                 sample=False,
                 remove_h=False,
                 no_feature=True,
                 ):

        self.processed_filename = processed_filename
        self.root = root
        self.name = f"{name}{'_no_feature' if no_feature else '_with_feature'}{'_no_h' if remove_h else '_with_h'}{'_sample' if sample else '_all'}"
        self.sample = sample
        self.num_conformations = 30
        self.remove_h = remove_h
        self.no_feature = no_feature

        super(DrugDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if osp.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process()

    @property
    def raw_dir(self):
        return osp.join(self.root)

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, self.name, name)

    @property
    def processed_file_names(self):
        return self.processed_filename

    def process(self):
        r"""Processes the dataset from raw data file to the :obj:`self.processed_dir` folder.
        """
        self.data, self.slices = self.pre_process()

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        print('making processed files:', self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get(self, idx):
        r"""Gets the data object at index :idx:.

        Args:
            idx: The index of the data that you want to reach.
        :rtype: A data object corresponding to the input index :obj:`idx` .
        """
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        return data

    def pre_process(self):
        data_list = []
        
        raw_file = osp.join(self.root, 'drugs_crude.msgpack')
        unpacker = msgpack.Unpacker(open(raw_file, "rb"))
        for i, drugs_1k in enumerate(unpacker):
            print(f"Unpacking file {i}...")
            for smiles, all_info in drugs_1k.items():
                conformers = all_info['conformers']
                # Get the energy of each conformer. Keep only the lowest values
                all_energies = []
                for conformer in conformers:
                    all_energies.append(conformer['totalenergy'])
                all_energies = np.array(all_energies)
                argsort = np.argsort(all_energies)
                lowest_energies = argsort[:self.num_conformations]
                for id in lowest_energies:
                    conformer = conformers[id]
                    coords = np.array(conformer['xyz']).astype(float)        # atom type + xyz
                    if self.remove_h:
                        mask = coords[:, 0] != 1.0
                        coords = coords[mask]
                    n = coords.shape[0]

                    data = Data()
                    data.smiles = smiles
                    data.z = torch.tensor(coords[:,0], dtype=torch.int64)
                    data.xyz = torch.tensor(coords[:,1:], dtype=torch.float32)
                    data.no = n
                    if not self.no_feature:
                        print('!!! include node and edge features, not supported !!!')
                        exit()
                    data_list.append(data)
            if self.sample:
                if len(data_list) > 100:
                    break
                
        data, slices = self.collate(data_list)
        return data, slices


from utils import relabel_undirected_graph, create_lab_ptn_from_weights, \
    coordinates_to_distances


def generate_tokens(graph_data):
    node_attr, edge_index, edge_attr, xyz, perms, label = graph_data
    n = node_attr.shape[0]
    canons = []
    labels = [perm[label] for perm in perms]
    for label in labels:
        canon = []
        node_perm = label[:n]
        edge_perm = label[n:] - n
        value_to_idx = {value: idx for idx, value in enumerate(edge_perm)}

        syms = node_attr[:, 0][node_perm]
        symbols = [periodictable.elements[an].symbol for an in syms]
        dist_token = edge_attr[:, 0][edge_perm].tolist()
        canon.append(symbols)
        canon.append(symbols + dist_token)
        third_labeling = []
        last_added_indices = set()
        for i in range(n):
            third_labeling.append(symbols[i])
            indices = np.where((edge_index[:, 0] == node_perm[i]) | (edge_index[:, 1] == node_perm[i]))[0]
            indices = set(indices)
            indices -= last_added_indices

            sorted_indices = sorted(indices, key=lambda x: value_to_idx[x])
            for idx in sorted_indices:
                third_labeling.append(dist_token[idx])
            last_added_indices = indices

        canon.append(third_labeling)
        canon.append([[symbols[i]] + xyz[node_perm][i].tolist() for i in range(n)])

        canons.append(canon)

    return canons


if __name__ == '__main__':    
    dataset = DrugDataset(root='data/geom/',
                           name='data',
                           processed_filename='data.pt',
                           sample=False,
                           remove_h=False,
                           no_feature=True)

    train_dataset, val_dataset, test_dataset = split_data(dataset)

    split = 'test'
    if split == 'train':
        dataset = train_dataset
    elif split == 'valid':
        dataset = val_dataset
    elif split == 'test':
        dataset = test_dataset
    else:
        raise NotImplementedError("error input")

    output_dir = 'data/Molecule3D/drugs/'
    chunk_files = [f for f in os.listdir(output_dir) if f.startswith(f'DRUG_{split}_labels_')]

    for chunk_file in tqdm(chunk_files):
        with open(os.path.join(output_dir, chunk_file), 'rb') as file:
            labelings = pickle.load(file)
        def process_graph_label(index):
            label = labelings[index]
            canons = generate_tokens(label)
            return canons

        num_items = len(labelings)
        with Pool(processes=32) as pool:
            labels = list(tqdm(pool.imap(process_graph_label, range(num_items)), total=num_items))

        # Save the processed labels in a corresponding chunked output file
        output_chunk_file = chunk_file.replace('labels', 'canons')
        with open(os.path.join(output_dir, output_chunk_file), 'wb') as file:
            pickle.dump(labels, file)
        chunk_index = int(chunk_file.split('_')[-1].split('.')[0])
        chunk_size = 1000  # size of each output chunk
        for i in tqdm(range(0, len(labels), chunk_size)):
            chunked_labels = labels[i:i + chunk_size]
            output_chunk_file = f'DRUG_{split}_canons_{chunk_index * 100 + i // chunk_size}.pkl'
            with open(os.path.join(output_dir, output_chunk_file), 'wb') as file:
                pickle.dump(chunked_labels, file)
            del chunked_labels
        del labelings
        del labels

