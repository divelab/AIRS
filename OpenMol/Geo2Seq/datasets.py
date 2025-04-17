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
import networkx as nx
from torch_geometric.utils import from_networkx
from copy import deepcopy
from space_filling_curve_sort import *
import pandas as pd

from utils import expand_edges_and_attributes, relabel_undirected_graph, create_lab_ptn_from_weights, \
    coordinates_to_distances

import periodictable


chirality_map = {
    0: "",
    1: "@",
    2: "@@",
    3: ""
    # "misc": "miscellaneous chiral configuration"
}

# allowable multiple choice node and edge features
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'
    ],
    'possible_formal_charge_list': [
        -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'
    ],
    'possible_numH_list': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'
    ],
    'possible_number_radical_e_list': [
        0, 1, 2, 3, 4, 'misc'
    ],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list': [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
        safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
        allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
        safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
        safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
        safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
        safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
        safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
        allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
        allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
    ]
    return atom_feature


def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
    ]))


# modified by ZX
def bond_to_feature_vector_ve(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
        allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
        allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
        0,  # Not a virtual edge
        1  # hop of real edge = 1
    ]
    return bond_feature


# original verison
def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
        allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
        allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
    ]
    return bond_feature


def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list']
    ]))


def atom_feature_vector_to_dict(atom_feature):
    [atomic_num_idx,
     chirality_idx,
     degree_idx,
     formal_charge_idx,
     num_h_idx,
     number_radical_e_idx,
     hybridization_idx,
     is_aromatic_idx,
     is_in_ring_idx] = atom_feature

    feature_dict = {
        'atomic_num': allowable_features['possible_atomic_num_list'][atomic_num_idx],
        'chirality': allowable_features['possible_chirality_list'][chirality_idx],
        'degree': allowable_features['possible_degree_list'][degree_idx],
        'formal_charge': allowable_features['possible_formal_charge_list'][formal_charge_idx],
        'num_h': allowable_features['possible_numH_list'][num_h_idx],
        'num_rad_e': allowable_features['possible_number_radical_e_list'][number_radical_e_idx],
        'hybridization': allowable_features['possible_hybridization_list'][hybridization_idx],
        'is_aromatic': allowable_features['possible_is_aromatic_list'][is_aromatic_idx],
        'is_in_ring': allowable_features['possible_is_in_ring_list'][is_in_ring_idx]
    }

    return feature_dict


def bond_feature_vector_to_dict(bond_feature):
    [bond_type_idx,
     bond_stereo_idx,
     is_conjugated_idx] = bond_feature

    feature_dict = {
        'bond_type': allowable_features['possible_bond_type_list'][bond_type_idx],
        'bond_stereo': allowable_features['possible_bond_stereo_list'][bond_stereo_idx],
        'is_conjugated': allowable_features['possible_is_conjugated_list'][is_conjugated_idx]
    }

    return feature_dict


def virtual_bond_features(hop):
    fbond = [
        len(allowable_features['possible_bond_type_list']),  # not belong to any type
        len(allowable_features['possible_bond_stereo_list']),  # not belong to any type
        len(allowable_features['possible_is_conjugated_list']),  # not belong to any type
        1,  # is a virtual edge
        int(hop)
    ]
    return fbond


def mol2graph_virtual_edge(mol):
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))

    G = nx.Graph()
    for i in range(len(atom_features_list)):
        G.add_node(i)
        G.nodes[i]['x'] = atom_features_list[i]

    # egdes
    for i in range(len(atom_features_list)):
        for j in range(i + 1, len(atom_features_list)):
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond:
                G.add_edge(i, j)
                G.edges[i, j]['edge_attr'] = bond_to_feature_vector_ve(bond)

    G_virtual = G.copy()

    for i in range(len(atom_features_list)):
        for j in range(i + 1, len(atom_features_list)):
            bond = mol.GetBondBetweenAtoms(i, j)
            if not bond:
                try:
                    hop = nx.shortest_path_length(G, source=i, target=j)
                    G_virtual.add_edge(i, j)
                    G_virtual.edges[i, j]['edge_attr'] = virtual_bond_features(hop)
                except:
                    continue

    # from nx_graph to pyg_graph
    graph = from_networkx(G_virtual)
    return graph


def mol2graph_virtual_edge_2hop(mol):
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))

    G = nx.Graph()
    for i in range(len(atom_features_list)):
        G.add_node(i)
        G.nodes[i]['x'] = atom_features_list[i]

    # egdes
    for i in range(len(atom_features_list)):
        for j in range(i + 1, len(atom_features_list)):
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond:
                G.add_edge(i, j)
                G.edges[i, j]['edge_attr'] = bond_to_feature_vector_ve(bond)

    G_virtual = G.copy()

    for i in range(len(atom_features_list)):
        for j in range(i + 1, len(atom_features_list)):
            bond = mol.GetBondBetweenAtoms(i, j)
            if not bond:
                try:
                    hop = nx.shortest_path_length(G, source=i, target=j)
                    if hop == 2:
                        G_virtual.add_edge(i, j)
                        G_virtual.edges[i, j]['edge_attr'] = virtual_bond_features(hop)
                except:
                    continue

    # from nx_graph to pyg_graph
    graph = from_networkx(G_virtual)
    return graph


def mol2graph(mol):
    """
    Converts molecule object to graph Data object
    :input: molecule object
    :return: graph object
    """

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph


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
        with open('data/Molecule3D/splits/random_split_inds.json') as f:
            inds = json.load(f)
        train_dataset, val_dataset, test_dataset = dataset[inds['train']], dataset[inds['valid']], dataset[inds['test']]

        if sample_ratio is not None:
            print("-----Random sampling-----")
            idx_train = shuffle(list(range(sample_ratio[0])), random_state=seed)
            idx_val = shuffle(list(range(sample_ratio[1])), random_state=seed)
            idx_test = shuffle(list(range(sample_ratio[2])), random_state=seed)
            train_dataset = train_dataset[idx_train]
            val_dataset = val_dataset[idx_val]
            test_dataset = test_dataset[idx_test]

    return train_dataset, val_dataset, test_dataset


def scanffold_split_data(dataset, seed=0, sample_ratio=[100, 150, 150]):
    with open('data/Molecule3D/splits/scaffold_split_inds.json') as f:
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


class Mol3dDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 processed_filename='data_replicate.pt',
                 sample=False,
                 ):

        self.processed_filename = processed_filename
        self.root = root
        self.name = name
        self.sample = sample

        super(Mol3dDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if osp.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process()

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
        name = 'processed'
        return osp.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        name = self.name + '.csv'
        return name

    @property
    def processed_file_names(self):
        return self.processed_filename

    def download(self):
        pass

    def process(self):
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

        raw_dir = '/mnt/dive/shared/kaleb/Datasets/PubChemQC/raw_08132021'

        sdf_paths = [osp.join(raw_dir, 'combined_mols_0_to_1000000.sdf'),
                     osp.join(raw_dir, 'combined_mols_1000000_to_2000000.sdf'),
                     osp.join(raw_dir, 'combined_mols_2000000_to_3000000.sdf'),
                     osp.join(raw_dir, 'combined_mols_3000000_to_3899647.sdf')]
        suppl_list = [Chem.SDMolSupplier(p, removeHs=False, sanitize=True) for p in sdf_paths]

        for i, suppl in enumerate(suppl_list):
            for j in tqdm(range(len(suppl)), desc=f'{i + 1}/{len(sdf_paths)}'):

                if self.sample:
                    if j > 100:
                        break

                mol = suppl[j]
                smiles = Chem.MolToSmiles(mol)
                coords = mol.GetConformer().GetPositions()
                z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

                graph = mol2graph(mol)
                data = Data()

                data.__num_nodes__ = int(graph['num_nodes'])
                data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                data.smiles = smiles
                data.xyz = torch.tensor(coords, dtype=torch.float32)
                data.z = torch.tensor(z, dtype=torch.int64)
                data.no = torch.arange(graph['num_nodes'])
                data_list.append(data)

        data, slices = self.collate(data_list)
        return data, slices


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

        syms = node_attr[node_perm][:, 0] + 1
        chiralities = node_attr[node_perm][:, 1]
        symbols = [periodictable.elements[an].symbol for an in syms]
        mapped_chirality = [s + chirality_map[chirality] for s, chirality in zip(symbols, chiralities)]
        dist_token = edge_attr[:, 3][edge_perm].tolist()
        canon.append(mapped_chirality)
        canon.append(mapped_chirality + dist_token)
        third_labeling = []
        last_added_indices = set()
        for i in range(n):
            third_labeling.append(symbols[i] + chirality_map[chiralities[i]])
            indices = np.where((edge_index[:, 0] == node_perm[i]) | (edge_index[:, 1] == node_perm[i]))[0]
            indices = set(indices)
            indices -= last_added_indices

            sorted_indices = sorted(indices, key=lambda x: value_to_idx[x])
            for idx in sorted_indices:
                third_labeling.append(dist_token[idx])
            last_added_indices = indices

        canon.append(third_labeling)
        canon.append([[symbols[i] + chirality_map[chiralities[i]]] + xyz[node_perm][i].tolist() for i in range(n)])

        canons.append(canon)


    return canons


if __name__ == '__main__':
    root_path = 'data/Molecule3D'
    target = 'gap'
    split = 'random'
    dataset = Mol3dDataset(root=root_path,
                           name='data_baseline',
                           processed_filename='data.pt',
                           sample=False)
    dataset.data.y = dataset.data[target]
    dataset.slices['y'] = dataset.slices[target]

    if split == 'random':
        train_dataset, val_dataset, test_dataset = split_data(dataset, from_json=True, sample_ratio=None)
    elif split == 'scaffold':
        train_dataset, val_dataset, test_dataset = scanffold_split_data(dataset, sample_ratio=None)
    else:
        print('split method not supported')
    print('train, validaion, test:', len(train_dataset), len(val_dataset), len(test_dataset))

    labelings = pickle.load(open("data/Molecule3D/test_labels.pkl", "rb"))
    def process_graph_label(index):
        label = labelings[index]
        canons = generate_tokens(label)
        return canons

    num_items = len(labelings)  # Replace with the actual number of items in the dataset
    with Pool(processes=32) as pool:
        labels = list(tqdm(pool.imap(process_graph_label, range(num_items)), total=num_items))

    with open('data/Molecule3D/test_canons.pkl', 'wb') as file:
        pickle.dump(labels, file)

