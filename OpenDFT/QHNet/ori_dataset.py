import os
import torch
import tarfile
import numpy as np
import os.path as osp
from tqdm import tqdm
from ase.db import connect
from argparse import Namespace
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url

import logging
logger = logging.getLogger()


def random_split(dataset, lengths, seed=None):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = np.random.RandomState(seed=seed).permutation(sum(lengths))
    return [torch.utils.data.Subset(dataset, indices[offset - length:offset])
            for offset, length in zip(torch._utils._accumulate(lengths), lengths)]


def get_mask(data):
    mask_period_group_1 = torch.tensor([1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    mask_period_group_2 = torch.ones(14)

    mask_row = []
    for atom in data.atoms:
        mask_row.append(mask_period_group_1 if atom < 2 else mask_period_group_2)

    data.mask_row = torch.stack(mask_row, dim=0)
    return data


def hamiltonian_transform(hamiltonian, atoms):
    conv = Namespace(
        atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5]},
    )

    orbitals = ''
    orbitals_order = []
    for a in atoms:
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a]]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append(np.array(map_idx) + offset)
        transform_signs.append(np.array(map_sign))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = np.concatenate(transform_indices).astype(np.int)
    transform_signs = np.concatenate(transform_signs)

    hamiltonian_new = hamiltonian[...,transform_indices, :]
    hamiltonian_new = hamiltonian_new[...,:, transform_indices]
    hamiltonian_new = hamiltonian_new * transform_signs[:, None]
    hamiltonian_new = hamiltonian_new * transform_signs[None, :]
    return hamiltonian_new


class MD17_DFT(InMemoryDataset):
    def __init__(self, root='dataset/', name='water',
                 transform=None, pre_transform=None,
                 pre_filter=None):

        # water, ethanol, malondialdehyde, uracil
        self.name = name
        self.folder = osp.join(root, self.name)
        self.url = 'http://quantum-machine.org/data/schnorb_hamiltonian'
        self.chemical_symbols = ['n', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O']
        self.atom_types = None

        orbitals_ref = {}
        orbitals_ref[1] = np.array([0, 0, 1])  # H: 2s 1p
        orbitals_ref[6] = np.array([0, 0, 0, 1, 1, 2])  # C: 3s 2p 1d
        orbitals_ref[7] = np.array([0, 0, 0, 1, 1, 2])  # N: 3s 2p 1d
        orbitals_ref[8] = np.array([0, 0, 0, 1, 1, 2])  # O: 3s 2p 1d
        self.orbitals_ref = orbitals_ref

        orbitals = []
        if name == 'water':
            atoms = [8, 1, 1]
        elif name == 'ethanol':
            atoms = [6, 6, 8, 1, 1, 1, 1, 1, 1]
        elif name == 'malondialdehyde':
            atoms = [6, 6, 6, 8, 8, 1, 1, 1, 1]
        elif name == 'uracil':
            atoms = [6, 6, 7, 6, 7, 6, 8, 8, 1, 1, 1, 1]
        elif name == 'aspirin':
            atoms = [6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8,
                     1, 1, 1, 1, 1, 1, 1, 1]
        for Z in atoms:
            orbitals.append(tuple((int(Z),int(l)) for l in self.orbitals_ref[Z]))
        self.orbitals = tuple(orbitals)

        super(MD17_DFT, self).__init__(self.folder, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if not self.atom_types:
            self.atom_types = ''.join([self.chemical_symbols[i] for i in self[0].atoms])

    @property
    def raw_file_names(self):
        if self.name == 'ethanol':
            return [f'schnorb_hamiltonian_{self.name}_dft.tgz',
                    f'schnorb_hamiltonian_{self.name}_dft.db']
        elif self.name == 'aspirin':
            return [f'schnorb_hamiltonian_{self.name}_quambo.db',
                    f'schnorb_hamiltonian_{self.name}_quambo.db']
        else:
            return [f'schnorb_hamiltonian_{self.name}.tgz',
                    f'schnorb_hamiltonian_{self.name}.db']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        if self.name == 'ethanol':
            url = f'{self.url}/schnorb_hamiltonian_{self.name}' + '_dft.tgz'
        else:
            url = f'{self.url}/schnorb_hamiltonian_{self.name}' + '.tgz'
        download_url(url, self.raw_dir)
        extract_path = self.raw_dir
        tar = tarfile.open(os.path.join(self.raw_dir, self.raw_file_names[0]), 'r')
        for item in tar:
            tar.extract(item, extract_path)

    def process(self):
        db = connect(osp.join(self.raw_dir, self.raw_file_names[1]))
        data_list = []
        if not getattr(self, "atom_types"):
            self.atom_types = ''.join([
                self.chemical_symbols[i] for i in next(db.select(1))['numbers']])

        for row in tqdm(db.select()):
            data_list.append(self.get_mol(row))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_mol(self, row):
        # from angstrom to bohr
        # make sure the original data type is float or double
        pos = torch.tensor(row['positions'] * 1.8897261258369282, dtype=torch.float64)
        atoms = torch.tensor(row['numbers'], dtype=torch.int64).view(-1, 1)
        energy = torch.tensor(row.data['energy'], dtype=torch.float64)
        force = torch.tensor(row.data['forces'], dtype=torch.float64)
        hamiltonian = torch.tensor(hamiltonian_transform(
            row.data['hamiltonian'], self.atom_types), dtype=torch.float64)
        overlap = torch.tensor(hamiltonian_transform(
            row.data['overlap'], self.atom_types), dtype=torch.float64)

        data = Data(pos=pos,
                    atoms=atoms,
                    energy=energy,
                    force=force,
                    hamiltonian=hamiltonian,
                    overlap=overlap)

        return data


class Mixed_MD17_DFT(InMemoryDataset):
    def __init__(self, root='dataset/', name='all',
                 transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = name
        self.folder = osp.join(root, self.name)
        if self.name == 'all_split':
            self.names = ['water']
        elif self.name == 'all':
            self.names = ['water', 'ethanol', 'malondialdehyde', 'uracil']
        else:
            raise NotImplementedError(
                f"wrong dataset name, please set it to all instead of {self.name}")

        self.url = 'http://quantum-machine.org/data/schnorb_hamiltonian'
        self.chemical_symbols = ['n', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O']
        self.orbital_mask = {}
        idx_1s_2s = torch.tensor([0, 1])
        idx_2p = torch.tensor([3, 4, 5])
        orbital_mask_line1 = torch.cat([idx_1s_2s, idx_2p])
        orbital_mask_line2 = torch.arange(14)
        for i in range(1, 11):
            self.orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2

        super(Mixed_MD17_DFT, self).__init__(self.folder, transform, pre_transform, pre_filter)
        self.data, self.slices, self.train_mask, self.val_mask, self.test_mask = \
            torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_file_list = []
        for name in self.names:
            if name == 'ethanol':
                raw_file_list.append(f'schnorb_hamiltonian_{name}_dft.tgz')
                raw_file_list.append(f'schnorb_hamiltonian_{name}_dft.db')
            else:
                raw_file_list.append(f'schnorb_hamiltonian_{name}.tgz')
                raw_file_list.append(f'schnorb_hamiltonian_{name}.db')
        return raw_file_list

    @property
    def processed_file_names(self):
        return ['mixed_md17_data.pt']

    def download(self):
        for name_idx, name in enumerate(self.names):
            if name != 'aspirin':
                if name == 'ethanol':
                    url = f'{self.url}/schnorb_hamiltonian_{name}' + '_dft.tgz'
                else:
                    url = f'{self.url}/schnorb_hamiltonian_{name}' + '.tgz'
                download_url(url, self.raw_dir)
                extract_path = self.raw_dir
                tar = tarfile.open(os.path.join(self.raw_dir, self.raw_file_names[name_idx * 2]), 'r')
                for item in tar:
                    tar.extract(item, extract_path)

    def process(self):
        seed = 42
        data_list = []
        lengths = {
            'water': [2000, 500, 1000],
            'ethanol': [10000, 500, 1000],
            'malondialdehyde': [10000, 500, 1000],
            'uracil': [10000, 500, 1000],
        }
        train_mask = []
        val_mask = []
        test_mask =[]
        start_index = 0
        for name_idx, name in enumerate(self.names[:2]):
            db = connect(osp.join(self.raw_dir, self.raw_file_names[name_idx * 2 + 1]))
            indices = np.random.RandomState(seed=seed).permutation(sum(lengths[name]))
            train_mask.append(indices[:lengths[name][0]] + start_index)
            val_mask.append(indices[lengths[name][0]: lengths[name][0] + lengths[name][1]] + start_index)
            test_mask.append(indices[lengths[name][0] + lengths[name][1]:
                                     lengths[name][0] + lengths[name][1] + lengths[name][2]] + start_index)
            start_index = start_index + sum(lengths[name])
            for row in tqdm(db.select()):
                data_list.append(self.get_mol(row))
        data, slices = self.collate(data_list)

        train_mask = torch.tensor(np.concatenate(train_mask))
        val_mask = torch.tensor(np.concatenate(val_mask))
        test_mask = torch.tensor(np.concatenate(test_mask))
        print('Saving...')
        torch.save((data, slices, train_mask, val_mask, test_mask), self.processed_paths[0])

    def get_mol(self, row):
        pos = torch.tensor(row['positions'] * 1.8897261258369282, dtype=torch.float64)
        atoms = torch.tensor(row['numbers'], dtype=torch.int64).view(-1, 1)
        energy = torch.tensor(row.data['energy'], dtype=torch.float64)
        force = torch.tensor(row.data['forces'], dtype=torch.float64)
        atom_types = ''.join([self.chemical_symbols[i] for i in atoms])
        hamiltonian = torch.tensor(hamiltonian_transform(
            row.data['hamiltonian'], atom_types), dtype=torch.float64)
        overlap = torch.tensor(hamiltonian_transform(
            row.data['overlap'], atom_types), dtype=torch.float64)

        hamiltonian_diagonal_blocks, hamiltonian_non_diagonal_blocks, \
        hamiltonian_diagonal_block_masks, hamiltonian_non_diagonal_block_masks = \
            self.cut_matrix(hamiltonian, atoms)
        overlap_diagonal_blocks, overlap_non_diagonal_blocks, \
        overlap_diagonal_block_masks, overlap_non_diagonal_block_masks  = \
            self.cut_matrix(overlap, atoms)
        data = Data(pos=pos,
                    atoms=atoms,
                    energy=energy,
                    force=force,
                    hamiltonian_diagonal_blocks=hamiltonian_diagonal_blocks,
                    hamiltonian_non_diagonal_blocks=hamiltonian_non_diagonal_blocks,
                    hamiltonian_diagonal_block_masks=hamiltonian_diagonal_block_masks,
                    hamiltonian_non_diagonal_block_masks=hamiltonian_non_diagonal_block_masks,
                    overlap_diagonal_blocks=overlap_diagonal_blocks,
                    overlap_non_diagonal_blocks=overlap_non_diagonal_blocks,
                    overlap_diagonal_block_masks=overlap_diagonal_block_masks,
                    overlap_non_diagonal_block_masks=overlap_non_diagonal_block_masks)
        return data

    def cut_matrix(self, matrix, atoms):
        all_diagonal_matrix_blocks = []
        all_non_diagonal_matrix_blocks = []
        all_diagonal_matrix_block_masks = []
        all_non_diagonal_matrix_block_masks = []
        col_idx = 0
        for idx_i, atom_i in enumerate(atoms): # (src)
            row_idx = 0
            atom_i = atom_i.item()
            mask_i = self.orbital_mask[atom_i]
            for idx_j, atom_j in enumerate(atoms): # (dst)
                atom_j = atom_j.item()
                mask_j = self.orbital_mask[atom_j]
                matrix_block = torch.zeros(14, 14).type(torch.float64)
                matrix_block_mask = torch.zeros(14, 14).type(torch.float64)
                extracted_matrix = \
                    matrix[row_idx: row_idx + len(mask_j), col_idx: col_idx + len(mask_i)]

                # for matrix_block
                tmp = matrix_block[mask_j]
                tmp[:, mask_i] = extracted_matrix
                matrix_block[mask_j] = tmp

                tmp = matrix_block_mask[mask_j]
                tmp[:, mask_i] = 1
                matrix_block_mask[mask_j] = tmp

                if idx_i == idx_j:
                    all_diagonal_matrix_blocks.append(matrix_block)
                    all_diagonal_matrix_block_masks.append(matrix_block_mask)
                else:
                    all_non_diagonal_matrix_blocks.append(matrix_block)
                    all_non_diagonal_matrix_block_masks.append(matrix_block_mask)
                row_idx = row_idx + len(mask_j)
            col_idx = col_idx + len(mask_i)
        return torch.stack(all_diagonal_matrix_blocks, dim=0), \
               torch.stack(all_non_diagonal_matrix_blocks, dim=0),\
               torch.stack(all_diagonal_matrix_block_masks, dim=0), \
               torch.stack(all_non_diagonal_matrix_block_masks, dim=0)
