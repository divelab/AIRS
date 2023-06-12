from typing import Optional, List

import os
import random
import torch
import numpy as np
import os.path as osp
from argparse import Namespace
import pickle

from tqdm import tqdm
from apsw import Connection
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip, Data)



# BOHR2ANG = 1.8897259886

convention_dict = {
    'pyscf_631G': Namespace(
        atom_to_orbitals_map={1: 'ss', 6: 'ssspp', 7: 'ssspp', 8: 'ssspp', 9: 'ssspp'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': 
                          [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1], 6: [0, 1, 2, 3, 4], 7:  [0, 1, 2, 3, 4],
            8:  [0, 1, 2, 3, 4], 9:  [0, 1, 2, 3, 4]
        },
    ),
    'pyscf_def2svp': Namespace(
        atom_to_orbitals_map={1: 'ssp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd', 9: 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2], 6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5], 9: [0, 1, 2, 3, 4, 5]
        },
    ),
}



atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}


HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414
conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs_tensor = torch.zeros(5, 19)
atomrefs_tensor[:, 7]  = torch.tensor(atomrefs[7])
atomrefs_tensor[:, 8]  = torch.tensor(atomrefs[8])
atomrefs_tensor[:, 9]  = torch.tensor(atomrefs[9])
atomrefs_tensor[:, 10] = torch.tensor(atomrefs[10])


def matrix_transform(matrices, atoms, convention='pyscf_631G'):
    conv = convention_dict[convention]
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
    transform_indices = np.concatenate(transform_indices).astype(np.int32)
    transform_signs = np.concatenate(transform_signs)

    matrices_new = matrices[...,transform_indices, :]
    matrices_new = matrices_new[...,:, transform_indices]
    matrices_new = matrices_new * transform_signs[:, None]
    matrices_new = matrices_new * transform_signs[None, :]
    return matrices_new



class QH9Stable(InMemoryDataset):
    def __init__(self, root='datasets/', split='random', transform=None, pre_transform=None, pre_filter=None):
        self.folder = osp.join(root, 'QH9Stable')
        self.split = split
        self.connections = {}
        self.db_dir = os.path.join(self.folder, 'processed')
        self.full_orbitals = 14
        self.orbital_mask = {}
        idx_1s_2s_2p = torch.tensor([0, 1, 3, 4, 5])
        orbital_mask_line1 = idx_1s_2s_2p
        orbital_mask_line2 = torch.arange(self.full_orbitals)
        for i in range(1, 11):
            self.orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2

        super(QH9Stable, self).__init__(self.folder, transform, pre_transform, pre_filter)
        self.data, self.slices, self.train_mask, self.val_mask, self.test_mask \
            = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'QH9Stable.db']

    @property
    def processed_file_names(self):
        if self.split=='random':
            return 'processed_QH9Stable_random.pt'
        elif self.split=='size_ood':
            return 'processed_QH9Stable_size_ood.pt'

    def process(self):
        data_list = []
        if self.split=='size_ood':
            num_nodes_list = []
        for raw_file_name in self.raw_file_names:
            connection = Connection(os.path.join(self.root, "raw", raw_file_name))
            cursor = connection.cursor()
            data = cursor.execute("select * from data").fetchall()
            for row in tqdm(data):
                mol_id, num_nodes, atoms, pos, Ham = \
                    row[0], row[1], np.frombuffer(row[2], np.int32), \
                    np.frombuffer(row[3], np.float64), \
                    np.frombuffer(row[4], np.float64)
                pos = pos.reshape(num_nodes, 3)
                num_orbitals = sum([5 if atom <= 2 else 14 for atom in atoms])
                Ham = Ham.reshape(num_orbitals, num_orbitals)
                data_list.append(self.get_mol(atoms, pos, Ham))
                if self.split=='size_ood':
                    num_nodes_list.append(num_nodes)

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        if self.split=='random':
            print('Random splitting...')
            data_ratio = [0.8, 0.1, 0.1]
            data_split = [int(len(data_list) * data_ratio[0]), int(len(data_list) * data_ratio[1])]
            data_split.append(len(data_list) - sum(data_split))
            indices = np.random.RandomState(seed=43).permutation(len(data_list))
            train_mask = indices[:data_split[0]]
            val_mask = indices[data_split[0]:data_split[0]+data_split[1]]
            test_mask = indices[data_split[0]+data_split[1]:]
        elif self.split=='size_ood':
            print('Size OOD splitting...')
            num_nodes_array = np.array(num_nodes_list)
            train_indices = np.where(num_nodes_array <= 20)
            val_condition = np.logical_and(num_nodes_array >= 21, num_nodes_array <= 22)
            val_indices = np.where(val_condition)
            test_indices = np.where(num_nodes_array >= 23)
            train_mask = train_indices[0].astype(np.int64)
            val_mask = val_indices[0].astype(np.int64)
            test_mask = test_indices[0].astype(np.int64)
            
        torch.save((data, slices, train_mask, val_mask, test_mask), self.processed_paths[0])

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
                matrix_block = torch.zeros(self.full_orbitals, self.full_orbitals).type(torch.float64)
                matrix_block_mask = torch.zeros(self.full_orbitals, self.full_orbitals).type(torch.float64)
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

    def get_mol(self, atoms, pos, Ham):
        hamiltonian = torch.tensor(
            matrix_transform(Ham, atoms, convention='pyscf_def2svp'), dtype=torch.float64)
        diagonal_hamiltonian, non_diagonal_hamiltonian, \
        diagonal_hamiltonian_mask, non_diagonal_hamiltonian_mask \
            = self.cut_matrix(hamiltonian, atoms)

        data = Data(
            pos=torch.tensor(pos, dtype=torch.float64),
            atoms= torch.tensor(atoms, dtype=torch.int64).view(-1, 1),
            diagonal_hamiltonian=diagonal_hamiltonian,
            non_diagonal_hamiltonian=non_diagonal_hamiltonian,
            diagonal_hamiltonian_mask=diagonal_hamiltonian_mask,
            non_diagonal_hamiltonian_mask=non_diagonal_hamiltonian_mask,
        )
        return data


class QH9Dynamic(InMemoryDataset):
    def __init__(self, root='datasets/', task='', split='geometry', transform=None, pre_transform=None, pre_filter=None):
        assert task in ['']
        self.folder = osp.join(root, 'QH9Dynamic')
        self.split = split
        self.connections = {}
        self.db_dir = os.path.join(self.folder, 'processed')
        self.full_orbitals = 14
        self.orbital_mask = {}
        idx_1s_2s_2p = torch.tensor([0, 1, 3, 4, 5])
        orbital_mask_line1 = idx_1s_2s_2p
        orbital_mask_line2 = torch.arange(self.full_orbitals)
        # orbital_mask_line2 = torch.arange(9)
        for i in range(1, 11):
            self.orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2

        super(QH9Dynamic, self).__init__(self.folder, transform, pre_transform, pre_filter)
        self.data, self.slices, self.train_mask, self.val_mask, self.test_mask \
            = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'QH9Dynamic.db']

    @property
    def processed_file_names(self):
        if self.split=='geometry':
            return 'processed_QH9Dynamic_geometry.pt'
        elif self.split=='mol':
            return 'processed_QH9Dynamic_mol.pt'

    def process(self, num_geometry_per_mol=60, num_train_geometry_per_mol=50, num_val_geometry_per_mol=5, num_mol=2399):
        data_list = []
        if self.split=='geometry':
            print('Geometry-wise splitting...')
            train_mask = np.array([], dtype=np.int64)
            val_mask = np.array([], dtype=np.int64)
            test_mask = np.array([], dtype=np.int64)
            cur_index = 0
            for raw_file_name in self.raw_file_names:
                connection = Connection(os.path.join(self.root, "raw", raw_file_name))
                cursor = connection.cursor()
                data = cursor.execute("select * from data").fetchall()
                for ind, row in enumerate(tqdm(data)):
                    mol_id, geo_id, num_nodes, atoms, pos, Ham = \
                        row[0], row[1], row[2], np.frombuffer(row[3], np.int32), \
                        np.frombuffer(row[4], np.float64), \
                        np.frombuffer(row[5], np.float64)
                    pos = pos.reshape(num_nodes, 3)
                    num_orbitals = sum([5 if atom <= 2 else 14 for atom in atoms])
                    Ham = Ham.reshape(num_orbitals, num_orbitals)
                    data_list.append(self.get_mol(atoms, pos, Ham))
                    if (ind + 1) % num_geometry_per_mol == 0: # Finish traversing one molecule
                        indices = np.random.RandomState(seed=ind).permutation(num_geometry_per_mol) # Different random split for different molecules
                        train_mask_cur_mol = cur_index + indices[:num_train_geometry_per_mol]
                        val_mask_cur_mol = cur_index + indices[num_train_geometry_per_mol:num_train_geometry_per_mol+num_val_geometry_per_mol]
                        test_mask_cur_mol = cur_index + indices[num_train_geometry_per_mol+num_val_geometry_per_mol:]
                        train_mask = np.concatenate((train_mask, train_mask_cur_mol))
                        val_mask = np.concatenate((val_mask, val_mask_cur_mol))
                        test_mask = np.concatenate((test_mask, test_mask_cur_mol))
                        cur_index += num_geometry_per_mol

                if self.pre_filter is not None:
                    data_list = [data for data in data_list if self.pre_filter(data)]
                if self.pre_transform is not None:
                    data_list = [self.pre_transform(data) for data in data_list]
                    
        elif self.split=='mol':
            print('Molecule-wise splitting...')
            for raw_file_name in self.raw_file_names:
                connection = Connection(os.path.join(self.root, "raw", raw_file_name))
                cursor = connection.cursor()
                data = cursor.execute("select * from data").fetchall()
                for ind, row in enumerate(tqdm(data)):
                    mol_id, geo_id, num_nodes, atoms, pos, Ham = \
                        row[0], row[1], row[2], np.frombuffer(row[3], np.int32), \
                        np.frombuffer(row[4], np.float64), \
                        np.frombuffer(row[5], np.float64)
                    pos = pos.reshape(num_nodes, 3)
                    num_orbitals = sum([5 if atom <= 2 else 14 for atom in atoms])
                    Ham = Ham.reshape(num_orbitals, num_orbitals)
                    data_list.append(self.get_mol(atoms, pos, Ham))
                if self.pre_filter is not None:
                    data_list = [data for data in data_list if self.pre_filter(data)]
                if self.pre_transform is not None:
                    data_list = [self.pre_transform(data) for data in data_list]
            
            mol_id_list = [i for i in range(num_mol) for _ in range(num_geometry_per_mol)]
            data_ratio = [0.8, 0.1, 0.1]
            index_list = [i for i in range(max(mol_id_list)+1)]
            random.seed(43)
            random.shuffle(index_list)
            train_mol_ids = np.array(index_list[:int(len(index_list) * data_ratio[0])])
            val_mol_ids = np.array(index_list[int(len(index_list) * data_ratio[0]):int(len(index_list) * data_ratio[0]) + int(len(index_list) * data_ratio[1])])
            test_mol_ids = np.array(index_list[int(len(index_list) * data_ratio[0]) + int(len(index_list) * data_ratio[1]):])
            mol_id_array = np.array(mol_id_list)
            train_mask = np.where(np.isin(mol_id_array, train_mol_ids))[0].astype(np.int64)
            val_mask = np.where(np.isin(mol_id_array, val_mol_ids))[0].astype(np.int64)
            test_mask = np.where(np.isin(mol_id_array, test_mol_ids))[0].astype(np.int64)
            
        data, slices = self.collate(data_list)

        print('Saving...') 
        torch.save((data, slices, train_mask, val_mask, test_mask), self.processed_paths[0])

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
                matrix_block = torch.zeros(self.full_orbitals, self.full_orbitals).type(torch.float64)
                matrix_block_mask = torch.zeros(self.full_orbitals, self.full_orbitals).type(torch.float64)
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

    def get_mol(self, atoms, pos, Ham):
        hamiltonian = torch.tensor(
            matrix_transform(Ham, atoms, convention='pyscf_def2svp'), dtype=torch.float64)
        diagonal_hamiltonian, non_diagonal_hamiltonian, \
        diagonal_hamiltonian_mask, non_diagonal_hamiltonian_mask \
            = self.cut_matrix(hamiltonian, atoms)

        data = Data(
            pos=torch.tensor(pos, dtype=torch.float64),
            atoms= torch.tensor(atoms, dtype=torch.int64).view(-1, 1),
            diagonal_hamiltonian=diagonal_hamiltonian,
            non_diagonal_hamiltonian=non_diagonal_hamiltonian,
            diagonal_hamiltonian_mask=diagonal_hamiltonian_mask,
            non_diagonal_hamiltonian_mask=non_diagonal_hamiltonian_mask,
        )
        return data


if __name__ == '__main__':
    # dataset = QH9Stable()
    dataset = QH9Dynamic()
    print(dataset)
