import torch
import os, json
import os.path as osp
from itertools import repeat
import numpy as np
from rdkit import Chem
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
import msgpack
import pickle

dict_drug = {1:'H', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 13:'Al', 14:'Si', 15:'P', 16:'S', 17:'Cl', 33:'As', 35:'Br', 53:'I', 80:'Hg', 83:'Bi'}

def nan_to_num(vec, num=0.0):
    idx = np.isnan(vec)
    vec[idx] = num
    return vec

def _normalize(vec, axis=-1):
    return nan_to_num(
        np.divide(vec, np.linalg.norm(vec, axis=axis, keepdims=True)))


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
                 reorder=True
                 ):

        self.processed_filename = processed_filename
        self.root = root
        self.name = f"{name}{'_no_feature' if no_feature else '_with_feature'}{'_no_h' if remove_h else '_with_h'}{'_reorder' if reorder else ''}{'_sample' if sample else ''}"
        self.sample = sample
        self.num_conformations = 30
        self.remove_h = remove_h
        self.no_feature = no_feature
        self.reorder = reorder

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
        mask_list = []
        
        raw_file = osp.join(self.root, 'drugs_crude.msgpack')
        unpacker = msgpack.Unpacker(open(raw_file, "rb"))
        drugs_file = os.path.join(self.root, "rdkit_folder/summary_drugs.json")
        with open(drugs_file, "r") as f:
            drugs_summ = json.load(f)
        count_error = 0
        count_num_conformer_error = 0
        count_conformer_error = 0
        for i, drugs_1k in enumerate(unpacker):
            print('no file, num_conf, conf_xyz:', count_error, count_num_conformer_error, count_conformer_error)
            print(f"Unpacking file {i}...")
            for smiles, all_info in tqdm(drugs_1k.items()):
                try:
                    pickle_path = os.path.join(self.root, 'rdkit_folder', drugs_summ[smiles]['pickle_path'])
                    with open(pickle_path, 'rb') as f:
                        rdkit_data = pickle.load(f) 
                except:
                    count_error += 1
                    rdkit_data = None

                
                # rdkit_data:
                # dict_keys(['totalconfs', 'temperature', 'uniqueconfs', 'lowestenergy', 'poplowestpct', 
                # 'ensembleenergy', 'ensembleentropy', 'ensemblefreeenergy', 'sars_cov_one_cl_protease_active', 
                # 'charge', 'datasets', 'conformers', 'smiles'])
                
                # an example
                # {'totalconfs': 4, 'temperature': 298.15, 'uniqueconfs': 2, 'lowestenergy': -44.15207, 'poplowestpct': 57.701, 
                #  'ensembleenergy': 0.078, 'ensembleentropy': 2.731, 'ensemblefreeenergy': -0.814, 'sars_cov_one_cl_prot...ase_active': 0, 
                #  'charge': 0, 'datasets': ['aid1706'], 'conformers': [{...}, {...}], 'smiles': 'c1ccc(-c2nc(-c3cccnc3)cs2)cc1'}

                # all_info:
                # dict_keys(['conformers', 
                # 'totalconfs', 'temperature', 'uniqueconfs', 'lowestenergy', 'poplowestpct', 
                # 'ensembleenergy', 'ensembleentropy', 'ensemblefreeenergy', 'sars_cov_one_pl_protease_active', 'sars_cov_one_cl_protease_active', 
                # 'charge', 'datasets'])

                # an example
                # all_info
                # {'conformers': [{...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, ...], 
                # 'totalconfs': 195, 'temperature': 298.15, 'uniqueconfs': 165, 'lowestenergy': -85.0142, 'poplowestpct': 18.706, 
                # 'ensembleenergy': 0.931, 'ensembleentropy': 6.452, 'ensemblefreeenergy': -1.924, 'sars_cov_one_pl_prot...ase_active': 0, 'sars_cov_one_cl_prot...ase_active': 0, 
                # 'charge': 0, 'datasets': ['plpro', 'aid1706']}

                conformers = all_info['conformers']
                flag = False
                if rdkit_data != None:
                    rdkit_conformers = rdkit_data['conformers']
                    try:
                        assert len(conformers) == len(rdkit_conformers)
                    except:
                        count_num_conformer_error += 1
                        flag = True

                all_energies = []
                for conformer in conformers:
                    all_energies.append(conformer['totalenergy'])
                all_energies = np.array(all_energies)
                argsort = np.argsort(all_energies)
                lowest_energies = argsort[:self.num_conformations]
                for id in lowest_energies:
                    conformer = conformers[id]
                    coords = np.array(conformer['xyz']).astype(float)       # conformer['xyz']: atom type + xyz
                    if rdkit_data != None and flag != True:
                        mol = rdkit_conformers[id]['rd_mol']
                        rdkit_xyz = mol.GetConformer().GetPositions()
                        try:
                            assert abs(coords[:,1:] - rdkit_xyz).sum() < 0.1
                        except:
                            count_conformer_error += 1
                            flag = True
                    
                        Chem.MolToSmiles(mol)
                        order = mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder']
                        reorder_mol = Chem.RenumberAtoms(mol,order)
                        atom_type = np.array([atom.GetSymbol() for atom in reorder_mol.GetAtoms()])
                        atomic_number = np.array([atom.GetAtomicNum() for atom in reorder_mol.GetAtoms()])
                        smiles_order_coords = reorder_mol.GetConformer().GetPositions()

                        data = Data()
                        data.smiles = smiles
                        data.z = torch.tensor(coords[:,0], dtype=torch.int64)
                        data.xyz = torch.tensor(coords[:,1:], dtype=torch.float32)
                        data.smiles_order_z = torch.tensor(atomic_number, dtype=torch.int64)
                        data.smiles_order_xyz = torch.tensor(smiles_order_coords, dtype=torch.float32)
                        data.no = len(coords)
                        data_list.append(data)
                        mask_list.append(False)
                    else:
                        mask_list.append(True)
                    
            if self.sample:
                if len(data_list) > 10000:
                    break
                
        data, slices = self.collate(data_list)
        torch.save(mask_list, 'mask_list.pt')
        print('no file, num_conf, conf_xyz:', count_error, count_num_conformer_error, count_conformer_error)
        return data, slices


if __name__ == '__main__':    
    dataset = DrugDataset(root='data/geom/',
                           name='data',
                           processed_filename='data.pt',
                           sample=False,
                           remove_h=False,
                           no_feature=True,
                           reorder=True)
    print(dataset)
    print(len(dataset))
    print(dataset[0])
    raw_dir = 'data/geom/'
    perm = np.load(raw_dir + 'geom_permutation.npy').astype(int)
    print('perm', len(perm))
    mask_list = torch.load('mask_list.pt')
    print('mask', len(mask_list))
    print()
    new_dataset = []
    j = 0
    for i in tqdm(range(len(perm))):
        if not mask_list[i]:
            new_dataset.append(dataset[j])
            j += 1
        else:
            new_dataset.append(None)
    num_mol = len(new_dataset)
    print('num_mol', num_mol)
    val_proportion = 0.1
    test_proportion = 0.1
    val_index = int(num_mol * val_proportion)
    test_index = val_index + int(num_mol * test_proportion)
    train_idx, val_idx, test_idx = perm[test_index:], perm[:val_index], perm[val_index:test_index]


    write_path = 'drug_seq/'
    order_type = 'order_type'
    remove_h = False
    symbols_beyond_type = False
    sample = False

    max_len = 0

    for split in ['train']:
        write_name_ori_coord = f"{order_type}{'_ori_cord'}{'_noH' if remove_h else '_adH'}{'_sample' if sample else ''}{'_seq'}"
        write_name_invariant = f"{order_type}{'_invariant_cord'}{'_noH' if remove_h else '_adH'}{'_sample' if sample else ''}{'_seq'}"
        write_name_spherical = f"{order_type}{'_spherical_cord'}{'_noH' if remove_h else '_adH'}{'_sample' if sample else ''}{'_seq'}"
        
        write_path_ori_coord = write_path + write_name_ori_coord + '.txt'
        write_path_invariant = write_path + write_name_invariant + '.txt'
        write_path_spherical = write_path + write_name_spherical + '.txt'

        smiles_seq = []
        atom_seq = []
        coords_seq = []
        invariant_coords_seq = []
        spherical_coords_seq = []

        if split == 'train':
            split_idx = train_idx
        else:
            split_idx = val_idx

        if sample:
            size = 10000
        else:
            size = len(split_idx)

        for i in tqdm(range(size)):
            mol = new_dataset[split_idx[i]]
            if mol == None:
                continue
            atom_type = np.array([dict_drug[key] for key in mol.smiles_order_z.numpy()])
            coords = mol.smiles_order_xyz.numpy()
            smiles = mol.smiles[0]

            num_atom = len(atom_type)
            if num_atom > max_len:
                max_len = num_atom

            centered_coords = coords - coords[0]
            invariant_coords = np.zeros_like(coords)
            spherical_coords = np.zeros_like(coords)

            # we have to select three nodes to build a global frame
            flag = False 

            if num_atom == 1:
                pass
            elif num_atom == 2:
                d = np.linalg.norm(coords[1] - coords[0], axis=-1)
                invariant_coords[1,0] = d
                spherical_coords[1,0] = d
            else:
                v1 = centered_coords[1] - centered_coords[0]
                for i in range(2, num_atom):
                    v2 = centered_coords[i] - centered_coords[0]
                    if np.linalg.norm(np.cross(v1, v2)) != 0:
                        flag = True # # can find the third node that is not on the same line as the first two nodes
                        break
                if flag == False and i == num_atom - 1: # cannot find the third node that is not on the same line as the first two nodes
                    invariant_coords = centered_coords
                else:
                    # build a global frame (xyz axis)
                    x = _normalize(v1)
                    y = _normalize(np.cross(v1, v2))
                    z = np.cross(x, y)
                    # invariant coords
                    invariant_coords = np.dot(centered_coords, np.stack((x, y, z)).T)
                d = np.linalg.norm(invariant_coords, axis=-1)
                theta = np.zeros_like(d)
                theta[1:] = np.arccos(invariant_coords[1:,2]/d[1:])
                phi = np.arctan2(invariant_coords[:,1], invariant_coords[:,0])
                # invariant_spherical_coords
                spherical_coords = np.stack((d, theta, phi)).T

            
            coords = np.array([["{:.2f}".format(value) for value in row] for row in coords])
            invariant_coords = np.array([["{:.2f}".format(value) for value in row] for row in invariant_coords])
            spherical_coords = np.array([["{:.2f}".format(value) for value in row] for row in spherical_coords])
            coords_seq.append(coords)
            invariant_coords_seq.append(invariant_coords)
            spherical_coords_seq.append(spherical_coords)

            smiles_seq.append(smiles)
            atom_seq.append(atom_type)
        
        with open(write_path_ori_coord, 'w') as file:
            for i in range(len(atom_seq)):
                for j in range(len(atom_seq[i])):
                    file.write(atom_seq[i][j])
                    file.write(' ')
                    file.write(str(coords_seq[i][j][0]) + ' ' + str(coords_seq[i][j][1]) + ' ' + str(coords_seq[i][j][2]) + ' ')
                file.write('\n')

        with open(write_path_invariant, 'w') as file:
            for i in range(len(atom_seq)):
                for j in range(len(atom_seq[i])):
                    file.write(atom_seq[i][j])
                    file.write(' ')
                    file.write(str(invariant_coords_seq[i][j][0]) + ' ' + str(invariant_coords_seq[i][j][1]) + ' ' + str(invariant_coords_seq[i][j][2]) + ' ')
                file.write('\n')
        
        with open(write_path_spherical, 'w') as file:
            for i in range(len(atom_seq)):
                for j in range(len(atom_seq[i])):
                    file.write(atom_seq[i][j])
                    file.write(' ')
                    file.write(str(spherical_coords_seq[i][j][0]) + ' ' + str(spherical_coords_seq[i][j][1]) + '° ' + str(spherical_coords_seq[i][j][2]) + '° ')
                file.write('\n')
        print()
    print('max num atom:',max_len)

