# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import argparse
import torch
import numpy as np
from tqdm import tqdm
from eval_follow_edm.datasets_config import get_dataset_info
from eval_follow_edm.analyze import analyze_stability_for_molecules
import re
import math

try:
    from eval_follow_edm import rdkit_functions
except ModuleNotFoundError:
    print('Not importing rdkit functions.')


def invariant_seq_for_edm_eval(dataset_name='qm9',
                               input_path = 'seq.txt',
                               remove_h = True, symbols_beyond_type = False):

    if dataset_name == 'molecule3d':
        if remove_h:
            dict = {'B': 0,'C': 1,'C@': 1,'C@@': 1, 'N': 2, 'O': 3, 'F': 4, 
                    'Si': 5, 'Si@': 5, 'Si@@': 5, 'P': 6, 'P@':6, 'P@@':6, 'S': 7, 'S@': 7, 'S@@': 7, 'Cl': 8,  'Br': 9}
        else:
            raise Exception('Not supported dataset name %s' % dataset_name)
    elif dataset_name == 'qm9':
        if remove_h:
            dict = {'C': 0, 'C@': 0,'C@@': 0, 'N': 1, 'O': 2, 'F': 3}
        else:
            dict = {'H': 0, 'C': 1, 'C@': 1,'C@@': 1, 'N': 2, 'O': 3, 'F': 4}
    elif dataset_name == 'geom':
        if remove_h:
            dict = {'B': 0, 'C': 1,'C@': 1,'C@@': 1, 'N': 2, 'O': 3, 'F': 4, 'Al': 5, 
                    'Si': 6, 'Si@': 6, 'Si@@': 6, 'P': 7, 'P@':7, 'P@@':7, 'S': 8, 'S@': 8, 'S@@': 8, 
                    'Cl': 9, 'As': 10, 'Br': 11, 'I': 12, 'Hg': 13, 'Bi': 14} 
        else:
            dict = {'H': 0, 'B': 1, 'C': 2,'C@': 2,'C@@': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 
                    'Si': 7, 'Si@': 7, 'Si@@': 7, 'P': 8, 'P@':8, 'P@@':8, 'S': 9, 'S@': 9, 'S@@': 9, 
                    'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15} 
    else:
        raise Exception('Not supported dataset name %s' % dataset_name)

    with open(input_path, 'r') as file:
        lines = file.readlines()
        num_samples = len(lines)
    file.close()
    all_len_files = [len(line.split()) for line in lines]
    max_num_atoms = math.ceil(max(all_len_files) / 4.0)
    print('max_num_atoms(max_len_seq/4):', max_num_atoms)
    
    all_len = []
    num_type = max(dict.values()) + 1
    count_invalid_len = 0
    count_invalid_seq = 0
    count_invalid_coords = 0
    one_hot = torch.zeros((num_samples, max_num_atoms, num_type), dtype=float)
    x = torch.zeros((num_samples, max_num_atoms, 3), dtype=float)
    node_mask = torch.zeros((num_samples, max_num_atoms, 1), dtype=float)
    idx = 0
    with open(input_path, 'r') as file:
        for num_line, line in enumerate(tqdm(file)):
            if num_line >= num_samples:
                break
            if not symbols_beyond_type:
                mol = np.array(line.split())
                try:
                    mol = mol.reshape(-1,4)
                except:
                    for cut_idx in range(int(len(mol)/4)-1):
                        vals = mol[4 * cut_idx:4 * cut_idx + 4]
                        try:
                            dict[vals[0]]
                            vals[1:4].astype(float)
                        except:
                            mol = mol[:4 * cut_idx].reshape(-1,4)
                            break
                        if cut_idx == int(len(mol)/4)-2:
                            mol = mol[:4 * cut_idx + 4].reshape(-1,4)
                    # print('invalid length')
                    count_invalid_len += 1
                    # continue
                seq = mol[:,0]
            else:
                try:
                    match = re.findall(r'\b[A-Za-z] [+-]?\d+.\d+ [+-]?\d+.\d+ [+-]?\d+.\d+\b', line)
                    mol = np.array([item.split() for item in match])
                    seq = mol[:,0]
                    # print('line, mol:', len(line.split()), len(mol)*4)
                except:
                    print('no invalid format')
                    continue
            try:
                one_hot_emb = torch.nn.functional.one_hot(torch.tensor([dict[key] for key in seq]), num_type)
            except:
                # print('invalid seq')
                count_invalid_seq += 1
                continue
            try:
                invariant_coords = mol[:,1:4].astype(float)
            except:
                # print('invalid coords')
                count_invalid_coords += 1
                continue   
            
            num_nodes = len(seq)
            all_len.append(num_nodes)
            one_hot[idx, :num_nodes] = one_hot_emb
            x[idx,:num_nodes] = torch.tensor(invariant_coords)
            node_mask[idx, :num_nodes] = 1.
            idx += 1
    one_hot, x, node_mask = one_hot[:idx], x[:idx], node_mask[:idx]
    print('max_num_atoms(after filter out invalid molecules):', 0 if len(all_len) == 0 else max(all_len))
    frequency_mol_len = {}
    for element in all_len:
        frequency_mol_len[element] = frequency_mol_len.get(element, 0) + 1
    molecules = {'one_hot': one_hot, 'x': x, 'node_mask': node_mask}
    # torch.save(molecules, write_path)
    print('invalid: 1. length is not a multiple of 4; 2. invalid atom type; 3. invalid coords:\n', 
          count_invalid_len, count_invalid_seq, count_invalid_coords)
    print('done')
    return molecules


def spherical_seq_for_edm_eval(dataset_name = 'qm9',
        input_path = 'seq.txt',
        remove_h = False, symbols_beyond_type = False):
    
    if dataset_name == 'molecule3d':
        if remove_h:
            dict = {'B': 0,'C': 1,'C@': 1,'C@@': 1, 'N': 2, 'O': 3, 'F': 4, 
                    'Si': 5, 'Si@': 5, 'Si@@': 5, 'P': 6, 'P@':6, 'P@@':6, 'S': 7, 'S@': 7, 'S@@': 7, 'Cl': 8,  'Br': 9}
        else:
            raise Exception('Not supported dataset name %s' % dataset_name)
    elif dataset_name == 'qm9':
        if remove_h:
            dict = {'C': 0, 'C@': 0,'C@@': 0, 'N': 1, 'O': 2, 'F': 3}
        else:
            dict = {'H': 0, 'C': 1, 'C@': 1,'C@@': 1, 'N': 2, 'O': 3, 'F': 4}
    elif dataset_name == 'geom':
        if remove_h:
            dict = {'B': 0, 'C': 1,'C@': 1,'C@@': 1, 'N': 2, 'O': 3, 'F': 4, 'Al': 5, 
                    'Si': 6, 'Si@': 6, 'Si@@': 6, 'P': 7, 'P@':7, 'P@@':7, 'S': 8, 'S@': 8, 'S@@': 8, 
                    'Cl': 9, 'As': 10, 'Br': 11, 'I': 12, 'Hg': 13, 'Bi': 14} 
        else:
            dict = {'H': 0, 'B': 1, 'C': 2,'C@': 2,'C@@': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 
                    'Si': 7, 'Si@': 7, 'Si@@': 7, 'P': 8, 'P@':8, 'P@@':8, 'S': 9, 'S@': 9, 'S@@': 9, 
                    'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15} 
    else:
        raise Exception('Not supported dataset name %s' % dataset_name)
    
    with open(input_path, 'r') as file:
        lines = file.readlines()
        num_samples = len(lines)
    file.close()
    all_len_files = [len(line.split()) for line in lines]
    max_num_atoms = math.ceil(max(all_len_files) / 4.0)
    print('max_num_atoms(max_len_seq/4):', max_num_atoms)

    all_len = []
    num_type = max(dict.values()) + 1
    count_invalid_len = 0
    count_invalid_seq = 0
    count_invalid_coords = 0
    one_hot = torch.zeros((num_samples, max_num_atoms, num_type), dtype=float)
    x = torch.zeros((num_samples, max_num_atoms, 3), dtype=float)
    node_mask = torch.zeros((num_samples, max_num_atoms, 1), dtype=float)
    idx = 0
    with open(input_path, 'r') as file:
        for num_line, line in enumerate(tqdm(file)):
            if num_line >= num_samples:
                break
            if not symbols_beyond_type:
                mol = np.array(line.split())
                try:
                    mol = mol.reshape(-1,4)
                except:
                    for cut_idx in range(int(len(mol)/4)-1):
                        vals = mol[4 * cut_idx:4 * cut_idx + 4]
                        if vals[2][-1] != '°' or vals[3][-1] != '°':
                            mol = mol[:4 * cut_idx].reshape(-1,4)
                            break
                        else:
                            try:
                                dict[vals[0]]
                                vals[1].astype(float)
                                np.str_(vals[2][:-1]).astype(float)
                                np.str_(vals[3][:-1]).astype(float)
                            except:
                                mol = mol[:4 * cut_idx].reshape(-1,4)
                                break
                    if cut_idx == int(len(mol)/4)-2:
                        mol = mol[:4 * cut_idx + 4].reshape(-1,4)
                    # print('invalid length')
                    count_invalid_len += 1
                    # continue
                if len(mol.shape) == 1:
                    count_invalid_seq += 1
                    continue
                seq = mol[:,0]
            else:
                try:
                    match = re.findall(r'\b[A-Za-z] [+-]?\d+.\d+ [+-]?\d+.\d+° [+-]?\d+.\d+°?\b', line)
                    mol = np.array([(item+'°').split() for item in match])
                    seq = mol[:,0]
                    # print('line, mol:', len(line.split()), len(mol)*4)
                except:
                    print('no invalid format')
                    continue
            try:
                one_hot_emb = torch.nn.functional.one_hot(torch.tensor([dict[key] for key in seq]), num_type)
            except:
                # print('invalid seq')
                count_invalid_seq += 1
                continue
            try:
                spherical_coords = mol[:,1:]
                d = spherical_coords[:,0].astype(float)
                theta = np.array([s[:-1] for s in spherical_coords[:,1]]).astype(float)
                phi = np.array([s[:-1] for s in spherical_coords[:,2]]).astype(float)
                invariant_coords = np.stack((d * np.sin(theta) * np.cos(phi), d * np.sin(theta) * np.sin(phi), d * np.cos(theta))).T
            except:
                # print('invalid coords')
                count_invalid_coords += 1
                continue   
            
            num_nodes = len(seq)
            all_len.append(num_nodes)
            one_hot[idx, :num_nodes] = one_hot_emb
            x[idx,:num_nodes] = torch.tensor(invariant_coords)
            node_mask[idx, :num_nodes] = 1.
            idx += 1
    one_hot, x, node_mask = one_hot[:idx], x[:idx], node_mask[:idx]
    print('max_num_atoms(after filter out invalid molecules):', 0 if len(all_len) == 0 else max(all_len))
    frequency_mol_len = {}
    for element in all_len:
        frequency_mol_len[element] = frequency_mol_len.get(element, 0) + 1
    molecules = {'one_hot': one_hot, 'x': x, 'node_mask': node_mask}
    print('invalid: 1. length is not a multiple of 4; 2. invalid atom type; 3. invalid coords:\n', 
          count_invalid_len, count_invalid_seq, count_invalid_coords)
    print('done')
    return molecules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="qm9")
    parser.add_argument('--rep_type', type=str, default="spherical")
    parser.add_argument('--remove_h', default=False, action='store_true')
    parser.add_argument('--symbols_beyond_type', default=False, action='store_true')
    parser.add_argument('--input_path', type=str, default="generated_samples_unconditional.txt")
    args = parser.parse_args()
    print(args)

    #### eval LM for molecule generation ####

    dataset_info = get_dataset_info(args.dataset_name, remove_h=args.remove_h)
    if args.rep_type == 'invariant':
        molecules = invariant_seq_for_edm_eval(dataset_name=args.dataset_name, 
                               input_path=args.input_path,  
                               remove_h=args.remove_h,
                               symbols_beyond_type=args.symbols_beyond_type)
    elif args.rep_type == 'spherical':
        molecules = spherical_seq_for_edm_eval(dataset_name=args.dataset_name, 
                               input_path=args.input_path,  
                               remove_h=args.remove_h,
                               symbols_beyond_type=args.symbols_beyond_type)

    stability_dict, rdkit_metrics = analyze_stability_for_molecules(
        molecules, dataset_info)
    print(stability_dict)

    if rdkit_metrics is not None:
        rdkit_metrics = rdkit_metrics[0]
        print("Validity %.4f, Uniqueness: %.4f, Novelty: %.4f" % (rdkit_metrics[0], rdkit_metrics[1], rdkit_metrics[2]))
    else:
        print("Install rdkit roolkit to obtain Validity, Uniqueness, Novelty")


if __name__ == "__main__":
    main()
