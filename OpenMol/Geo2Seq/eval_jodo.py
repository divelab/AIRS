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
from eval_follow_jodo.stability import get_edm_metric, get_2D_edm_metric
from eval_follow_jodo.mose_metric import get_moses_metrics
from eval_follow_jodo.cal_geometry import get_sub_geometry_metric
import re
import math
import logging


try:
    from eval_follow_edm import rdkit_functions
except ModuleNotFoundError:
    print('Not importing rdkit functions.')
from rdkit.Chem import rdDetermineBonds, rdmolops

# dict_qm9 = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F'}
dict_qm9 = {0: 'H', 1: 'C', 2: 'N', 3: 'O', 4: 'F'}

def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)

def write_xyz_file(atom_types, atom_coordinates, file_path):
    with open(file_path, 'w') as file:
        num_atoms = len(atom_types)
        file.write(f"{num_atoms}\n")
        file.write('\n')

        for atom_type, coords in zip(atom_types, atom_coordinates):
            x, y, z = coords
            file.write(f"{atom_type} {np.format_float_positional(x)} {np.format_float_positional(y)} {np.format_float_positional(z)}\n")

def midi_for_eval(input_path='generated_samples.txt'):
    with open(input_path, 'r') as file:
        txt = file.read()
        # num_samples = len(lines)
    file.close()
    lines = txt.split('\n\n')[:-1] # each line is a molecule
    mol_for_our_edm_eval = dict()
    mol_for_jodo_eval = []
    mol_for_jodo_eval_rdkit_e = []
    for line in tqdm(lines, disable=True):
        all = np.array(line.split()) # 0:'N=X' 1:'X:' 2 to X+1: (X)  X+2: 'charges:' X+3 to 2*X+2: (X) 2*X+3: 'pos:' 2*X+4 to 5*X+3: (3*X) 5*X+4: 'E' 5*X+5 to X^2+5*X+4
        num_atom = int(all[0][2:])
        x = torch.tensor(all[2:num_atom + 2].astype(int), dtype=torch.int64)
        charges = torch.tensor(all[num_atom + 3: 2 * num_atom + 3].astype(int), dtype=torch.int64)
        pos = torch.tensor(all[2 * num_atom + 4 : 5 * num_atom + 4].astype(float).reshape(num_atom, 3), dtype=torch.float32)
        e = torch.tensor(all[5 * num_atom + 5: num_atom * num_atom + 5 * num_atom + 5].astype(float).reshape(num_atom, num_atom), dtype=torch.float32)
        # e values: 0/1/2/3
        mol_for_jodo_eval.append((pos, x, e, charges))

        file_path = 'example_t.xyz'
        ori_z = x
        ori_coords = pos
        
        write_xyz_file([dict_qm9[key] for key in ori_z.numpy()], ori_coords, file_path)
        raw_mol = Chem.MolFromXYZFile(file_path)
        if raw_mol == None:
            continue
        mol = Chem.Mol(raw_mol)
        # print(Chem.MolToSmiles(mol, canonical=True)) # no bonds
        try:
            rdDetermineBonds.DetermineBonds(mol)
        except:
            pass
        # print(Chem.MolToSmiles(mol, canonical=True)) # with bonds
        smiles = Chem.MolToSmiles(mol)
        e_rdkit = torch.tensor(rdmolops.GetAdjacencyMatrix(mol), dtype=torch.float32)
        mol_for_jodo_eval_rdkit_e.append((pos, x, e_rdkit, charges))
        # print()
    # print()
    return mol_for_jodo_eval, mol_for_jodo_eval_rdkit_e

def jodo_for_eval(input_path='generated_samples.txt'):
    with open(input_path, 'r') as file:
        txt = file.read()
        # num_samples = len(lines)
    file.close()
    lines = txt.split('\n\n')[:-1] # each line is a molecule
    mol_for_our_edm_eval = dict()
    mol_for_jodo_eval = []
    mol_for_jodo_eval_rdkit_e = []
    for line in tqdm(lines, disable=True):
        all = np.array(line.split()) # 0:'N=X' 1:'X:' 2 to X+1: (X)  X+2: 'charges:' X+3 to 2*X+2: (X) 2*X+3: 'pos:' 2*X+4 to 5*X+3: (3*X) 5*X+4: 'E' 5*X+5 to X^2+5*X+4
        num_atom = int(all[0][2:])
        x = torch.tensor(all[2:num_atom + 2].astype(int), dtype=torch.int64)
        charges = torch.tensor(all[num_atom + 3: 2 * num_atom + 3].astype(int), dtype=torch.int64)
        pos = torch.tensor(all[2 * num_atom + 4 : 5 * num_atom + 4].astype(float).reshape(num_atom, 3), dtype=torch.float32)
        e = torch.tensor(all[5 * num_atom + 5: num_atom * num_atom + 5 * num_atom + 5].astype(float).reshape(num_atom, num_atom), dtype=torch.float32)
        # e values: 0/1/2/3
        mol_for_jodo_eval.append((pos, x, e, charges))

        file_path = 'example_t.xyz'
        ori_z = x
        ori_coords = pos
        
        write_xyz_file([dict_qm9[key] for key in ori_z.numpy()], ori_coords, file_path)
        raw_mol = Chem.MolFromXYZFile(file_path)
        if raw_mol == None:
            continue
        mol = Chem.Mol(raw_mol)
        # print(Chem.MolToSmiles(mol, canonical=True)) # no bonds
        try:
            rdDetermineBonds.DetermineBonds(mol)
        except:
            pass
        # print(Chem.MolToSmiles(mol, canonical=True)) # with bonds
        smiles = Chem.MolToSmiles(mol)
        e_rdkit = torch.tensor(rdmolops.GetAdjacencyMatrix(mol), dtype=torch.float32)
        mol_for_jodo_eval_rdkit_e.append((pos, x, e_rdkit, charges))
        # print()
    # print()
    return mol_for_jodo_eval, mol_for_jodo_eval_rdkit_e

def spherical_seq_for_eval(dataset_name = 'qm9',
        input_path = 'QM9_seq/seq.txt',
        remove_h = False):
    
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
    
    mol_for_jodo_eval_rdkit_e = []

    with open(input_path, 'r') as file:
        lines = file.readlines()
        num_samples = len(lines)
    file.close()
    all_len_files = [len(line.split()) for line in lines]
    max_num_atoms = math.ceil(max(all_len_files) / 4.0)
    print('max_num_atoms(max_len_seq/4):', max_num_atoms)

    count_invalid_len = 0
    count_invalid_seq = 0
    count_invalid_coords = 0
    
    with open(input_path, 'r') as file:
        for num_line, line in enumerate(tqdm(file, disable=True)):
            if num_line >= num_samples:
                break
            mol = np.array(line.split())
            try:
                mol = mol.reshape(-1,4)
            except:
                count_invalid_len += 1
                continue
            seq = mol[:,0]
            try:
                x = torch.tensor([dict[key] for key in seq])
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
            
            pos = torch.tensor(invariant_coords, dtype=torch.float32)
        
            file_path = 'example_' + args.method + '.xyz'
            ori_z = x
            ori_coords = pos
            
            write_xyz_file([dict_qm9[key] for key in ori_z.numpy()], ori_coords, file_path)
            raw_mol = Chem.MolFromXYZFile(file_path)
            mol = Chem.Mol(raw_mol)
            # print(Chem.MolToSmiles(mol, canonical=True)) # no bonds
            try:
                rdDetermineBonds.DetermineBonds(mol)
            except:
                pass
            # print(Chem.MolToSmiles(mol, canonical=True)) # with bonds
            smiles = Chem.MolToSmiles(mol)
            e_rdkit = torch.tensor(rdmolops.GetAdjacencyMatrix(mol), dtype=torch.float32)
            mol_for_jodo_eval_rdkit_e.append((pos, x, e_rdkit, x))
    
    print('invalid: 1. length is not a multiple of 4; 2. invalid atom type; 3. invalid coords:\n', 
          count_invalid_len, count_invalid_seq, count_invalid_coords)
    print('done')
    return None, mol_for_jodo_eval_rdkit_e

def edm_for_eval(input_path = 'generated_example_edm.pt'):
    molecules = torch.load(input_path)
    # molecules['one_hot'], molecules['x'], molecules['node_mask']
    mol_for_jodo_eval = []
    mol_for_jodo_eval_rdkit_e = []

def geoldm_for_eval(input_path = 'generated_example_geoldm.pt'):
    data = torch.load(input_path)
    mol_for_jodo_eval = []
    mol_for_jodo_eval_rdkit_e = []

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default="ours") # ours, jodo, midi, edm, geoldm
parser.add_argument('--input_path', type=str, default="generated_samples_.txt")
args = parser.parse_args()
print(args)

if args.method == 'ours':
    processed_mols, processed_mols_e_rdkit = spherical_seq_for_eval(input_path=args.input_path)
elif args.method == 'midi':
    processed_mols, processed_mols_e_rdkit = midi_for_eval()
elif args.method == 'jodo':
    processed_mols, processed_mols_e_rdkit = jodo_for_eval()
elif args.method == 'edm':
    processed_mols, processed_mols_e_rdkit = edm_for_eval()
elif args.method == 'geoldm':
    processed_mols, processed_mols_e_rdkit = geoldm_for_eval()

dataset_info = get_dataset_info('qm9', remove_h=False)
print('============================================================')
print('dataset_info:', dataset_info)
# Build evaluation metrics
# # ###################################### smiles from rdDetermineBonds.DetermineBonds(mol) ############################################################
###################################### smiles from original QM9 rdkit file ############################################################
qm9_data = torch.load('QM9/processed/data_qm9.pt')
split = torch.load('QM9/processed/split_dict_qm9.pt')
train_mols = [qm9_data[0].rdmol[i] for i in split['train']]
test_mols = [qm9_data[0].rdmol[i] for i in split['test']]
train_smiles = [mol2smiles(mol) for mol in train_mols]
test_smiles = [mol2smiles(mol) for mol in test_mols]
EDM_metric = get_edm_metric(dataset_info, train_smiles)
EDM_metric_2D = get_2D_edm_metric(dataset_info, train_smiles)
mose_metric = get_moses_metrics(test_smiles, n_jobs=32, device='cpu')
sub_geo_mmd_metric = get_sub_geometry_metric(test_mols, dataset_info, 'JODO/data/QM9')

def eval(data):
    # EDM evaluation metrics
    stability_res, rdkit_res, sample_rdmols = EDM_metric(data)
    print('============================================================')
    print(stability_res, rdkit_res)
    print('Number of molecules: %d' % len(sample_rdmols))
    print("Metric-3D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f," %
                    (stability_res['atom_stable'], stability_res['mol_stable'], rdkit_res['Validity'],
                    rdkit_res['Complete']))

    # Mose evaluation metrics
    mose_res = mose_metric(sample_rdmols)
    print('============================================================')
    print(mose_res)
    print("Metric-3D || 3D FCD: %.4f" % (mose_res['FCD']))

    # 2D evaluation metrics
    stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(data)
    print('============================================================')
    print(stability_res, rdkit_res)
    print("Metric-2D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f,"
                    " unique & valid: %.4f, unique & valid & novelty: %.4f" % (stability_res['atom_stable'],
                    stability_res['mol_stable'], rdkit_res['Validity'], rdkit_res['Complete'], rdkit_res['Unique'],
                    rdkit_res['Novelty']))
    mose_res = mose_metric(complete_rdmols)
    print('============================================================')
    print(mose_res)
    print("Metric-2D || FCD: %.4f, SNN: %.4f, Frag: %.4f, Scaf: %.4f, IntDiv: %.4f" % (mose_res['FCD'],
                    mose_res['SNN'], mose_res['Frag'], mose_res['Scaf'], mose_res['IntDiv']))
    
    sub_geo_mmd_res = sub_geo_mmd_metric(complete_rdmols)
    print('============================================================')
    print(sub_geo_mmd_res)
    print("Metric-Align || Bond Length MMD: %.4f, Bond Angle MMD: %.4f, Dihedral Angle MMD: %.6f" % (
        sub_geo_mmd_res['bond_length_mean'], sub_geo_mmd_res['bond_angle_mean'],
        sub_geo_mmd_res['dihedral_angle_mean']))
    ## bond length
    bond_length_str = ''
    for sym in dataset_info['top_bond_sym']:
        bond_length_str += f"{sym}: %.4f " % sub_geo_mmd_res[sym]
    print('============================================================')
    print('bond_length_str: ', bond_length_str)
    ## bond angle
    bond_angle_str = ''
    for sym in dataset_info['top_angle_sym']:
        bond_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
    print('============================================================')
    print('bond_angle_str: ', bond_angle_str)
    ## dihedral angle
    dihedral_angle_str = ''
    for sym in dataset_info['top_dihedral_sym']:
        dihedral_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
    print('============================================================')
    print('dihedral_angle_str: ', dihedral_angle_str)


if processed_mols != None:
    print('======================== processed_mols =============================')
    eval(processed_mols)
if processed_mols_e_rdkit != None:
    print('==================== processed_mols_e_rdkit =========================')
    eval(processed_mols_e_rdkit)
        