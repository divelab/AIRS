from pathlib import Path
from time import time
import argparse
import shutil
import random

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import numpy as np

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa
from rdkit import Chem
from scipy.ndimage import gaussian_filter

import torch

from analysis.molecule_builder import build_molecule
from analysis.metrics import rdmol_to_smiles
import constants
from constants import covalent_radii, dataset_params

import warnings
import tempfile

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, UFFHasAllMoleculeParams
from openbabel import openbabel

from analysis.molecule_builder import process_molecule

import pickle
import glob
import os


from constants import bonds1, bonds2, bonds3, margin1, margin2, margin3, \
    bond_dict
    
# 
import copy
import collections
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from rdkit.Chem import PeriodicTable
from rdkit.Chem.Lipinski import RotatableBondSmarts
import scipy
from scipy import spatial as sci_spatial
import torch
from tqdm.auto import tqdm
import seaborn as sns
from collections import Counter


def rotation_vector_to_axis_angle(v):
    theta = np.linalg.norm(v)
    u = v / theta if theta > 0 else np.array([1, 0, 0])  # Default axis if theta is 0
    return u, theta

def axis_angle_to_rotation_matrix(axis, angle):
    ux, uy, uz = axis
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    
    # Skew-symmetric matrix for axis
    u_skew = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0]
    ])
    
    # Outer product of axis with itself
    uuT = np.outer(axis, axis)
    
    # Rodrigues' rotation formula
    rotation_matrix = cos_theta * np.eye(3) + sin_theta * u_skew + (1 - cos_theta) * uuT
    
    return rotation_matrix

def seq_to_mol(seq, fragments_dict, R_m2w, t_m2w, gt_atom_positions_list=None, no_SPH=False):
    
    mol = seq.split()
    
    try:
        mol = np.array(mol).reshape(-1, 7)
    except:
        # invalid_seq += 1
        return None

    position_list = []
    atom_type_list = []
    
    for i in range(mol.shape[0]):
        # centroid_coord = np.array(seq_out.split()[1:4])
        # centroid_coord = centroid_coord.astype(float)
        spherical_coord = np.array(mol[i, 1:4])
        # print(spherical_coord)
        if not no_SPH:
            if (spherical_coord[0].endswith('°') or spherical_coord[0].endswith('*')) or (not spherical_coord[1].endswith('°')) or (not spherical_coord[2].endswith('°')):
                return None
            else:
                spherical_coord = [s[:-1] if s.endswith('°') else s for s in spherical_coord]
        try:
            spherical_coord = np.array(spherical_coord).astype(float)
        except:
            return None
        centroid_coord = np.zeros_like(spherical_coord)
        
        if no_SPH:
            centroid_coord[0], centroid_coord[1], centroid_coord[2] = spherical_coord[0], spherical_coord[1], spherical_coord[2]
        else:
            r = spherical_coord[0]
            theta = spherical_coord[1]
            phi = spherical_coord[2]
            centroid_coord[0], centroid_coord[1], centroid_coord[2] = spherical_to_cartesian(r, theta, phi)
        
        
        frag_type = mol[i, 0]
        # get atom_type and local coordinate of each atom in the fragment
        try:
            frag = fragments_dict[frag_type]
        except:
            return None
        frag_atom_coord = frag["coord"]
        frag_atom_type = frag["atom_type"]
        t_f2c = frag["t_f2c"]


        
        R_vector = mol[i, 4:]
        if (not R_vector[0].endswith('*')) or (not R_vector[1].endswith('*')) or (not R_vector[2].endswith('*')):
            return None
        else:
            R_vector = [s[:-1] if s.endswith('*') else s for s in R_vector]
        R_vector = np.array(R_vector).astype(float)
        R_axis, R_angle = rotation_vector_to_axis_angle(R_vector)
        R_f2m_from_seq = axis_angle_to_rotation_matrix(R_axis, R_angle)
        t_f2m_from_seq = centroid_coord - t_f2c
        
        
        # get atom position in the molecule local frame
        frag_atom_coord_in_molecule = (frag_atom_coord @ R_f2m_from_seq.T) + t_f2m_from_seq
        # get atom position in the world frame
        frag_atom_coord_world = (frag_atom_coord_in_molecule @ R_m2w.T) + t_m2w
        if gt_atom_positions_list:
            print(np.allclose(gt_atom_positions_list[i], frag_atom_coord_world))
            
            
        position_list.append(frag_atom_coord_world)
        atom_type_list.extend(frag_atom_type)
            

    mol_dict = {}
    try:
        mol_dict["positions"] = np.concatenate(position_list, axis=0)
    except:
        import pdb; pdb.set_trace()
    mol_dict["atom_type"] = atom_type_list
    
    return mol_dict


def get_bond_order(atom1, atom2, distance):
    distance = 100 * distance  # We change the metric

    if atom1 in bonds3 and atom2 in bonds3[atom1] and distance < bonds3[atom1][atom2] + margin3:
        return 3  # Triple

    if atom1 in bonds2 and atom2 in bonds2[atom1] and distance < bonds2[atom1][atom2] + margin2:
        return 2  # Double

    if atom1 in bonds1 and atom2 in bonds1[atom1] and distance < bonds1[atom1][atom2] + margin1:
        return 1  # Single

    return 0      # No bond


def get_bond_order_batch(atoms1, atoms2, distances, dataset_info):
    if isinstance(atoms1, np.ndarray):
        atoms1 = torch.from_numpy(atoms1)
    if isinstance(atoms2, np.ndarray):
        atoms2 = torch.from_numpy(atoms2)
    if isinstance(distances, np.ndarray):
        distances = torch.from_numpy(distances)

    distances = 100 * distances  # We change the metric

    bonds1 = torch.tensor(dataset_info['bonds1'], device=atoms1.device)
    bonds2 = torch.tensor(dataset_info['bonds2'], device=atoms1.device)
    bonds3 = torch.tensor(dataset_info['bonds3'], device=atoms1.device)

    bond_types = torch.zeros_like(atoms1)  # 0: No bond

    # Single
    bond_types[distances < bonds1[atoms1, atoms2] + margin1] = 1

    # Double (note that already assigned single bonds will be overwritten)
    bond_types[distances < bonds2[atoms1, atoms2] + margin2] = 2

    # Triple
    bond_types[distances < bonds3[atoms1, atoms2] + margin3] = 3

    return bond_types

def make_mol_edm(positions, atom_types, dataset_info, add_coords):
    """
    Equivalent to EDM's way of building RDKit molecules
    """
    n = len(positions)

    # (X, A, E): atom_types, adjacency matrix, edge_types
    # X: N (int)
    # A: N x N (bool) -> (binary adjacency matrix)
    # E: N x N (int) -> (bond type, 0 if no bond)
    positions = torch.tensor(positions, dtype=torch.float32)
    tmp = []
    for i in range(len(atom_types)):
        tmp.append(dataset_info['atom_encoder'][atom_types[i]])
    atom_types = torch.tensor(tmp, dtype=torch.int64)

    
    pos = positions.unsqueeze(0)  # add batch dim
    dists = torch.cdist(pos, pos, p=2).squeeze(0).view(-1)  # remove batch dim & flatten
    atoms1, atoms2 = torch.cartesian_prod(atom_types, atom_types).T
    E_full = get_bond_order_batch(atoms1, atoms2, dists, dataset_info).view(n, n)
    E = torch.tril(E_full, diagonal=-1)  # Warning: the graph should be DIRECTED
    A = E.bool()
    X = atom_types

    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(dataset_info["atom_decoder"][atom.item()])
        mol.AddAtom(a)


    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(),
                    bond_dict[E[bond[0], bond[1]].item()])

    if add_coords:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (positions[i, 0].item(),
                                     positions[i, 1].item(),
                                     positions[i, 2].item()))
        mol.AddConformer(conf)
    # import pdb; pdb.set_trace()
    return mol

def is_directory_empty(directory):
    return len(os.listdir(directory)) == 0

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

def write_sdf_file(sdf_path, molecules):
    # NOTE Changed to be compatitble with more versions of rdkit
    #with Chem.SDWriter(str(sdf_path)) as w:
    #    for mol in molecules:
    #        w.write(mol)

    w = Chem.SDWriter(str(sdf_path))
    w.SetKekulize(False)
    for m in molecules:
        if m is not None:
            w.write(m)

def write_xyz_file(coords, atom_types, filename):
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, 'w') as f:
        f.write(out)

def make_mol_openbabel(positions, atom_types, atom_decoder):
    """
    Build an RDKit molecule using openbabel for creating bonds
    Args:
        positions: N x 3
        atom_types: N
        atom_decoder: maps indices to atom types
    Returns:
        rdkit molecule
    """
    # atom_types = [atom_decoder[x] for x in atom_types]
    # import pdb; pdb.set_trace()
    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name

        # Write xyz file
        write_xyz_file(positions, atom_types, tmp_file)

        # Convert to sdf file with openbabel
        # openbabel will add bonds
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")
        ob_mol = openbabel.OBMol()
        obConversion.ReadFile(ob_mol, tmp_file)

        obConversion.WriteFile(ob_mol, tmp_file)

        # Read sdf file with RDKit
        tmp_mol = Chem.SDMolSupplier(tmp_file, sanitize=False)[0]

    # Build new molecule. This is a workaround to remove radicals.
    mol = Chem.RWMol()
    for atom in tmp_mol.GetAtoms():
        mol.AddAtom(Chem.Atom(atom.GetSymbol()))
    mol.AddConformer(tmp_mol.GetConformer(0))
    # import pdb; pdb.set_trace()
    for bond in tmp_mol.GetBonds():
        mol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                    bond.GetBondType())

    return mol

def build_molecule(positions, atom_types, dataset_info, add_coords=False,
                   use_openbabel=True):
    """
    Build RDKit molecule
    Args:
        positions: N x 3
        atom_types: N
        dataset_info: dict
        add_coords: Add conformer to mol (always added if use_openbabel=True)
        use_openbabel: use OpenBabel to create bonds
    Returns:
        RDKit molecule
    """
    if use_openbabel:
        mol = make_mol_openbabel(positions, atom_types,
                                 dataset_info["atom_decoder"])
    else:
        mol = make_mol_edm(positions, atom_types, dataset_info, add_coords)

    return mol


if __name__ == '__main__':
    parser = argparse.ArgumentParser('evaluation')
    parser.add_argument('--save_dir', type=str, help='')
    parser.add_argument('--ca_only', action='store_true')
    parser.add_argument('--largest_frag', action='store_true')
    parser.add_argument('--use_openbabel', action='store_true')
    parser.add_argument('--transformation_file_path', type=str, help='')
    parser.add_argument('--no_SPH', action='store_true')
    parser.add_argument('--train_data_foler', type=str, help='')
    parser.add_argument('--min_size', type=int, help='', default=15)
    parser.add_argument('--max_size', type=int, help='', default=35)
    
    args = parser.parse_args()


# load test data
test_data = np.load("./crossdock2020/processed_crossdock_noH_ca_only_temp_smiles_reorder_cutoff_15/test.npz")

# load dataset info
ca_only = args.ca_only
if ca_only:
    dataset_info = dataset_params['crossdock']
else:
    dataset_info = dataset_params['crossdock_full']

amino_acid_dict = dataset_info['aa_encoder']
atom_dict = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']


# post processing of generated molecules
base_path = Path('./sample_output/')
if not os.path.exists(base_path / args.save_dir / 'raw_seq'):
    os.makedirs(base_path / args.save_dir / 'raw_seq')


if not os.path.exists(base_path / args.save_dir / 'processed_mol_sdf'):
    os.makedirs(base_path / args.save_dir / 'processed_mol_sdf')


delete_files = base_path / args.save_dir / 'processed_mol_sdf' / '*.sdf'
os.system(f'rm {delete_files}')
delete_files = base_path / args.save_dir / 'coord_seq' / '*.txt'
os.system(f'rm {delete_files}')


with open(f"./seq/{args.train_data_foler}/fragments_dict.pickle", 'rb') as file:
    fragments_dict = pickle.load(file)
    
with open(f"./seq/{args.train_data_foler}/transformation_matrix.pickle", 'rb') as file:
    transformation_matrix = pickle.load(file)

assert len(transformation_matrix['R_m2w']) == 100
assert len(transformation_matrix['t_m2w']) == 100
R_m2w_list = transformation_matrix['R_m2w']
t_m2w_list = transformation_matrix['t_m2w']



path_list = [f'./sample_output/{args.save_dir}/raw_seq/gen_mol_{i}.txt' for i in range(100)]


print("=== Buidling molecules from sequence ===")


pocket_mol = []

for idx, path in tqdm(enumerate(path_list), total=len(path_list)):
    
    R_m2w = R_m2w_list[idx]
    t_m2w = t_m2w_list[idx]
    assert R_m2w.dtype == np.float64
    assert t_m2w.dtype == np.float64
    
    with open(path, 'r') as file:
        lines = file.readlines()
        mol_list = []
        for line in lines:
            if line == '\n':
                continue
            
            mol_dict = seq_to_mol(line, fragments_dict, R_m2w, t_m2w, gt_atom_positions_list=None, no_SPH=args.no_SPH)
            if mol_dict is None:
                continue
                
            positions = mol_dict["positions"]
            atom_type = mol_dict["atom_type"]


            assert len(atom_type) == len(positions)
            try:
                mol = build_molecule(positions, atom_type, dataset_info, add_coords=True, use_openbabel=args.use_openbabel)
            except:
                raise ValueError('cannot build molecule')

            mol = process_molecule(mol, add_hydrogens=False, sanitize=True, relax_iter=200, largest_frag=args.largest_frag)
            
            if mol is not None:
                # make average molecule size similar to other methods and testset
                if mol.GetNumAtoms() < args.min_size or mol.GetNumAtoms() > args.max_size:
                    continue
            
            if mol is not None:
                mol_list.append(mol)
            if len(mol_list) == 100:
                break

        pocket_mol.append(mol_list)


torch.save(pocket_mol, './sample_output/{args.save_dir}/LM_generate_mol.pt')


write_path = f'./sample_output/{args.save_dir}/log.txt'


# evaluate QED, SA, LogP, Lipinski, Diversity
print("=== Calculating QED, SA, LogP, Lipinski, Diversity ===")
from analysis.metrics import MoleculeProperties
mol_metrics = MoleculeProperties()
all_qed, all_sa, all_logp, all_lipinski, per_pocket_diversity = mol_metrics.evaluate(pocket_mol, write_path)


##### evaluate docking score            

# molecule file naming: {pocket_PDBid}_pocket_gen_mol.sdf
# pdb file naming: {pocket_PDBid}_pocket.pdb

# get name of test file
print("=== Writing RDKit mol to SDF ===")
test_file_names = test_data['names']
for idx, test_file_name in enumerate(test_file_names):
    name = test_data['names'][idx]
    pdb_id = name.split('/')[-1][:4]   
    
    # write molecule to sdf file
    outdir = f'./sample_output/{args.save_dir}/processed_mol_sdf/'
    if len(pocket_mol[idx]) != 0:
        write_sdf_file(Path(outdir, f'{pdb_id}_pocket_gen.sdf'), pocket_mol[idx])
    


