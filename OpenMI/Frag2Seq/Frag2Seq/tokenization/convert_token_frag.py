from tqdm import tqdm
import numpy as np
import argparse

from pathlib import Path
from time import time
import argparse
import shutil
import random
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import numpy as np

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa, one_to_three
from rdkit import Chem
from scipy.ndimage import gaussian_filter

import torch

from analysis.molecule_builder import build_molecule
from analysis.metrics import rdmol_to_smiles
import constants
from constants import covalent_radii, dataset_params

import lmdb
import gzip
import pickle
import os

import biotite.database.rcsb as rcsb
import biotite.structure.io.pdb as pdb
import biotite.structure.io as strucio

import esm.inverse_folding

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem
import numpy as np

        
def is_file_empty(file_path):
    """Check if file is empty by reading first character in it"""
    # Open the file in read mode ('r')
    with open(file_path, 'r') as file:
        # Read the first character
        first_char = file.read(1)
        # If not able to read 1 character, file is empty
        if not first_char:
            return True
    return False


def nan_to_num(vec, num=0.0):
    idx = np.isnan(vec)
    vec[idx] = num
    return vec

def _normalize(vec, axis=-1):
    return nan_to_num(
        np.divide(vec, np.linalg.norm(vec, axis=axis, keepdims=True)))



def create_local_coordinate_system(point1, point2):
    # Calculate the local Z-axis vector and normalize it
    z_axis = point2 - point1
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # Find a vector orthogonal to Z-axis for X-axis
    # Check if Z-axis is parallel to global Z-axis
    if np.allclose(z_axis, [0, 0, 1]) or np.allclose(z_axis, [0, 0, -1]):
        # Use global Y-axis if Z-axis is parallel/near to global Z-axis
        x_axis = np.cross(z_axis, [0, 1, 0])
    else:
        # Typically, use global Z-axis
        x_axis = np.cross(z_axis, [0, 0, 1])
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Calculate Y-axis as a cross product of Z-axis and X-axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # The local coordinate system's axes
    return x_axis, y_axis, z_axis

def create_affine_matrix(R, t):
    """Create a 4x4 affine transformation matrix from a rotation matrix R and a translation vector t."""
    A = np.eye(4)  # Start with an identity matrix
    A[:3, :3] = R  # Set the top-left 3x3 submatrix as the rotation matrix
    A[:3, 3] = t   # Set the top-right 3x1 column as the translation vector
    return A

def inverse_affine_matrix(A):
    """Compute the inverse of a 4x4 affine transformation matrix."""
    A_inv = np.linalg.inv(A)
    return A_inv

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def axis_angle_to_rotation_vector(u, theta):
    v = theta * u
    return v

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

def rotation_matrix_to_axis_angle(R):
    # Ensure R is a numpy array
    R = np.array(R)
    
    # Calculate the angle of rotation
    theta = np.arccos((np.trace(R) - 1) / 2.0)

    # Avoid division by zero by handling the case when theta is 0
    if theta > 1e-6:  # Use a small threshold to handle numerical precision issues
        # Calculate the components of the rotation axis
        ux = (R[2, 1] - R[1, 2]) / (2 * np.sin(theta))
        uy = (R[0, 2] - R[2, 0]) / (2 * np.sin(theta))
        uz = (R[1, 0] - R[0, 1]) / (2 * np.sin(theta))
        axis = np.array([ux, uy, uz])
    else:
        # For small angles, approximate the axis with any unit vector
        axis = np.array([1, 0, 0])  # Arbitrary choice

    return axis, theta

def isRotationMatrix(R):
    # square matrix test
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return False
    should_be_identity = np.allclose(R.dot(R.T), np.identity(R.shape[0], float))
    should_be_one = np.allclose(np.linalg.det(R), 1)
    return should_be_identity and should_be_one

def calculate_centroid(positions):
    """Calculate the centroid from a list of positions."""
    return np.mean(positions, axis=0)


def construct_local_frame(coords):
    centered_coords = coords - coords[0]
    num_nodes = len(centered_coords)

    invariant_spherical_coords = np.zeros_like(coords)
    
    # we have to select three nodes to build a global frame
    # when selecting the third node, we have to make sure this node is not on the same line as the first two nodes
    flag = False 

    if num_nodes == 1:
        x = np.array([1,0,0])
        y = np.array([0,1,0])
        z = np.array([0,0,1])
        local_coords = centered_coords
        # raise ValueError("does not support one node for now")
    elif num_nodes == 2:
        d = np.linalg.norm(coords[1] - coords[0], axis=-1)
        invariant_spherical_coords[1,0] = d
        
        x, y, z = create_local_coordinate_system(coords[1], coords[0])
        local_coords = np.dot(centered_coords, np.stack((x, y, z)).T)
        # raise ValueError("does not support two nodes for now")
    else:
        v1 = centered_coords[1] - centered_coords[0]
        for i in range(2, num_nodes):
            v2 = centered_coords[i] - centered_coords[0]
            if np.linalg.norm(np.cross(v1, v2)) != 0:
                flag = True # # can find the third node that is not on the same line as the first two nodes
                break
        if flag == False and i == num_nodes-1: # cannot find the third node that is not on the same line as the first two nodes
            local_coords = centered_coords
        else:
            # build a global frame (xyz axis)
            x = _normalize(v1)
            y = _normalize(np.cross(v1, v2))
            z = np.cross(x, y)
            # invariant coords
            local_coords = np.dot(centered_coords, np.stack((x, y, z)).T)
            # save rotation matrix and translation vector
    
        # convert to spherical coordinate
        d = np.linalg.norm(local_coords, axis=-1)
        theta = np.zeros_like(d)
        theta[1:] = np.arccos(local_coords[1:,2]/d[1:])
        phi = np.arctan2(local_coords[:,1], local_coords[:,0])
        # invariant_spherical_coords
        invariant_spherical_coords = np.stack((d, theta, phi)).T
        
    rotation_matrix = np.column_stack((x, y, z))
    translation = coords[0]
    # transform_matrix['translation'].append(coords[0])
    # transform_matrix['rotation'].append(T)
    
    return local_coords, invariant_spherical_coords, rotation_matrix, translation

def frags_to_seq(frags, centroid_list, fragments_dict):
    
    centroid_frag_coords = np.array(centroid_list).reshape(-1,3)
    frag_centroid_local_coords, frag_centroid_spherical_coords, R_m2w, t_m2w = construct_local_frame(centroid_frag_coords)
    assert isRotationMatrix(R_m2w), "R_m2w is not a valid rotation matrix"
    
    # Dictionary to store fragments
    # fragments_dict = {}
    seq_out_list = []
        
    for i, frag in enumerate(frags):
        # print(f"{Chem.MolToSmiles(frag)}")
        conf = frag.GetConformer()
        atom_positions = conf.GetPositions()
        
        frag_atom_local_coords, _, R_f2w, t_f2w = construct_local_frame(atom_positions)
        assert isRotationMatrix(R_f2w), "R_f2w is not a valid rotation matrix"
        
        R_f2m = np.dot(R_m2w.T, R_f2w)
        t_f2m = np.dot(R_m2w.T, t_f2w - t_m2w)
        t_f2c = frag_centroid_local_coords[i] - t_f2m
        
        R_axis, R_angle = rotation_matrix_to_axis_angle(R_f2m)
        R_vector = axis_angle_to_rotation_vector(R_axis, R_angle)
              


        # Getting the SMILES as the key
        smiles_key = Chem.MolToSmiles(frag, canonical=True)
        # conf = frag.GetConformer()
        # atom_positions = conf.GetPositions()


        # Inner dictionary: atom type to coordinates
        atom_dict = {}
        atom_type_list = []
        for atom in frag.GetAtoms():
            atom_idx = atom.GetIdx()
            # print(atom_idx)
            atom_type = atom.GetSymbol()  # Using atomic symbol as atom type
            atom_type_list.append(atom_type)

        atom_dict["coord"] = frag_atom_local_coords
        atom_dict["atom_type"] = atom_type_list
        atom_dict["t_f2c"] = t_f2c

        # Populate the outer dictionary
        if smiles_key not in fragments_dict:
            fragments_dict[smiles_key] = atom_dict

        # print(fragments_dict[smiles_key])
        
        seq_out = ""
        seq_out = seq_out + smiles_key + " " \
                f"{frag_centroid_spherical_coords[i][0]:.2f}" + " " + f"{frag_centroid_spherical_coords[i][1]:.2f}°" + " " + f"{frag_centroid_spherical_coords[i][2]:.2f}°" + " " + \
                f"{R_vector[0]:.2f}*" + " " + f"{R_vector[1]:.2f}*" + " " + f"{R_vector[2]:.2f}*" 
        # seq_out = seq_out + smiles_key + " " \
        #         f"{frag_centroid_spherical_coords[i][0]}" + " " + f"{frag_centroid_spherical_coords[i][1]}°" + " " + f"{frag_centroid_spherical_coords[i][2]}°" + " " + \
        #         f"{R_vector[0]}*" + " " + f"{R_vector[1]}*" + " " + f"{R_vector[2]}*" 
        
        # print(seq_out)
        seq_out_list.append(seq_out)
        
    return seq_out_list, R_m2w, t_m2w


def get_fragment_order(mol):
    # Identify rotatable bonds excluding those in rings
    rotatable_bond_indices = []
    for bond in mol.GetBonds():
        if not bond.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE:
            beginAtom = bond.GetBeginAtom()
            endAtom = bond.GetEndAtom()
            if beginAtom.GetDegree() > 1 and endAtom.GetDegree() > 1:
                rotatable_bond_indices.append(bond.GetIdx())

    # Fragment the molecule on the identified bonds
    try:
        fragmented_mol = Chem.FragmentOnBonds(mol, rotatable_bond_indices, addDummies=False)
    except:
        # print(Chem.MolToSmiles(mol, canonical=True))
        # raise ValueError("Can not split fragment")
        print("Can not split fragment")
        return None, None, None
        

    # Convert the fragmented molecule to individual fragments
    frags = Chem.GetMolFrags(fragmented_mol, asMols=True, sanitizeFrags=True)

    # Sort fragments based on the original molecule's atom indices to maintain connectivity order
    frags_sorted_by_first_atom_idx = sorted(frags, key=lambda frag: min(atom.GetIdx() for atom in frag.GetAtoms()))

    # Convert sorted fragments into SMILES strings
    ordered_frags_smiles = [Chem.MolToSmiles(frag, canonical=True) for frag in frags_sorted_by_first_atom_idx]


    return ordered_frags_smiles, rotatable_bond_indices, frags_sorted_by_first_atom_idx

def split_frag_gen_seq(mol, fragments_dict):
    
    ordered_frags_smiles, rotatable_bond_indices, ordered_frags = get_fragment_order(mol)
    if ordered_frags is None:
        R_m2w = np.eye(3)
        t_m2w = mol.GetConformer().GetPositions()[0]
        return None, R_m2w, t_m2w
    
    frag_list = []
    centroid_list = []
    # gt_atom_positions_list = []

    for i, frag in enumerate(ordered_frags):
        # print(f"Fragment {i + 1}: {Chem.MolToSmiles(frag)}")
        conf = frag.GetConformer()
        atom_positions = conf.GetPositions()
        # print(atom_positions)
        # gt_atom_positions_list.append(atom_positions)
        # frag_list.append(ordered_frags_smiles[i])
        # Calculate centroid of the fragment
        centroid = calculate_centroid(atom_positions)
        # print(f"Centroid coordinates of fragment: {centroid}")
        centroid_list.extend(centroid)
        
    seq_out_list, R_m2w, t_m2w = frags_to_seq(ordered_frags, centroid_list, fragments_dict)
    
    return seq_out_list, R_m2w, t_m2w


def generate_frag_seq(data, base_dir, test=False, process_lig_only=False, max_len=560, fragments_dict=False, save_transformation_matrix=False):
    pocket_word_count_list = []
    prefix = 'test' if test else 'train'
    
    names = data['names']
    lig_coords = data['lig_coords']
    lig_one_hot = data['lig_one_hot']
    lig_mask = data['lig_mask']
    pocket_coords = data['pocket_coords']
    pocket_one_hot = data['pocket_one_hot']
    pocket_mask = data['pocket_mask']
    
    pocket_length_list = []
    ligand_length_list = []
    
    transform_matrix = {'R_m2w': [], 't_m2w': []}
    
    if process_lig_only:
        split = 'test' if test else 'train'
        filename = f"protein_embedding_{split}.lmdb"
        delete_file = str(base_dir / filename)
        os.system(f"rm -f {delete_file}")
        file_path = str(base_dir / filename)
        env_new = lmdb.open(
            f"{file_path}",
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(100e9),
        )
        
        txn_write = env_new.begin(write=True)
        
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model = model.eval()
        
    count = 0
    
    
    with open(str(base_dir / f'{prefix}_frag_seq.txt'), 'w') as f:
        for i in tqdm(range(len(names))):
            
            valid_mol = True
            valid_protein = True
            
            name = names[i]
            sdf_data_root = Path('../crossdock2020/crossdocked_pocket10') / name.split('.pdb_')[1].strip()
            # print(str(data_root))
            supplier = Chem.SDMolSupplier(str(sdf_data_root))
            mol = supplier[0]
            
            seq_out_list, R_m2w, t_m2w = split_frag_gen_seq(mol, fragments_dict)
            if save_transformation_matrix:
                transform_matrix['R_m2w'].append(R_m2w)
                transform_matrix['t_m2w'].append(t_m2w)
            
            if seq_out_list is None:
                valid_mol = False
                if not test:
                    continue
            
            if not process_lig_only:
                pocket_atom_type_index = np.argmax(pocket_one_hot[pocket_mask==i], axis=1)
                pocket_atom_pos = pocket_coords[pocket_mask==i]

                pocket_length_list.append(pocket_atom_pos.shape[0])

                
                protein_local_coords, protein_spherical_coords, _, _ = construct_local_frame(pocket_atom_pos)
                
                for j in range(pocket_atom_type_index.shape[0]):
                    pocket_atom_type = one_to_three(amino_acid_swapped_dict[pocket_atom_type_index[j]])
                    
                    # f.write(f'{pocket_atom_type} {pocket_atom_pos[j][0]:.2f} {pocket_atom_pos[j][1]:.2f} {pocket_atom_pos[j][2]:.2f} ')
                    
                    f.write(f'{pocket_atom_type} {protein_spherical_coords[j][0]:.2f} {protein_spherical_coords[j][1]:.2f}° {protein_spherical_coords[j][2]:.2f}° ')
                
                pocket_word_count = (j+1) * 4
                pocket_word_count_list.append(pocket_word_count)
                
            if (test and process_lig_only):
                
                with open(str(base_dir / f'{prefix}_names.txt'), 'a') as file:
                    file.write(names[i] + '\n')
                
                if seq_out_list is None:
                    seq = " "
                else:
                    seq = " ".join(seq_out_list)
                f.write(seq)
                f.write('\n')
            
            
            if process_lig_only:
                root = "../crossdock2020/crossdocked_pocket10"
                path = Path(data['names'][i].split('.pdb')[0] + '.pdb')
                fpath = root / path
                # generate_embedding(txn_write, count=i, split=prefix, data=data, max_len=560)
                
                pdb_file = strucio.load_structure(fpath)

                chain_ids = pdb_file.chain_id
                chain_ids = list(set(chain_ids))


                try:
                    structure = esm.inverse_folding.util.load_structure(str(fpath), chain_ids)
                    coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
                except:
                    print(f"idx: {i} Error in parsing {fpath}")
                    if test:
                        embedding_dict = {"padded_embedding": None, "mask": None, "length": None}
                        key = int(count).to_bytes(4, byteorder="big")
                        txn_write.put(key, gzip.compress(pickle.dumps(embedding_dict)))
                        count += 1
                    continue
                
                
                # extracrt encoder output for multiple chains
                rep_list = []
                for chain_id in chain_ids:
                    rep = esm.inverse_folding.multichain_util.get_encoder_output_for_complex(model, alphabet, coords, chain_id)
                    rep_list.append(rep)
                    
                rep = torch.concat(rep_list, axis=0)
                
                m = rep.shape[0]
                padding = torch.zeros(max_len - m, 512)
                padded_embedding = torch.cat([rep, padding], dim=0)
                mask = torch.ones(max_len)
                mask[m:] = 0
                
                embedding_dict = {"padded_embedding": padded_embedding, "mask": mask, "length": m}
                
                if (not test) and (not valid_mol):
                    continue
                
                key = int(count).to_bytes(4, byteorder="big")
                txn_write.put(key, gzip.compress(pickle.dumps(embedding_dict)))
                
                if count % 10000 == 0:
                    txn_write.commit()
                    txn_write = env_new.begin(write=True)
            
                count += 1
        
                
        
            if not test:
                with open(str(base_dir / f'{prefix}_names.txt'), 'a') as file:
                    file.write(names[i] + '\n')
                
                
                if seq_out_list is not None:
                    seq = " ".join(seq_out_list)
                    f.write(seq)
                    f.write('\n')
            else:
                f.write('\n')
        
            
    if process_lig_only:
        txn_write.commit()
        env_new.close()

    if test:
        with open(str(base_dir / f'{prefix}_mol_coord_seq.txt'), 'w') as f:
            for i in tqdm(range(len(names))):
                lig_atom_type_index = np.argmax(lig_one_hot[lig_mask==i], axis=1)
                lig_atom_pos = lig_coords[lig_mask==i]
                ligand_length_list.append(lig_atom_pos.shape[0])
                for k in range(lig_atom_type_index.shape[0]):
                    lig_atom_type = atom_swapped_dict[lig_atom_type_index[k]]
                    f.write(f'{lig_atom_type} {lig_atom_pos[k][0]:.6f} {lig_atom_pos[k][1]:.6f} {lig_atom_pos[k][2]:.6f} ')
                f.write('\n')

    
    if (not test) and (not process_lig_only):
        with open(str(base_dir / f'train_pocket_word_count.txt'), 'w') as f:
            for pocket_word_count in tqdm(pocket_word_count_list):
                f.write(f'{pocket_word_count}\n')


    with open(str(base_dir / 'transformation_matrix.pickle'), 'wb') as file:
        pickle.dump(transform_matrix, file)


    return pocket_length_list, ligand_length_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str, default='')
    parser.add_argument('--save_folder', type=str, default='./seq/')
    parser.add_argument('--seperate_prot_lig', action='store_true')
    parser.add_argument('--process_lig_only', action='store_true')
    parser.add_argument('--input_path', type=str, default='./train_coord_seq.txt')
    parser.add_argument('--write_path', type=str, default='./train_spherical_seq.txt')
    args = parser.parse_args()
    
    import os
    
    delete_file = os.path.join(args.save_folder, 'train_frag_seq.txt')
    os.system(f'rm {delete_file}')
    delete_file = os.path.join(args.save_folder, 'train_spherical_seq.txt')
    os.system(f'rm {delete_file}')
    delete_file = os.path.join(args.save_folder, 'train_pocket_word_count.txt')
    os.system(f'rm {delete_file}')
    delete_file = os.path.join(args.save_folder, 'train_names.txt')
    os.system(f'rm {delete_file}')
    delete_file = os.path.join(args.save_folder, 'test_frag_seq.txt')
    os.system(f'rm {delete_file}')
    delete_file = os.path.join(args.save_folder, 'test_spherical_seq.txt')
    os.system(f'rm {delete_file}')
    delete_file = os.path.join(args.save_folder, 'test_names.txt')
    os.system(f'rm {delete_file}')
    delete_file = os.path.join(args.save_folder, 'pocket_length_dict.pickle')
    os.system(f'rm {delete_file}')
    delete_file = os.path.join(args.save_folder, 'ligand_length_dict.pickle')
    os.system(f'rm {delete_file}')
    delete_file = os.path.join(args.save_folder, 'transformation_matrix.pickle')
    os.system(f'rm {delete_file}')
    delete_file = os.path.join(args.save_folder, 'fragments_dict.pickle')
    os.system(f'rm {delete_file}')


    base_dir = Path(args.save_folder)
    
    if not base_dir.exists():
        base_dir.mkdir()
    
    ca_only = True
    if ca_only:
        dataset_info = dataset_params['crossdock']
    else:
        dataset_info = dataset_params['crossdock_full']
        
    amino_acid_dict = dataset_info['aa_encoder']
    atom_dict = dataset_info['atom_encoder']
    atom_decoder = dataset_info['atom_decoder']
    
    
    train_data = np.load(os.path.join("../crossdock2020/", f"{args.base_folder}", "train.npz"))
    test_data = np.load(os.path.join("../crossdock2020/", f"{args.base_folder}", "test.npz"))
    
    
    atom_swapped_dict = {value: key for key, value in atom_dict.items()}
    amino_acid_swapped_dict = {value: key for key, value in amino_acid_dict.items()}
    
    
    pocket_length_dict = {'train': [], 'test': []}
    ligand_length_dict = {'train': [], 'test': []}
    
    fragments_dict = {}
    
    pocket_length, ligand_length = generate_frag_seq(train_data, base_dir=base_dir, test=False, process_lig_only=args.process_lig_only, fragments_dict=fragments_dict)
    pocket_length_dict['train'].extend(pocket_length)
    ligand_length_dict['train'].extend(ligand_length)
    
    pocket_length, ligand_length = generate_frag_seq(test_data, base_dir=base_dir, test=True, process_lig_only=args.process_lig_only, fragments_dict=fragments_dict, save_transformation_matrix=True)
    pocket_length_dict['test'].extend(pocket_length)
    ligand_length_dict['test'].extend(ligand_length)
    
    
    with open(f'{args.save_folder}fragments_dict.pickle', 'wb') as file:
        pickle.dump(fragments_dict, file)
    
    with open(f'{args.save_folder}pocket_length_dict.pickle', 'wb') as file:
        pickle.dump(pocket_length_dict, file)
        
    with open(f'{args.save_folder}ligand_length_dict.pickle', 'wb') as file:
        pickle.dump(ligand_length_dict, file)
    
    
