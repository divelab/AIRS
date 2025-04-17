# code adapted from https://github.com/arneschneuing/DiffSBDD/blob/main/process_crossdock.py

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


def process_ligand_and_pocket(pdbfile, sdffile,
                              atom_dict, dist_cutoff, ca_only, reorder):
    pdb_struct = PDBParser(QUIET=True).get_structure('', pdbfile)

    try:
        ligand = Chem.SDMolSupplier(str(sdffile))[0]
    except:
        raise Exception(f'cannot read sdf mol ({sdffile})')

    if reorder:
        # reorder atoms in ligand to match the order in SMILES
        Chem.MolToSmiles(ligand)
        order = ligand.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder']
        ligand = Chem.RenumberAtoms(ligand,order)
        # atom_type = np.array([atom.GetSymbol() for atom in reorder_mol.GetAtoms()])
        # atomic_number = np.array([atom.GetAtomicNum() for atom in reorder_mol.GetAtoms()])
        # smiles_order_coords = reorder_mol.GetConformer().GetPositions()



    # remove H atoms if not in atom_dict, other atom types that aren't allowed
    # should stay so that the entire ligand can be removed from the dataset
    lig_atoms = [a.GetSymbol() for a in ligand.GetAtoms()
                 if (a.GetSymbol().capitalize() in atom_dict or a.element != 'H')]
    lig_coords = np.array([list(ligand.GetConformer(0).GetAtomPosition(idx))
                           for idx in range(ligand.GetNumAtoms())])

    try:
        lig_one_hot = np.stack([
            np.eye(1, len(atom_dict), atom_dict[a.capitalize()]).squeeze()
            for a in lig_atoms
        ])
    except KeyError as e:
        raise KeyError(
            f'{e} not in atom dict ({sdffile})')

    # Find interacting pocket residues based on distance cutoff
    pocket_residues = []
    total_residues = 0
    for residue in pdb_struct[0].get_residues():
        total_residues += 1
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
        if is_aa(residue.get_resname(), standard=True) and \
                (((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(
                    -1) ** 0.5).min() < dist_cutoff:
            pocket_residues.append(residue)
            
    pocket_residues_cutoff_percentage = round(len(pocket_residues) / total_residues, 2)

    pocket_ids = [f'{res.parent.id}:{res.id[1]}' for res in pocket_residues]
    ligand_data = {
        'lig_coords': lig_coords,
        'lig_one_hot': lig_one_hot,
    }
    if ca_only:
        try:
            pocket_one_hot = []
            full_coords = []
            for res in pocket_residues:
                for atom in res.get_atoms():
                    if atom.name == 'CA':
                        pocket_one_hot.append(np.eye(1, len(amino_acid_dict),
                                                     amino_acid_dict[three_to_one(res.get_resname())]).squeeze())
                        full_coords.append(atom.coord)
            pocket_one_hot = np.stack(pocket_one_hot)
            full_coords = np.stack(full_coords)
        except KeyError as e:
            raise KeyError(
                f'{e} not in amino acid dict ({pdbfile}, {sdffile})')
        pocket_data = {
            'pocket_coords': full_coords,
            'pocket_one_hot': pocket_one_hot,
            'pocket_ids': pocket_ids
        }
    else:
        full_atoms = np.concatenate(
            [np.array([atom.element for atom in res.get_atoms()])
             for res in pocket_residues], axis=0)
        full_coords = np.concatenate(
            [np.array([atom.coord for atom in res.get_atoms()])
             for res in pocket_residues], axis=0)
        try:
            pocket_one_hot = []
            for a in full_atoms:
                if a in amino_acid_dict:
                    atom = np.eye(1, len(amino_acid_dict),
                                  amino_acid_dict[a.capitalize()]).squeeze()
                elif a != 'H':
                    atom = np.eye(1, len(amino_acid_dict),
                                  len(amino_acid_dict)).squeeze()
                pocket_one_hot.append(atom)
            pocket_one_hot = np.stack(pocket_one_hot)
        except KeyError as e:
            raise KeyError(
                f'{e} not in atom dict ({pdbfile})')
        pocket_data = {
            'pocket_coords': full_coords,
            'pocket_one_hot': pocket_one_hot,
            'pocket_ids': pocket_ids
        }
    return ligand_data, pocket_data, pocket_residues_cutoff_percentage


def compute_smiles(positions, one_hot, mask):
    print("Computing SMILES ...")

    atom_types = np.argmax(one_hot, axis=-1)

    sections = np.where(np.diff(mask))[0] + 1
    positions = [torch.from_numpy(x) for x in np.split(positions, sections)]
    atom_types = [torch.from_numpy(x) for x in np.split(atom_types, sections)]

    mols_smiles = []

    pbar = tqdm(enumerate(zip(positions, atom_types)),
                total=len(np.unique(mask)))
    for i, (pos, atom_type) in pbar:
        mol = build_molecule(pos, atom_type, dataset_info)
        mol = rdmol_to_smiles(mol)
        if mol is not None:
            mols_smiles.append(mol)
        pbar.set_description(f'{len(mols_smiles)}/{i + 1} successful')

    return mols_smiles


def get_n_nodes(lig_mask, pocket_mask, smooth_sigma=None):
    # Joint distribution of ligand's and pocket's number of nodes
    idx_lig, n_nodes_lig = np.unique(lig_mask, return_counts=True)
    idx_pocket, n_nodes_pocket = np.unique(pocket_mask, return_counts=True)
    assert np.all(idx_lig == idx_pocket)

    joint_histogram = np.zeros((np.max(n_nodes_lig) + 1,
                                np.max(n_nodes_pocket) + 1))

    for nlig, npocket in zip(n_nodes_lig, n_nodes_pocket):
        joint_histogram[nlig, npocket] += 1

    print(f'Original histogram: {np.count_nonzero(joint_histogram)}/'
          f'{joint_histogram.shape[0] * joint_histogram.shape[1]} bins filled')

    # Smooth the histogram
    if smooth_sigma is not None:
        filtered_histogram = gaussian_filter(
            joint_histogram, sigma=smooth_sigma, order=0, mode='constant',
            cval=0.0, truncate=4.0)

        print(f'Smoothed histogram: {np.count_nonzero(filtered_histogram)}/'
              f'{filtered_histogram.shape[0] * filtered_histogram.shape[1]} bins filled')

        joint_histogram = filtered_histogram

    return joint_histogram


def get_bond_length_arrays(atom_mapping):
    bond_arrays = []
    for i in range(3):
        bond_dict = getattr(constants, f'bonds{i + 1}')
        bond_array = np.zeros((len(atom_mapping), len(atom_mapping)))
        for a1 in atom_mapping.keys():
            for a2 in atom_mapping.keys():
                if a1 in bond_dict and a2 in bond_dict[a1]:
                    bond_len = bond_dict[a1][a2]
                else:
                    bond_len = 0
                bond_array[atom_mapping[a1], atom_mapping[a2]] = bond_len

        assert np.all(bond_array == bond_array.T)
        bond_arrays.append(bond_array)

    return bond_arrays


def get_lennard_jones_rm(atom_mapping):
    # Bond radii for the Lennard-Jones potential
    LJ_rm = np.zeros((len(atom_mapping), len(atom_mapping)))

    for a1 in atom_mapping.keys():
        for a2 in atom_mapping.keys():
            all_bond_lengths = []
            for btype in ['bonds1', 'bonds2', 'bonds3']:
                bond_dict = getattr(constants, btype)
                if a1 in bond_dict and a2 in bond_dict[a1]:
                    all_bond_lengths.append(bond_dict[a1][a2])

            if len(all_bond_lengths) > 0:
                # take the shortest possible bond length because slightly larger
                # values aren't penalized as much
                bond_len = min(all_bond_lengths)
            else:
                if a1 == 'others' or a2 == 'others':
                    bond_len = 0
                else:
                    # Replace missing values with sum of average covalent radii
                    bond_len = covalent_radii[a1] + covalent_radii[a2]

            LJ_rm[atom_mapping[a1], atom_mapping[a2]] = bond_len

    assert np.all(LJ_rm == LJ_rm.T)
    return LJ_rm


def get_type_histograms(lig_one_hot, pocket_one_hot, atom_encoder, aa_encoder):
    atom_decoder = list(atom_encoder.keys())
    atom_counts = {k: 0 for k in atom_encoder.keys()}
    for a in [atom_decoder[x] for x in lig_one_hot.argmax(1)]:
        atom_counts[a] += 1

    aa_decoder = list(aa_encoder.keys())
    aa_counts = {k: 0 for k in aa_encoder.keys()}
    for r in [aa_decoder[x] for x in pocket_one_hot.argmax(1)]:
        aa_counts[r] += 1

    return atom_counts, aa_counts


def saveall(filename, pdb_and_mol_ids, lig_coords, lig_one_hot, lig_mask,
            pocket_coords, pocket_one_hot, pocket_mask):
    np.savez(filename,
             names=pdb_and_mol_ids,
             lig_coords=lig_coords,
             lig_one_hot=lig_one_hot,
             lig_mask=lig_mask,
             pocket_coords=pocket_coords,
             pocket_one_hot=pocket_one_hot,
             pocket_mask=pocket_mask
             )
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', type=Path)
    parser.add_argument('--outdir', type=Path, default=None)
    parser.add_argument('--no_H', action='store_true')
    parser.add_argument('--ca_only', action='store_true')
    parser.add_argument('--dist_cutoff', type=float, default=8.0)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--reorder', action='store_true')
    
    args = parser.parse_args()

    datadir = args.basedir / 'crossdocked_pocket10/'

    if args.ca_only:
        dataset_info = dataset_params['crossdock']
    else:
        dataset_info = dataset_params['crossdock_full']
    amino_acid_dict = dataset_info['aa_encoder']
    atom_dict = dataset_info['atom_encoder']
    atom_decoder = dataset_info['atom_decoder']

    # Make output directory
    if args.outdir is None:
        suffix = '_crossdock' if 'H' in atom_dict else '_crossdock_noH'
        suffix += '_ca_only_temp' if args.ca_only else '_full_temp'
        processed_dir = Path(args.basedir, f'processed{suffix}')
    else:
        processed_dir = args.outdir

    processed_dir.mkdir(exist_ok=True, parents=True)

    # Read data split
    split_path = Path(args.basedir, 'split_by_name.pt')
    data_split = torch.load(split_path)

    # There is no validation set, copy 300 training examples (the validation set
    # is not very important in this application)
    # Note: before we had a data leak but it should not matter too much as most
    # metrics monitored during training are independent of the pockets
    data_split['val'] = random.sample(data_split['train'], 300)

    n_train_before = len(data_split['train'])
    n_val_before = len(data_split['val'])
    n_test_before = len(data_split['test'])

    failed_save = []

    n_samples_after = {}
    for split in data_split.keys():
        lig_coords = []
        lig_one_hot = []
        lig_mask = []
        pocket_coords = []
        pocket_one_hot = []
        pocket_mask = []
        pdb_and_mol_ids = []
        count_protein = []
        count_ligand = []
        count_total = []
        count = 0
        pocket_residues_cutoff_percentage_list = []

        pdb_sdf_dir = processed_dir / split
        pdb_sdf_dir.mkdir(exist_ok=True)

        tic = time()
        num_failed = 0
        pbar = tqdm(data_split[split])
        pbar.set_description(f'#failed: {num_failed}')
        for pocket_fn, ligand_fn in pbar:

            sdffile = datadir / f'{ligand_fn}'
            pdbfile = datadir / f'{pocket_fn}'

            try:
                struct_copy = PDBParser(QUIET=True).get_structure('', pdbfile)
            except:
                num_failed += 1
                failed_save.append((pocket_fn, ligand_fn))
                print(failed_save[-1])
                pbar.set_description(f'#failed: {num_failed}')
                continue

            try:
                ligand_data, pocket_data, pocket_residues_cutoff_percentage = process_ligand_and_pocket(
                    pdbfile, sdffile,
                    atom_dict=atom_dict, dist_cutoff=args.dist_cutoff,
                    ca_only=args.ca_only, reorder=args.reorder)
                
                pocket_residues_cutoff_percentage_list.append(pocket_residues_cutoff_percentage)
                
            except (KeyError, AssertionError, FileNotFoundError, IndexError,
                    ValueError) as e:
                print(type(e).__name__, e, pocket_fn, ligand_fn)
                num_failed += 1
                pbar.set_description(f'#failed: {num_failed}')
                continue

            pdb_and_mol_ids.append(f"{pocket_fn}_{ligand_fn}")
            lig_coords.append(ligand_data['lig_coords'])
            lig_one_hot.append(ligand_data['lig_one_hot'])
            lig_mask.append(count * np.ones(len(ligand_data['lig_coords'])))
            pocket_coords.append(pocket_data['pocket_coords'])
            pocket_one_hot.append(pocket_data['pocket_one_hot'])
            pocket_mask.append(
                count * np.ones(len(pocket_data['pocket_coords'])))
            count_protein.append(pocket_data['pocket_coords'].shape[0])
            count_ligand.append(ligand_data['lig_coords'].shape[0])
            count_total.append(pocket_data['pocket_coords'].shape[0] +
                               ligand_data['lig_coords'].shape[0])
            count += 1

            if split in {'val', 'test'}:
                # Copy PDB file
                new_rec_name = Path(pdbfile).stem.replace('_', '-')
                pdb_file_out = Path(pdb_sdf_dir, f"{new_rec_name}.pdb")
                shutil.copy(pdbfile, pdb_file_out)

                # Copy SDF file
                new_lig_name = new_rec_name + '_' + Path(sdffile).stem.replace('_', '-')
                sdf_file_out = Path(pdb_sdf_dir, f'{new_lig_name}.sdf')
                shutil.copy(sdffile, sdf_file_out)

                # specify pocket residues
                with open(Path(pdb_sdf_dir, f'{new_lig_name}.txt'), 'w') as f:
                    f.write(' '.join(pocket_data['pocket_ids']))
        
        if split in {'train'}:
            import pickle
            with open(Path(pdb_sdf_dir, f'pocket_residues_cutoff_percentage.pickle'), 'wb') as file:
                pickle.dump(pocket_residues_cutoff_percentage_list, file)

        lig_coords = np.concatenate(lig_coords, axis=0)
        lig_one_hot = np.concatenate(lig_one_hot, axis=0)
        lig_mask = np.concatenate(lig_mask, axis=0)
        pocket_coords = np.concatenate(pocket_coords, axis=0)
        pocket_one_hot = np.concatenate(pocket_one_hot, axis=0)
        pocket_mask = np.concatenate(pocket_mask, axis=0)

        saveall(processed_dir / f'{split}.npz', pdb_and_mol_ids, lig_coords,
                lig_one_hot, lig_mask, pocket_coords,
                pocket_one_hot, pocket_mask)

        n_samples_after[split] = len(pdb_and_mol_ids)
        print(f"Processing {split} set took {(time() - tic) / 60.0:.2f} minutes")