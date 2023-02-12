import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd

import networkx as nx
from networkx.algorithms import tree
from math import pi
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from Bio.PDB import PDBParser
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# lig_elem_range = [
#        #B, C, N, O, F, Mg, Al, Si, P,  S, Cl, Sc, V, Fe, Cu, Zn, As, Se, Br, Y, Mo, Ru, Rh, Sb, I, W, Au
#         5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 21, 23, 26, 29, 30, 33, 34, 35, 39, 42, 44, 45, 51, 53, 74, 79
#     ]
# rec_elem_range = [
#        #C, N, O, Na, Mg,  P,  S, Cl,  K, Ca, Mn, Co, Cu, Zn, Se, Cd, I, Cs, Hg
#         6, 7, 8, 11, 12, 15, 16, 17, 19, 20, 25, 27, 29, 30, 34, 48, 53, 55, 80
#     ]


def collate_mols(mol_dicts):
    data_batch = {}

    for key in ['atom_type', 'position', 'rec_mask', 'cannot_contact', 'new_atom_type', 'new_dist', 'new_angle', 'new_torsion', 'cannot_focus']:
        data_batch[key] = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0)
    
    num_steps_list = torch.tensor([0]+[len(mol_dicts[i]['new_atom_type']) for i in range(len(mol_dicts)-1)])
    batch_idx_offsets = torch.cumsum(num_steps_list, dim=0)
    repeats = torch.tensor([len(mol_dict['batch']) for mol_dict in mol_dicts])
    batch_idx_repeated_offsets = torch.repeat_interleave(batch_idx_offsets, repeats)
    batch_offseted = torch.cat([mol_dict['batch'] for mol_dict in mol_dicts], dim=0) + batch_idx_repeated_offsets
    data_batch['batch'] = batch_offseted

    num_atoms_list = torch.tensor([0]+[len(mol_dicts[i]['atom_type']) for i in range(len(mol_dicts)-1)])
    atom_idx_offsets = torch.cumsum(num_atoms_list, dim=0)
    for key in ['focus', 'c1_focus', 'c2_c1_focus', 'contact_y_or_n']:
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts])
        atom_idx_repeated_offsets = torch.repeat_interleave(atom_idx_offsets, repeats)
        if key == 'contact_y_or_n':
            atom_offseted = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0) + atom_idx_repeated_offsets
        else:
            atom_offseted = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0) + atom_idx_repeated_offsets[:,None]
        data_batch[key] = atom_offseted

    return data_batch