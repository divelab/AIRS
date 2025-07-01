from tqdm import tqdm
import random
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolTransforms import GetBondLength, GetAngleDeg, GetDihedralDeg
from .mmd import compute_mmd
# from mmd import compute_mmd
import torch
import time
import numpy as np
import os
import pickle


# Bond distance
def get_bond_symbol(bond_n):
    """
    Return the symbol representation of a bond
    """
    a0 = bond_n.GetBeginAtom().GetSymbol()
    a1 = bond_n.GetEndAtom().GetSymbol()
    b = str(int(bond_n.GetBondType()))  # single:1, double:2, triple:3, aromatic: 12
    return ''.join([a0, b, a1]), ''.join([a1, b, a0])


def cal_bond_distance(mol_list, top_bond_syms):
    """
    Return bond distance statistic dict
    """
    bond_distance_dict = dict()
    for bond_sym in top_bond_syms:
        bond_distance_dict[bond_sym] = []
    for mol in mol_list:
        conf = mol.GetConformer()
        for bond in mol.GetBonds():
            atom_id0, atom_id1 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt, reverse_bt = get_bond_symbol(bond)
            # reverse_bt = bt[::-1]
            if bt in top_bond_syms:
                bond_distance_dict[bt].append(GetBondLength(conf, atom_id0, atom_id1))
            elif reverse_bt in top_bond_syms:
                bond_distance_dict[reverse_bt].append(GetBondLength(conf, atom_id1, atom_id0))
    return bond_distance_dict


# Bond Angle
def get_bond_pairs(mol):
    """Get all the bond pairs in a molecule"""
    valid_bond_pairs = []
    for idx_bond, bond in enumerate(mol.GetBonds()):
        idx_end_atom = bond.GetEndAtomIdx()
        end_atom = mol.GetAtomWithIdx(idx_end_atom)
        end_bonds = end_atom.GetBonds()
        for end_bond in end_bonds:
            if end_bond.GetIdx() == idx_bond:
                continue
            else:
                valid_bond_pairs.append([bond, end_bond])
                # bond_idx.append((bond.GetIdx(), end_bond.GetIdx()))
    return valid_bond_pairs


def get_bond_pair_symbol(bond_pairs):
    """Return the symbol representation of a bond angle"""
    atom0_0 = bond_pairs[0].GetBeginAtomIdx()
    atom0_1 = bond_pairs[0].GetEndAtomIdx()
    atom0_0_sym = bond_pairs[0].GetBeginAtom().GetSymbol()
    atom0_1_sym = bond_pairs[0].GetEndAtom().GetSymbol()
    bond_left = str(int(bond_pairs[0].GetBondType()))

    atom1_0 = bond_pairs[1].GetBeginAtomIdx()
    atom1_1 = bond_pairs[1].GetEndAtomIdx()
    atom1_0_sym = bond_pairs[1].GetBeginAtom().GetSymbol()
    atom1_1_sym = bond_pairs[1].GetEndAtom().GetSymbol()
    bond_right = str(int(bond_pairs[1].GetBondType()))

    if atom0_0 == atom1_0:
        sym = ''.join([atom0_1_sym, bond_left, atom0_0_sym]) + '-' + ''.join([atom1_0_sym, bond_right, atom1_1_sym])
        ijk = (atom0_1, atom0_0, atom1_1)
    elif atom0_0 == atom1_1:
        sym = ''.join([atom0_1_sym, bond_left, atom0_0_sym]) + '-' + ''.join([atom1_1_sym, bond_right, atom1_0_sym])
        ijk = (atom0_1, atom0_0, atom1_0)
    elif atom0_1 == atom1_0:
        sym = ''.join([atom0_0_sym, bond_left, atom0_1_sym]) + '-' + ''.join([atom1_0_sym, bond_right, atom1_1_sym])
        ijk = (atom0_0, atom0_1, atom1_1)
    elif atom0_1 == atom1_1:
        sym = ''.join([atom0_0_sym, bond_left, atom0_1_sym]) + '-' + ''.join([atom1_1_sym, bond_right, atom1_0_sym])
        ijk = (atom0_0, atom0_1, atom1_0)
    else:
        raise ValueError("Bond pair error.")

    return sym, ijk

def cal_bond_angle(mol_list, top_angle_syms):
    """
    Return bond angle statistic dict
    """
    bond_angle_dict = dict()
    for angle_sym in top_angle_syms:
        bond_angle_dict[angle_sym] = []
    for mol in mol_list:
        conf = mol.GetConformer()
        bond_pairs = get_bond_pairs(mol)
        for bond_pair in bond_pairs:
            angle_sym, ijk = get_bond_pair_symbol(bond_pair)
            i, j, k = ijk
            # reverse_angle_sym = angle_sym[::-1]
            reverse_angle_sym, _ = get_bond_pair_symbol(bond_pair[::-1])
            if angle_sym in top_angle_syms:
                bond_angle_dict[angle_sym].append(GetAngleDeg(conf, i, j, k))
            elif reverse_angle_sym in top_angle_syms:
                bond_angle_dict[reverse_angle_sym].append(GetAngleDeg(conf, k, j, i))
    return bond_angle_dict


# Dihedral Angle
def get_triple_bonds(mol):
    """Get all the bond triples in a molecule"""
    valid_triple_bonds = []
    for idx_bond, bond in enumerate(mol.GetBonds()):
        idx_begin_atom = bond.GetBeginAtomIdx()
        idx_end_atom = bond.GetEndAtomIdx()
        begin_atom = mol.GetAtomWithIdx(idx_begin_atom)
        end_atom = mol.GetAtomWithIdx(idx_end_atom)
        begin_bonds = begin_atom.GetBonds()
        valid_left_bonds = []
        for begin_bond in begin_bonds:
            if begin_bond.GetIdx() == idx_bond:
                continue
            else:
                valid_left_bonds.append(begin_bond)
        if len(valid_left_bonds) == 0:
            continue

        end_bonds = end_atom.GetBonds()
        for end_bond in end_bonds:
            if end_bond.GetIdx() == idx_bond:
                continue
            else:
                for left_bond in valid_left_bonds:
                    valid_triple_bonds.append([left_bond, bond, end_bond])

    return valid_triple_bonds


def get_triple_bond_symbol(triple_bonds):
    """Return the symbol representation of a dihedral angle"""
    atom0_0 = triple_bonds[0].GetBeginAtomIdx()
    atom0_1 = triple_bonds[0].GetEndAtomIdx()
    atom0_0_sym = triple_bonds[0].GetBeginAtom().GetSymbol()
    atom0_1_sym = triple_bonds[0].GetEndAtom().GetSymbol()
    bond_left = str(int(triple_bonds[0].GetBondType()))

    atom1_0 = triple_bonds[1].GetBeginAtomIdx()
    atom1_1 = triple_bonds[1].GetEndAtomIdx()
    atom1_0_sym = triple_bonds[1].GetBeginAtom().GetSymbol()
    atom1_1_sym = triple_bonds[1].GetEndAtom().GetSymbol()
    bond_mid = str(int(triple_bonds[1].GetBondType()))

    atom2_0 = triple_bonds[2].GetBeginAtomIdx()
    atom2_1 = triple_bonds[2].GetEndAtomIdx()
    atom2_0_sym = triple_bonds[2].GetBeginAtom().GetSymbol()
    atom2_1_sym = triple_bonds[2].GetEndAtom().GetSymbol()
    bond_right = str(int(triple_bonds[2].GetBondType()))

    ijkl = []
    if atom0_0 == atom1_0:
        sym = ''.join([atom0_1_sym, bond_left, atom0_0_sym]) + '-' + ''.join([atom1_0_sym, bond_mid, atom1_1_sym])
        last_id = atom1_1
        ijkl += [atom0_1, atom0_0, atom1_1]
    elif atom0_0 == atom1_1:
        sym = ''.join([atom0_1_sym, bond_left, atom0_0_sym]) + '-' + ''.join([atom1_1_sym, bond_mid, atom1_0_sym])
        last_id = atom1_0
        ijkl += [atom0_1, atom0_0, atom1_0]
    elif atom0_1 == atom1_0:
        sym = ''.join([atom0_0_sym, bond_left, atom0_1_sym]) + '-' + ''.join([atom1_0_sym, bond_mid, atom1_1_sym])
        last_id = atom1_1
        ijkl += [atom0_0, atom0_1, atom1_1]
    elif atom0_1 == atom1_1:
        sym = ''.join([atom0_0_sym, bond_left, atom0_1_sym]) + '-' + ''.join([atom1_1_sym, bond_mid, atom1_0_sym])
        last_id = atom1_0
        ijkl += [atom0_0, atom0_1, atom1_0]
    else:
        raise ValueError("Left and middle bonds error.")

    if atom2_0 == last_id:
        sym = sym + '-' + ''.join([atom2_0_sym, bond_right, atom2_1_sym])
        ijkl.append(atom2_1)
    elif atom2_1 == last_id:
        sym = sym + '-' + ''.join([atom2_1_sym, bond_right, atom2_0_sym])
        ijkl.append(atom2_0)
    else:
        raise ValueError("Right bond error.")

    return sym, ijkl


def cal_dihedral_angle(mol_list, top_dihedral_syms):
    """
    Return dihedral angle statistic dict
    """
    dihedral_angle_dict = dict()
    for dihedral_sym in top_dihedral_syms:
        dihedral_angle_dict[dihedral_sym] = []
    for mol in mol_list:
        conf = mol.GetConformer()
        triple_bonds = get_triple_bonds(mol)
        for triple_bond in triple_bonds:
            dihedral_sym, ijkl = get_triple_bond_symbol(triple_bond)
            i, j, k, l = ijkl
            # reverse_dihedral_sym = dihedral_sym[::-1]
            reverse_dihedral_sym, _ = get_triple_bond_symbol(triple_bond[::-1])
            if dihedral_sym in dihedral_angle_dict:
                dihedral_angle_dict[dihedral_sym].append(GetDihedralDeg(conf, i, j, k, l))
            elif reverse_dihedral_sym in dihedral_angle_dict:
                dihedral_angle_dict[reverse_dihedral_sym].append(GetDihedralDeg(conf, l, k, j, i))
    return dihedral_angle_dict


def load_target_geometry(mols, info, dataset_root):
    """Save and load target geometry statistic"""
    file_path = os.path.join(dataset_root, 'target_geometry_stat.pk')
    # if file exist, load pickle
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            geo_stat = pickle.load(f)
        return geo_stat
    # if file not exist, calculate statistic
    bond_dict = cal_bond_distance(mols, info['top_bond_sym'])
    angle_dict = cal_bond_angle(mols, info['top_angle_sym'])
    dihedral_dict = cal_dihedral_angle(mols, info['top_dihedral_sym'])

    geo_stat = {**bond_dict, **angle_dict, **dihedral_dict}
    with open(file_path, 'wb') as f:
        pickle.dump(geo_stat, f)
    return geo_stat


# pipeline for dataset bound
def report_dataset_bound(trains, tests, cal_fn, top_geometry):
    res_dict = dict()
    for sym in top_geometry:
        res_dict[sym] = []

    target_geo = cal_fn(tests, top_geometry)
    for i in range(5):
        print(i)
        random_trains = random.sample(trains, 10000)
        train_geo = cal_fn(random_trains, top_geometry)
        for sym in top_geometry:
            time1 = time.time()
            tar = target_geo[sym]
            des = train_geo[sym]
            if len(tar) > 20000:
                tar = random.sample(tar, 20000)
            if len(des) > 20000:
                des = random.sample(des, 20000)
            res_dict[sym].append(compute_mmd(
                torch.tensor(des), torch.tensor(tar), batch_size=10000))
            time2 = time.time()
            if i == 0:
                print(sym, len(train_geo[sym]), len(target_geo[sym]), 'time:', time2- time1)

    for sym in top_geometry:
        print(sym, np.mean(res_dict[sym]), np.std(res_dict[sym]))


def compute_geo_mmd(gen_mols, tar_geo, cal_fn, top_geo_syms, mean_name):
    res_dict = dict()
    gen_geo = cal_fn(gen_mols, top_geo_syms)
    for geo_sym in top_geo_syms:
        tar = tar_geo[geo_sym]
        gen = gen_geo[geo_sym]
        if len(gen) == 0:
            res_dict[geo_sym] = float('nan')
            continue
        if len(tar) > 20000:
            tar = random.sample(tar, 20000)
        if len(gen) > 20000:
            gen = random.sample(gen, 20000)
        res_dict[geo_sym] = compute_mmd(torch.tensor(gen), torch.tensor(tar), batch_size=10000)

    res_dict[mean_name] = np.nanmean(list(res_dict.values()))
    return res_dict


# get sub geometry eval func
def get_sub_geometry_metric(test_mols, dataset_info, root_path):
    tar_geo_stat = load_target_geometry(test_mols, dataset_info, root_path)

    def sub_geometry_metric(gen_mols):
        bond_length_dict = compute_geo_mmd(gen_mols, tar_geo_stat, cal_bond_distance,
                                           dataset_info['top_bond_sym'], mean_name='bond_length_mean')
        bond_angle_dict = compute_geo_mmd(gen_mols, tar_geo_stat, cal_bond_angle,
                                          dataset_info['top_angle_sym'], mean_name='bond_angle_mean')
        dihedral_angle_dict = compute_geo_mmd(gen_mols, tar_geo_stat, cal_dihedral_angle,
                                              dataset_info['top_dihedral_sym'], mean_name='dihedral_angle_mean')
        metric = {**bond_length_dict, **bond_angle_dict, **dihedral_angle_dict}

        return metric

    return sub_geometry_metric