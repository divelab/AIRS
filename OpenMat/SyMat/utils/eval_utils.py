import numpy as np
import itertools
import smact
import torch
from tqdm import tqdm
from smact.screening import pauling_test
from scipy.spatial.distance import pdist, cdist
from scipy.stats import wasserstein_distance
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty
from collections import Counter
from .constants import CompScalerMeans, CompScalerStds, chemical_symbols
from .data_utils import StandardScaler
from .mat_utils import frac_to_cart_coords, distance_matrix_pbc

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

COV_Cutoffs = {
    'mp_20': {'struc': 0.6, 'comp': 12.},
    'carbon_24': {'struc': 1.0, 'comp': 4.},
    'perov_5': {'struc': 0.8, 'comp': 6},
}

def smact_validity(atom_types_list,
                   use_pauling_test=True,
                   include_alloys=True):
    is_valid, num_valid = [], 0
    for atom_types in tqdm(atom_types_list):
        elem_counter = Counter(atom_types)
        composition = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
        comp, count = list(zip(*composition))
        count = np.array(count)
        count = count / np.gcd.reduce(count)
        count = tuple(count.astype('int').tolist())

        elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
        space = smact.element_dictionary(elem_symbols)
        smact_elems = [e[1] for e in space.items()]
        electronegs = [e.pauling_eneg for e in smact_elems]
        ox_combos = [e.oxidation_states for e in smact_elems]
        if len(set(elem_symbols)) == 1:
            is_valid.append(True)
            num_valid += 1
            continue
        if include_alloys:
            is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
            if all(is_metal_list):
                is_valid.append(True)
                num_valid += 1
                continue

        threshold = np.max(count)
        compositions = []
        for ox_states in itertools.product(*ox_combos):
            stoichs = [(c,) for c in count]
            # Test for charge balance
            cn_e, cn_r = smact.neutral_ratios(
                ox_states, stoichs=stoichs, threshold=threshold)
            # Electronegativity test
            if cn_e:
                if use_pauling_test:
                    try:
                        electroneg_OK = pauling_test(ox_states, electronegs)
                    except TypeError:
                        # if no electronegativity data, assume it is okay
                        electroneg_OK = True
                else:
                    electroneg_OK = True
                if electroneg_OK:
                    for ratio in cn_r:
                        compositions.append(
                            tuple([elem_symbols, ox_states, ratio]))
        compositions = [(i[0], i[2]) for i in compositions]
        compositions = list(set(compositions))
        if len(compositions) > 0:
            is_valid.append(True)
            num_valid += 1
        else:
            is_valid.append(False)
    
    return is_valid, num_valid / len(atom_types_list)


def compute_elem_type_num_wdist(gen_atom_types_list, gt_atom_types_list):
    gt_elem_type_nums = []
    for gt_atom_types in gt_atom_types_list:
        gt_elem_type_nums.append(len(set(gt_atom_types)))
    
    gen_elem_type_nums = []
    for gen_atom_types in gen_atom_types_list:
        gen_elem_type_nums.append(len(set(gen_atom_types)))
    
    return wasserstein_distance(gen_elem_type_nums, gt_elem_type_nums)


def get_comp_fp(atom_types_list):
    comp_fps = []
    for atom_types in atom_types_list:
        elem_counter = Counter(atom_types)
        comp = Composition(elem_counter)
        try:
            comp_fp = CompFP.featurize(comp)
        except:
            comp_fp = None
        comp_fps.append(comp_fp)
    return comp_fps


def structure_validity(atom_types_list, lengths_list, angles_list, frac_coords_list, structure_list, cutoff=0.5):
    is_valid, num_valid = [], 0
    
    for i in tqdm(range(len(atom_types_list))):
        if structure_list[i] is None:
            is_valid.append(False)
            continue

        length = torch.from_numpy(lengths_list[i]).view(1,-1)
        angle = torch.from_numpy(angles_list[i]).view(1,-1)
        frac_coords = torch.from_numpy(frac_coords_list[i])
        num_atom = len(atom_types_list[i])
        cart_coord = frac_to_cart_coords(frac_coords, length, angle, torch.tensor([num_atom]))
        dist_mat = distance_matrix_pbc(cart_coord, length, angle)
        
        dist_mat += torch.diag(torch.ones([num_atom]) * cutoff)
        min_dist = dist_mat.min()
        if min_dist >= cutoff:
            is_valid.append(True)
            num_valid += 1
        else:
            is_valid.append(False)
    
    return is_valid, num_valid / len(is_valid)


def get_structure(atom_types_list, lengths_list, angles_list, frac_coords_list):
    structure_list = []
    for i in range(len(atom_types_list)):
        try:
            atom_types, lengths, angles, frac_coords = atom_types_list[i], lengths_list[i], angles_list[i], frac_coords_list[i]
            structure = Structure(lattice=Lattice.from_parameters(*(lengths.tolist() + angles.tolist())), species=atom_types, coords=frac_coords, coords_are_cartesian=False)
            if structure.volume < 0.1:
                structure = None
        except:
            structure = None
        
        structure_list.append(structure)
    
    return structure_list


def compute_density_wdist(gen_structure_list, gt_structure_list):
    gen_densities = [gen_structure.density for gen_structure in gen_structure_list if gen_structure is not None]
    gt_densities = [gt_structure.density for gt_structure in gt_structure_list if gt_structure is not None]
    density_wdist = wasserstein_distance(gen_densities, gt_densities)
    return density_wdist


def get_structure_fp(structure_list):
    structure_fps = []
    for structure in structure_list:
        if structure is None:
            struct_fp = None
            structure_fps.append(struct_fp)
            continue

        try:
            site_fps = [CrystalNNFP.featurize(structure, i) for i in range(len(structure))]
            struct_fp = np.array(site_fps).mean(axis=0)
        except:
            struct_fp = None
        structure_fps.append(struct_fp)
    return structure_fps


def get_fp_pdist(fp_array):
    if isinstance(fp_array, list):
        fp_array = np.array(fp_array)
    fp_pdists = pdist(fp_array)
    return fp_pdists.mean()


def filter_fps(struc_fps, comp_fps):
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps


def compute_cov(gen_atom_types_list, gt_atom_types_list, gen_structure_list, gt_structure_list, data_name):
    assert len(gen_atom_types_list) == len(gen_structure_list)
    assert len(gt_atom_types_list) == len(gt_structure_list)

    gen_comp_fps = get_comp_fp(gen_atom_types_list)
    gt_comp_fps = get_comp_fp(gt_atom_types_list)
    gen_structure_fps = get_structure_fp(gen_structure_list)
    gt_structure_fps = get_structure_fp(gt_structure_list)
    num_gen_crystals = len(gen_comp_fps)

    filtered_gen_comp_fps, filtered_gen_structure_fps = [], []
    for comp_fp, structure_fp in zip(gen_comp_fps, gen_structure_fps):
        if comp_fp is not None and structure_fp is not None:
            filtered_gen_comp_fps.append(comp_fp)
            filtered_gen_structure_fps.append(structure_fp)
    print(len(filtered_gen_comp_fps))
    CompScaler = StandardScaler(means=np.array(CompScalerMeans), stds=np.array(CompScalerStds), replace_nan_token=0.)
    gen_comp_fps = CompScaler.transform(filtered_gen_comp_fps)
    gt_comp_fps = CompScaler.transform(gt_comp_fps)
    gen_structure_fps = np.array(filtered_gen_structure_fps)
    gt_structure_fps = np.array(gt_structure_fps)

    comp_pdist = cdist(gen_comp_fps, gt_comp_fps)
    structure_pdist = cdist(gen_structure_fps, gt_structure_fps)

    structure_recall_dist = structure_pdist.min(axis=0)
    structure_precision_dist = structure_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    comp_cutoff = COV_Cutoffs[data_name]['comp']
    structure_cutoff = COV_Cutoffs[data_name]['struc']
    cov_recall = np.mean(np.logical_and(
        structure_recall_dist <= structure_cutoff,
        comp_recall_dist <= comp_cutoff))
    cov_precision = np.sum(np.logical_and(
        structure_precision_dist <= structure_cutoff,
        comp_precision_dist <= comp_cutoff)) / num_gen_crystals
    
    return cov_recall, cov_precision