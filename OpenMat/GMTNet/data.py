import pickle as pk
import spglib
import numpy as np
from tqdm import tqdm
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from e3nn import o3
from e3nn.io import CartesianTensor
import torch
import json
from pymatgen.io.vasp import Poscar
import pdb
from pymatgen.io.jarvis import JarvisAtomsAdaptor
from jarvis.core.atoms import Atoms
jarvis_adpt = JarvisAtomsAdaptor()

irreps_output = o3.Irreps('1x0e + 1x0o + 1x1e + 1x1o + 1x2e + 1x2o + 1x3e + 1x3o')
converter = CartesianTensor("ij")
E_matrix = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two 3D vectors."""
    return sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5

def are_vectors_equal(vec1, vec2, tolerance=1e-5):
    for v1, v2 in zip(vec1, vec2):
        diff = abs(v1 - v2)
        if not (diff < tolerance or abs(diff - 1) < tolerance):
            return False
    return True


def are_almost_equal(list1, list2, tolerance=1e-5):
    """Check if two lists of 3D vectors are equal within a given tolerance."""
    if len(list1) != len(list2):
        return False
    matched = []
    for v1 in list1:
        found_match = False
        for j, v2 in enumerate(list2):
            if j in matched:
                continue
            if euclidean_distance(v1, v2) <= tolerance:
                matched.append(j)
                found_match = True
                break
            if are_vectors_equal(v1, v2):
                matched.append(j)
                found_match = True
                break
        if not found_match:
            return False
    return True


def rm_duplicates(vectors):
    vecs = vectors.reshape(-1, 9)
    seen = set()
    duplicates = set()
    
    for i in range(vecs.shape[0]):
        vector = vecs[i]
        vt = tuple(vector)  # Convert to tuple for hashability
        if vt in seen:
            duplicates.add(vt)
        else:
            seen.add(vt)
    seen = list(seen)
    vector_list = np.array([list(vector) for vector in seen]).reshape(-1, 3, 3)

    return vector_list

def is_group(rots):
    length = rots.shape[0]
    tmp_list = [rots[idx] for idx in range(length)]
    for ix in range(length):
        for iy in range(length):
            tmp_mul = np.matmul(rots[ix], rots[iy])
            is_present = any(np.sum(abs(tmp_mul - v)) < 1e-5 for v in tmp_list)
            if not is_present:
                print(tmp_mul, "not in the list", tmp_list)
                return False
        tmp_inv = rots[ix].T
        is_present = any(np.sum(abs(tmp_inv - v)) < 1e-5 for v in tmp_list)
        if not is_present:
            print(tmp_inv, "not in the list", tmp_list)
            return False
    return True

def get_symmetry_dataset(structure, symprec=1e-5):
    """
    Get space group for a pymatgen Structure object.

    Parameters:
    - structure: pymatgen Structure object
    - symprec: float, the symmetry precision for determining the space group

    Returns:
    - symmetry: dict
    """
    # Convert pymatgen structure to tuple format suitable for spglib
    lattice = structure.lattice.matrix
    positions = structure.frac_coords
    atomic_numbers = structure.atomic_numbers

    cell = (lattice, positions, atomic_numbers)
    # Determine space group
    symmetry = spglib.get_symmetry_dataset(cell, symprec=symprec)
    return symmetry


def find_almost_equal_entries(matrix):
    """
    Find entries in each matrix that are almost equal to each other with less than absolute 0.01% difference.
    """
    h, w = matrix.shape
    matrix = matrix.view(-1)
    mask = torch.abs(matrix.unsqueeze(0) - matrix.unsqueeze(1)) < (0.0001 * torch.abs(matrix.unsqueeze(0) + matrix.unsqueeze(1)) / 2)
    return mask

def get_dataset(
    dataset_name="dielectric",
    symprec=1e-5, # Euclidean distance tolerance to determine the space group operations
    use_corrected_structure=False,
    load_preprocessed=False,
):
    if load_preprocessed:
        with open("yourpath/preprocessed_%s_dataset_elec.pkl"%dataset_name, 'rb') as f:
            dataset = pk.load(f)
            dat = []
            f_norm=[]
            
            for i in tqdm(range(len(dataset))):
                dataset[i]['reduce_rotations'] = None
                dataset[i]['wigner_D_per_atom'] = None
                dataset[i]['wigner_D_num'] = None
                dataset[i]['p_input'] = {}
                dataset[i]['p_input']['structure'] = dataset[i]['structure']
                dataset[i]['p_input']['equivalent_atoms'] = dataset[i]['equivalent_atoms']
                dataset[i]['matrix_equal'] = find_almost_equal_entries(dataset[i]['ideal_matrix'])
                f_norm.append((torch.tensor(dataset[i]['dielectric']) ** 2).sum() ** 0.5) 
            print("dataset fnorm mean", torch.tensor(f_norm).mean(), "std", torch.tensor(f_norm).std())


            cubic_cnt = 0
            hexa_cnt = 0
            tetr_cnt = 0
            orth_cnt = 0
            mono_cnt = 0
            tric_cnt = 0
            for i in tqdm(range(len(dataset))):
                space_g = dataset[i]['sym_dataset']['number']
                if space_g >= 195:
                    cubic_cnt += 1
                elif space_g >= 143:
                    hexa_cnt += 1
                elif space_g >= 75:
                    tetr_cnt += 1
                elif space_g >= 16:
                    orth_cnt += 1
                elif space_g >= 3:
                    mono_cnt += 1
                else:
                    tric_cnt += 1
            print("cubic_cnt ", cubic_cnt, "hexa_cnt ", hexa_cnt, "tetr_cnt ", tetr_cnt, "orth_cnt ", orth_cnt, "mono_cnt ", mono_cnt, "tric_cnt ", tric_cnt)
            # dataset = dat
        return dataset
    # load higher tensor order property dataset
    with open("yourpath/jarvis_diele_piezo.pkl", 'rb') as f:
        dataset = pk.load(f)

    # Screen process
    print("Screening and filtering process: filter out too large entries")
    dat = []
    data_cnt = 0
    for i in tqdm(range(len(dataset))):
        if dataset[i]['dielectric']:
            dielectric = torch.tensor(dataset[i]['dielectric'])
            if abs(dielectric).max() < 100:
                data_cnt += 1
                dat.append(dataset[i])
    print(data_cnt)

    dataset = dat

    # store space group operations for every crystal in the dataset
    print("Beginning preprocess: Step 1 - determine space group operations...")
    rotation_list = []
    trans_list = []
    ideal_matrixs = []
    dat = []
    cnt = 0
    error = 0
    for i in tqdm(range(len(dataset))):
        structure = jarvis_adpt.get_structure(Atoms.from_dict(dataset[i]['atoms']))
        dataset[i]['structure'] = structure
        sym_dataset = get_symmetry_dataset(structure, symprec)
        # transform the structure accordingly to make space group operations valid, this will erase the distortions in the structure
        dataset[i]['equivalent_atoms'] = sym_dataset['equivalent_atoms']
        dataset[i]['sym_dataset'] = sym_dataset
        dataset[i]['corrected_structure'] = Structure(lattice=sym_dataset['std_lattice'], species=sym_dataset['std_types'], coords=sym_dataset['std_positions'])
        dataset[i]['corrected_rotation'] = sym_dataset['std_rotation_matrix']
        
        if use_corrected_structure:
            # remove the rotation transformation
            Rot = np.array(sym_dataset['std_rotation_matrix'])
            target_tmp = np.array(dataset[i][dataset_name])
            dataset[i][dataset_name] = np.dot(Rot, np.dot(target_tmp, Rot.T))
            sym_dataset = get_symmetry_dataset(dataset[i]['corrected_structure'], symprec)
            dataset[i]['equivalent_atoms'] = sym_dataset['equivalent_atoms']
            dataset[i]['sym_dataset'] = sym_dataset
            dataset[i]['structure'] = dataset[i]['corrected_structure']

        # check the transformed structure - labels satisfy symmetry or not
        mask = (torch.arange(32)+10.)
        mask[8:] *= 100
        rots = np.array(sym_dataset['rotations'])
        rots = rm_duplicates(rots)
        Lat = dataset[i]['structure'].lattice.matrix.T
        L_inv = np.linalg.inv(Lat)
        D_x = torch.zeros(32, 32)
        tmp_rot = np.matmul(Lat, np.matmul(rots, L_inv))
        assert is_group(tmp_rot), ("Found non_group rots", tmp_rot)
        D_tmp = irreps_output.D_from_matrix(torch.Tensor(tmp_rot))
        assert (((abs(D_tmp[:,5:8,5:8] - tmp_rot)).sum(dim=-1).sum(dim=-1) > 1e-2).sum() < 1e-5), (abs(D_tmp[:,5:8,5:8] - tmp_rot).sum(dim=-1).sum(dim=-1))
        D_x = D_tmp.sum(dim=0)
        feature_mask = torch.matmul(D_x, mask)
        mask_total = feature_mask[[0, 2, 3, 4, 8, 9, 10, 11, 12]]
        ideal_matrix = converter.to_cartesian(mask_total)
        # print(sym_dataset['number'], ideal_matrix)
        ideal_matrixs.append(ideal_matrix)
        dataset[i]['ideal_matrix'] = ideal_matrix
        D_x = D_x / D_tmp.shape[0]
        zero_mask = (D_x > 1e-5).float()
        D_x *= zero_mask
        dataset[i]['feature_mask'] = D_x
        dataset[i]['feature_mask_ori'] = feature_mask
        dataset[i]['rot_list'] = tmp_rot
    
    error_cnt = 0
    for i in tqdm(range(len(dataset))):
        # item 1: zero investigation
        ideal_mask = (abs(ideal_matrixs[i]) < 1.).float()
        dielectric = torch.tensor(dataset[i]['dielectric'])
        if (abs(dielectric * ideal_mask)).sum() > 1e-4:
            error_cnt += 1
            print(dielectric, ideal_mask)

    print("zero investigation", error_cnt)

    for i in tqdm(range(len(dataset))):
        dataset[i]['reduce_rotations'] = None
        dataset[i]['wigner_D_per_atom'] = None
        dataset[i]['wigner_D_num'] = None
        dataset[i]['p_input'] = {}
        dataset[i]['p_input']['structure'] = dataset[i]['structure']
        dataset[i]['p_input']['equivalent_atoms'] = dataset[i]['equivalent_atoms']
        dataset[i]['matrix_equal'] = find_almost_equal_entries(dataset[i]['ideal_matrix'])

    with open("yourpath/preprocessed_%s_dataset_elec.pkl"%dataset_name, 'wb') as f:
        pk.dump(dataset, f)

    return dataset


