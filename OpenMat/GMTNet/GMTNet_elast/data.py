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
from multiprocessing import Pool
jarvis_adpt = JarvisAtomsAdaptor()

irreps_output = o3.Irreps('2x0e + 2x0o + 2x1e + 2x1o + 2x2e + 2x2o + 2x3e + 2x3o + 1x4e')
converter = CartesianTensor("ijkl=ijlk=jikl=klij") # 2x0e+2x2e+1x4e
E_matrix = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
symprec=1e-5
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
        vt = tuple(vector) 
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

def contract_tensor(d34):
    """
    Contract a 3x3x3 tensor into a 6x6 matrix representation.
    """
    d66 = np.zeros((6, 6))
    mapping = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
    for i, (m, n) in enumerate(mapping):
        for j, (k, l) in enumerate(mapping):
            d66[i, j] = d34[m, n, k, l]
    return d66

def find_almost_equal_entries(matrix):
    """
    Find entries in each matrix that are almost equal to each other with less than absolute 0.01% difference.
    """
    h, w = matrix.shape
    matrix = matrix.view(-1)
    mask = torch.abs(matrix.unsqueeze(0) - matrix.unsqueeze(1)) < (0.0001 * torch.abs(matrix.unsqueeze(0) + matrix.unsqueeze(1)) / 2)
    mask2 = torch.abs(matrix.unsqueeze(0) + matrix.unsqueeze(1)) < (0.0001 * torch.abs(matrix.unsqueeze(0) - matrix.unsqueeze(1)) / 2)
    return torch.stack([mask, mask2])

def process_chunk(dataset):
    for i in tqdm(range(len(dataset))):
        structure = jarvis_adpt.get_structure(Atoms.from_dict(dataset[i]['atoms']))
        dataset[i]['structure'] = structure
        sym_dataset = get_symmetry_dataset(structure, symprec)
        # transform the structure accordingly to make space group operations valid, this will erase the distortions in the structure
        dataset[i]['equivalent_atoms'] = sym_dataset['equivalent_atoms']
        dataset[i]['sym_dataset'] = sym_dataset
        dataset[i]['corrected_structure'] = Structure(lattice=sym_dataset['std_lattice'], species=sym_dataset['std_types'], coords=sym_dataset['std_positions'])
        dataset[i]['corrected_rotation'] = sym_dataset['std_rotation_matrix']

        # check the transformed structure - labels satisfy symmetry or not
        mask = (torch.arange(73)+10.)
        mask[16:] *= 100
        rots = np.array(sym_dataset['rotations'])
        rots = rm_duplicates(rots)
        Lat = dataset[i]['structure'].lattice.matrix.T
        L_inv = np.linalg.inv(Lat)
        D_x = torch.zeros(73, 73)
        tmp_rot = np.matmul(Lat, np.matmul(rots, L_inv))
        assert is_group(tmp_rot), ("Found non_group rots", tmp_rot)
        D_tmp = irreps_output.D_from_matrix(torch.Tensor(tmp_rot))
        assert (((abs(D_tmp[:,10:13,10:13] - tmp_rot)).sum(dim=-1).sum(dim=-1) > 1e-2).sum() < 1e-5)
        D_x = D_tmp.sum(dim=0)
        feature_mask = torch.matmul(D_x, mask)
        # 2x0e+2x2e+1x4e '2x0e + 2x0o + 2x1e + 2x1o + 2x2e + 2x2o + 2x3e + 2x3o + 2x4e + 2x4o'
        mask_total = feature_mask[[0, 1, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 64, 65, 66, 67, 68, 69, 70, 71, 72]]
        ideal_matrix = converter.to_cartesian(mask_total)
        ideal_matrix = contract_tensor(ideal_matrix)
        dataset[i]['ideal_matrix'] = ideal_matrix
        D_x = D_x / D_tmp.shape[0]
        zero_mask = (D_x > 1e-5).float()
        D_x *= zero_mask
        dataset[i]['feature_mask'] = D_x
        dataset[i]['feature_mask_ori'] = feature_mask
    return dataset

def get_dataset(
    dataset_name="elastic",
    symprec=1e-5, # Euclidean distance tolerance to determine the space group operations
    use_corrected_structure=False,
    load_preprocessed=False,
    chunk=None,
):
    if load_preprocessed:
        with open("/yourpath/all_elastic_preprocessed.pkl", 'rb') as f:
            dataset = pk.load(f)
            dat = []
            f_norm = []
            for i in tqdm(range(len(dataset))):
                dataset[i]['reduce_rotations'] = None
                dataset[i]['wigner_D_per_atom'] = None
                dataset[i]['wigner_D_num'] = None
                dataset[i]['p_input'] = {}
                dataset[i]['p_input']['structure'] = dataset[i]['structure']
                dataset[i]['p_input']['equivalent_atoms'] = dataset[i]['equivalent_atoms']
                dataset[i]['matrix_equal'] = find_almost_equal_entries(torch.tensor(dataset[i]['ideal_matrix']))
                ideal_mask = torch.tensor(abs(dataset[i]['ideal_matrix']) < 1.).float()
                elastic = torch.tensor(dataset[i]['elastic'])
                if (abs(elastic * ideal_mask)).max() < 1e-4:
                    dataset[i]['elastic'] = elastic * torch.tensor(abs(dataset[i]['ideal_matrix']) > 1.).float()
                    dat.append(dataset[i])
                    f_norm.append(((dataset[i]['elastic']) ** 2).sum() ** 0.5)
            dataset = dat
            print("dataset f_norm mean", torch.tensor(f_norm).mean(), "std", torch.tensor(f_norm).std())
        return dataset
    # load higher tensor order property dataset
    with open("/yourpath/jarvis_elastic.pkl", 'rb') as f:
        dataset = pk.load(f)

    # Screen process
    print("Screening and filtering process: filter out too large entries")
    dat = []
    data_cnt = 0
    for i in tqdm(range(len(dataset))):
        if dataset[i]['elastic_total_kbar']:
            elastic = torch.tensor(dataset[i]['elastic_total_kbar']) / 10.
            if abs(elastic).max() < 1500.:
                data_cnt += 1
                dat.append(dataset[i])
    print(data_cnt)

    dataset = dat

    # store space group operations for every crystal in the dataset
    print("Beginning preprocess: Step 1 - determine space group operations...")
    ideal_matrixs = []
    dat = []
    cnt = 0
    error = 0
    def chunkify(lst, n):
        # This function splits the list into n chunks
        return [lst[i::n] for i in range(n)]
    
    num_processes = 8
    dataset_chunks = chunkify(dataset, num_processes)

    # with Pool(num_processes) as p:
    #     chunk_results = list(tqdm(p.imap(process_chunk, dataset_chunks), total=num_processes))
    chunk_results = process_chunk(dataset_chunks[chunk])

    # flat_results = []
    # for jj in range(len(chunk_results)):
    #     flat_results += chunk_results[jj]

    dataset = chunk_results
    
    error_cnt = 0
    for i in tqdm(range(len(dataset))):
        # item 1: zero investigation
        ideal_mask = torch.tensor(abs(dataset[i]['ideal_matrix']) < 1.).float()
        elastic = torch.tensor(dataset[i]['elastic_total_kbar']) / 10.
        if (abs(elastic * ideal_mask)).max() > 2e-4:
            error_cnt += 1
            print(elastic, ideal_mask)

    print("zero investigation", error_cnt)

    for i in tqdm(range(len(dataset))):
        dataset[i]['reduce_rotations'] = None
        dataset[i]['wigner_D_per_atom'] = None
        dataset[i]['wigner_D_num'] = None
        dataset[i]['p_input'] = {}
        dataset[i]['elastic'] = np.array(dataset[i]['elastic_total_kbar']) / 10.
        dataset[i]['p_input']['structure'] = dataset[i]['structure']
        dataset[i]['p_input']['equivalent_atoms'] = dataset[i]['equivalent_atoms']
        dataset[i]['matrix_equal'] = find_almost_equal_entries(torch.tensor(dataset[i]['ideal_matrix']))

    with open("/yourpath/preprocessed_%s_dataset_%d.pkl"%(dataset_name, chunk), 'wb') as f:
        pk.dump(dataset, f)

    return dataset

if __name__ == "__main__":
    get_dataset(dataset_name="elastic",use_corrected_structure=False,load_preprocessed=False, chunk=7)