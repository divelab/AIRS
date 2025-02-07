import sys
sys.path.append(".")
import pickle as pk
import gzip
import tarfile
from collections import Counter
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
import numpy as np
import itertools
import tarfile
from tqdm import tqdm
import smact
from smact.screening import pauling_test
import os

import warnings
warnings.filterwarnings("ignore")

generated_cifs = {}

# from https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/eval_utils.py
def smact_validity(comp, count, use_pauling_test=True, include_alloys=True):
    elem_symbols = tuple(comp)
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True
    threshold = np.max(count)
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
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
                return True
    return False


# from https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/eval_utils.py
def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True

def extract_cif_id(filepath):
    """
    Parses a filename assumed to be in the format "id__n.cif",
    returning the "id".

    :param filepath: a filename assumed to be in the format "id__n.cif"
    :return: the extracted values of `id`
    """
    filename = os.path.basename(filepath)
    # split from the right, once
    parts = filename.rsplit("__", 1)
    if len(parts) == 2:
        id_part, _ = parts
        return id_part
    else:
        raise ValueError(f"'{filename}' does not conform to expected format 'id__n.cif'")

def read_true_cifs(input_path):
    true_cifs = {}
    with tarfile.open(input_path, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="extracting true CIFs..."):
            f = tar.extractfile(member)
            if f is not None:
                cif = f.read().decode("utf-8")
                filename = os.path.basename(member.name)
                cif_id = filename.replace(".cif", "")
                true_cifs[cif_id] = cif
    return true_cifs

def get_structs(id_to_gen_cifs, id_to_true_cifs, n_gens, length_lo, length_hi, angle_lo, angle_hi):
    gen_structs = []
    true_structs = []
    for id, structure in tqdm(id_to_gen_cifs.items(), desc="converting CIFs to Structures..."):
        if id not in id_to_true_cifs:
            print(f"could not find ID `{id}` in true CIFs")

        structs = []
        for stru in structure:
            structs.append(stru)
        gen_structs.append(structs)

        true_structs.append(Structure.from_str(id_to_true_cifs[id], fmt="cif"))
    return gen_structs, true_structs

def is_valid(struct):
    atom_types = [str(specie) for specie in struct.species]
    elem_counter = Counter(atom_types)
    composition = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
    elems, counts = list(zip(*composition))
    counts = np.array(counts)
    counts = counts / np.gcd.reduce(counts)
    comps = tuple(counts.astype("int").tolist())

    comp_valid = smact_validity(elems, comps)
    struct_valid = structure_validity(struct)
    return comp_valid and struct_valid

# adapted from
#  https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/compute_metrics.py

# def process_one(pred, gt, is_pred_valid):
#     if not is_pred_valid:
#         return None
#     try:
#         rms_dist = matcher.get_rms_dist(pred, gt)
#         rms_dist = None if rms_dist is None else rms_dist[0]
#         return rms_dist
#     except Exception:
#         return None

import signal
import numpy as np
from tqdm import tqdm

# Timeout exception
class TimeoutException(Exception):
    pass

# Timeout handler function
def timeout_handler(signum, frame):
    raise TimeoutException

# Set the signal alarm to use the timeout handler
signal.signal(signal.SIGALRM, timeout_handler)


def already_exists(candidate_vec, current_list):
    flag = False
    for vec in current_list:
        x = abs(candidate_vec[0] - vec[0])
        y = abs(candidate_vec[1] - vec[1])
        z = abs(candidate_vec[2] - vec[2])
        if x > 0.5: x = 1 - x
        if y > 0.5: y = 1 - y
        if z > 0.5: z = 1 - z
        dist = (x**2 + y**2 + z**2) ** 0.5
        if dist < 0.01: 
            flag = True
            return True
    return False


def get_match_rate_and_rms(gen_structs, true_structs, matcher, data_len):

    def process_one(pred, gt, is_pred_valid):
        if not is_pred_valid:
            return None
        try:
            signal.alarm(60)  # Set the timeout for 60 seconds
            rms_dist = matcher.get_rms_dist(pred, gt)
            signal.alarm(0)  # Disable the alarm after the function call
            rms_dist = None if rms_dist is None else rms_dist[0]
            return rms_dist
        except TimeoutException:
            print("Skipping comparison due to timeout.")
            return None
        except Exception:
            return None
        finally:
            signal.alarm(0)  # Ensure the alarm is disabled
        
    rms_dists = []
    for i in tqdm(range(len(gen_structs)), desc="comparing structures..."):
        tmp_rms_dists = []
        for j in range(len(gen_structs[i])):
            try:
                struct_valid = is_valid(gen_structs[i][j])
                rmsd = process_one(gen_structs[i][j], true_structs[i], struct_valid)
                if rmsd is not None:
                    tmp_rms_dists.append(rmsd)
            except Exception:
                print("match failed")
                pass
        if len(tmp_rms_dists) == 0:
            rms_dists.append(None)
        else:
            rms_dists.append(np.min(tmp_rms_dists))

    rms_dists = np.array(rms_dists)
    match_rate = sum(rms_dists != None) / data_len
    mean_rms_dist = rms_dists[rms_dists != None].mean()
    return {"match_rate": match_rate, "rms_dist": mean_rms_dist}

num_gen = 20
# with tarfile.open("/mnt/data/shared/keqiangyan/gen_mp20_int_3w5_20_t16.tar.gz", "r:gz") as tar:
# with tarfile.open("small_330_3w5_mp20_raw.tar.gz", "r:gz") as tar:
# with tarfile.open("small_330_3w_mp20_raw_gen20_t12.tar.gz", "r:gz") as tar:
# with tarfile.open("mp_20_large_temp4top5.tar.gz", "r:gz") as tar:
with tarfile.open("mp_20_large_temp15_20.tar.gz", "r:gz") as tar:
    for member in tar.getmembers():
        file1 = tar.extractfile(member)
        cif_id = extract_cif_id(member.name)
        cif_str = file1.read().decode()
        if cif_id not in generated_cifs:
            generated_cifs[cif_id] = []
        generated_cifs[cif_id].append(cif_str)

# print(len(cif_ids), cif_ids)
gen_structures = {}
form_error_cnt = 0

for id, cifs in tqdm(generated_cifs.items(), desc="converting CIFs to Structures..."):
    for cif in cifs[:num_gen]:
        structure = {}
        lines = cif.split('\n')
        try:
            structure["formula"] = lines[0].split()[1:]
            structure["space_group"] = lines[1].split()[1]
            structure["a"] = float(lines[2].split()[2])
            structure["b"] = float(lines[2].split()[4])
            structure["c"] = float(lines[2].split()[6])
            structure["alpha"] = float(lines[2].split()[8])
            structure["beta"] = float(lines[2].split()[10])
            structure["gamma"] = float(lines[2].split()[12])
            structure["atoms"] = []
            structure["coords"] = []
            for ii in range(3, len(lines)):
                if len(lines[ii]) > 3:
                    structure["atoms"].append(lines[ii].split()[0])
                    structure["coords"].append(np.array([float(lines[ii].split()[2]), float(lines[ii].split()[3]),float(lines[ii].split()[4])]))
            space_group = SpaceGroup(structure["space_group"])
        
            atom_pos_dict = {}
            space_ops = list(space_group.symmetry_ops)
            for ii in range(len(structure["atoms"])):
                atom = structure["atoms"][ii]
                if atom not in atom_pos_dict:
                    atom_pos_dict[atom] = []
                    atom_pos_dict[atom].append(structure['coords'][ii])
                else:
                    current_list = np.array(atom_pos_dict[atom])
                    if already_exists(structure['coords'][ii], current_list):
                        continue
                    else:
                        atom_pos_dict[atom].append(structure['coords'][ii])
                for op in space_ops:
                    new_pos = op.operate(structure['coords'][ii]) % 1.0
                    current_list = np.array(atom_pos_dict[atom])
                    frac_distance = np.linalg.norm((current_list - new_pos), axis=-1)
                    if already_exists(new_pos, current_list):
                        continue
                    else:
                        atom_pos_dict[atom].append(new_pos)
            keys = atom_pos_dict.keys()
            atom_type_list = []
            frac_coords = []
            for key in keys:
                for pos in atom_pos_dict[key]:
                    atom_type_list.append(key)
                    frac_coords.append(pos)
            nlattice = Lattice.from_parameters(structure["a"], structure["b"], structure["c"], structure["alpha"], structure["beta"], structure["gamma"])
            nstructure = Structure(lattice = nlattice, species=atom_type_list, coords=frac_coords)
            if id not in gen_structures:
                gen_structures[id] = []
            gen_structures[id].append(nstructure)
        except:
            # print(cif)
            form_error_cnt +=1
            continue

print("format syntax error count ", form_error_cnt)
# defaults taken from DiffCSP
struct_matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
id_to_gen_cifs = gen_structures
id_to_true_cifs = read_true_cifs("/mnt/data/shared/keqiangyan/cryllm_results/mp_20_result/mp20_314/mp_20_test_cin.tar.gz")
# print(len(id_to_true_cifs.keys()))

gen_structs, true_structs = get_structs(id_to_gen_cifs, id_to_true_cifs, 20, 0.5, 1000., 10., 170.)
metrics = get_match_rate_and_rms(gen_structs, true_structs, struct_matcher, data_len=len(id_to_true_cifs.keys()))
print(metrics)
