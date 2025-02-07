import sys
sys.path.append(".")
import os
import argparse
from collections import Counter
import tarfile

import numpy as np
import itertools
from tqdm import tqdm

import smact
from smact.screening import pauling_test

from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

from crystallm import is_sensible

import warnings
warnings.filterwarnings("ignore")


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
def get_match_rate_and_rms(gen_structs, true_structs, matcher):
    def process_one(pred, gt, is_pred_valid):
        if not is_pred_valid:
            return None
        try:
            rms_dist = matcher.get_rms_dist(pred, gt)
            rms_dist = None if rms_dist is None else rms_dist[0]
            return rms_dist
        except Exception:
            return None

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
                pass
        if len(tmp_rms_dists) == 0:
            rms_dists.append(None)
        else:
            rms_dists.append(np.min(tmp_rms_dists))

    rms_dists = np.array(rms_dists)
    match_rate = sum(rms_dists != None) / len(gen_structs)
    mean_rms_dist = rms_dists[rms_dists != None].mean()
    return {"match_rate": match_rate, "rms_dist": mean_rms_dist}


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


def read_generated_cifs(input_path):
    generated_cifs = {}
    with tarfile.open(input_path, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="extracting generated CIFs..."):
            f = tar.extractfile(member)
            if f is not None:
                cif = f.read().decode("utf-8")
                cif_id = extract_cif_id(member.name)
                if cif_id not in generated_cifs:
                    generated_cifs[cif_id] = []
                generated_cifs[cif_id].append(cif)
    return generated_cifs


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
    for id, cifs in tqdm(id_to_gen_cifs.items(), desc="converting CIFs to Structures..."):
        if id not in id_to_true_cifs:
            raise Exception(f"could not find ID `{id}` in true CIFs")

        structs = []
        for cif in cifs[:n_gens]:
            try:
                if not is_sensible(cif, length_lo, length_hi, angle_lo, angle_hi):
                    continue
                structs.append(Structure.from_str(cif, fmt="cif"))
            except Exception:
                pass
        gen_structs.append(structs)

        true_structs.append(Structure.from_str(id_to_true_cifs[id], fmt="cif"))
    return gen_structs, true_structs


"""
This script performs the CDVAE and DiffCSP benchmark analysis, as described in:
https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/compute_metrics.py.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform benchmark analysis.")
    parser.add_argument("gen_cifs",
                        help="Path to the .tar.gz file containing the generated CIF files.")
    parser.add_argument("true_cifs",
                        help="Path to the .tar.gz file containing the true CIF files.")
    parser.add_argument("--num-gens", required=False, default=0, type=int,
                        help="The maximum number of generations to use per structure. Default is 0, which means "
                             "use all of the available generations.")
    parser.add_argument("--length_lo", required=False, default=0.5, type=float,
                        help="The smallest cell length allowable for the sensibility check")
    parser.add_argument("--length_hi", required=False, default=1000., type=float,
                        help="The largest cell length allowable for the sensibility check")
    parser.add_argument("--angle_lo", required=False, default=10., type=float,
                        help="The smallest cell angle allowable for the sensibility check")
    parser.add_argument("--angle_hi", required=False, default=170., type=float,
                        help="The largest cell angle allowable for the sensibility check")
    args = parser.parse_args()

    gen_cifs_path = args.gen_cifs
    true_cifs_path = args.true_cifs
    n_gens = args.num_gens
    length_lo = args.length_lo
    length_hi = args.length_hi
    angle_lo = args.angle_lo
    angle_hi = args.angle_hi

    if n_gens == 0:
        n_gens = None
        print("using all available generations...")
    else:
        if n_gens < 0:
            raise Exception(f"invalid value for n_gens: {n_gens}")
        print(f"using a maximum of {n_gens} generation(s) per compound...")

    # defaults taken from DiffCSP
    struct_matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    id_to_gen_cifs = read_generated_cifs(gen_cifs_path)
    id_to_true_cifs = read_true_cifs(true_cifs_path)

    gen_structs, true_structs = get_structs(
        id_to_gen_cifs, id_to_true_cifs, n_gens, length_lo, length_hi, angle_lo, angle_hi
    )

    metrics = get_match_rate_and_rms(gen_structs, true_structs, struct_matcher)

    print(metrics)
