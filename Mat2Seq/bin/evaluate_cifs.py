import sys
sys.path.append(".")
import argparse
import tarfile
import queue
import multiprocessing as mp

from tqdm import tqdm
import numpy as np
import pandas as pd

from crystallm import (
    CIFTokenizer,
    bond_length_reasonableness_score,
    extract_data_formula,
    extract_numeric_property,
    extract_space_group_symbol,
    extract_volume,
    get_unit_cell_volume,
    is_atom_site_multiplicity_consistent,
    is_space_group_consistent,
    is_sensible,
    is_valid,
    replace_symmetry_operators,
)

import warnings
warnings.filterwarnings("ignore")


def read_generated_cifs(input_path):
    generated_cifs = []
    with tarfile.open(input_path, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="extracting generated CIFs..."):
            f = tar.extractfile(member)
            if f is not None:
                cif = f.read().decode("utf-8")
                generated_cifs.append(cif)
    return generated_cifs


def progress_listener(queue, n):
    pbar = tqdm(total=n)
    tot = 0
    while True:
        message = queue.get()
        tot += message
        pbar.update(message)
        if tot == n:
            break


def eval_cif(progress_queue, task_queue, result_queue, length_lo, length_hi, angle_lo, angle_hi):
    tokenizer = CIFTokenizer()
    n_atom_site_multiplicity_consistent = 0
    n_space_group_consistent = 0
    bond_length_reasonableness_scores = []
    is_valid_and_len = []

    while not task_queue.empty():
        try:
            cif = task_queue.get_nowait()
        except queue.Empty:
            break

        try:
            if not is_sensible(cif, length_lo, length_hi, angle_lo, angle_hi):
                raise Exception

            gen_len = len(tokenizer.tokenize_cif(cif))

            space_group_symbol = extract_space_group_symbol(cif)
            if space_group_symbol is not None and space_group_symbol != "P 1":
                cif = replace_symmetry_operators(cif, space_group_symbol)

            if is_atom_site_multiplicity_consistent(cif):
                n_atom_site_multiplicity_consistent += 1

            if is_space_group_consistent(cif):
                n_space_group_consistent += 1

            score = bond_length_reasonableness_score(cif)
            bond_length_reasonableness_scores.append(score)

            a = extract_numeric_property(cif, "_cell_length_a")
            b = extract_numeric_property(cif, "_cell_length_b")
            c = extract_numeric_property(cif, "_cell_length_c")
            alpha = extract_numeric_property(cif, "_cell_angle_alpha")
            beta = extract_numeric_property(cif, "_cell_angle_beta")
            gamma = extract_numeric_property(cif, "_cell_angle_gamma")
            implied_vol = get_unit_cell_volume(a, b, c, alpha, beta, gamma)

            gen_vol = extract_volume(cif)
            data_formula = extract_data_formula(cif)

            valid = is_valid(cif, bond_length_acceptability_cutoff=1.0)

            is_valid_and_len.append((data_formula, space_group_symbol, valid, gen_len, implied_vol, gen_vol))

        except Exception:
            pass

        progress_queue.put(1)

    result = (
        n_atom_site_multiplicity_consistent,
        n_space_group_consistent,
        bond_length_reasonableness_scores,
        is_valid_and_len,
    )
    result_queue.put(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated structures.")
    parser.add_argument("gen_cifs",
                        help="Path to the .tar.gz file containing the generated CIF files.")
    parser.add_argument("--out", "-o", action="store",
                        required=True,
                        help="Path to the .csv file where the results will be stored.")
    parser.add_argument("--length_lo", required=False, default=0.5, type=float,
                        help="The smallest cell length allowable for the sensibility check")
    parser.add_argument("--length_hi", required=False, default=1000., type=float,
                        help="The largest cell length allowable for the sensibility check")
    parser.add_argument("--angle_lo", required=False, default=10., type=float,
                        help="The smallest cell angle allowable for the sensibility check")
    parser.add_argument("--angle_hi", required=False, default=170., type=float,
                        help="The largest cell angle allowable for the sensibility check")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of workers to use for processing.")
    args = parser.parse_args()

    gen_cifs_path = args.gen_cifs
    out_fname = args.out
    length_lo = args.length_lo
    length_hi = args.length_hi
    angle_lo = args.angle_lo
    angle_hi = args.angle_hi
    workers = args.workers

    cifs = read_generated_cifs(gen_cifs_path)

    manager = mp.Manager()
    progress_queue = manager.Queue()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    n = len(cifs)
    for cif in cifs:
        task_queue.put(cif)

    watcher = mp.Process(target=progress_listener, args=(progress_queue, n,))

    processes = [
        mp.Process(
            target=eval_cif,
            args=(progress_queue, task_queue, result_queue, length_lo, length_hi, angle_lo, angle_hi)
        ) for _ in range(workers)
    ]
    processes.append(watcher)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    n_atom_site_multiplicity_consistent = 0
    n_space_group_consistent = 0
    bond_length_reasonableness_scores = []
    is_valid_and_lens = []

    while not result_queue.empty():
        n_atom_site_occ, n_space_group, scores, is_valid_and_len = result_queue.get()
        n_atom_site_multiplicity_consistent += n_atom_site_occ
        n_space_group_consistent += n_space_group
        bond_length_reasonableness_scores.extend(scores)
        is_valid_and_lens.extend(is_valid_and_len)

    n_valid = 0
    valid_gen_lens = []
    results_data = {
        "comp": [],
        "sg": [],
        "is_valid": [],
        "gen_len": [],
        "implied_vol": [],
        "gen_vol": [],
    }
    for comp, sg, valid, gen_len, implied_vol, gen_vol in is_valid_and_lens:
        if valid:
            n_valid += 1
            valid_gen_lens.append(gen_len)
        results_data["comp"].append(comp)
        results_data["sg"].append(sg)
        results_data["is_valid"].append(valid)
        results_data["gen_len"].append(gen_len)
        results_data["implied_vol"].append(implied_vol)
        results_data["gen_vol"].append(gen_vol)

    print(f"space_group_consistent: {n_space_group_consistent}/{n} ({n_space_group_consistent / n:.3f})\n "
          f"atom_site_multiplicity_consistent: "
          f"{n_atom_site_multiplicity_consistent}/{n} ({n_atom_site_multiplicity_consistent / n:.3f})\n "
          f"bond length reasonableness score: "
          f"{np.mean(bond_length_reasonableness_scores):.4f} ± {np.std(bond_length_reasonableness_scores):.4f}\n "
          f"bond lengths reasonable: "
          f"{bond_length_reasonableness_scores.count(1.)}/{n} ({bond_length_reasonableness_scores.count(1.) / n:.3f})")
    print(f"num valid: {n_valid} / {n} ({n_valid / n:.2f})")
    print(f"longest valid generated length: {np.max(valid_gen_lens):,}")
    print(f"avg. valid generated length: {np.mean(valid_gen_lens):.3f} ± {np.std(valid_gen_lens):.3f}")

    pd.DataFrame(results_data).to_csv(out_fname)
