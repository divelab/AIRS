import sys
sys.path.append(".")
import argparse
import gzip
from tqdm import tqdm
import multiprocessing as mp
from queue import Empty

from crystallm import (
    semisymmetrize_cif,
    replace_data_formula_with_nonreduced_formula,
    add_atomic_props_block,
    round_numbers,
    extract_formula_units,
)

try:
    import cPickle as pickle
except ImportError:
    import pickle

import warnings
warnings.filterwarnings("ignore")


def progress_listener(queue, n):
    pbar = tqdm(total=n)
    tot = 0
    while True:
        message = queue.get()
        tot += message
        pbar.update(message)
        if tot == n:
            break


def augment_cif(progress_queue, task_queue, result_queue, oxi, decimal_places):
    augmented_cifs = []

    while not task_queue.empty():
        try:
            id, cif_str = task_queue.get_nowait()
        except Empty:
            break
            
        try:
            formula_units = extract_formula_units(cif_str)
            # exclude CIFs with formula units (Z) = 0, which are erroneous
            if formula_units == 0:
                raise Exception()

            cif_str = replace_data_formula_with_nonreduced_formula(cif_str)
            cif_str = semisymmetrize_cif(cif_str)
            cif_str = add_atomic_props_block(cif_str, oxi)
            cif_str = round_numbers(cif_str, decimal_places=decimal_places)
            augmented_cifs.append((id, cif_str))
        except Exception:
            pass

        progress_queue.put(1)

    result_queue.put(augmented_cifs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process CIF files.")
    parser.add_argument("name", type=str,
                        help="Path to the file with the CIFs to be pre-processed. It is expected that the file "
                             "contains the gzipped contents of a pickled Python list of tuples, of (id, cif) "
                             "pairs.")
    parser.add_argument("--out", "-o", action="store",
                        required=True,
                        help="Path to the file where the pre-processed CIFs will be stored. "
                             "The file will contain the gzipped contents of a pickle dump. It is "
                             "recommended that the filename end in `.pkl.gz`.")
    parser.add_argument("--decimal-places", type=int, default=4,
                        help="The number of decimal places to round the floating point numbers to in the CIF. "
                             "Default is 4.")
    parser.add_argument("--workers", type=int, default=4,
                        help="The number of workers to use for processing. Default is 4.")

    args = parser.parse_args()

    cifs_fname = args.name
    out_fname = args.out
    oxi = False
    decimal_places = args.decimal_places
    workers = args.workers

    print(f"loading data from {cifs_fname}...")
    with gzip.open(cifs_fname, "rb") as f:
        cifs = pickle.load(f)

    manager = mp.Manager()
    progress_queue = manager.Queue()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    for id, cif in cifs:
        task_queue.put((id, cif))

    watcher = mp.Process(target=progress_listener, args=(progress_queue, len(cifs),))

    processes = [mp.Process(target=augment_cif, args=(progress_queue, task_queue, result_queue, oxi, decimal_places))
                 for _ in range(workers)]
    processes.append(watcher)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    modified_cifs = []

    while not result_queue.empty():
        modified_cifs.extend(result_queue.get())

    print(f"number of CIFs: {len(modified_cifs)}")

    print(f"saving data to {out_fname}...")
    with gzip.open(out_fname, "wb") as f:
        pickle.dump(modified_cifs, f, protocol=pickle.HIGHEST_PROTOCOL)
