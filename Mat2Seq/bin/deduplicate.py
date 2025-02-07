import sys
sys.path.append(".")
import argparse
from tqdm import tqdm
import gzip

from crystallm import (
    extract_formula_nonreduced,
    extract_space_group_symbol,
    extract_volume,
    extract_formula_units,
)

try:
    import cPickle as pickle
except ImportError:
    import pickle

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate CIF files.")
    parser.add_argument("name", type=str,
                        help="Path to the file with the CIFs to be deduplicated. It is expected that the file "
                             "contains the gzipped contents of a pickled Python list of tuples, of (id, cif) pairs.")
    parser.add_argument("--out", "-o", action="store",
                        required=True,
                        help="Path to the file where the deduplicated CIFs will be stored. "
                             "The file will contain the gzipped contents of a pickle dump. It is "
                             "recommended that the filename end in `.pkl.gz`.")
    args = parser.parse_args()

    cifs_fname = args.name
    out_fname = args.out

    print(f"loading data from {cifs_fname}...")
    with gzip.open(cifs_fname, "rb") as f:
        cifs = pickle.load(f)

    print(f"number of CIFs to deduplicate: {len(cifs):,}")

    lowest_vpfu = {}

    for id, cif in tqdm(cifs):
        formula = extract_formula_nonreduced(cif)
        space_group = extract_space_group_symbol(cif)
        formula_units = extract_formula_units(cif)
        if formula_units == 0:
            formula_units = 1
        vpfu = extract_volume(cif) / formula_units

        key = (formula, space_group)
        if key not in lowest_vpfu:
            lowest_vpfu[key] = (id, cif, vpfu)
        else:
            existing_vpfu = lowest_vpfu[key][2]
            if vpfu < existing_vpfu:
                lowest_vpfu[key] = (id, cif, vpfu)

    selected_entries = [(id, cif) for id, cif, _ in lowest_vpfu.values()]

    print(f"number of entries to write: {len(selected_entries):,}")

    print(f"saving data to {out_fname}...")
    with gzip.open(out_fname, "wb") as f:
        pickle.dump(selected_entries, f, protocol=pickle.HIGHEST_PROTOCOL)
