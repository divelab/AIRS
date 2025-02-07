import argparse
import gzip
import pickle
from tqdm import tqdm
from pymatgen.core import Composition

from crystallm import (
    extract_formula_nonreduced,
    extract_formula_units,
    extract_space_group_symbol,
)

import warnings
warnings.filterwarnings("ignore")


def load_cifs(fname):
    print(f"loading data from {fname}...")
    with gzip.open(fname, "rb") as f:
        return pickle.load(f)


def load_cifs_multi(fnames):
    cifs = []
    if fnames:
        for fname in fnames:
            cifs.extend(load_cifs(fname))
    return cifs


def extract_key(cif):
    comp = extract_formula_nonreduced(cif)
    reduced_comp = Composition(comp).reduced_formula
    Z = extract_formula_units(cif)
    space_group = extract_space_group_symbol(cif)
    return reduced_comp, Z, space_group


def extract_all_keys_to_cifs(cifs):
    keys_to_cifs = {}
    for id, cif in tqdm(cifs, desc="extracting keys..."):
        try:
            key = extract_key(cif)
            if key in keys_to_cifs:
                print(f"WARNING: key already exists: {key}")
            keys_to_cifs[key] = (id, cif)
        except Exception as e:
            print(f"ERROR: could not parse CIF with ID '{id}': {e}")
    return keys_to_cifs


"""
This script takes a base dataset of CIF files, each considered unique according to its reduced formula,
Z, and space group, and takes a dataset of CIF files to include, and a dataset of CIF files to exclude.
A new dataset will be produced which includes the CIF files to include (if not already present), and 
is missing the CIF files to exclude.
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge CIF file datasets.")
    parser.add_argument("--base", "-b", type=str, required=True,
                        help="Path to the file with the CIFs which form the base dataset. "
                             "It is expected that the file contains the gzipped contents of a pickled Python "
                             "list of tuples, of (id, cif) pairs.")
    parser.add_argument("--include", "-i", type=str, action="append",
                        help="Path to the file with the CIFs which are to be included. "
                             "It is expected that the file contains the gzipped contents of a pickled Python "
                             "list of tuples, of (id, cif) pairs.")
    parser.add_argument("--exclude", "-x", type=str, action="append",
                        help="Path to the file with the CIFs which are to be excluded. "
                             "It is expected that the file contains the gzipped contents of a pickled Python "
                             "list of tuples, of (id, cif) pairs.")
    parser.add_argument("--out", "-o", type=str, action="store", required=True,
                        help="Path to the file where the merged CIFs will be stored. "
                             "The file will contain the gzipped contents of a pickle dump. It is "
                             "recommended that the filename end in `.pkl.gz`.")

    args = parser.parse_args()

    base_fname = args.base
    include_fnames = args.include
    exclude_fnames = args.exclude
    out_fname = args.out

    base_cifs = load_cifs(base_fname)
    include_cifs = load_cifs_multi(include_fnames)
    exclude_cifs = load_cifs_multi(exclude_fnames)

    print(f"total base CIFs: {len(base_cifs):,}")
    print(f"total CIFs to include: {len(include_cifs):,}")
    print(f"total CIFs to exclude: {len(exclude_cifs):,}")

    # create map {(reduced comp, Z, spacegroup) -> (id, cif)}
    base_keys_to_cifs = extract_all_keys_to_cifs(base_cifs)
    include_keys_to_cifs = extract_all_keys_to_cifs(include_cifs)
    exclude_keys_to_cifs = extract_all_keys_to_cifs(exclude_cifs)

    base_keys = set(base_keys_to_cifs)
    include_keys = set(include_keys_to_cifs)
    exclude_keys = set(exclude_keys_to_cifs)

    final_keys = base_keys.union(include_keys - base_keys).difference(exclude_keys)

    final_cifs = []
    added_keys = set()
    added_ids = set()

    for key in tqdm(final_keys, desc="processing final CIFs..."):
        # the key must be in either the 'base' CIFs or the 'include' CIFs
        id, cif = base_keys_to_cifs[key] if key in base_keys_to_cifs else include_keys_to_cifs[key]

        if key in added_keys:
            print(f"WARNING: key '{key}' has already been added")

        if id in added_ids:
            print(f"WARNING: ID '{id}' has already been added")

        final_cifs.append((id, cif))
        added_keys.add(key)
        added_ids.add(id)

    print(f"total final CIFs: {len(final_cifs):,}")

    print(f"saving data to {out_fname}...")
    with gzip.open(out_fname, "wb") as f:
        pickle.dump(final_cifs, f, protocol=pickle.HIGHEST_PROTOCOL)
