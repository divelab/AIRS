import os
import tarfile
import argparse
import pickle
import gzip
from tqdm import tqdm


def load_data_from_tar(tar_gz_filename):
    print(f"loading data from {tar_gz_filename}...")
    cif_data = []
    with tarfile.open(tar_gz_filename, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="extracting files..."):
            f = tar.extractfile(member)
            if f is not None:
                content = f.read().decode("utf-8")
                filename = os.path.basename(member.name)
                cif_id = filename.replace(".cif", "")
                cif_data.append((cif_id, content))
    return cif_data


def save_data_to_pickle(cif_data, pickle_filename):
    print(f"saving data to {pickle_filename}...")
    with gzip.open(pickle_filename, "wb") as f:
        pickle.dump(cif_data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .tar.gz file to .pkl.gz file.")
    parser.add_argument("name", help="Path to the input .tar.gz file")
    parser.add_argument("out",
                        help="Path to the output .pkl.gz file. The resulting file will contain a pickled "
                             "Python list of tuples, of (id, cif) pairs, where the id is the filename without "
                             "the .cif extension.")
    args = parser.parse_args()

    cif_data = load_data_from_tar(args.name)

    save_data_to_pickle(cif_data, args.out)

    print("conversion complete!")
