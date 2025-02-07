import argparse
import gzip
import pickle
import tarfile
import io
from tqdm import tqdm


def load_data_from_pickle(pickle_gz_filename):
    print(f"loading data from {pickle_gz_filename}...")
    with gzip.open(pickle_gz_filename, "rb") as f:
        cif_data = pickle.load(f)
    return cif_data


def save_data_to_tar(cif_data, tar_gz_filename):
    print(f"saving data to {tar_gz_filename}...")
    with tarfile.open(tar_gz_filename, "w:gz") as tar:
        for item_id, cif_content in tqdm(cif_data):
            cif_file = io.BytesIO(cif_content.encode("utf-8"))
            tarinfo = tarfile.TarInfo(name=f"{item_id}.cif")
            tarinfo.size = len(cif_file.getvalue())
            cif_file.seek(0)
            tar.addfile(tarinfo, cif_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a .pkl.gz file to a .tar.gz file.")
    parser.add_argument("name",
                        help="Path to the input .pkl.gz file. It is expected that the file contains the gzipped "
                             "contents of a pickled Python list of tuples, of (id, cif) pairs.")
    parser.add_argument("out", help="Path to the output .tar.gz file")
    args = parser.parse_args()

    cif_data = load_data_from_pickle(args.name)

    save_data_to_tar(cif_data, args.out)

    print("conversion complete!")