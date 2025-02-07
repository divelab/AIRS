import gzip
import pickle
import argparse
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split CIF data into train, validation, and test sets.")
    parser.add_argument("name", type=str,
                        help="Path to the file with the CIFs to be split. It is expected that the file "
                             "contains the gzipped contents of a pickled Python list of tuples, of (id, cif) "
                             "pairs.")
    parser.add_argument("--train_out", type=str, required=True,
                        help="Path to the file where the training set CIFs will be stored. "
                             "The file will contain the gzipped contents of a pickle dump. It is "
                             "recommended that the filename end in `.pkl.gz`.")
    parser.add_argument("--val_out", type=str, required=True,
                        help="Path to the file where the validation set CIFs will be stored. "
                             "The file will contain the gzipped contents of a pickle dump. It is "
                             "recommended that the filename end in `.pkl.gz`.")
    parser.add_argument("--test_out", type=str, required=True,
                        help="Path to the file where the test set CIFs will be stored. "
                             "The file will contain the gzipped contents of a pickle dump. It is "
                             "recommended that the filename end in `.pkl.gz`.")
    parser.add_argument("--random_state", type=int, default=20230610,
                        help="Random state for train-test split.")
    parser.add_argument("--validation_size", type=float, default=0.10,
                        help="Size of the validation set as a fraction.")
    parser.add_argument("--test_size", type=float, default=0.0045,
                        help="Size of the test set as a fraction.")
    args = parser.parse_args()

    cifs_fname = args.name
    train_fname = args.train_out
    val_fname = args.val_out
    test_fname = args.test_out
    random_state = args.random_state
    validation_size = args.validation_size
    test_size = args.test_size

    print(f"loading data from {cifs_fname}...")
    with gzip.open(cifs_fname, "rb") as f:
        cifs = pickle.load(f)

    print("splitting dataset...")

    cifs_train, cifs_test = train_test_split(cifs, test_size=test_size,
                                             shuffle=True, random_state=random_state)

    cifs_train, cifs_val = train_test_split(cifs_train, test_size=validation_size,
                                            shuffle=True, random_state=random_state)

    print(f"number of CIFs in train set: {len(cifs_train):,}")
    print(f"number of CIFs in validation set: {len(cifs_val):,}")
    print(f"number of CIFs in test set: {len(cifs_test):,}")

    print("writing train set...")
    with gzip.open(train_fname, "wb") as f:
        pickle.dump(cifs_train, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("writing validation set...")
    with gzip.open(val_fname, "wb") as f:
        pickle.dump(cifs_val, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("writing test set...")
    with gzip.open(test_fname, "wb") as f:
        pickle.dump(cifs_test, f, protocol=pickle.HIGHEST_PROTOCOL)
