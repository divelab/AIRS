import sys
sys.path.append(".")
import os
import numpy as np
import random
import gzip
import argparse
import tarfile
import multiprocessing as mp
from tqdm import tqdm
try:
    import cPickle as pickle
except ImportError:
    import pickle

from crystallm import (
    CIFTokenizer,
    array_split,
)


def progress_listener(queue, n):
    pbar = tqdm(total=n, desc="tokenizing...")
    while True:
        message = queue.get()
        if message == "kill":
            break
        pbar.update(message)


def tokenize(chunk_of_cifs, queue=None):
    tokenizer = CIFTokenizer()
    tokenized = []
    for cif in tqdm(chunk_of_cifs, disable=queue is not None, desc="tokenizing..."):
        if queue:
            queue.put(1)
        tokenized.append(tokenizer.tokenize_cif(cif))
    return tokenized


def preprocess(cifs_raw):
    cifs = []
    for _, cif in tqdm(cifs_raw, desc="preparing files..."):
        # filter out some lines in the CIF
        lines = cif.split('\n')
        cif_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 0 and not line.startswith("#") and "pymatgen" not in line:
                cif_lines.append(line)
        cif_lines.append("\n")
        cifs.append("\n".join(cif_lines))
    return cifs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize CIF files.")
    parser.add_argument("--train_fname", type=str, required=True,
                        help="Path to the file with the training set CIFs to be tokenized. It is expected that "
                             "the file contains the gzipped contents of a pickled Python list of tuples, of (id, cif) "
                             "pairs.")
    parser.add_argument("--val_fname", type=str, default="",
                        help="Path to the file with the validation set CIFs to be tokenized. It is expected that "
                             "the file contains the gzipped contents of a pickled Python list of tuples, of (id, cif) "
                             "pairs.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory to store processed files.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of workers to use for processing.")
    args = parser.parse_args()

    train_fname = args.train_fname
    val_fname = args.val_fname
    out_dir = args.out_dir
    workers = args.workers

    has_val = len(val_fname) > 0

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"loading data from {train_fname}...")
    with gzip.open(train_fname, "rb") as f:
        cifs_raw_train = pickle.load(f)

    if has_val:
        print(f"loading data from {val_fname}...")
        with gzip.open(val_fname, "rb") as f:
            cifs_raw_val = pickle.load(f)

    # shuffle the order of the train CIFs
    random.shuffle(cifs_raw_train)

    cifs_train = preprocess(cifs_raw_train)
    if has_val:
        cifs_val = preprocess(cifs_raw_val)

    # tokenize the train CIFs in parallel
    chunks = array_split(cifs_train, workers)
    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(workers + 1)  # add an extra worker for the watcher
    watcher = pool.apply_async(progress_listener, (queue, len(cifs_train),))

    jobs = []
    for i in range(workers):
        chunk = chunks[i]
        job = pool.apply_async(tokenize, (chunk, queue))
        jobs.append(job)

    tokenized_cifs_train = []
    for job in jobs:
        tokenized_cifs_train.extend(job.get())

    queue.put("kill")
    pool.close()
    pool.join()

    lens = [len(t) for t in tokenized_cifs_train]
    unk_counts = [t.count("<unk>") for t in tokenized_cifs_train]
    print(f"train min tokenized length: {np.min(lens):,}")
    print(f"train max tokenized length: {np.max(lens):,}")
    print(f"train mean tokenized length: {np.mean(lens):.2f} +/- {np.std(lens):.2f}")
    print(f"train total unk counts: {np.sum(unk_counts)}")

    if has_val:
        # tokenize the validation CIFs
        tokenized_cifs_val = tokenize(cifs_val)

        lens = [len(t) for t in tokenized_cifs_val]
        unk_counts = [t.count("<unk>") for t in tokenized_cifs_val]
        print(f"val min tokenized length: {np.min(lens):,}")
        print(f"val max tokenized length: {np.max(lens):,}")
        print(f"val mean tokenized length: {np.mean(lens):.2f} +/- {np.std(lens):.2f}")
        print(f"val total unk counts: {np.sum(unk_counts)}")

    # create a single stream of tokens that will be the dataset
    train_data = []
    for t in tqdm(tokenized_cifs_train, desc="concatenating train tokens..."):
        train_data.extend(t)

    if has_val:
        val_data = []
        for t in tqdm(tokenized_cifs_val, desc="concatenating val tokens..."):
            val_data.extend(t)

    print("encoding...")
    tokenizer = CIFTokenizer()
    train_ids = tokenizer.encode(train_data)
    print(f"train has {len(train_ids):,} tokens")
    if has_val:
        val_ids = tokenizer.encode(val_data)
        print(f"val has {len(val_ids):,} tokens")
    print(f"vocab size: {len(tokenizer.token_to_id)}")

    print("exporting to .bin files...")
    train_ids = np.array(train_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(out_dir, "train.bin"))
    if has_val:
        val_ids = np.array(val_ids, dtype=np.uint16)
        val_ids.tofile(os.path.join(out_dir, "val.bin"))

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": len(tokenizer.token_to_id),
        "itos": tokenizer.id_to_token,
        "stoi": tokenizer.token_to_id,
    }
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("creating tar.gz archive...")
    subdir_name = os.path.basename(os.path.normpath(out_dir))
    tar_gz_filename = os.path.join(out_dir, f"{subdir_name}.tar.gz")
    with tarfile.open(tar_gz_filename, "w:gz") as tar:
        for filename in ["train.bin", "val.bin", "meta.pkl"]:
            file_path = os.path.join(out_dir, filename)
            if os.path.exists(file_path):
                arcname = os.path.join(subdir_name, filename)
                tar.add(file_path, arcname=arcname)

    print(f"tarball created at {tar_gz_filename}")
