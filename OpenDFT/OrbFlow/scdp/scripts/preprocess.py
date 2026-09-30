from pathlib import Path
import lz4, gzip, zlib
import lz4.frame
import lzma
import os
import pickle
import tarfile
import argparse
import multiprocessing as mp

import lmdb
import numpy as np
import torch
from tqdm import tqdm

from scdp.data.data import AtomicData, AtomicNumberTable


def get_atomic_number_table_from_zs(zs) -> AtomicNumberTable:
    z_set = set()
    for z in zs:
        z_set.add(z)
    return AtomicNumberTable(sorted(list(z_set)))

def decompress_tarmember(tar, tarinfo):
    """Extract compressed tar file member and return a bytes object with the content"""
    bytesobj = tar.extractfile(tarinfo).read()
    if tarinfo.name.endswith(".zz"):
        filecontent = zlib.decompress(bytesobj)
    elif tarinfo.name.endswith(".lz4"):
        filecontent = lz4.frame.decompress(bytesobj)
    elif tarinfo.name.endswith(".gz"):
        filecontent = gzip.decompress(bytesobj)
    elif tarinfo.name.endswith(".xz"):
        filecontent = lzma.decompress(bytesobj)
    else:
        filecontent = bytesobj

    return filecontent

def lmdb_shard_is_complete(db_path: os.PathLike | str) -> bool:
    """True if LMDB exists and has a valid ``length`` metadata key."""
    path = Path(db_path)
    if not path.is_file():
        return False
    env = None
    try:
        env = lmdb.open(
            str(path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
        )
        with env.begin() as txn:
            length_raw = txn.get(b"length")
            if length_raw is None:
                return False
            length = pickle.loads(length_raw)
        return int(length) > 0
    except lmdb.Error:
        return False
    finally:
        if env is not None:
            env.close()


def remove_lmdb_shard(db_path: os.PathLike | str) -> None:
    path = Path(db_path)
    lock = Path(f"{db_path}-lock")
    for p in (path, lock):
        if p.exists():
            p.unlink()


def process_tar_file_to_lmdb(mp_arg):
    z_table, tarpath, db_path, pid, idx, args = mp_arg
    n_device = torch.cuda.device_count()
    device = f"cuda:{pid % n_device}" if args.device == "cuda" else "cpu"
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 16,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    member_list = []
    read_cfg = "r:gz" if tarpath.parts[-1].endswith('.gz') else "r:"
    with tarfile.open(tarpath, read_cfg) as tar:
        for member in tqdm(tar.getmembers(), position=pid, desc=f"processing job {pid} on {device}"):
            member_list.append(member)

            filecontent = decompress_tarmember(tar, member)
            fileinfo = member
            data_object = AtomicData.from_file(
                fcontent=filecontent,
                finfo=fileinfo,
                build_method=args.build_method,
                z_table=z_table,
                atom_cutoff=args.atom_cutoff,
                probe_cutoff=args.probe_cutoff,
                vnode_method=args.vnode_method,
                vnode_factor=args.vnode_factor,
                vnode_res=args.vnode_res,
                disable_pbc=args.disable_pbc,
                max_neighbors=args.max_neighbors,
                device=device,
            )

            txn = db.begin(write=True)
            txn.put(
                f"{idx}".encode("ascii"),
                pickle.dumps(data_object, protocol=-1),
            )
            txn.commit()
            idx += 1

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

    return idx

def main_tar(args: argparse.Namespace) -> None:
    tar_files = sorted(list(Path(args.data_path).glob("*.tar*")))
    if not tar_files:
        raise FileNotFoundError(f"No *.tar* files found under {args.data_path}")
    os.makedirs(os.path.join(args.out_path), exist_ok=True)

    start = max(0, int(args.start_index))
    end = len(tar_files) - 1 if args.end_index is None else min(int(args.end_index), len(tar_files) - 1)
    if start > end:
        raise ValueError(f"start_index ({start}) must be <= end_index ({end})")

    db_paths = [
        os.path.join(args.out_path, "data.%04d.lmdb" % i) for i in range(len(tar_files))
    ]

    z_table = get_atomic_number_table_from_zs(np.arange(100).tolist())

    mp_args = []
    skipped = 0
    for i in range(start, end + 1):
        db_path = db_paths[i]
        if args.skip_existing and lmdb_shard_is_complete(db_path):
            print(f"Skipping complete shard {i}: {db_path}")
            skipped += 1
            continue
        remove_lmdb_shard(db_path)
        mp_args.append(
            (
                z_table,
                tar_files[i],
                db_path,
                i,
                0,
                args,
            )
        )

    print(
        f"Tar shards {start}..{end} ({end - start + 1} total in range): "
        f"{len(mp_args)} to process, {skipped} skipped (complete)."
    )
    if not mp_args:
        print("Nothing to do.")
        return

    pool = mp.Pool(args.num_workers)
    list([*pool.imap(process_tar_file_to_lmdb, mp_args)])

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        help="path to charge density tar files",
    )
    parser.add_argument(
        "--out_path",
        help="Directory to save extracted features. Will create if doesn't exist",
    )
    parser.add_argument(
        "--num_dbs",
        type=int,
        default=32,
        help="No. of feature-extracting processes or no. of dataset chunks",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="No. of processes to use for feature extraction",
    )
    parser.add_argument(
        "--build_method",
        type=str,
        default='vnode',
        choices=["vnode", "probe"],
        help="Method to use for building graphs",
    )
    parser.add_argument(
        "--atom_cutoff",
        type=float,
        default=4.0,
        help="Cutoff radius for atom graph",
    )
    parser.add_argument(
        "--max_neighbors",
        type=int,
        default=None,
        help="Max number of neighbors for each atom in the graph",
    )
    parser.add_argument(
        "--probe_cutoff",
        type=float,
        default=4.0,
        help="Cutoff radius for probe edges",
    )
    parser.add_argument(
        "--vnode_method",
        type=str,
        default='bond',
        choices=["bond", "none"],
        help="VO centers: bond | none",
    )
    parser.add_argument(
        "--vnode_factor",
        type=int,
        default=3,
        help="Maximum number of iterations for virtual node generation as a factor of number of atoms",
    )
    parser.add_argument(
        "--vnode_res",
        type=float,
        default=0.8,
        help="vnode resolution in the obb method",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for feature extraction",
    )
    parser.add_argument(
        "--disable_pbc",
        action="store_true",
        help="Disable periodic boundary conditions",
    )
    parser.add_argument(
        "--tar",
        action="store_true",
        help="process tar files if <True>, else process pkl files",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip tar indices whose output LMDB already has a valid length key",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="First tar shard index to process (0 -> 000xxx.tar)",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="Last tar shard index to process, inclusive (default: last tar)",
    )
    return parser


if __name__ == "__main__":
    parser: argparse.ArgumentParser = get_parser()
    args: argparse.Namespace = parser.parse_args()
    if args.tar:
        main_tar(args)
    else:
        raise NotImplementedError