"""
===============================================================================
File:           data
Date:           6/6/2024
Description:    Database utilities
Notes:
  Structure of LMDB:
      * Keys are cids as strings, ex: b'000015111' (strings enoded with string.encode or b'string')
      * Values are nested dictionaries in the form:
      
          b'000015111' : 
          {   'pm3' : [{Step 1}, {Step 2}, ..., {Step n}],
               'hf' : [{Step 1}, {Step 2}, ..., {Step m}],
               'DFT_1st' : [{Step 1}, {Step 2}, ..., {Step z}]
               'DFT_2nd' : [{Step 1}, {Step 2}, ..., {Step k}]
          }
          
          
          
          
          
      * Each calculation method's corresponding value is a list of dictionaries where each dictionary is the values at a given step
      * The final step value (last dictionary in the list) is the optimized geomoetry for it's corresponding calculation method
      * The optimized geometry for one step is the input of the next step, DFT_2nd's last step being the complete optimized geometry for the molecule
      * Each step dictionary is structered in following ways:

          - PM3 and HF: {  'coordiates' : {'atom' : f'{element_letter}, 'charge' : float(charge_val), 'x' : float(x_val), 'y' : float(y_val), 'z' : float{z_val}} ,
                              'energy'     : float(energy_val) ,
                              'gradient'   : {{'atom' : f'{element_letter}, 'charge' : float(charge_val), 'dx' : float(dx_val), 'dy' : float(dy_val), 'dz' : float{dz_val}}}
                          }

          - DFT_1st has two variants, one parsed from Firefly method, and one parsed from Smash method ::
              ~ Firefly: Same as PM3 and HF format:
                         {  'coordiates' : {'atom' : f'{element_letter}, 'charge' : float(charge_val), 'x' : float(x_val), 'y' : float(y_val), 'z' : float{z_val}} ,
                              'energy'     : float(energy_val) ,
                              'gradient'   : {{'atom' : f'{element_letter}, 'charge' : float(charge_val), 'dx' : float(dx_val), 'dy' : float(dy_val), 'dz' : float{dz_val}}}
                          }
              ~ Smash: Does not have a charge value for a coordinate and gradient step:
                          {  'coordiates' : {'atom' : f'{element_letter}, 'x' : float(x_val), 'y' : float(y_val), 'z' : float{z_val}} ,
                              'energy'     : float(energy_val) ,
                              'gradient'   : {{'atom' : f'{element_letter}, 'dx' : float(dx_val), 'dy' : float(dy_val), 'dz' : float{dz_val}}}
                          }

         - DFT_2nd (same format as PM3, HF, and DFT_1st) ::
                          {  'coordiates' : {'atom' : f'{element_letter}, 'charge' : float(charge_val), 'x' : float(x_val), 'y' : float(y_val), 'z' : float{z_val}} ,
                              'energy'     : float(energy_val) ,
                              'gradient'   : {{'atom' : f'{element_letter}, 'charge' : float(charge_val), 'dx' : float(dx_val), 'dy' : float(dy_val), 'dz' : float{dz_val}}}
                          }

      * Each cacluation method's value is an array of those step dictionaries listed above
      * IMPORTANT:: sometimes hf files are duplicated to DFT_1st file, so in those cases DFT_1st : None


  APPENDEX::
  * There's a few important things to mention
  * In general, it seems best practice to open environments and start transactions within "with" blocks
  * I ran into issues w/ error 'MAX Readers limit reached' three seperate times
      - I'm not entirely sure what causes the issue, it would seem to resolve on its own after some time
      - My theory is that, there's sometimes a delay between closing transactiosn in code and the lmdb registering transactions as being closed
      - But even then, increasing the max_readers did not always resolve this issue
      - Sometimes setting lock to False would, once I increased max_readers to 10, the issue fixed, I halfed it until I got back to one and it ran no problem
      - I do not have much experience with lmdb databases, so if someone knows that causes the issue please let us know so everyone can benefit
      - To hopefully avoid this, I would try to use as little transactions as possbible (transactions being the txn variable, object returned from env.begin()
      - Should set max_readers to at least 256
  * What docs have to say about transactions:
      - While any reader exists, writers cannot reuse space in the database file that has become unused in later versions. Due to this, continual use of long-lived read transactions may cause the database to grow without bound. A lost reference to a read transaction will simply be aborted (and its reader slot freed) when the Transaction is eventually garbage collected. This should occur immediately on CPython, but may be deferred indefinitely on PyPy.
      - However the same is not true for write transactions: losing a reference to a write transaction can lead to deadlock, particularly on PyPy, since if the same process that lost the Transaction reference immediately starts another write transaction, it will deadlock on its own lock. Subsequently the lost transaction may never be garbage collected (since the process is now blocked on itself) and the database will become unusable.
      - To prevent this, should probably use with blocks anytime accessing lmdb database so that all memory and transactions are cleaned up.
  * LMDB PYTHON DOCS: https://lmdb.readthedocs.io/en/release/

===============================================================================
"""
"""
LMDB dataset utilities.

Each LMDB entry is a gzipped+pickled dict containing trajectories for stages like:
pm3, hf, DFT_1st, DFT_2nd.

`data_to_pyg` converts trajectory steps into PyG `Data` objects.
"""
import bisect
import gzip
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import lmdb
import numpy as np
import periodictable
import torch
from torch.utils.data import Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

HARTREE_2_EV = 27.2114
BOHR_2_ANGSTROM = 1.8897

_MEAN_ENERGY = -0.0
_STD_ENERGY = 1.0
_STD_FORCE_SCALE = 1.0


atomic_number_mapping: Dict[str, int] = {}
for element in periodictable.elements:
    sym = getattr(element, "symbol", None)
    num = getattr(element, "number", None)
    if not sym or num is None:
        continue
    atomic_number_mapping[sym] = int(num)
    atomic_number_mapping[sym.upper()] = int(num)
    atomic_number_mapping[sym.lower()] = int(num)


atom_energy: Dict[int, float] = {
    1: -0.5002727762,
    4: -14.6684425428,
    5: -24.6543539532,
    6: -37.8462799513,
    7: -54.5844893657,
    8: -75.0606214015,
    9: -99.7155354215,
    14: -289.3723539998,
    15: -341.2580898032,
    16: -398.1049925382,
    17: -460.1362417086,
    21: -760.5813501324,
    22: -849.3013849537,
    23: -943.8255794204,
    24: -1044.2810289455,
    25: -1150.8680174849,
    26: -1263.5207828239406,
    27: -1382.5485719267936,
    28: -1508.0542451335,
    29: -1640.1731641564784,
    31: -1924.5926070018,
    32: -2076.6914561594,
    33: -2235.5683127287,
    34: -2401.2347730327,
    35: -2573.8397377628,
}


def find_last_index_with_key(objects: Sequence[Dict[str, Any]], key: str) -> int:
    for i in range(len(objects) - 1, -1, -1):
        if key in objects[i] and objects[i][key] is not None:
            return i
    return -1


def _phase_key_for_stage(stage: str) -> str:
    if stage in {"1st", "1st_smash"}:
        return "DFT_1st"
    if stage == "2nd":
        return "DFT_2nd"
    if stage in {"hf", "pm3"}:
        return stage
    raise ValueError(f"Unknown stage: {stage}")


def _select_reference_step(record: Dict[str, Any], stage: str) -> Optional[Dict[str, Any]]:
    def last_valid(steps: Optional[Sequence[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        if not steps:
            return None
        idx = find_last_index_with_key(steps, "energy")
        return None if idx == -1 else steps[idx]

    if stage == "mixing":
        step = last_valid(record.get("DFT_2nd"))
        if step is None:
            step = last_valid(record.get("DFT_1st"))
        return step

    phase_key = _phase_key_for_stage(stage)
    return last_valid(record.get(phase_key))


def _to_cid(key: Any) -> str:
    if isinstance(key, (bytes, bytearray)):
        try:
            return key.decode("utf-8")
        except UnicodeDecodeError:
            return key.hex()
    return str(key)


def data_to_pyg(record: Dict[str, Any], key: Any, stage: str = "1st", filter: bool = False) -> List[Data]:
    _ = filter  # kept for compatibility (not used here)

    ref_step = _select_reference_step(record, stage)
    if ref_step is None:
        return []

    ref_coords = ref_step.get("coordinates")
    ref_energy = ref_step.get("energy")
    if not ref_coords or ref_energy is None:
        return []

    if stage == "1st_smash":
        if "charge" in ref_coords[0]:
            return []

    def build_from_phase(phase: Optional[Sequence[Dict[str, Any]]]) -> List[Data]:
        if not phase:
            return []

        out: List[Data] = []
        for step in phase:
            coords = step.get("coordinates")
            energy = step.get("energy")
            gradient = step.get("gradient")

            if not coords or energy is None or not gradient:
                continue
            if len(coords) != len(ref_coords) or len(gradient) != len(coords):
                continue

            if stage == "1st_smash" and "charge" in coords[0]:
                continue

            atomic_numbers: List[int] = []
            positions: List[List[float]] = []
            last_positions: List[List[float]] = []
            forces: List[List[float]] = []
            formation_energies: List[float] = []

            for i, atom_info in enumerate(coords):
                atom = atom_info["atom"]
                atomic_number = atomic_number_mapping[atom]

                x = float(atom_info["x"])
                y = float(atom_info["y"])
                z = float(atom_info["z"])

                atomic_numbers.append(atomic_number)
                formation_energies.append(atom_energy[atomic_number])
                positions.append([x, y, z])

                rc = ref_coords[i]
                last_positions.append([float(rc["x"]), float(rc["y"]), float(rc["z"])])

                g = gradient[i]
                forces.append(
                    [
                        -float(g["dx"]) * HARTREE_2_EV * BOHR_2_ANGSTROM,
                        -float(g["dy"]) * HARTREE_2_EV * BOHR_2_ANGSTROM,
                        -float(g["dz"]) * HARTREE_2_EV * BOHR_2_ANGSTROM,
                    ]
                )

            x_t = torch.tensor(atomic_numbers, dtype=torch.long).view(-1, 1)
            pos_t = torch.tensor(positions, dtype=torch.float32)
            last_pos_t = torch.tensor(last_positions, dtype=torch.float32)

            natoms = x_t.size(0)
            fe_sum = float(sum(formation_energies))

            y_t = torch.tensor([(float(energy) - fe_sum) * HARTREE_2_EV / natoms], dtype=torch.float32)
            last_y_t = torch.tensor([(float(ref_energy) - fe_sum) * HARTREE_2_EV / natoms], dtype=torch.float32)
            f_t = torch.tensor(forces, dtype=torch.float32)

            if not (
                torch.isfinite(pos_t).all()
                and torch.isfinite(last_pos_t).all()
                and torch.isfinite(y_t).all()
                and torch.isfinite(last_y_t).all()
                and torch.isfinite(f_t).all()
            ):
                continue

            out.append(
                Data(
                    x=x_t,
                    natoms=natoms,
                    pos=pos_t,
                    last_pos=last_pos_t,
                    y=y_t,
                    last_y=last_y_t,
                    y_force=f_t,
                    cid=_to_cid(key),
                )
            )

        return out

    if stage == "mixing":
        return build_from_phase(record.get("DFT_1st")) + build_from_phase(record.get("DFT_2nd"))

    phase_key = _phase_key_for_stage(stage)
    return build_from_phase(record.get(phase_key))


def _open_readonly_env(db_path: Path) -> lmdb.Environment:
    return lmdb.open(
        str(db_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=512,
    )


def process_key(key: bytes, db_path: Path, stage: str, filtering: bool) -> Optional[bytes]:
    env = _open_readonly_env(db_path)
    try:
        with env.begin(write=False) as txn:
            payload = txn.get(key)
        if not payload:
            return None
        record = pickle.loads(gzip.decompress(payload))
        data_objects = data_to_pyg(record, key, stage=stage, filter=filtering)
        return key if data_objects else None
    finally:
        env.close()


def process_num(key: bytes, db_path: Path, stage: str, filtering: bool) -> Optional[int]:
    env = _open_readonly_env(db_path)
    try:
        with env.begin(write=False) as txn:
            payload = txn.get(key)
        if not payload:
            return None
        record = pickle.loads(gzip.decompress(payload))
        data_objects = data_to_pyg(record, key, stage=stage, filter=filtering)
        return len(data_objects) if data_objects else None
    finally:
        env.close()


def get_valid_nums(db_path: Path, keys: Sequence[bytes], stage: str, filtering: bool) -> List[int]:
    valid_nums: List[int] = []
    worker_func = partial(process_num, db_path=db_path, stage=stage, filtering=filtering)

    with ProcessPoolExecutor(max_workers=32) as executor:
        results = executor.map(worker_func, keys)
        for maybe_len in tqdm(results, total=len(keys), desc="Get valid numbers"):
            if maybe_len is not None:
                valid_nums.append(maybe_len)

    return valid_nums


def filter_valid_keys(db_path: Path, keys: Sequence[bytes], stage: str, filtering: bool) -> List[bytes]:
    valid_keys: List[bytes] = []
    worker_func = partial(process_key, db_path=db_path, stage=stage, filtering=filtering)

    with ProcessPoolExecutor(max_workers=32) as executor:
        results = executor.map(worker_func, keys)
        for maybe_key in tqdm(results, total=len(keys), desc="Filtering valid keys"):
            if maybe_key is not None:
                valid_keys.append(maybe_key)

    return valid_keys


class LMDBDataset(Dataset):
    def __init__(
        self,
        path: str,
        transform=None,
        keys_file: str = "valid_keys",
        stage: str = "1st",
        total_traj: bool = True,
        subset: bool = False,
    ) -> None:
        super().__init__()

        self.path = Path(path)
        self.keys_file = keys_file
        self.stage = stage
        self.total_traj = total_traj
        self.subset = subset

        assert self.path.is_dir(), "Path is not a directory"

        db_paths = sorted(self.path.glob("*.lmdb"))
        assert db_paths, f"No LMDBs found in '{self.path}'"

        self._keys: List[List[bytes]] = []
        self._nums: List[List[int]] = [] if total_traj else []
        self.envs: List[lmdb.Environment] = []

        self.postfix = "_subset" if subset else ""

        for i, db_path in enumerate(db_paths):
            if subset and "Data06.lmdb" not in str(db_path):
                continue

            env = self.connect_db(db_path)
            self.envs.append(env)

            keys_path = self.path / Path(f"{self.keys_file}_{i}_{self.stage}{self.postfix}.txt")
            if keys_path.exists():
                keys = self.load_keys(i)
            else:
                with env.begin(write=False) as txn:
                    all_keys = list(tqdm(txn.cursor().iternext(values=False), desc=f"Read keys {db_path.name}"))
                keys = filter_valid_keys(db_path, all_keys, self.stage, not self.subset)
                self.save_keys(keys, i)

            self._keys.append(keys)

            if total_traj:
                nums_path = self.path / Path(f"{self.keys_file}_{i}_{self.stage}{self.postfix}_number.txt")
                if nums_path.exists():
                    nums = self.load_nums(i)
                else:
                    nums = get_valid_nums(db_path, self._keys[-1], self.stage, not self.subset)
                    self.save_numbers(nums, i)
                self._nums.append(nums)

        if not total_traj:
            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = int(sum(keylens))
        else:
            keylens = [sum(k) for k in self._nums]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self._num_cumulative = [np.cumsum(k).tolist() for k in self._nums]
            self.num_samples = int(sum(keylens))

            nums_flat = np.concatenate([np.asarray(nums, dtype=np.int64) for nums in self._nums]) if self._nums else np.array([], dtype=np.int64)
            cumulative_nums = np.cumsum(nums_flat)
            start_indices = np.concatenate(([0], cumulative_nums[:-1])) if len(cumulative_nums) else np.array([], dtype=np.int64)
            self.trajectory_indices = list(zip(start_indices.tolist(), cumulative_nums.tolist()))

        self.transform = transform
        self.maximum_dist = 0

    def save_keys(self, keys: Sequence[bytes], index: int) -> None:
        p = self.path / Path(f"{self.keys_file}_{index}_{self.stage}{self.postfix}.txt")
        with open(p, "w") as f:
            for key in keys:
                f.write(key.hex() + "\n")

    def save_numbers(self, numbers: Sequence[int], index: int) -> None:
        p = self.path / Path(f"{self.keys_file}_{index}_{self.stage}{self.postfix}_number.txt")
        with open(p, "w") as f:
            for num in numbers:
                f.write(str(num) + "\n")

    def load_keys(self, index: int) -> List[bytes]:
        p = self.path / Path(f"{self.keys_file}_{index}_{self.stage}{self.postfix}.txt")
        with open(p, "r") as f:
            return [bytes.fromhex(line.strip()) for line in f]

    def load_nums(self, index: int) -> List[int]:
        p = self.path / Path(f"{self.keys_file}_{index}_{self.stage}{self.postfix}_number.txt")
        with open(p, "r") as f:
            return [int(line.strip()) for line in f]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        db_idx = bisect.bisect(self._keylen_cumulative, idx)
        el_idx = idx if db_idx == 0 else idx - self._keylen_cumulative[db_idx - 1]
        assert el_idx >= 0

        if not self.total_traj:
            with self.envs[db_idx].begin(write=False) as txn:
                payload = txn.get(self._keys[db_idx][el_idx])
            if not payload:
                return None

            record = pickle.loads(gzip.decompress(payload))
            key = self._keys[db_idx][el_idx]
            data_objects = data_to_pyg(record, key, stage=self.stage, filter=not self.subset)
            if not data_objects:
                return None

            if self.transform is not None:
                data_objects = [self.transform(d) for d in data_objects]

            return random.choice(data_objects)

        num_idx = bisect.bisect(self._num_cumulative[db_idx], el_idx)
        data_idx = el_idx if num_idx == 0 else el_idx - self._num_cumulative[db_idx][num_idx - 1]
        assert data_idx >= 0

        key = self._keys[db_idx][num_idx]
        with self.envs[db_idx].begin(write=False) as txn:
            payload = txn.get(key)
        if not payload:
            return None

        record = pickle.loads(gzip.decompress(payload))
        data_objects = data_to_pyg(record, key, stage=self.stage, filter=not self.subset)
        if not data_objects or data_idx >= len(data_objects):
            return None

        data_object = data_objects[data_idx]
        if self.transform is not None:
            data_object = self.transform(data_object)
        return data_object

    def connect_db(self, lmdb_path: Optional[Path] = None) -> lmdb.Environment:
        return lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=512,
        )

    def close_db(self) -> None:
        for env in getattr(self, "envs", []):
            env.close()


class CommonLMDBDataset(Dataset):
    def __init__(self, path: str, transform=None) -> None:
        super().__init__()
        self.path = Path(path)
        assert self.path.is_file(), "Path is not a file"

        self.env = self.connect_db(self.path)
        with self.env.begin(write=False) as txn:
            self.all_keys = list(tqdm(txn.cursor().iternext(values=False), desc="Read keys"))

        self.transform = transform

    def __len__(self) -> int:
        return len(self.all_keys)

    def __getitem__(self, idx: int):
        with self.env.begin(write=False) as txn:
            payload = txn.get(self.all_keys[idx])
        if not payload:
            return None

        data_object = pickle.loads(gzip.decompress(payload))

        if hasattr(data_object, "pos") and isinstance(data_object.pos, torch.Tensor) and data_object.pos.ndim >= 3:
            t = random.randrange(data_object.pos.size(0))
            data_object.pos = data_object.pos[t]

        if self.transform is not None:
            data_object = self.transform(data_object)

        return data_object

    def connect_db(self, lmdb_path: Optional[Path] = None) -> lmdb.Environment:
        return lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=512,
        )

    def close_db(self) -> None:
        self.env.close()


def initialize_datasets(root, train_val_ratio, transform, stage, total_traj, subset):
    lmdb_dataset = LMDBDataset(root, transform=transform, stage=stage, total_traj=total_traj, subset=subset)

    if not total_traj:
        train_size = int(train_val_ratio[0] * len(lmdb_dataset))
        val_size = int(train_val_ratio[1] * len(lmdb_dataset))
        test_size = len(lmdb_dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            lmdb_dataset,
            [train_size, val_size, test_size],
            torch.Generator().manual_seed(42),
        )
    else:
        num_trajectories = len(lmdb_dataset.trajectory_indices)
        trajectory_indices = list(range(num_trajectories))
        random.seed(42)
        random.shuffle(trajectory_indices)

        train_size = int(train_val_ratio[0] * num_trajectories)
        val_size = int(train_val_ratio[1] * num_trajectories)

        train_traj = trajectory_indices[:train_size]
        val_traj = trajectory_indices[train_size : train_size + val_size]
        test_traj = trajectory_indices[train_size + val_size :]

        def expand(traj_ids: List[int]) -> List[int]:
            out: List[int] = []
            for tid in traj_ids:
                start_idx, end_idx = lmdb_dataset.trajectory_indices[tid]
                out.extend(range(start_idx, end_idx))
            return out

        train_dataset = Subset(lmdb_dataset, expand(train_traj))
        val_dataset = Subset(lmdb_dataset, expand(val_traj))
        test_dataset = Subset(lmdb_dataset, expand(test_traj))

    return {"train": train_dataset, "val": val_dataset, "test": test_dataset}


def scale_transform(data: Data) -> Data:
    data.y = (data.y - _MEAN_ENERGY) / _STD_ENERGY
    data.y_force = data.y_force / _STD_FORCE_SCALE
    data.pos = data.pos - data.pos.mean(0, keepdim=True)
    data.num_atoms = data.pos.size(0)
    return data


class LMDBDataLoader:
    def __init__(
        self,
        root,
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_ratio: Tuple[float, float] = (0.8, 0.1),
        stage: str = "1st",
        total_traj: bool = False,
        subset: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        assert len(train_val_ratio) == 2

        self.datasets = initialize_datasets(
            root,
            train_val_ratio,
            scale_transform,
            stage,
            total_traj,
            subset=subset,
        )

    def train_loader(self, distributed: bool = False):
        if distributed:
            sampler = DistributedSampler(self.datasets["train"])
        else:
            subset_indices = torch.randperm(len(self.datasets["train"]))
            sampler = SubsetRandomSampler(subset_indices)

        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            sampler=sampler,
            pin_memory=True,
        )

    def val_loader(self, distributed: bool = False):
        if distributed:
            sampler = DistributedSampler(self.datasets["val"])
            return DataLoader(
                self.datasets["val"],
                batch_size=self.batch_size,
                drop_last=True,
                num_workers=self.num_workers,
                sampler=sampler,
                pin_memory=True,
            )

        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_loader(self, distributed: bool = False):
        if distributed:
            sampler = DistributedSampler(self.datasets["test"])
            return DataLoader(
                self.datasets["test"],
                batch_size=self.batch_size,
                drop_last=False,
                num_workers=self.num_workers,
                sampler=sampler,
                pin_memory=True,
            )

        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
