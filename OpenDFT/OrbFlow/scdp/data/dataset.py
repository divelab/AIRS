from pathlib import Path
import bisect
import pickle
from typing import List, Optional, Union

import lmdb
import numpy as np
import torch

from torch.utils.data import Dataset
from scdp.common.typing import assert_is_instance as aii


def decode_coeff_sidecar_payload(raw: bytes, field: str) -> torch.Tensor:
    """Decode a pickled tensor (or dict/object) sidecar entry."""
    obj = pickle.loads(raw)
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict) and field in obj:
        return obj[field]
    if hasattr(obj, field):
        return getattr(obj, field)
    raise TypeError(f"Unexpected {field} sidecar payload type: {type(obj)}")


def decode_gt_coeffs_payload(raw: bytes) -> torch.Tensor:
    return decode_coeff_sidecar_payload(raw, "gt_coeffs")


def decode_sad_coeffs_payload(raw: bytes) -> torch.Tensor:
    return decode_coeff_sidecar_payload(raw, "sad_coeffs")


class LmdbDataset(Dataset):
    def __init__(
        self,
        path: Union[str, Path],
        gt_coeffs_path: Optional[Union[str, Path]] = None,
        sad_coeffs_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.path = Path(path)
        self.gt_coeffs_path = Path(gt_coeffs_path) if gt_coeffs_path is not None else None
        self.sad_coeffs_path = Path(sad_coeffs_path) if sad_coeffs_path is not None else None

        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"
            self._keys: List[List[int]] = []
            self.envs = []
            for db_path in db_paths:
                cur_env = self.connect_db(db_path)
                self.envs.append(cur_env)
                num_entries = self._read_length(cur_env)
                self._keys.append(list(range(num_entries)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
        else:
            self.env = self.connect_db(self.path)
            num_entries = self._read_length(self.env)
            self._keys = list(range(num_entries))
            self.num_samples = num_entries
            self._keylen_cumulative = []
            self.envs = []

        self.gt_coeffs_envs: Optional[List[lmdb.Environment]] = None
        self._gt_coeffs_single_env: Optional[lmdb.Environment] = None
        self.sad_coeffs_envs: Optional[List[lmdb.Environment]] = None
        self._sad_coeffs_single_env: Optional[lmdb.Environment] = None
        if self.gt_coeffs_path is not None:
            self._open_coeff_sidecar_envs("gt_coeffs")
        if self.sad_coeffs_path is not None:
            self._open_coeff_sidecar_envs("sad_coeffs")

    def _read_length(self, env: lmdb.Environment) -> int:
        with env.begin() as txn:
            length_entry = txn.get(b"length")
            if length_entry is not None:
                return int(pickle.loads(length_entry))
            return aii(env.stat()["entries"], int)

    def _sidecar_path(self, which: str) -> Optional[Path]:
        if which == "gt_coeffs":
            return self.gt_coeffs_path
        if which == "sad_coeffs":
            return self.sad_coeffs_path
        raise ValueError(which)

    def _open_coeff_sidecar_envs(self, which: str) -> None:
        sidecar_path = self._sidecar_path(which)
        assert sidecar_path is not None
        if not sidecar_path.is_file():
            side_paths = sorted(sidecar_path.glob("*.lmdb"))
            assert len(side_paths) > 0, f"No LMDBs found in {which}_path '{sidecar_path}'"
            if not self.path.is_file():
                main_names = [p.name for p in sorted(self.path.glob("*.lmdb"))]
                side_names = [p.name for p in side_paths]
                if main_names != side_names:
                    raise ValueError(
                        f"{which} sidecar shards must match main LMDB shard names:\n"
                        f"  main: {main_names}\n"
                        f"  {which}: {side_names}"
                    )
            envs = [self.connect_db(p) for p in side_paths]
        else:
            envs = None
            single = self.connect_db(sidecar_path)
        if which == "gt_coeffs":
            if envs is not None:
                self.gt_coeffs_envs = envs
            else:
                self._gt_coeffs_single_env = single
        else:
            if envs is not None:
                self.sad_coeffs_envs = envs
            else:
                self._sad_coeffs_single_env = single

    def _attach_coeff_sidecar(self, data_object, db_idx: int, el_idx: int, which: str) -> None:
        sidecar_path = self._sidecar_path(which)
        if sidecar_path is None:
            return
        key = f"{el_idx}".encode("ascii")
        if which == "gt_coeffs":
            envs = self.gt_coeffs_envs
            single = self._gt_coeffs_single_env
            decode = decode_gt_coeffs_payload
        else:
            envs = self.sad_coeffs_envs
            single = self._sad_coeffs_single_env
            decode = decode_sad_coeffs_payload
        env = envs[db_idx] if envs is not None else single
        assert env is not None
        with env.begin() as txn:
            raw = txn.get(key)
        if raw is None:
            raise KeyError(
                f"Missing {which} sidecar entry shard={db_idx} key={el_idx} under {sidecar_path}"
            )
        setattr(data_object, which, decode(raw))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not self.path.is_file():
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pickle.loads(datapoint_pickled)
            data_object.id = f"{db_idx}_{el_idx}"
            self._attach_coeff_sidecar(data_object, db_idx, el_idx, "gt_coeffs")
            self._attach_coeff_sidecar(data_object, db_idx, el_idx, "sad_coeffs")
        else:
            datapoint_pickled = self.env.begin().get(
                f"{self._keys[idx]}".encode("ascii")
            )
            data_object = pickle.loads(datapoint_pickled)
            self._attach_coeff_sidecar(data_object, 0, idx, "gt_coeffs")
            self._attach_coeff_sidecar(data_object, 0, idx, "sad_coeffs")

        return data_object

    def get_metadata(self, num_samples):
        pass

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()
        if self.gt_coeffs_envs is not None:
            for env in self.gt_coeffs_envs:
                env.close()
        if self._gt_coeffs_single_env is not None:
            self._gt_coeffs_single_env.close()
        if self.sad_coeffs_envs is not None:
            for env in self.sad_coeffs_envs:
                env.close()
        if self._sad_coeffs_single_env is not None:
            self._sad_coeffs_single_env.close()
