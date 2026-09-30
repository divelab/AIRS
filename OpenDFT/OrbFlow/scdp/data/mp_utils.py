"""Helpers for Materials Project (MP) charge-density benchmarks (GPWNO splits)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

# Crystal families in GPWNO / data_splits/mp_splits/
MP_LATTICE_FAMILIES = (
    "cubic",
    "hexagonal",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "triclinic",
    "mixed",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MP_SPLITS_DIR = REPO_ROOT / "data_splits" / "mp_splits"


def mp_split_path(lattice: str) -> Path:
    lattice = lattice.lower()
    if lattice not in MP_LATTICE_FAMILIES:
        raise ValueError(
            f"Unknown lattice {lattice!r}; expected one of {MP_LATTICE_FAMILIES}"
        )
    path = MP_SPLITS_DIR / f"mp_{lattice}_split.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def mp_download_list_path(lattice: str) -> Path:
    lattice = lattice.lower()
    path = MP_SPLITS_DIR / f"mp_{lattice}_download.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_mp_split(lattice: str) -> dict[str, list[str]]:
    with open(mp_split_path(lattice), encoding="utf-8") as handle:
        data = json.load(handle)
    for key in ("train", "validation", "test"):
        if key not in data:
            raise KeyError(f"split missing {key!r} in {mp_split_path(lattice)}")
    return {k: [str(x) for x in data[k]] for k in ("train", "validation", "test")}


def load_mp_download_ids(lattice: str) -> list[str]:
    """Material IDs from ``mp_<lattice>_download.txt`` (``mp-123.chgcar`` lines)."""
    ids: list[str] = []
    with open(mp_download_list_path(lattice), encoding="utf-8") as handle:
        for line in handle:
            name = line.strip()
            if not name:
                continue
            stem = Path(name).stem
            if not stem.startswith("mp-"):
                raise ValueError(f"unexpected download list entry: {name!r}")
            ids.append(stem)
    return ids


def load_mp_id_list(path: Path) -> list[str]:
    """Load mp-ids from a text file (one mp-<id> or mp-<id>.chgcar per line)."""
    ids: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stem = Path(line.strip()).stem
            if stem.startswith("mp-"):
                ids.append(stem)
    return ids


def mp_ids_in_splits(split: dict[str, list[str]]) -> list[str]:
    """Stable union of train / validation / test IDs (first-seen order)."""
    seen: set[str] = set()
    ordered: list[str] = []
    for key in ("train", "validation", "test"):
        for mp_id in split[key]:
            if mp_id not in seen:
                seen.add(mp_id)
                ordered.append(mp_id)
    return ordered


def chgcar_path(chgcar_dir: Path, mp_id: str) -> Path:
    return chgcar_dir / f"{mp_id}.chgcar"


def existing_chgcar_ids(chgcar_dir: Path, mp_ids: Iterable[str]) -> list[str]:
    return [mp_id for mp_id in mp_ids if chgcar_path(chgcar_dir, mp_id).is_file()]


def filter_split_to_available(
    split: dict[str, list[str]], available: set[str]
) -> dict[str, list[str]]:
    return {k: [x for x in split[k] if x in available] for k in split}


def mp_id_to_index_path(lmdb_dir: Path) -> Path:
    return lmdb_dir / "mp_id_index.json"


def shard_db_name(shard_id: int) -> str:
    return f"data.{shard_id:04d}.lmdb"


def shard_mp_ids_sidecar_path(shard_path: Path) -> Path:
    """Light per-shard mp-id list aligned with local keys 0..N-1."""
    return shard_path.with_suffix(".mp_ids.json")


def list_lmdb_shards(lmdb_dir: Path) -> list[Path]:
    shards = [
        p
        for p in lmdb_dir.glob("data.*.lmdb")
        if not p.name.endswith(".monolith")
    ]
    return sorted(shards, key=lambda p: int(p.name.split(".")[1]))


def mp_sharded_datapath(lattice: str, datapath: Path | None = None) -> Path:
    """Default LMDB root: ``<datapath>/<lattice>/scdp_lmdb``."""
    base = datapath if datapath is not None else Path(
        os.environ.get("DATAPATH", "data")
    )
    return base / lattice.lower() / "scdp_lmdb"
