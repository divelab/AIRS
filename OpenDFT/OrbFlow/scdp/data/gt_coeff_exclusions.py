"""Shared helpers for ``gt_coeff_excluded.json`` (preprocess skip + training filter).

Global indices are **cumulative** over LMDB shard lengths (same convention as
``datasplits.json``, ``LmdbDataset.__getitem__``, and ``preprocess_mp_scdp.scan_sharded_lmdb``):

    global_idx = sum(len(shard_i) for i < shard) + local_idx

Legacy files used ``shard * 1000 + local_idx`` (nominal). That only matches
cumulative indices when every shard before the last has exactly 1000 graphs.
Use ``normalize_gt_coeff_excluded_document`` / ``migrate_gt_coeff_excluded_file``
to upgrade old sidecars (required for variable-size shards such as MP mixed).
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import lmdb

from scdp.data.mp_utils import list_lmdb_shards

GT_COEFF_EXCLUDED_FILENAME = "gt_coeff_excluded.json"
INDEX_CONVENTION_CUMULATIVE = "cumulative"
INDEX_CONVENTION_NOMINAL_LEGACY = "nominal_shard_x_1000"


def read_lmdb_shard_lengths(lmdb_dir: Path) -> List[int]:
    """Return graph counts per ``data.NNNN.lmdb`` shard (sorted by shard id)."""
    lengths: List[int] = []
    for shard_path in list_lmdb_shards(lmdb_dir):
        env = lmdb.open(
            str(shard_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
        )
        try:
            with env.begin() as txn:
                length_entry = txn.get(b"length")
                if length_entry is None:
                    n = int(env.stat()["entries"]) - 1
                else:
                    n = int(pickle.loads(length_entry))
            lengths.append(n)
        finally:
            env.close()
    return lengths


def shard_offsets(shard_lengths: Iterable[int]) -> List[int]:
    """Cumulative start index per shard; ``offsets[shard+1] - offsets[shard]`` is length."""
    offsets = [0]
    for n in shard_lengths:
        offsets.append(offsets[-1] + int(n))
    return offsets


def global_idx_from_shard_local(shard: int, local_idx: int, offsets: List[int]) -> int:
    """Cumulative global index for ``(shard, local_idx)``."""
    s, local = int(shard), int(local_idx)
    if s < 0 or s + 1 >= len(offsets):
        raise IndexError(f"shard {s} out of range (n_shards={len(offsets) - 1})")
    shard_len = offsets[s + 1] - offsets[s]
    if local < 0 or local >= shard_len:
        raise IndexError(f"local_idx {local} out of range for shard {s} (len={shard_len})")
    return offsets[s] + local


def nominal_global_idx(shard: int, local_idx: int) -> int:
    """Legacy ``shard * 1000 + local`` (valid only for fixed 1000-wide shards)."""
    return int(shard) * 1000 + int(local_idx)


def global_idx(
    shard: int,
    local_idx: int,
    *,
    lmdb_in: Optional[Path] = None,
    offsets: Optional[List[int]] = None,
) -> int:
    """Cumulative global index; pass ``lmdb_in`` or precomputed ``offsets``."""
    if offsets is not None:
        return global_idx_from_shard_local(shard, local_idx, offsets)
    if lmdb_in is not None:
        return global_idx_from_shard_local(
            shard, local_idx, shard_offsets(read_lmdb_shard_lengths(lmdb_in))
        )
    return nominal_global_idx(shard, local_idx)


def load_gt_coeff_excluded_document(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"global_indices": [], "entries": []}
    with open(path) as fp:
        data = json.load(fp)
    data.setdefault("global_indices", [])
    data.setdefault("entries", [])
    return data


def save_gt_coeff_excluded_document(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def normalize_gt_coeff_excluded_document(
    data: Dict[str, Any],
    lmdb_in_path: Path,
    *,
    lmdb_in_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a copy with cumulative ``global_indices`` and updated entry metadata."""
    offsets = shard_offsets(read_lmdb_shard_lengths(lmdb_in_path))
    out = dict(data)
    entries = list(out.get("entries") or [])
    global_indices: List[int] = []
    seen: Set[int] = set()

    new_entries: List[Dict[str, Any]] = []
    for entry in entries:
        shard = int(entry["shard"])
        local = int(entry["local_idx"])
        gidx = global_idx_from_shard_local(shard, local, offsets)
        new_entry = dict(entry)
        new_entry["global_idx"] = gidx
        new_entries.append(new_entry)
        if gidx not in seen:
            seen.add(gidx)
            global_indices.append(gidx)

    # Preserve orphan globals only when already cumulative (no entry remap available).
    if not entries:
        for raw in out.get("global_indices") or []:
            g = int(raw)
            if g not in seen:
                seen.add(g)
                global_indices.append(g)
    else:
        out["entries"] = new_entries

    out["global_indices"] = sorted(global_indices)
    out["index_convention"] = INDEX_CONVENTION_CUMULATIVE
    out["lmdb_in"] = lmdb_in_hint or str(lmdb_in_path)
    if "description" not in out:
        out["description"] = (
            "Graphs skipped during GT coeff generation (OOM/timeout/manual). "
            "global_indices use cumulative LMDB indexing (see scdp/data/gt_coeff_exclusions.py)."
        )
    return out


def migrate_gt_coeff_excluded_file(
    excluded_path: Path,
    lmdb_in_path: Path,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Rewrite ``gt_coeff_excluded.json`` to cumulative indexing."""
    data = load_gt_coeff_excluded_document(excluded_path)
    normalized = normalize_gt_coeff_excluded_document(data, lmdb_in_path)
    if not dry_run:
        save_gt_coeff_excluded_document(excluded_path, normalized)
    return normalized


def load_gt_coeff_excluded_indices(
    gt_coeffs_path: Optional[Path] = None,
    excluded_file: Optional[Path] = None,
    lmdb_in_path: Optional[Path] = None,
) -> Set[int]:
    """Global dataset indices from ``gt_coeff_excluded.json``.

    When ``lmdb_in_path`` is set, indices are normalized to cumulative convention
    (migrating legacy nominal entries via shard/local when present).
    """
    if excluded_file is not None:
        path = Path(excluded_file)
    elif gt_coeffs_path is not None:
        path = Path(gt_coeffs_path) / GT_COEFF_EXCLUDED_FILENAME
    else:
        return set()
    if not path.is_file():
        return set()

    data = load_gt_coeff_excluded_document(path)
    if lmdb_in_path is None:
        return set(int(x) for x in data.get("global_indices", []))

    if data.get("index_convention") == INDEX_CONVENTION_CUMULATIVE and data.get("entries"):
        normalized = normalize_gt_coeff_excluded_document(data, Path(lmdb_in_path))
    elif data.get("index_convention") == INDEX_CONVENTION_CUMULATIVE:
        return set(int(x) for x in data.get("global_indices", []))
    else:
        normalized = normalize_gt_coeff_excluded_document(data, Path(lmdb_in_path))

    return set(int(x) for x in normalized.get("global_indices", []))


def excluded_local_indices_for_shard(
    global_excluded: Set[int],
    shard: int,
    offsets: List[int],
) -> Set[int]:
    """Map cumulative ``global_indices`` to local LMDB keys for one shard."""
    s = int(shard)
    base = offsets[s]
    end = offsets[s + 1]
    return {int(g) - base for g in global_excluded if base <= int(g) < end}


def load_excluded_local_for_shard(
    lmdb_in_path: Path,
    gt_coeffs_path: Path,
    shard: int,
) -> Set[int]:
    """Convenience wrapper for GT preprocess / shell helpers."""
    global_excluded = load_gt_coeff_excluded_indices(
        gt_coeffs_path, lmdb_in_path=lmdb_in_path
    )
    offsets = shard_offsets(read_lmdb_shard_lengths(lmdb_in_path))
    return excluded_local_indices_for_shard(global_excluded, shard, offsets)


def filter_split_indices(indices, excluded: Set[int]):
    if not excluded:
        return [int(i) for i in indices]
    return [int(i) for i in indices if int(i) not in excluded]


def append_gt_coeff_exclusion(
    excluded_path: Path,
    *,
    shard: int,
    local_idx: int,
    source_file: str,
    reason: str,
    lmdb_in_path: Path,
    metrics: Optional[dict] = None,
) -> int:
    """Record a frame for preprocess skip and DataModule split filtering."""
    offsets = shard_offsets(read_lmdb_shard_lengths(lmdb_in_path))
    gidx = global_idx_from_shard_local(shard, local_idx, offsets)
    data = load_gt_coeff_excluded_document(excluded_path)
    existing: Set[int] = set(int(x) for x in data["global_indices"])
    entry = {
        "shard": int(shard),
        "local_idx": int(local_idx),
        "global_idx": gidx,
        "source_file": source_file,
        "reason": reason,
        "metrics": metrics or {},
        "excluded_at": datetime.now(timezone.utc).isoformat(),
    }
    if gidx not in existing:
        data["global_indices"].append(gidx)
        data["entries"].append(entry)
    else:
        for i, old in enumerate(data["entries"]):
            if int(old.get("global_idx", -1)) == gidx:
                data["entries"][i] = entry
                break
    data["index_convention"] = INDEX_CONVENTION_CUMULATIVE
    data["lmdb_in"] = str(lmdb_in_path)
    data.setdefault(
        "description",
        (
            "Graphs skipped during GT coeff generation (OOM/timeout/manual). "
            "global_indices use cumulative LMDB indexing."
        ),
    )
    save_gt_coeff_excluded_document(excluded_path, data)
    return gidx
