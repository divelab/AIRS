"""
Write ``gt_coeffs`` into a **coeffs-only sidecar LMDB** (same shard keys as ``lmdb_in``).

The main LMDB is read-only. Each sidecar entry stores a pickled ``gt_coeffs`` tensor
(small) keyed ``"0"`` … ``"N-1"`` plus ``"length"`` per shard.

Resume: ``--skip_existing`` skips graphs (or whole shards) already present in the sidecar.
Shard range: ``--start_shard`` / ``--end_shard`` (inclusive, from ``data.0042.lmdb`` → 42).

Performance knobs (defaults tuned for 8 Slurm CPUs):
  --num_workers 8       parallel graphs per shard (CPU only)
  --gpu_workers 4       parallel graphs on cuda:0..3 (device=cuda only; default 1)
  --torch_threads 8     BLAS threads when num_workers=1; use 1 per worker when num_workers>1
  --device cuda         GPU matmul/solve
  --dtype float32       faster than float64; ridge keeps system stable
  --chunk_size 4096     GTO design-matrix probe chunks
  --max_probe_samples   -1 = all probes (production); 16000 = fast debug only (~7% nMAPE)
  --probe_subsample_above  -1 = disabled; if n_probe > threshold, cap at max_probe_samples

Example:
  python scdp/scripts/compute_gt_coeffs_lmdb.py \\
    --lmdb_in /path/to/lmdb \\
    --lmdb_out /path/to/lmdb_gt \\
    --metadata_json /path/to/lmdb/metadata.json \\
    --skip_existing --num_workers 8 --dtype float32
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Deque, Dict, List, Optional, Tuple

import lmdb
import torch
from tqdm import tqdm

from scdp.data.gt_coeff_exclusions import (
    append_gt_coeff_exclusion,
    excluded_local_indices_for_shard,
    global_idx_from_shard_local,
    load_gt_coeff_excluded_indices,
    read_lmdb_shard_lengths,
    shard_offsets,
)
from scdp.data.coeff_sidecar_inline_validate import (
    SidecarInlineValidator,
    inline_validate_full_grid_chunk_from_env,
    inline_validate_full_grid_from_env,
    inline_validate_interval_from_env,
    inline_validate_probe_samples_from_env,
)
from scdp.data.gt_coeff_projection import (
    compute_gt_coeffs_payload,
    configure_linear_algebra_threads,
    load_pbc_from_lmdb_dir,
    make_pack_for_lmdb,
)


@dataclass
class ShardStats:
    computed: int = 0
    skipped: int = 0
    excluded: int = 0
    timed_out: int = 0
    failed: int = 0
    total: int = 0


@dataclass
class ProjectionConfig:
    scale: float
    ridge: float
    ridge_diag_frac: float
    probe_weighting: Optional[str]
    max_probe_samples: int
    device: str
    dtype_name: str
    chunk_size: int
    accumulate_on_cpu: Optional[bool] = None
    density_synthesis_mode: str = "linear"
    coeff_nonneg: bool = False
    compact_active_orbitals: bool = False
    lmdb_pbc: bool = False  # MP units.json fallback when graph has no data.pbc
    probe_subsample_above: int = -1  # full grid when n_probe <= threshold; else cap
    oom_policy: str = "cpu_fallback"


_WORKER_PACK = None
_WORKER_CFG: Optional[ProjectionConfig] = None


def _coeff_nonneg_from_env(default: bool = False) -> bool:
    v = os.environ.get("COEFF_NONNEG")
    if v is None or not str(v).strip():
        return bool(default)
    return str(v).strip().lower() not in ("0", "false", "no")


def shard_number(path: Path) -> int:
    parts = path.name.split(".")
    if len(parts) >= 2 and parts[0] == "data" and parts[1].isdigit():
        return int(parts[1])
    raise ValueError(f"Unexpected LMDB shard name (expected data.NNNN.lmdb): {path.name}")


def read_shard_length(env: lmdb.Environment) -> int:
    with env.begin() as txn:
        length_entry = txn.get(b"length")
        if length_entry is None:
            return env.stat()["entries"] - 1
        return int(pickle.loads(length_entry))


def sidecar_shard_complete(
    env: lmdb.Environment,
    n: int,
    excluded_local: Optional[set] = None,
) -> bool:
    excluded_local = excluded_local or set()
    with env.begin() as txn:
        length_entry = txn.get(b"length")
        if length_entry is None:
            return False
        if int(pickle.loads(length_entry)) != n:
            return False
        for i in range(n):
            if i in excluded_local:
                continue
            if txn.get(f"{i}".encode("ascii")) is None:
                return False
    return True


def encode_gt_coeffs(gt: torch.Tensor) -> bytes:
    return pickle.dumps(gt, protocol=-1)


def _init_worker(
    pack_kwargs: Dict[str, Any],
    proj_cfg: ProjectionConfig,
    worker_threads: int,
) -> None:
    global _WORKER_PACK, _WORKER_CFG
    configure_linear_algebra_threads(worker_threads)
    _WORKER_PACK = make_pack_for_lmdb(**pack_kwargs)
    _WORKER_CFG = proj_cfg


def _payload_kwargs(cfg: ProjectionConfig, seed: int) -> Dict[str, Any]:
    return dict(
        scale=cfg.scale,
        ridge=cfg.ridge,
        ridge_diag_frac=cfg.ridge_diag_frac,
        probe_weighting=cfg.probe_weighting,
        max_probe_samples=cfg.max_probe_samples,
        seed=seed,
        device=cfg.device,
        dtype_name=cfg.dtype_name,
        chunk_size=cfg.chunk_size,
        accumulate_on_cpu=cfg.accumulate_on_cpu,
        density_synthesis_mode=cfg.density_synthesis_mode,
        coeff_nonneg=cfg.coeff_nonneg,
        compact_active_orbitals=cfg.compact_active_orbitals,
        lmdb_pbc=cfg.lmdb_pbc,
        probe_subsample_above=cfg.probe_subsample_above,
        oom_policy=cfg.oom_policy,
    )


def _worker_task_with_pack(
    task: Tuple[int, bytes, int],
    pack,
    cfg: ProjectionConfig,
) -> Tuple[int, bytes, float]:
    idx, raw, seed = task
    t0 = time.monotonic()
    payload = compute_gt_coeffs_payload(raw, pack, **_payload_kwargs(cfg, seed))
    elapsed = time.monotonic() - t0
    if str(cfg.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return idx, payload, elapsed


def _worker_task(task: Tuple[int, bytes, int]) -> Tuple[int, bytes, float]:
    idx, raw, seed = task
    assert _WORKER_PACK is not None and _WORKER_CFG is not None
    return _worker_task_with_pack(task, _WORKER_PACK, _WORKER_CFG)


def _graph_progress_log_every() -> int:
    """Emit plain-text graph lines every N graphs (Slurm-friendly). 0 disables."""
    return max(0, int(os.environ.get("GT_PROGRESS_LOG_EVERY", "5")))


def _monitor_graph_progress(
    progress_queue: Any,
    pbar: tqdm,
    futures: List[Any],
) -> None:
    """Drain per-graph completion signals from GPU workers into one tqdm bar."""
    while True:
        try:
            progress_queue.get(timeout=0.5)
            pbar.update(1)
        except Empty:
            if all(f.done() for f in futures):
                break
    while True:
        try:
            progress_queue.get_nowait()
            pbar.update(1)
        except Empty:
            break


def _task_graph_summary(raw: bytes) -> Dict[str, Any]:
    try:
        data = pickle.loads(raw)
        return {
            "n_atoms": int(data.atom_types.shape[0]),
            "n_probe": int(data.probe_coords.shape[0]),
        }
    except Exception:
        return {}


def _gpu_worker_main(
    gpu_id: int,
    in_queue: Any,
    out_queue: Any,
    pack_kwargs: Dict[str, Any],
    proj_cfg: ProjectionConfig,
    worker_threads: int,
) -> None:
    configure_linear_algebra_threads(worker_threads)
    dev = f"cuda:{int(gpu_id)}"
    pk = {**pack_kwargs, "device": dev}
    cfg = ProjectionConfig(
        scale=proj_cfg.scale,
        ridge=proj_cfg.ridge,
        ridge_diag_frac=proj_cfg.ridge_diag_frac,
        probe_weighting=proj_cfg.probe_weighting,
        max_probe_samples=proj_cfg.max_probe_samples,
        device=dev,
        dtype_name=proj_cfg.dtype_name,
        chunk_size=proj_cfg.chunk_size,
        accumulate_on_cpu=proj_cfg.accumulate_on_cpu,
        density_synthesis_mode=proj_cfg.density_synthesis_mode,
        coeff_nonneg=proj_cfg.coeff_nonneg,
        compact_active_orbitals=proj_cfg.compact_active_orbitals,
        lmdb_pbc=proj_cfg.lmdb_pbc,
        probe_subsample_above=proj_cfg.probe_subsample_above,
        oom_policy=proj_cfg.oom_policy,
    )
    pack = make_pack_for_lmdb(**pk)
    while True:
        item = in_queue.get()
        if item is None:
            break
        local_idx, raw, seed = item
        try:
            t0 = time.monotonic()
            payload = compute_gt_coeffs_payload(
                raw, pack, **_payload_kwargs(cfg, seed)
            )
            elapsed = time.monotonic() - t0
            out_queue.put((int(local_idx), "ok", payload, elapsed))
        except Exception as exc:
            out_queue.put((int(local_idx), "error", repr(exc), 0.0))
        if str(cfg.device).startswith("cuda"):
            torch.cuda.empty_cache()


@dataclass
class _GpuWorker:
    gpu_id: int
    process: mp.Process
    in_queue: Any
    out_queue: Any


def _start_gpu_worker(
    mp_ctx: mp.context.BaseContext,
    gpu_id: int,
    pack_kwargs: Dict[str, Any],
    proj_cfg: ProjectionConfig,
    worker_threads: int,
) -> _GpuWorker:
    in_q = mp_ctx.Queue()
    out_q = mp_ctx.Queue()
    proc = mp_ctx.Process(
        target=_gpu_worker_main,
        args=(gpu_id, in_q, out_q, pack_kwargs, proj_cfg, worker_threads),
        daemon=True,
    )
    proc.start()
    return _GpuWorker(gpu_id=gpu_id, process=proc, in_queue=in_q, out_queue=out_q)


def _stop_gpu_worker(worker: _GpuWorker) -> None:
    try:
        worker.in_queue.put(None)
    except Exception:
        pass
    worker.process.join(timeout=2.0)
    if worker.process.is_alive():
        worker.process.terminate()
        worker.process.join(timeout=5.0)
    if worker.process.is_alive():
        worker.process.kill()
        worker.process.join()


def _record_timed_out_graph(
    *,
    excluded_path: Path,
    lmdb_in_path: Path,
    shard_num: int,
    src_name: str,
    local_idx: int,
    gpu_id: int,
    elapsed_s: float,
    timeout_s: float,
    raw: bytes,
    excluded_local: set,
    record_exclusions: bool,
    shard_offsets_table: List[int],
) -> int:
    summary = _task_graph_summary(raw)
    reason = (
        f"gt projection exceeded {timeout_s:.0f}s on cuda:{gpu_id} "
        f"(elapsed={elapsed_s:.1f}s)"
    )
    gidx = global_idx_from_shard_local(shard_num, local_idx, shard_offsets_table)
    if record_exclusions:
        gidx = append_gt_coeff_exclusion(
            excluded_path,
            shard=shard_num,
            local_idx=local_idx,
            source_file=src_name,
            reason=reason,
            lmdb_in_path=lmdb_in_path,
            metrics={
                "timeout_sec": float(timeout_s),
                "elapsed_sec": float(elapsed_s),
                "gpu_id": int(gpu_id),
                **summary,
            },
        )
        excluded_local.add(int(local_idx))
        print(
            "gt_coeff TIMEOUT: "
            f"shard={shard_num} local_idx={local_idx} global_idx={gidx} "
            f"gpu=cuda:{gpu_id} elapsed={elapsed_s:.1f}s limit={timeout_s:.0f}s "
            f"n_atoms={summary.get('n_atoms', '?')} n_probe={summary.get('n_probe', '?')} "
            f"source={src_name} -> {excluded_path}",
            flush=True,
        )
    else:
        print(
            "gt_coeff TIMEOUT (no coeff written; retry on next run): "
            f"shard={shard_num} local_idx={local_idx} global_idx={gidx} "
            f"gpu=cuda:{gpu_id} elapsed={elapsed_s:.1f}s limit={timeout_s:.0f}s "
            f"n_atoms={summary.get('n_atoms', '?')} n_probe={summary.get('n_probe', '?')} "
            f"source={src_name}",
            flush=True,
        )
    return gidx


def _process_gpu_tasks_with_timeout(
    tasks: List[Tuple[int, bytes, int]],
    src_path: Path,
    lmdb_in_path: Path,
    gpu_workers: int,
    pack_kwargs: Dict[str, Any],
    proj_cfg: ProjectionConfig,
    worker_threads: int,
    graph_timeout_sec: float,
    excluded_path: Path,
    shard_num: int,
    excluded_local: set,
    record_exclusions: bool,
    shard_offsets_table: List[int],
    on_success: Optional[Any] = None,
) -> Tuple[List[Tuple[int, bytes]], int]:
    mp_ctx = mp.get_context("spawn")
    workers = [
        _start_gpu_worker(mp_ctx, gpu_id, pack_kwargs, proj_cfg, worker_threads)
        for gpu_id in range(gpu_workers)
    ]
    pending: Deque[Tuple[int, bytes, int]] = deque(tasks)
    in_flight: Dict[int, Tuple[int, bytes, int, float]] = {}
    results: List[Tuple[int, bytes]] = []
    timed_out = 0
    task_by_idx = {idx: (idx, raw, seed) for idx, raw, seed in tasks}

    desc = f"{src_path.name} graphs ({gpu_workers} gpus, timeout={graph_timeout_sec:.0f}s)"
    try:
        with tqdm(total=len(tasks), desc=desc, unit="graph", mininterval=10.0, file=sys.stdout) as pbar:
            while pending or in_flight:
                for gpu_id, worker in enumerate(workers):
                    if gpu_id in in_flight or not pending:
                        continue
                    local_idx, raw, seed = pending.popleft()
                    worker.in_queue.put((local_idx, raw, seed))
                    in_flight[gpu_id] = (local_idx, raw, seed, time.monotonic())

                for gpu_id in list(in_flight.keys()):
                    worker = workers[gpu_id]
                    try:
                        local_idx, status, payload, elapsed = worker.out_queue.get_nowait()
                    except Empty:
                        local_idx, raw, seed, t0 = in_flight[gpu_id]
                        if time.monotonic() - t0 < graph_timeout_sec:
                            continue
                        _stop_gpu_worker(worker)
                        workers[gpu_id] = _start_gpu_worker(
                            mp_ctx, gpu_id, pack_kwargs, proj_cfg, worker_threads
                        )
                        _record_timed_out_graph(
                            excluded_path=excluded_path,
                            lmdb_in_path=lmdb_in_path,
                            shard_num=shard_num,
                            src_name=src_path.name,
                            local_idx=local_idx,
                            gpu_id=gpu_id,
                            elapsed_s=time.monotonic() - t0,
                            timeout_s=graph_timeout_sec,
                            raw=raw,
                            excluded_local=excluded_local,
                            record_exclusions=record_exclusions,
                            shard_offsets_table=shard_offsets_table,
                        )
                        del in_flight[gpu_id]
                        timed_out += 1
                        pbar.update(1)
                        continue

                    del in_flight[gpu_id]
                    pbar.update(1)
                    if status == "ok":
                        raw = task_by_idx[local_idx][1]
                        results.append((local_idx, payload))
                        if on_success is not None:
                            on_success(local_idx, raw, payload, float(elapsed))
                    else:
                        raw = task_by_idx[local_idx][1]
                        _record_failed_graph(
                            excluded_path=excluded_path,
                            lmdb_in_path=lmdb_in_path,
                            shard_num=shard_num,
                            src_name=src_path.name,
                            local_idx=local_idx,
                            gpu_id=gpu_id,
                            raw=raw,
                            error=str(payload),
                            excluded_local=excluded_local,
                            record_exclusions=record_exclusions,
                            shard_offsets_table=shard_offsets_table,
                        )
                        timed_out += 1

                if in_flight:
                    time.sleep(0.2)
    finally:
        for worker in workers:
            _stop_gpu_worker(worker)

    return results, timed_out


def _process_gpu_shard(
    gpu_id: int,
    task_list: List[Tuple[int, bytes, int]],
    pack_kwargs: Dict[str, Any],
    proj_cfg: ProjectionConfig,
    worker_threads: int,
    progress_queue: Optional[Any] = None,
) -> List[Tuple[int, bytes, float]]:
    """Process a batch of graphs on ``cuda:{gpu_id}`` (one process per GPU)."""
    configure_linear_algebra_threads(worker_threads)
    dev = f"cuda:{int(gpu_id)}"
    pk = {**pack_kwargs, "device": dev}
    cfg = ProjectionConfig(
        scale=proj_cfg.scale,
        ridge=proj_cfg.ridge,
        ridge_diag_frac=proj_cfg.ridge_diag_frac,
        probe_weighting=proj_cfg.probe_weighting,
        max_probe_samples=proj_cfg.max_probe_samples,
        device=dev,
        dtype_name=proj_cfg.dtype_name,
        chunk_size=proj_cfg.chunk_size,
        accumulate_on_cpu=proj_cfg.accumulate_on_cpu,
        density_synthesis_mode=proj_cfg.density_synthesis_mode,
        coeff_nonneg=proj_cfg.coeff_nonneg,
        compact_active_orbitals=proj_cfg.compact_active_orbitals,
        lmdb_pbc=proj_cfg.lmdb_pbc,
        probe_subsample_above=proj_cfg.probe_subsample_above,
        oom_policy=proj_cfg.oom_policy,
    )
    pack = make_pack_for_lmdb(**pk)
    out: List[Tuple[int, bytes, float]] = []
    desc = f"cuda:{int(gpu_id)}"
    n = len(task_list)
    log_every = _graph_progress_log_every()
    for i, task in enumerate(task_list):
        try:
            out.append(_worker_task_with_pack(task, pack, cfg))
        except Exception as exc:
            local_idx = int(task[0])
            print(
                f"[{desc}] gt_coeff ERROR local_idx={local_idx}: {exc!r} "
                f"(no coeff written; continue)",
                flush=True,
            )
            if str(cfg.device).startswith("cuda"):
                torch.cuda.empty_cache()
        if progress_queue is not None:
            progress_queue.put(1)
        if log_every > 0 and ((i + 1) % log_every == 0 or i + 1 == n):
            print(f"[{desc}] graph {i + 1}/{n}", flush=True)
    return out


def _split_tasks_evenly(
    tasks: List[Tuple[int, bytes, int]], n: int
) -> List[List[Tuple[int, bytes, int]]]:
    n = max(1, int(n))
    chunks: List[List[Tuple[int, bytes, int]]] = [[] for _ in range(n)]
    for i, task in enumerate(tasks):
        chunks[i % n].append(task)
    return [c for c in chunks if c]


def _graph_seed(data, fallback: int) -> int:
    return (hash(data.metadata) & 0x7FFFFFFF) if hasattr(data, "metadata") else fallback


def _record_failed_graph(
    *,
    excluded_path: Path,
    lmdb_in_path: Path,
    shard_num: int,
    src_name: str,
    local_idx: int,
    gpu_id: int,
    raw: bytes,
    error: str,
    excluded_local: set,
    record_exclusions: bool,
    shard_offsets_table: List[int],
) -> int:
    summary = _task_graph_summary(raw)
    gidx = global_idx_from_shard_local(shard_num, local_idx, shard_offsets_table)
    if record_exclusions:
        gidx = append_gt_coeff_exclusion(
            excluded_path,
            shard=shard_num,
            local_idx=local_idx,
            source_file=src_name,
            reason=f"gt projection failed on cuda:{gpu_id}: {error}",
            lmdb_in_path=lmdb_in_path,
            metrics={"gpu_id": int(gpu_id), "error": error, **summary},
        )
        excluded_local.add(int(local_idx))
        print(
            "gt_coeff ERROR: "
            f"shard={shard_num} local_idx={local_idx} global_idx={gidx} "
            f"gpu=cuda:{gpu_id} n_atoms={summary.get('n_atoms', '?')} "
            f"n_probe={summary.get('n_probe', '?')} source={src_name} "
            f"err={error} -> {excluded_path}",
            flush=True,
        )
    else:
        print(
            "gt_coeff ERROR (no coeff written; retry on next run): "
            f"shard={shard_num} local_idx={local_idx} global_idx={gidx} "
            f"gpu=cuda:{gpu_id} n_atoms={summary.get('n_atoms', '?')} "
            f"n_probe={summary.get('n_probe', '?')} source={src_name} "
            f"err={error}",
            flush=True,
        )
    return gidx


def _store_graph_and_maybe_validate(
    dst_txn: lmdb.Transaction,
    idx: int,
    raw: bytes,
    payload: bytes,
    inline_validator: Optional[SidecarInlineValidator],
    shard_name: str,
    elapsed_sec: Optional[float] = None,
) -> None:
    dst_txn.put(f"{idx}".encode("ascii"), payload)
    if inline_validator is not None:
        inline_validator.on_graph(
            shard_name=shard_name,
            local_idx=int(idx),
            raw=raw,
            coeff_payload=payload,
            elapsed_sec=elapsed_sec,
        )


def process_one_db(
    src_path: Path,
    dst_path: Path,
    pack,
    proj_cfg: ProjectionConfig,
    skip_existing: bool,
    num_workers: int,
    gpu_workers: int,
    worker_threads: int,
    pack_kwargs: Dict[str, Any],
    excluded_local: Optional[set] = None,
    graph_timeout_sec: float = 0.0,
    excluded_path: Optional[Path] = None,
    record_exclusions: bool = False,
    inline_validator: Optional[SidecarInlineValidator] = None,
) -> ShardStats:
    stats = ShardStats()
    excluded_local = set(excluded_local or set())
    shard_num = shard_number(src_path)
    lmdb_in_path = src_path.parent
    shard_offsets_table = shard_offsets(read_lmdb_shard_lengths(lmdb_in_path))
    if excluded_path is None:
        excluded_path = dst_path.parent / "gt_coeff_excluded.json"
    src_env = lmdb.open(str(src_path), subdir=False, readonly=True, lock=False, readahead=True)
    n = read_shard_length(src_env)
    stats.total = n
    stats.excluded = len(excluded_local)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() and skip_existing:
        probe_env = lmdb.open(str(dst_path), subdir=False, readonly=True, lock=False, readahead=False)
        try:
            if sidecar_shard_complete(probe_env, n, excluded_local):
                stats.skipped = n - len(excluded_local)
                src_env.close()
                return stats
        finally:
            probe_env.close()

    dst_env = lmdb.open(
        str(dst_path),
        map_size=1099511627776,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    tasks: List[Tuple[int, bytes, int]] = []
    try:
        with src_env.begin() as src_txn:
            for i in range(n):
                if i in excluded_local:
                    continue
                key = f"{i}".encode("ascii")
                if skip_existing:
                    with dst_env.begin() as dst_txn:
                        if dst_txn.get(key) is not None:
                            stats.skipped += 1
                            continue
                raw = src_txn.get(key)
                if raw is None:
                    continue
                data = pickle.loads(raw)
                seed = _graph_seed(data, i)
                tasks.append((i, raw, seed))

        if not tasks:
            with dst_env.begin(write=True) as dst_txn:
                dst_txn.put(b"length", pickle.dumps(n, protocol=-1))
            return stats

        results: List[Tuple[int, bytes]] = []
        task_raw = {idx: raw for idx, raw, _seed in tasks}
        wrote_incrementally = False
        use_cuda = proj_cfg.device.startswith("cuda")
        parallel = gpu_workers if use_cuda else num_workers

        if parallel <= 1:
            configure_linear_algebra_threads(worker_threads)
            with dst_env.begin(write=True) as dst_txn:
                for idx, raw, seed in tqdm(tasks, desc=f"{src_path.name}"):
                    try:
                        t0 = time.monotonic()
                        payload = compute_gt_coeffs_payload(
                            raw, pack, **_payload_kwargs(proj_cfg, seed)
                        )
                        elapsed = time.monotonic() - t0
                        _store_graph_and_maybe_validate(
                            dst_txn,
                            idx,
                            raw,
                            payload,
                            inline_validator,
                            src_path.name,
                            elapsed_sec=elapsed,
                        )
                        stats.computed += 1
                    except Exception as exc:
                        stats.failed += 1
                        _record_failed_graph(
                            excluded_path=excluded_path,
                            lmdb_in_path=lmdb_in_path,
                            shard_num=shard_num,
                            src_name=src_path.name,
                            local_idx=int(idx),
                            gpu_id=0,
                            raw=raw,
                            error=repr(exc),
                            excluded_local=excluded_local,
                            record_exclusions=record_exclusions,
                            shard_offsets_table=shard_offsets_table,
                        )
                    finally:
                        if use_cuda:
                            torch.cuda.empty_cache()
                dst_txn.put(b"length", pickle.dumps(n, protocol=-1))
            wrote_incrementally = True
        elif use_cuda:
            n_gpu = torch.cuda.device_count()
            if gpu_workers > n_gpu:
                raise RuntimeError(
                    f"gpu_workers={gpu_workers} but only {n_gpu} CUDA device(s) visible"
                )
            if graph_timeout_sec > 0:

                def _on_graph_success(
                    local_idx: int, raw: bytes, payload: bytes, elapsed_sec: float = 0.0
                ) -> None:
                    with dst_env.begin(write=True) as dst_txn:
                        _store_graph_and_maybe_validate(
                            dst_txn,
                            local_idx,
                            raw,
                            payload,
                            inline_validator,
                            src_path.name,
                            elapsed_sec=elapsed_sec,
                        )

                results, stats.timed_out = _process_gpu_tasks_with_timeout(
                    tasks,
                    src_path,
                    lmdb_in_path,
                    gpu_workers,
                    pack_kwargs,
                    proj_cfg,
                    worker_threads,
                    graph_timeout_sec,
                    excluded_path,
                    shard_num,
                    excluded_local,
                    record_exclusions,
                    shard_offsets_table,
                    on_success=_on_graph_success,
                )
                stats.computed += len(results)
                stats.excluded = len(excluded_local)
                with dst_env.begin(write=True) as dst_txn:
                    dst_txn.put(b"length", pickle.dumps(n, protocol=-1))
                wrote_incrementally = True
            else:
                chunks = _split_tasks_evenly(tasks, gpu_workers)
                results = []
                mp_ctx = mp.get_context("spawn")
                with mp_ctx.Manager() as manager:
                    progress_queue = manager.Queue()
                    with ProcessPoolExecutor(max_workers=len(chunks), mp_context=mp_ctx) as pool:
                        futures = [
                            pool.submit(
                                _process_gpu_shard,
                                gpu_id,
                                chunk,
                                pack_kwargs,
                                proj_cfg,
                                worker_threads,
                                progress_queue,
                            )
                            for gpu_id, chunk in enumerate(chunks)
                        ]
                        desc = f"{src_path.name} graphs ({len(chunks)} gpus)"
                        with tqdm(
                            total=len(tasks),
                            desc=desc,
                            unit="graph",
                            mininterval=10.0,
                            dynamic_ncols=True,
                            file=sys.stdout,
                        ) as pbar:
                            monitor = threading.Thread(
                                target=_monitor_graph_progress,
                                args=(progress_queue, pbar, futures),
                                daemon=True,
                            )
                            monitor.start()
                            for fut in futures:
                                results.extend(fut.result())
                            monitor.join(timeout=60.0)
        else:
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_worker,
                initargs=(pack_kwargs, proj_cfg, worker_threads),
            ) as pool:
                results = list(
                    tqdm(
                        pool.map(_worker_task, tasks, chunksize=1),
                        total=len(tasks),
                        desc=f"{src_path.name}",
                    )
                )

        if not wrote_incrementally:
            with dst_env.begin(write=True) as dst_txn:
                for idx, payload, elapsed in results:
                    _store_graph_and_maybe_validate(
                        dst_txn,
                        idx,
                        task_raw[idx],
                        payload,
                        inline_validator,
                        src_path.name,
                        elapsed_sec=elapsed,
                    )
                    stats.computed += 1
                dst_txn.put(b"length", pickle.dumps(n, protocol=-1))

        if inline_validator is not None:
            inline_validator.flush(force=True)
    finally:
        src_env.close()
        dst_env.sync()
        dst_env.close()

    return stats


def _print_missing_gt_summary(
    lmdb_in: Path,
    lmdb_out: Path,
    shard_jobs: List[Path],
) -> None:
    """List local/global indices still missing GT coeffs after this run."""
    missing_rows: List[str] = []
    n_missing = 0
    for src in shard_jobs:
        sn = shard_number(src)
        src_env = lmdb.open(str(src), subdir=False, readonly=True, lock=False, readahead=False)
        try:
            n = read_shard_length(src_env)
        finally:
            src_env.close()
        dst = lmdb_out / src.name
        if not dst.is_file():
            locals_missing = list(range(n))
        else:
            dst_env = lmdb.open(
                str(dst), subdir=False, readonly=True, lock=False, readahead=False
            )
            try:
                with dst_env.begin() as txn:
                    locals_missing = [
                        i
                        for i in range(n)
                        if txn.get(f"{i}".encode("ascii")) is None
                    ]
            finally:
                dst_env.close()
        if not locals_missing:
            continue
        n_missing += len(locals_missing)
        gidxs = [sn * 1000 + i for i in locals_missing]
        preview = ", ".join(
            f"local={i}/global={g}" for i, g in zip(locals_missing[:20], gidxs[:20])
        )
        more = "" if len(locals_missing) <= 20 else f" ... (+{len(locals_missing) - 20} more)"
        missing_rows.append(
            f"  {src.name}: missing={len(locals_missing)}/{n} [{preview}{more}]"
        )
    print(
        f"Missing GT summary ({lmdb_out}): "
        f"{n_missing} graphs across {len(missing_rows)}/{len(shard_jobs)} shards"
    )
    for row in missing_rows:
        print(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lmdb_in", type=Path, required=True, help="Main LMDB directory (read-only)")
    p.add_argument(
        "--lmdb_out",
        type=Path,
        required=True,
        help="Sidecar directory for gt_coeffs-only shards (created)",
    )
    p.add_argument(
        "--metadata_json",
        type=Path,
        required=True,
        help="metadata.json with target_var and unique_atom_types",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip graphs/shards already present in the sidecar (default: true)",
    )
    p.add_argument(
        "--no_skip_existing",
        action="store_false",
        dest="skip_existing",
        help="Recompute all gt_coeffs even if sidecar entries exist",
    )
    p.add_argument(
        "--start_shard",
        type=int,
        default=0,
        help="First shard index to process (data.NNNN.lmdb NNNN)",
    )
    p.add_argument(
        "--end_shard",
        type=int,
        default=None,
        help="Last shard index inclusive (default: last shard)",
    )
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument(
        "--ridge_diag_frac",
        type=float,
        default=0.0,
        help="LM-style diag(Phi^T Phi) damping; 0.01 adds ~0.5-1%% nMAPE vs labels",
    )
    p.add_argument(
        "--probe_weighting",
        type=str,
        default="uniform",
        choices=["uniform", "abs_rho"],
    )
    p.add_argument(
        "--max_probe_samples",
        type=int,
        default=-1,
        help="Probes used in ridge fit; -1 = full grid (match baseline/validate nMAPE)",
    )
    p.add_argument(
        "--probe_subsample_above",
        type=int,
        default=-1,
        help=(
            "When n_probe exceeds this threshold, cap fit probes at max_probe_samples; "
            "graphs at or below threshold use the full grid (-1 disables)"
        ),
    )
    p.add_argument("--dft_basis_set", type=str, default="def2-qzvppd")
    p.add_argument("--dft_wt_aug", action="store_true", default=True)
    p.add_argument("--no_dft_wt_aug", action="store_true")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument(
        "--density_synthesis_mode",
        type=str,
        default="linear",
        choices=["linear", "squared"],
        help="linear: Phi @ c; squared: Phi_sq @ d (gt_coeffs stores d)",
    )
    p.add_argument(
        "--coeff_nonneg",
        action=argparse.BooleanOptionalAction,
        default=_coeff_nonneg_from_env(False),
        help="Squared mode: enforce d>=0 (PGD NNLS). Default: unconstrained cuSOLVER ridge LS.",
    )
    p.add_argument("--lmax_restriction", action="store_true", default=True)
    p.add_argument("--no_lmax_restriction", action="store_true")
    p.add_argument("--uncontracted", action="store_true", default=True)
    p.add_argument("--contracted", action="store_true")
    p.add_argument("--orb_cutoff", type=float, default=5.0)
    p.add_argument("--vnode_elem", type=int, default=8)
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for GTO eval and linear solve",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Arithmetic dtype (float64 default on CPU; try float32 with --device cuda)",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Parallel worker processes per shard (CPU only; use 8 on Slurm)",
    )
    p.add_argument(
        "--gpu_workers",
        type=int,
        default=1,
        help="Parallel GPU processes per shard when --device cuda (1 graph/GPU at a time each)",
    )
    p.add_argument(
        "--torch_threads",
        type=int,
        default=0,
        help="BLAS/torch threads per process (0=auto: 8 if num_workers=1 else 1)",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        default=4096,
        help="Probe chunk size when building the design matrix",
    )
    p.add_argument(
        "--accumulate_on_cpu",
        action="store_true",
        help="Force Phi^T W Phi accumulation on CPU (default: GPU with OOM policy)",
    )
    p.add_argument(
        "--oom_policy",
        type=str,
        default=os.environ.get("GT_OOM_POLICY", "cpu_fallback"),
        choices=("cpu_fallback", "skip"),
        help="On CUDA OOM during GPU accumulate: cpu_fallback (default) or skip graph "
        "after clearing the GPU (env: GT_OOM_POLICY)",
    )
    p.add_argument(
        "--compact_active_orbitals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Active-orbital normal equations (lower RAM; MP cubic via GT_COMPACT_ACTIVE_ORBITALS=1)",
    )
    p.add_argument(
        "--excluded_json",
        type=Path,
        default=None,
        help="gt_coeff_excluded.json (default: <lmdb_out>/gt_coeff_excluded.json)",
    )
    p.add_argument(
        "--respect_excluded",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip frames listed in gt_coeff_excluded.json (default: false, process all)",
    )
    p.add_argument(
        "--record_exclusions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append timeout/error frames to gt_coeff_excluded.json (default: false, retry later)",
    )
    p.add_argument(
        "--graph_timeout_sec",
        type=float,
        default=float(os.environ.get("GT_GRAPH_TIMEOUT_SEC", "600")),
        help="Per-graph GPU projection timeout (0=disable). Timed-out frames are "
        "logged and appended to gt_coeff_excluded.json.",
    )
    p.add_argument(
        "--inline_validate_every",
        type=int,
        default=inline_validate_interval_from_env(100),
        help="Reconstruct density every N newly computed graphs (0=off). "
        "Env: COEFF_INLINE_VALIDATE_EVERY or GT_INLINE_VALIDATE_EVERY.",
    )
    p.add_argument(
        "--inline_validate_probe_samples",
        type=int,
        default=inline_validate_probe_samples_from_env(8192),
        help="Probes per graph for fast inline QA subsample (-1=full grid only). "
        "Env: COEFF_INLINE_VALIDATE_PROBE_SAMPLES.",
    )
    p.add_argument(
        "--inline_validate_full_grid",
        action=argparse.BooleanOptionalAction,
        default=inline_validate_full_grid_from_env(True),
        help="When subsampling inline QA, also report nMAPE on the full probe grid. "
        "Env: GT_INLINE_VALIDATE_FULL_GRID.",
    )
    p.add_argument(
        "--inline_validate_full_grid_chunk",
        type=int,
        default=inline_validate_full_grid_chunk_from_env(8192),
        help="Probe chunk size for full-grid inline QA inference. "
        "Env: GT_INLINE_VALIDATE_FULL_GRID_CHUNK.",
    )
    p.add_argument(
        "--inline_validate_nmape_warn",
        type=float,
        default=0.02,
        help="Append WARNING to inline QA log when batch mean nMAPE exceeds this (percent).",
    )
    args = p.parse_args()
    excluded_json = args.excluded_json or (args.lmdb_out / "gt_coeff_excluded.json")
    shard_offsets_table = shard_offsets(read_lmdb_shard_lengths(args.lmdb_in))
    global_excluded: set[int] = set()
    if args.respect_excluded:
        global_excluded = load_gt_coeff_excluded_indices(
            args.lmdb_out, lmdb_in_path=args.lmdb_in
        )
        if args.excluded_json is not None and excluded_json.is_file():
            with open(excluded_json) as fp:
                data = json.load(fp)
            global_excluded |= set(int(x) for x in data.get("global_indices", []))
        if global_excluded:
            print(
                f"gt_coeff_excluded: skipping {len(global_excluded)} global indices "
                f"({excluded_json})"
            )
    else:
        print(
            "gt_coeff_excluded: ignored (--no_respect_excluded); only --skip_existing skips graphs"
        )

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
        if int(args.gpu_workers) < 1:
            raise ValueError("--gpu_workers must be >= 1")
        if int(args.num_workers) > 1:
            print("NOTE: --num_workers ignored when --device cuda (use --gpu_workers instead)")
    elif int(args.gpu_workers) > 1:
        print("NOTE: --gpu_workers ignored when --device cpu")

    dft_wt_aug = not args.no_dft_wt_aug
    lmax_restriction = not args.no_lmax_restriction
    uncontracted = not args.contracted

    worker_threads = int(args.torch_threads)
    if worker_threads <= 0:
        if args.device == "cuda":
            n_gpu = max(1, int(args.gpu_workers))
            cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
            worker_threads = max(1, cpus // n_gpu)
        else:
            worker_threads = 8 if int(args.num_workers) <= 1 else 1
    configure_linear_algebra_threads(worker_threads)

    with open(args.metadata_json, "r") as fp:
        meta = json.load(fp)
    unique_atom_types = [int(x) for x in meta["unique_atom_types"]]
    if 0 not in unique_atom_types:
        unique_atom_types = [0] + unique_atom_types

    target_var = float(meta["target_var"])
    scale = target_var**0.5

    use_multi_gpu = args.device == "cuda" and int(args.gpu_workers) > 1
    pack_device = "cuda:0" if args.device == "cuda" else "cpu"
    pack_kwargs = dict(
        unique_atom_types=unique_atom_types,
        dft_basis_set=args.dft_basis_set,
        dft_wt_aug=dft_wt_aug,
        beta=args.beta,
        lmax_restriction=lmax_restriction,
        uncontracted=uncontracted,
        orb_cutoff=args.orb_cutoff,
        vnode_elem=args.vnode_elem,
        device=pack_device,
        density_synthesis_mode=str(args.density_synthesis_mode),
    )
    # Multi-GPU workers build packs in spawned children; avoid CUDA init in parent.
    if use_multi_gpu:
        pack = None
    else:
        pack = make_pack_for_lmdb(**pack_kwargs)

    lmdb_pbc = load_pbc_from_lmdb_dir(args.lmdb_in)
    if lmdb_pbc:
        print(f"lmdb_pbc=True (from {args.lmdb_in / 'units.json'}, MP fallback when graph has no data.pbc)")

    probe_pw = None if args.probe_weighting == "uniform" else "abs_rho"
    accum_cpu: Optional[bool] = True if args.accumulate_on_cpu else None
    if int(args.probe_subsample_above) >= 0 and int(args.max_probe_samples) < 0:
        raise ValueError(
            "--probe_subsample_above requires --max_probe_samples >= 0 "
            f"(got max_probe_samples={args.max_probe_samples})"
        )
    proj_cfg = ProjectionConfig(
        scale=scale,
        ridge=float(args.ridge),
        ridge_diag_frac=float(args.ridge_diag_frac),
        probe_weighting=probe_pw,
        max_probe_samples=int(args.max_probe_samples),
        device=pack_device if not use_multi_gpu else "cuda",
        dtype_name=args.dtype,
        chunk_size=int(args.chunk_size),
        accumulate_on_cpu=accum_cpu,
        density_synthesis_mode=str(args.density_synthesis_mode),
        coeff_nonneg=bool(args.coeff_nonneg),
        compact_active_orbitals=bool(args.compact_active_orbitals),
        lmdb_pbc=bool(lmdb_pbc),
        probe_subsample_above=int(args.probe_subsample_above),
        oom_policy=str(args.oom_policy),
    )

    if args.accumulate_on_cpu:
        accum_mode = "cpu"
    elif args.device == "cuda":
        accum_mode = (
            "auto(gpu,oom->skip)"
            if args.oom_policy == "skip"
            else "auto(gpu,oom->cpu) accumulate+solve"
        )
    else:
        accum_mode = "cpu"
    print(
        f"projection: device={args.device} dtype={args.dtype} "
        f"num_workers={args.num_workers} gpu_workers={args.gpu_workers} "
        f"torch_threads={worker_threads} accumulate={accum_mode} "
        f"oom_policy={args.oom_policy} "
        f"compact_active_orbitals={args.compact_active_orbitals} "
        f"max_probe_samples={args.max_probe_samples} "
        f"probe_subsample_above={args.probe_subsample_above} chunk_size={args.chunk_size} "
        f"graph_timeout_sec={args.graph_timeout_sec} orb_cutoff={args.orb_cutoff} "
        f"dft_basis_set={args.dft_basis_set} beta={args.beta}"
    )
    if args.device == "cuda":
        print(f"cuda_devices_visible={torch.cuda.device_count()}")

    db_paths = sorted(args.lmdb_in.glob("data.*.lmdb"), key=shard_number)
    if not db_paths:
        raise FileNotFoundError(f"No data.*.lmdb under {args.lmdb_in}")

    end_shard = (
        shard_number(db_paths[-1]) if args.end_shard is None else int(args.end_shard)
    )
    if args.start_shard > end_shard:
        raise ValueError(f"start_shard ({args.start_shard}) > end_shard ({end_shard})")

    inline_validator = SidecarInlineValidator.from_env_and_args(
        metadata=meta,
        scale=scale,
        beta=float(args.beta),
        density_synthesis_mode=str(args.density_synthesis_mode),
        coeff_field="gt_coeffs",
        device=pack_device if not use_multi_gpu else "cuda:0",
        vnode_elem=int(args.vnode_elem),
        dft_basis_set=args.dft_basis_set,
        dft_wt_aug=dft_wt_aug,
        lmax_restriction=lmax_restriction,
        uncontracted=uncontracted,
        orb_cutoff=float(args.orb_cutoff),
        interval=int(args.inline_validate_every),
        probe_samples=int(args.inline_validate_probe_samples),
        full_grid_validate=bool(args.inline_validate_full_grid),
        full_grid_chunk=int(args.inline_validate_full_grid_chunk),
        nmape_warn_threshold=float(args.inline_validate_nmape_warn),
        sidecar_label="gt_coeffs",
        lmdb_pbc=lmdb_pbc,
    )

    totals = ShardStats()
    selected = 0
    shard_jobs = [
        src
        for src in db_paths
        if args.start_shard <= shard_number(src) <= end_shard
    ]
    for src in tqdm(shard_jobs, desc="LMDB shards", unit="shard"):
        sn = shard_number(src)
        selected += 1
        dst = args.lmdb_out / src.name
        shard_excluded = excluded_local_indices_for_shard(
            global_excluded, sn, shard_offsets_table
        )
        stats = process_one_db(
            src,
            dst,
            pack,
            proj_cfg,
            args.skip_existing,
            int(args.num_workers),
            int(args.gpu_workers),
            worker_threads,
            pack_kwargs,
            excluded_local=shard_excluded,
            graph_timeout_sec=float(args.graph_timeout_sec),
            excluded_path=excluded_json,
            record_exclusions=bool(args.record_exclusions),
            inline_validator=inline_validator,
        )
        totals.computed += stats.computed
        totals.skipped += stats.skipped
        totals.excluded += stats.excluded
        totals.timed_out += stats.timed_out
        totals.failed += stats.failed
        totals.total += stats.total
        print(
            f"{src.name}: computed={stats.computed} skipped={stats.skipped} "
            f"excluded={stats.excluded} timed_out={stats.timed_out} "
            f"failed={stats.failed} total={stats.total}"
        )

    print(
        f"Sidecar {args.lmdb_out}: shards={selected} "
        f"computed={totals.computed} skipped={totals.skipped} "
        f"excluded={totals.excluded} timed_out={totals.timed_out} "
        f"failed={totals.failed} total={totals.total}"
    )
    _print_missing_gt_summary(args.lmdb_in, args.lmdb_out, shard_jobs)

    projection_meta = {
        "dft_basis_set": str(args.dft_basis_set),
        "orb_cutoff": float(args.orb_cutoff),
        "beta": float(args.beta),
        "ridge": float(args.ridge),
        "ridge_diag_frac": float(args.ridge_diag_frac),
        "density_synthesis_mode": str(args.density_synthesis_mode),
        "max_probe_samples": int(args.max_probe_samples),
        "probe_subsample_above": int(args.probe_subsample_above),
    }
    args.lmdb_out.mkdir(parents=True, exist_ok=True)
    with open(args.lmdb_out / "gt_projection.json", "w", encoding="utf-8") as fp:
        json.dump(projection_meta, fp, indent=2)
        fp.write("\n")


if __name__ == "__main__":
    main()
