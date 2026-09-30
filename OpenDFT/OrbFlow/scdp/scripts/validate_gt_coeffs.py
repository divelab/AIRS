#!/usr/bin/env python3
"""
Validate precomputed ``gt_coeffs`` by reconstructing charge density on probe grids.

If ridge-projected coefficients cannot represent ``chg_labels`` well, flow matching
targets are unreliable. This script does **not** use a neural network — only stored
``gt_coeffs`` + the same GTO ``orbital_inference`` path as training.

Metrics (per molecule, then aggregated):
  - nMAPE: 100 * sum(|rho_pred - rho_gt|) / sum(|rho_gt|)
  - mae_scaled: mean |pred/scale - target/scale| (training loss scale)
  - rmse_scaled: sqrt mean (pred/scale - target/scale)^2

Optional ``--recompute_n`` graphs: re-run ridge projection and report
  - coeff_storage_error: ||gt_stored - gt_recomputed|| / ||gt_stored||
  - fit_mse_on_fit_probes: MSE on probes used for the LS fit (should be low)

With ``--compute_flow_prior_stats`` (default), scan the training split and report
masked normalized-coefficient mean/std for ``gt_coeffs`` (and ``gt - sad`` if
``--sad_coeffs_path`` is set). Use ``suggested_flow_prior_mu`` / ``suggested_flow_prior_sigma``
as ``model.flow_prior_mu`` / ``model.flow_prior_sigma`` in flow training.

Example:
  python scdp/scripts/validate_gt_coeffs.py \\
    --lmdb_path /path/to/lmdb \\
    --gt_coeffs_path /path/to/lmdb_gt \\
    --metadata_json /path/to/lmdb/metadata.json \\
    --split_file /path/to/datasplits.json \\
    --tag val --max_n_graphs 200 --random_seed 42 --beta 1.3

  # Squared sidecar (d on phi^2): must pass --density_synthesis_mode squared
  python scdp/scripts/validate_gt_coeffs.py ... \\
    --gt_coeffs_path /path/to/lmdb_gt_ridge_sq \\
    --density_synthesis_mode squared --ridge 1e-6 --recompute_n 10

  # SAD sidecar vs superposed atomic density
  python scdp/scripts/validate_gt_coeffs.py ... \\
    --coeff_field sad_coeffs --sad_coeffs_path /path/to/lmdb_sad_ridge_sq \\
    --atomic_chgcar_dir /path/to/atomic_chgcar \\
    --density_synthesis_mode squared --no-compute_flow_prior_stats

  # Match QM9 flow/baseline training (beta=1.3):
  python scdp/scripts/validate_gt_coeffs.py ... --beta 1.3 --vnode_elem 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Subset
from tqdm.auto import tqdm

from scdp.common.pyg import DataLoader
from scdp.data.datamodule import (
    filter_split_indices,
    load_gt_coeff_excluded_indices,
    worker_init_fn,
)
from scdp.data.gt_coeff_exclusions import read_lmdb_shard_lengths, shard_offsets
from scdp.data.dataset import LmdbDataset
from scdp.data.gt_coeff_projection import (
    load_pbc_from_lmdb_dir,
    load_target_var_from_metadata,
    make_pack_for_lmdb,
    project_labels_to_gt_coeffs,
)
from scdp.data.sad_density import AtomicDensityLibrary, superposed_sad_density_at_probes
from scdp.model.density_synthesis import resolve_density_synthesis_mode
from scdp.model.coeff_flow_matching import (
    FlowPriorStatsAccumulator,
    accumulate_flow_prior_stats_batch,
    coeff_norm_denominator,
    num_graphs_in_batch,
)
from scdp.model.module import ChgLightningModule
from scdp.model.utils import get_nmape, get_probe_chunks

pylogger = logging.getLogger(__name__)


def _setup():
    torch.set_float32_matmul_precision("high")


def batched_n_probe(batch) -> torch.Tensor:
    """``get_probe_chunks`` expects a 1D tensor (one count per graph)."""
    n = batch.n_probe
    dev = batch.coords.device
    if isinstance(n, int):
        return torch.tensor([n], device=dev, dtype=torch.long)
    if torch.is_tensor(n):
        if n.dim() == 0:
            return n.reshape(1).to(device=dev, dtype=torch.long)
        return n.to(device=dev, dtype=torch.long)
    return torch.tensor([int(n)], device=dev, dtype=torch.long)


@torch.no_grad()
def coeffs_for_density(
    model: ChgLightningModule,
    batch,
    coeff_field: str = "gt_coeffs",
) -> torch.Tensor:
    """Normalize stored raw sidecar coeffs like flow / predict_coeffs."""
    raw = batch[coeff_field].to(device=batch.coords.device, dtype=batch.coords.dtype)
    norm = coeff_norm_denominator(batch, model.n_orbitals, model.unique_atom_types)
    return raw / norm


@torch.no_grad()
def predict_density_from_stored_coeffs(
    model: ChgLightningModule,
    batch,
    max_n_probe_per_pass: int,
    coeff_field: str = "gt_coeffs",
) -> torch.Tensor:
    coeffs = coeffs_for_density(model, batch, coeff_field=coeff_field)
    n_pass, n_per_pass, probes_to_process = get_probe_chunks(
        batched_n_probe(batch), max_n_probe_per_pass
    )
    preds = []
    for i_pass in range(n_pass):
        n_probe = n_per_pass[i_pass]
        probe_idx = probes_to_process[i_pass]
        probe_coords = batch.probe_coords[probe_idx]
        preds.append(
            model.orbital_inference(batch, coeffs, None, n_probe, probe_coords)
        )
    return torch.cat(preds, dim=0)


@torch.no_grad()
def predict_density_from_gt_coeffs(
    model: ChgLightningModule,
    batch,
    max_n_probe_per_pass: int,
) -> torch.Tensor:
    return predict_density_from_stored_coeffs(
        model, batch, max_n_probe_per_pass, coeff_field="gt_coeffs"
    )


@torch.no_grad()
def sad_density_labels_batched(
    batch,
    library: AtomicDensityLibrary,
) -> torch.Tensor:
    """Superposed atomic density on the same probe layout as ``batch.chg_labels``."""
    n_probe = batched_n_probe(batch)
    offsets = torch.zeros(n_probe.shape[0] + 1, dtype=torch.long, device=n_probe.device)
    offsets[1:] = n_probe.cumsum(0)
    parts = []
    is_vnode_all = getattr(batch, "is_vnode", None)
    for g in range(num_graphs_in_batch(batch)):
        mask = batch.batch == g
        atom_types = batch.atom_types[mask]
        coords = batch.coords[mask]
        start, end = int(offsets[g]), int(offsets[g + 1])
        probe_coords = batch.probe_coords[start:end]
        is_vnode = is_vnode_all[mask] if is_vnode_all is not None else None
        rho = superposed_sad_density_at_probes(
            atom_types, coords, probe_coords, library, is_vnode=is_vnode
        )
        parts.append(rho.to(device=batch.coords.device, dtype=batch.coords.dtype))
    return torch.cat(parts, dim=0)


def target_density_for_batch(
    batch,
    coeff_field: str,
    atomic_library: Optional[AtomicDensityLibrary],
) -> torch.Tensor:
    if coeff_field == "gt_coeffs":
        return batch.chg_labels
    if coeff_field == "sad_coeffs":
        if atomic_library is None:
            raise ValueError("atomic_library is required when coeff_field='sad_coeffs'")
        return sad_density_labels_batched(batch, atomic_library)
    raise ValueError(f"coeff_field must be 'gt_coeffs' or 'sad_coeffs', got {coeff_field!r}")


def graph_index_tensor(batch) -> torch.Tensor:
    ng = num_graphs_in_batch(batch)
    return torch.arange(ng, device=batch.coords.device).repeat_interleave(
        batched_n_probe(batch)
    )


@torch.no_grad()
def metrics_for_batch(
    model: ChgLightningModule,
    batch,
    max_n_probe_per_pass: int,
    coeff_field: str = "gt_coeffs",
    atomic_library: Optional[AtomicDensityLibrary] = None,
) -> dict:
    pred = predict_density_from_stored_coeffs(
        model, batch, max_n_probe_per_pass, coeff_field=coeff_field
    )
    target = target_density_for_batch(batch, coeff_field, atomic_library)
    gidx = graph_index_tensor(batch)
    scale = model.scale

    nmape_per = get_nmape(pred, target, gidx).cpu()
    diff_scaled = pred / scale - target / scale
    mae_scaled = diff_scaled.abs().mean().item()
    rmse_scaled = diff_scaled.pow(2).mean().sqrt().item()

    raw = batch[coeff_field].to(device=batch.coords.device, dtype=batch.coords.dtype)
    norm = coeff_norm_denominator(batch, model.n_orbitals, model.unique_atom_types)
    coeff_norm = raw / norm
    mask = model.orb_index[batch.atom_types.long()].bool()
    active = coeff_norm[mask]
    out = {
        "nmape": nmape_per.numpy().tolist(),
        "mae_scaled": float(mae_scaled),
        "rmse_scaled": float(rmse_scaled),
        "nmape_mean": float(nmape_per.mean().item()),
        "coeff_raw_min": float(raw.min().item()),
        "coeff_norm_min": float(active.min().item()) if active.numel() else float("nan"),
    }
    return out


@torch.no_grad()
def recompute_coeff_metrics(
    model: ChgLightningModule,
    data,
    scale: float,
    ridge: float,
    ridge_diag_frac: float,
    probe_weighting: Optional[str],
    max_probe_samples: int,
    device: str,
    dtype_name: str,
    chunk_size: int,
    accumulate_on_cpu: Optional[bool] = None,
    coeff_field: str = "gt_coeffs",
    density_synthesis_mode: str = "linear",
    atomic_library: Optional[AtomicDensityLibrary] = None,
) -> dict:
    dev = torch.device(device)
    data = data.clone().to(dev)
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.atom_types.shape[0], dtype=torch.long, device=dev)
    else:
        data.batch = data.batch.to(dev)
    pack = make_pack_for_lmdb(
        model.unique_atom_types.cpu().tolist(),
        dft_basis_set=model.hparams.dft_basis_set,
        dft_wt_aug=model.hparams.dft_wt_aug,
        beta=float(model.hparams.beta),
        lmax_restriction=model.hparams.lmax_restriction,
        uncontracted=model.hparams.uncontracted,
        orb_cutoff=float(model.hparams.orb_cutoff),
        vnode_elem=int(model.hparams.vnode_elem),
        device=device,
        density_synthesis_mode=density_synthesis_mode,
    )
    cell = data.cell
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
    if coeff_field == "sad_coeffs":
        if atomic_library is None:
            raise ValueError("atomic_library is required when coeff_field='sad_coeffs'")
        is_vnode = getattr(data, "is_vnode", None)
        label_rho = superposed_sad_density_at_probes(
            data.atom_types,
            data.coords,
            data.probe_coords,
            atomic_library,
            is_vnode=is_vnode,
        )
    else:
        label_rho = data.chg_labels
    gt_re = project_labels_to_gt_coeffs(
        data.atom_types,
        data.coords,
        cell,
        bool(getattr(data, "pbc", False)),
        data.probe_coords,
        label_rho,
        pack,
        scale=scale,
        ridge=ridge,
        ridge_diag_frac=ridge_diag_frac,
        probe_weighting=probe_weighting,
        max_probe_samples=max_probe_samples,
        seed=0,
        device=device,
        dtype_name=dtype_name,
        chunk_size=chunk_size,
        accumulate_on_cpu=accumulate_on_cpu,
        density_synthesis_mode=density_synthesis_mode,
    ).to(device=dev, dtype=data.coords.dtype)

    gt_stored = data[coeff_field].to(device=dev, dtype=data.coords.dtype)
    diff = (gt_re - gt_stored).pow(2).sum()
    denom = gt_stored.pow(2).sum().clamp(min=1e-20)
    coeff_rel_err = float((diff / denom).sqrt().item())

    data_re = data.clone() if hasattr(data, "clone") else data
    data_re[coeff_field] = gt_re
    data_re.batch = data.batch
    batch_re = data_re
    n_probe_t = data.n_probe
    if isinstance(n_probe_t, int):
        n_probe_val = n_probe_t
    elif n_probe_t.dim() == 0:
        n_probe_val = int(n_probe_t.item())
    else:
        n_probe_val = int(n_probe_t.sum())
    pred_re = predict_density_from_stored_coeffs(
        model,
        batch_re,
        max_n_probe_per_pass=max(n_probe_val, 1),
        coeff_field=coeff_field,
    )
    target_re = target_density_for_batch(batch_re, coeff_field, atomic_library)
    nmape_re = float(
        get_nmape(
            pred_re,
            target_re,
            torch.zeros(target_re.shape[0], dtype=torch.long, device=pred_re.device),
        )
        .mean()
        .item()
    )

    return {
        "coeff_storage_rel_l2": coeff_rel_err,
        "nmape_recomputed_coeffs": nmape_re,
        "coeff_recomputed_min": float(gt_re.min().item()),
    }


def build_model(metadata: dict, args: argparse.Namespace, device: str) -> ChgLightningModule:
    if "unique_atom_types" not in metadata:
        raise KeyError("metadata.json must contain 'unique_atom_types'")
    if "avg_num_neighbors" not in metadata:
        raise KeyError("metadata.json must contain 'avg_num_neighbors'")

    train_cfg = OmegaConf.create(
        {
            "ema": {"decay": 0.995},
            "trainer": {"strategy": "auto", "devices": 1},
        }
    )
    model = ChgLightningModule(
        model={
            "_target_": "scdp.model.scn.eSCN",
            "num_layers": 1,
            "lmax_list": [args.lmax],
            "mmax_list": [2],
            "cutoff": 6.0,
            "sphere_channels": 32,
            "hidden_channels": 64,
            "edge_channels": 32,
            "num_sphere_samples": 32,
            "enable_flow_coeff_entangle": False,
        },
        train=train_cfg,
        criterion="mae",
        vnode_elem=args.vnode_elem,
        pbc=bool(getattr(args, "pbc", False)),
        dft_basis_set=args.dft_basis_set,
        dft_wt_aug=not args.no_dft_wt_aug,
        beta=args.beta,
        lmax_restriction=not args.no_lmax_restriction,
        lmax_relax=0,
        uncontracted=not args.contracted,
        orb_cutoff=args.orb_cutoff,
        expo_trainable=False,
        magnitude_weighting=False,
        density_synthesis_mode=resolve_density_synthesis_mode(
            getattr(args, "density_synthesis_mode", None)
        ),
        metadata=metadata,
    )
    return model.to(device).eval()


def _resolve_split_indices(
    split_file: Path,
    tag: str,
    gt_coeffs_path: Optional[Path] = None,
    lmdb_path: Optional[Path] = None,
) -> List[int]:
    with open(split_file, "r") as fp:
        splits = json.load(fp)
    key = tag
    if key == "val":
        key = "validation"
    if key not in splits:
        raise KeyError(f"split '{tag}' not in {split_file}; keys={list(splits)}")
    indices = list(splits[key])
    excluded = load_gt_coeff_excluded_indices(
        gt_coeffs_path, lmdb_in_path=lmdb_path
    )
    if excluded:
        indices = filter_split_indices(indices, excluded)
    return indices


@torch.no_grad()
def compute_flow_prior_stats(
    dataset: LmdbDataset,
    model: ChgLightningModule,
    indices: List[int],
    max_graphs: int,
    batch_size: int,
    device: str,
    *,
    include_residual: bool,
) -> dict:
    """Aggregate masked normalized coeff stats on ``indices`` (typically train split)."""
    n_use = min(max_graphs, len(indices))
    subset = Subset(dataset, indices[:n_use])
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=worker_init_fn,
    )
    accum_gt = FlowPriorStatsAccumulator()
    accum_res = FlowPriorStatsAccumulator() if include_residual else None
    n_graphs = 0
    for batch in tqdm(loader, desc="flow prior stats"):
        batch = batch.to(device)
        accumulate_flow_prior_stats_batch(
            batch,
            accum_gt,
            accum_res,
            model.n_orbitals,
            model.unique_atom_types,
            model.orb_index,
        )
        n_graphs += batch.num_graphs

    gt_stats = accum_gt.finalize()
    out = {
        "n_graphs": n_graphs,
        "n_coeff_elements": gt_stats.get("count", 0),
        "space": "normalized_gt_coeffs (gt_raw / coeff_norm_denominator, masked active GTOs)",
        "gt": gt_stats,
        "hydra_overrides": {},
    }
    if gt_stats:
        out["hydra_overrides"] = {
            "model.flow_prior_mu": gt_stats["suggested_flow_prior_mu"],
            "model.flow_prior_sigma": gt_stats["suggested_flow_prior_sigma"],
        }

    if accum_res is not None:
        res_stats = accum_res.finalize()
        out["residual_gt_minus_sad"] = res_stats
        if res_stats:
            out["hydra_overrides_res_target"] = {
                "model.flow_prior_mu": res_stats["suggested_flow_prior_mu"],
                "model.flow_prior_sigma": res_stats["suggested_flow_prior_sigma"],
            }
    else:
        out["residual_gt_minus_sad"] = None
        out["note_residual"] = (
            "Pass --sad_coeffs_path to also estimate prior stats for flow_target_mode=res "
            "(normalized c_gt - c_sad)."
        )
    return out


def main():
    _setup()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lmdb_path", type=Path, required=True)
    p.add_argument("--gt_coeffs_path", type=Path, default=None)
    p.add_argument(
        "--sad_coeffs_path",
        type=Path,
        default=None,
        help="Optional SAD sidecar for residual prior stats (flow_target_mode=res)",
    )
    p.add_argument("--metadata_json", type=Path, required=True)
    p.add_argument("--split_file", type=Path, required=True)
    p.add_argument(
        "--tag",
        type=str,
        default="validation",
        help="Split key in datasplits.json (train / validation / test; alias val→validation)",
    )
    p.add_argument(
        "--shard",
        type=int,
        default=None,
        help="Validate only graphs in data.NNNN.lmdb (e.g. 0 → global indices in shard 0)",
    )
    p.add_argument(
        "--graph_indices",
        type=str,
        default=None,
        help="Comma-separated global LMDB indices (overrides --tag and --shard)",
    )
    p.add_argument("--max_n_graphs", type=int, default=500)
    p.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Randomly sample max_n_graphs with this seed; omit for first-N order in split",
    )
    p.add_argument(
        "--coeff_field",
        type=str,
        default="gt_coeffs",
        choices=("gt_coeffs", "sad_coeffs"),
        help="Sidecar field to validate (gt vs DFT labels, sad vs superposed atomic density)",
    )
    p.add_argument(
        "--atomic_chgcar_dir",
        type=Path,
        default=None,
        help="Required when --coeff_field sad_coeffs (isolated-atom CHGCAR refs)",
    )
    p.add_argument(
        "--density_synthesis_mode",
        choices=("linear", "squared"),
        default="linear",
        help="Must match sidecar precompute (squared: d on phi^2 via NNLS)",
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_n_probe_per_pass", type=int, default=400000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dir", type=Path, default=None, help="Defaults to gt_coeffs_path parent")
    p.add_argument("--beta", type=float, default=1.3, help="Must match gt_coeffs projection + training")
    p.add_argument("--vnode_elem", type=int, default=8)
    p.add_argument("--dft_basis_set", type=str, default="def2-qzvppd")
    p.add_argument("--orb_cutoff", type=float, default=5.0)
    p.add_argument("--lmax", type=int, default=6)
    p.add_argument(
        "--pbc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="PBC for orbital_inference validation (default: read units.json next to LMDB)",
    )
    p.add_argument("--no_dft_wt_aug", action="store_true")
    p.add_argument("--no_lmax_restriction", action="store_true")
    p.add_argument("--contracted", action="store_true")
    p.add_argument("--recompute_n", type=int, default=0, help="Re-project N random graphs for sanity")
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument(
        "--ridge_diag_frac",
        type=float,
        default=0.0,
        help="Set 0 for tight GT fit; 0.01 biases coefficients away from label grid",
    )
    p.add_argument("--probe_weighting", type=str, default="uniform", choices=("uniform", "abs_rho"))
    p.add_argument(
        "--max_probe_samples",
        type=int,
        default=-1,
        help="Probes for --recompute_n fresh projection (-1 = full grid)",
    )
    p.add_argument("--recompute_dtype", type=str, default="float64", choices=("float32", "float64"))
    p.add_argument("--chunk_size", type=int, default=4096)
    p.add_argument(
        "--nmape_threshold",
        type=float,
        default=0.02,
        help="Fail if summary nmape_mean exceeds this (percent); 0.02 = strict GT bar",
    )
    p.add_argument(
        "--fail_if_above_threshold",
        action="store_true",
        help="Exit 1 when nmape_mean > nmape_threshold",
    )
    p.add_argument(
        "--compute_flow_prior_stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan training split for suggested flow_prior_mu / flow_prior_sigma (default: on)",
    )
    p.add_argument(
        "--prior_stats_tag",
        type=str,
        default="train",
        help="Split key for prior statistics (default: train)",
    )
    p.add_argument(
        "--prior_stats_max_graphs",
        type=int,
        default=10000,
        help="Max training graphs for prior stats (default: 10000, use all if smaller)",
    )
    args = p.parse_args()
    coeff_field = str(args.coeff_field)
    synth_mode = resolve_density_synthesis_mode(args.density_synthesis_mode)

    if coeff_field == "gt_coeffs" and args.gt_coeffs_path is None:
        raise ValueError("--gt_coeffs_path is required when --coeff_field gt_coeffs")
    if coeff_field == "sad_coeffs":
        if args.sad_coeffs_path is None:
            raise ValueError("--sad_coeffs_path is required when --coeff_field sad_coeffs")
        if args.atomic_chgcar_dir is None:
            raise ValueError("--atomic_chgcar_dir is required when --coeff_field sad_coeffs")

    with open(args.metadata_json, "r") as fp:
        metadata = json.load(fp)
    target_var = float(metadata["target_var"])
    scale = target_var**0.5
    lmdb_root = args.lmdb_path if args.lmdb_path.is_dir() else args.lmdb_path.parent
    if args.pbc is None:
        args.pbc = load_pbc_from_lmdb_dir(lmdb_root)
    pylogger.info(
        "target_var=%.6e scale=%.6e beta=%.3f pbc=%s coeff_field=%s density_synthesis_mode=%s",
        target_var,
        scale,
        args.beta,
        args.pbc,
        coeff_field,
        synth_mode,
    )

    lmdb_path = args.lmdb_path
    sidecar_path = args.gt_coeffs_path
    if coeff_field == "sad_coeffs":
        sidecar_path = args.sad_coeffs_path
    if args.shard is not None:
        sn = int(args.shard)
        if sn < 0:
            raise ValueError("--shard must be >= 0")
        shard_name = f"data.{sn:04d}.lmdb"
        lmdb_path = args.lmdb_path / shard_name if args.lmdb_path.is_dir() else args.lmdb_path
        sidecar_path = (
            sidecar_path / shard_name if sidecar_path.is_dir() else sidecar_path
        )
        if not lmdb_path.is_file():
            raise FileNotFoundError(f"Main LMDB shard not found: {lmdb_path}")
        if not sidecar_path.is_file():
            raise FileNotFoundError(
                f"Sidecar shard not found: {sidecar_path} "
                "(build this shard before validating it)"
            )

    dataset = LmdbDataset(
        path=lmdb_path,
        gt_coeffs_path=args.gt_coeffs_path if coeff_field == "gt_coeffs" else None,
        sad_coeffs_path=args.sad_coeffs_path if coeff_field == "sad_coeffs" else args.sad_coeffs_path,
    )
    if args.graph_indices is not None:
        indices = [int(x.strip()) for x in args.graph_indices.split(",") if x.strip()]
        tag = args.tag if args.tag else "custom"
    elif args.shard is not None:
        sn = int(args.shard)
        gt_path = args.gt_coeffs_path if coeff_field == "gt_coeffs" else args.sad_coeffs_path
        excluded = load_gt_coeff_excluded_indices(gt_path, lmdb_in_path=args.lmdb_path)
        offsets = shard_offsets(read_lmdb_shard_lengths(args.lmdb_path))
        base, end = offsets[sn], offsets[sn + 1]
        indices = [g for g in range(base, end) if g not in excluded]
        tag = f"shard{sn:04d}"
    else:
        indices = _resolve_split_indices(
            args.split_file,
            args.tag,
            gt_coeffs_path=args.gt_coeffs_path if coeff_field == "gt_coeffs" else args.sad_coeffs_path,
            lmdb_path=args.lmdb_path,
        )
    pool = list(indices)
    n_eval = min(args.max_n_graphs, len(pool))
    if args.random_seed is not None:
        rng = np.random.default_rng(int(args.random_seed))
        pick = rng.choice(len(pool), size=n_eval, replace=False)
        eval_indices = [pool[int(i)] for i in pick]
    else:
        eval_indices = pool[:n_eval]
    subset = Subset(dataset, eval_indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=worker_init_fn,
    )

    device = args.device if torch.cuda.is_available() else "cpu"
    model = build_model(metadata, args, device)

    atomic_library = None
    if coeff_field == "sad_coeffs":
        atomic_library = AtomicDensityLibrary.from_directory(args.atomic_chgcar_dir)

    out_dir = args.out_dir or sidecar_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{args.tag}_{coeff_field}_{synth_mode}"
    per_graph_path = out_dir / f"coeff_validate_{run_tag}.jsonl"
    nmape_txt = out_dir / f"coeff_nmape_{run_tag}.txt"

    all_nmape: List[float] = []
    all_mae: List[float] = []
    all_rmse: List[float] = []
    all_coeff_min: List[float] = []
    n_graphs = 0

    with open(per_graph_path, "w") as jf, open(nmape_txt, "w") as nf:
        for batch in tqdm(loader, desc=f"validate {coeff_field} ({run_tag})"):
            batch = batch.to(device)
            m = metrics_for_batch(
                model,
                batch,
                args.max_n_probe_per_pass,
                coeff_field=coeff_field,
                atomic_library=atomic_library,
            )
            for v in m["nmape"]:
                all_nmape.append(float(v))
                nf.write(f"{v}\n")
            all_mae.append(m["mae_scaled"])
            all_rmse.append(m["rmse_scaled"])
            all_coeff_min.append(m["coeff_norm_min"])
            n_graphs += batch.num_graphs
            rec = {
                "nmape": m["nmape"],
                "nmape_mean": m["nmape_mean"],
                "mae_scaled": m["mae_scaled"],
                "rmse_scaled": m["rmse_scaled"],
                "coeff_raw_min": m["coeff_raw_min"],
                "coeff_norm_min": m["coeff_norm_min"],
            }
            jf.write(json.dumps(rec) + "\n")

    summary = {
        "tag": args.tag,
        "coeff_field": coeff_field,
        "density_synthesis_mode": synth_mode,
        "random_seed": args.random_seed,
        "n_graphs": n_graphs,
        "beta": args.beta,
        "ridge": args.ridge,
        "target_var": target_var,
        "sidecar_path": str(sidecar_path),
        "nmape_mean": float(np.mean(all_nmape)),
        "nmape_std": float(np.std(all_nmape)),
        "nmape_median": float(np.median(all_nmape)),
        "nmape_p95": float(np.percentile(all_nmape, 95)),
        "nmape_max": float(np.max(all_nmape)),
        "mae_scaled_mean": float(np.mean(all_mae)),
        "rmse_scaled_mean": float(np.mean(all_rmse)),
        "coeff_norm_min_over_graphs": float(np.min(all_coeff_min)),
        "nmape_threshold": args.nmape_threshold,
        "pass_threshold": float(np.mean(all_nmape)) <= args.nmape_threshold,
        "interpretation": (
            f"Reconstruct density from stored {coeff_field} via orbital_inference "
            f"(mode={synth_mode}) vs "
            f"{'chg_labels' if coeff_field == 'gt_coeffs' else 'superposed SAD density'}. "
            f"Target nMAPE mean < {args.nmape_threshold}% on full probe grid. "
            "Large errors usually mean basis/beta/ridge mismatch, subsampled fit, or "
            "density_synthesis_mode mismatch with the sidecar."
        ),
    }
    if synth_mode == "squared" and summary["coeff_norm_min_over_graphs"] < -1e-6:
        summary["nonneg_warning"] = (
            "Squared mode expects d >= 0; negative normalized coeffs found in sample."
        )
    summary_path = out_dir / f"coeff_validate_{run_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== {coeff_field} validation summary ({synth_mode}) ===")
    for k, v in summary.items():
        if k != "interpretation":
            print(f"  {k}: {v}")
    print(f"  {summary['interpretation']}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {nmape_txt}")

    if args.fail_if_above_threshold:
        if summary["nmape_mean"] > args.nmape_threshold:
            print(
                f"\nFAILED: nmape_mean={summary['nmape_mean']:.5f}% > "
                f"threshold={args.nmape_threshold}%"
            )
            sys.exit(1)
        print(
            f"\nPASSED threshold: nmape_mean={summary['nmape_mean']:.5f}% "
            f"<= {args.nmape_threshold}%"
        )

    if args.compute_flow_prior_stats and coeff_field == "gt_coeffs":
        train_indices = _resolve_split_indices(
            args.split_file,
            args.prior_stats_tag,
            gt_coeffs_path=args.gt_coeffs_path,
            lmdb_path=args.lmdb_path,
        )
        include_residual = args.sad_coeffs_path is not None
        prior_stats = compute_flow_prior_stats(
            dataset,
            model,
            train_indices,
            max_graphs=args.prior_stats_max_graphs,
            batch_size=args.batch_size,
            device=device,
            include_residual=include_residual,
        )
        prior_stats["prior_stats_tag"] = args.prior_stats_tag
        prior_stats["beta"] = args.beta
        prior_path = out_dir / "gt_coeff_flow_prior_stats.json"
        prior_path.write_text(json.dumps(prior_stats, indent=2))
        print("\n=== flow prior stats (normalized masked gt_coeffs) ===")
        gt = prior_stats.get("gt") or {}
        if gt:
            print(f"  n_graphs={prior_stats['n_graphs']}  coeff_elements={gt.get('count')}")
            print(f"  mean={gt.get('mean'):.6g}  std={gt.get('std'):.6g}  rms={gt.get('rms'):.6g}")
            print(
                "  suggested: "
                f"model.flow_prior_mu={gt.get('suggested_flow_prior_mu'):.6g} "
                f"model.flow_prior_sigma={gt.get('suggested_flow_prior_sigma'):.6g}"
            )
        res = prior_stats.get("residual_gt_minus_sad")
        if res:
            print("  residual (gt - sad):")
            print(
                f"    mean={res.get('mean'):.6g}  std={res.get('std'):.6g}  "
                f"sigma_suggest={res.get('suggested_flow_prior_sigma'):.6g}"
            )
        elif prior_stats.get("note_residual"):
            print(f"  {prior_stats['note_residual']}")
        print(f"Wrote {prior_path}")

    if args.recompute_n > 0:
        pylogger.info("Recomputing projection on %s random graphs", args.recompute_n)
        rng = np.random.default_rng(
            int(args.random_seed) if args.random_seed is not None else 0
        )
        pick = rng.choice(eval_indices, size=min(args.recompute_n, len(eval_indices)), replace=False)
        recomp = []
        for i in pick:
            data = dataset[int(i)].to(device)
            recomp.append(
                recompute_coeff_metrics(
                    model,
                    data,
                    scale=scale,
                    ridge=args.ridge,
                    ridge_diag_frac=args.ridge_diag_frac,
                    probe_weighting=args.probe_weighting
                    if args.probe_weighting != "uniform"
                    else None,
                    max_probe_samples=args.max_probe_samples,
                    device=device,
                    dtype_name=args.recompute_dtype,
                    chunk_size=args.chunk_size,
                    coeff_field=coeff_field,
                    density_synthesis_mode=synth_mode,
                    atomic_library=atomic_library,
                )
            )
        rc_summary = {
            "coeff_field": coeff_field,
            "density_synthesis_mode": synth_mode,
            "n": len(recomp),
            "coeff_storage_rel_l2_mean": float(
                np.mean([r["coeff_storage_rel_l2"] for r in recomp])
            ),
            "nmape_recomputed_mean": float(
                np.mean([r["nmape_recomputed_coeffs"] for r in recomp])
            ),
            "coeff_recomputed_min": float(
                np.min([r["coeff_recomputed_min"] for r in recomp])
            ),
        }
        rc_path = out_dir / f"coeff_recompute_{run_tag}.json"
        rc_path.write_text(json.dumps(rc_summary, indent=2))
        print("\n=== recompute sanity (stored vs fresh projection) ===")
        print(json.dumps(rc_summary, indent=2))
        print(f"Wrote {rc_path}")


if __name__ == "__main__":
    main()
