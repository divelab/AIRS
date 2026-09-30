#!/usr/bin/env python3
"""CUDA-synced inference timing for an SCDP (ChgLightningModule) checkpoint.

Mirrors scdp/scripts/bench_flow_infer.py but uses direct coeff regression.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import omegaconf
import torch
from lightning.pytorch import seed_everything
from torch.utils.data import Subset

from scdp.common.checkpoint_utils import resolve_eval_ckpt
from scdp.common.pyg import DataLoader
from scdp.data.datamodule import worker_init_fn
from scdp.data.dataset import LmdbDataset
from scdp.model.module import ChgLightningModule
from scdp.model.utils import get_nmape, get_probe_chunks
from scdp.model.gtos import apply_inference_gto_kernels


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def _time_batch(model, batch, max_n_probe: int):
    batch = batch.to("cuda", non_blocking=True)
    n_graphs = int(batch.num_graphs)

    _sync()
    t0 = time.perf_counter()
    coeffs, expo_scaling = model.predict_coeffs(batch)
    _sync()
    t1 = time.perf_counter()

    n_pass, n_per_pass, probes_to_process = get_probe_chunks(batch.n_probe, max_n_probe)
    all_preds = []
    for i_pass in range(n_pass):
        n_probe = n_per_pass[i_pass]
        probe_idx = probes_to_process[i_pass]
        probe_coords = batch.probe_coords[probe_idx]
        pred = model.orbital_inference(
            batch, coeffs, expo_scaling, n_probe, probe_coords
        )
        all_preds.append(pred)
    all_preds = torch.cat(all_preds, dim=0)
    nmape = (
        get_nmape(
            all_preds,
            batch.chg_labels,
            torch.arange(len(batch), device=all_preds.device).repeat_interleave(
                batch.n_probe
            ),
        )
        .detach()
        .cpu()
        .numpy()
        .tolist()
    )
    _sync()
    t2 = time.perf_counter()

    n_probe_total = (
        int(batch.n_probe.sum().item())
        if torch.is_tensor(batch.n_probe)
        else int(sum(batch.n_probe))
    )
    return {
        "n_graphs": n_graphs,
        "coeff_s": t1 - t0,
        "density_s": t2 - t1,
        "total_s": t2 - t0,
        "nmapes": nmape,
        "n_probe": n_probe_total,
    }


def _summarize(times_s, label: str):
    arr = np.asarray(times_s, dtype=np.float64)
    return {
        "label": label,
        "n": int(arr.size),
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std()),
        "median_s": float(np.median(arr)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "p90_s": float(np.percentile(arr, 90)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=Path, required=True)
    p.add_argument("--data_path", type=Path, required=True)
    p.add_argument("--split_file", type=Path, required=True)
    p.add_argument("--tag", type=str, default="test")
    p.add_argument("--max_n_graphs", type=int, default=0, help="0 = full split after warmup")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_n_probe", type=int, default=500000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_last", action="store_true")
    p.add_argument("--use_latest_epoch", action="store_true")
    p.add_argument("--model_name", type=str, default="scdp")
    p.add_argument("--molecule", type=str, default=None)
    p.add_argument("--json_out", type=Path, default=None)
    args = p.parse_args()

    seed_everything(args.seed)
    assert torch.cuda.is_available(), "CUDA required for this benchmark"

    cfg = omegaconf.OmegaConf.load(args.ckpt_path / "config.yaml")
    ds_cfg = cfg.data.dataset
    dataset_kw = {"path": str(args.data_path)}
    if ds_cfg.get("sad_coeffs_path"):
        dataset_kw["sad_coeffs_path"] = ds_cfg.sad_coeffs_path
    dataset = LmdbDataset(**dataset_kw)

    with open(args.split_file) as fp:
        splits = json.load(fp)
    split_indices = list(splits[args.tag])
    max_n = len(split_indices) - args.warmup if args.max_n_graphs <= 0 else args.max_n_graphs
    n_need = min(max_n + args.warmup, len(split_indices))
    selected = split_indices[:n_need]
    warmup_idx = selected[: args.warmup]
    timed_idx = selected[args.warmup : args.warmup + max_n]

    use_best = not args.use_latest_epoch and not args.use_last
    ckpt, score = resolve_eval_ckpt(
        args.ckpt_path, use_last=args.use_last, use_best=use_best
    )
    print(f"checkpoint: {ckpt}", flush=True)
    if score is not None:
        print(f"monitor_score: {score:.6f}", flush=True)

    model = ChgLightningModule.load_from_checkpoint(checkpoint_path=ckpt).to("cuda")
    model.eval()
    model.ema.copy_to(model.parameters())
    triton_on = apply_inference_gto_kernels(model)
    print(f"gto_triton_coeff_eval: {triton_on}", flush=True)

    gpu_name = torch.cuda.get_device_name(0)
    print(
        f"GPU={gpu_name}  batch_size={args.batch_size}  "
        f"max_n_probe={args.max_n_probe}  warmup={len(warmup_idx)}  "
        f"timed_graphs={len(timed_idx)}",
        flush=True,
    )

    def _make_loader(indices):
        return DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            worker_init_fn=worker_init_fn,
        )

    for batch in _make_loader(warmup_idx):
        stats = _time_batch(model, batch, args.max_n_probe)
        print(
            f"warmup: graphs={stats['n_graphs']}  total={stats['total_s']:.4f}s  "
            f"nmape_mean={np.mean(stats['nmapes']):.4f}",
            flush=True,
        )

    timed_coeff = []
    timed_density = []
    timed_total = []
    timed_nmapes = []
    n_probe_list = []
    for batch in _make_loader(timed_idx):
        stats = _time_batch(model, batch, args.max_n_probe)
        per_mol = stats["n_graphs"]
        timed_coeff.append(stats["coeff_s"] / per_mol)
        timed_density.append(stats["density_s"] / per_mol)
        timed_total.append(stats["total_s"] / per_mol)
        timed_nmapes.extend(stats["nmapes"])
        n_probe_list.append(stats["n_probe"] / per_mol)

    summaries = {
        "coeff_per_mol": _summarize(timed_coeff, "predict_coeffs"),
        "density_per_mol": _summarize(timed_density, "orbital_inference"),
        "total_per_mol": _summarize(timed_total, "coeff+density"),
    }

    print("\n=== SCDP inference timing (CUDA-synced, amortized per molecule) ===", flush=True)
    print(f"checkpoint_dir: {args.ckpt_path}", flush=True)
    print(f"ckpt_file: {ckpt}", flush=True)
    print(f"gpu: {gpu_name}", flush=True)
    print(f"batch_size: {args.batch_size}", flush=True)
    print(f"n_timed_molecules: {len(timed_nmapes)}", flush=True)
    print(f"mean_probes_per_mol: {float(np.mean(n_probe_list)):.0f}", flush=True)
    if timed_nmapes:
        print(
            f"nMAPE%: mean={np.mean(timed_nmapes):.4f}  "
            f"std={np.std(timed_nmapes):.4f}",
            flush=True,
        )
    for key in ("coeff_per_mol", "density_per_mol", "total_per_mol"):
        s = summaries[key]
        print(
            f"{s['label']}: mean={s['mean_s']:.4f}s  median={s['median_s']:.4f}s  "
            f"std={s['std_s']:.4f}s  (n={s['n']})",
            flush=True,
        )

    out = {
        "model_name": args.model_name,
        "molecule": args.molecule,
        "ckpt_path": str(args.ckpt_path),
        "ckpt_file": str(ckpt),
        "gpu": gpu_name,
        "batch_size": args.batch_size,
        "max_n_probe": args.max_n_probe,
        "max_n_graphs": len(timed_idx),
        "warmup": args.warmup,
        "mean_probes_per_mol": float(np.mean(n_probe_list)),
        "nmape_mean": float(np.mean(timed_nmapes)) if timed_nmapes else None,
        "nmape_std": float(np.std(timed_nmapes)) if timed_nmapes else None,
        "summaries": summaries,
        "per_batch_total_s_per_mol": timed_total,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json_out}", flush=True)

    t = summaries["total_per_mol"]
    print(
        f"HEADLINE: mean_infer_s_per_mol={t['mean_s']:.6f}  "
        f"median={t['median_s']:.6f}  batch_size={args.batch_size}  gpu={gpu_name}",
        flush=True,
    )


if __name__ == "__main__":
    main()
