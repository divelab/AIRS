"""
Evaluate a flow-matching checkpoint on a data split (default: test).

Uses ``ChgFlowMatchingModule.predict_coeffs`` (Euler CFM integration by default) then
``orbital_inference`` for nMAPE — same probe chunking as ``scdp/scripts/test.py``.

Euler time grids: ``--integrate_schedule uniform|endpoint_refine|power`` (see
``build_flow_integrate_t_grid`` in ``coeff_flow_matching.py``).
"""

import logging
import random
import time
import json
import argparse
import contextlib
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

import omegaconf
import torch
from lightning.pytorch import seed_everything

from torch.utils.data import Subset
from scdp.common.pyg import DataLoader
from scdp.data.dataset import LmdbDataset
from scdp.data.datamodule import worker_init_fn
from scdp.model.utils import get_nmape, get_probe_chunks
from scdp.common.checkpoint_utils import resolve_eval_ckpt
from scdp.model.coeff_flow_matching import FLOW_INTEGRATE_SCHEDULES, resolve_flow_integrate_schedule
from scdp.model.module_flow import ChgFlowMatchingModule
from scdp.model.gtos import apply_inference_gto_kernels

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def get_data_probe_chunk(input_data, indices):
    data = deepcopy(input_data)
    data["chg_labels"] = input_data.chg_labels[indices]
    data["probe_coords"] = input_data.probe_coords[indices]
    data["n_probe"] = len(indices)
    data["sampled"] = True
    return data


def needs_sad_coeffs(model: ChgFlowMatchingModule) -> bool:
    if model.flow_target_mode == "res":
        return True
    if model.flow_prior_mode == "sad_coeffs":
        return True
    return False


def _select_graph_indices(
    indices: List[int],
    max_graphs: int,
    *,
    random_sample: bool,
    sample_seed: int,
) -> Tuple[List[int], dict]:
    """Pick up to ``max_graphs`` dataset indices from the split."""
    n_pool = len(indices)
    n_use = min(max_graphs, n_pool)
    meta = {
        "split_size": n_pool,
        "n_graphs_selected": n_use,
        "random_sample": random_sample,
        "sample_seed": sample_seed if random_sample and n_use < n_pool else None,
    }
    if n_use >= n_pool:
        meta["sampling"] = "all"
        return list(indices), meta
    if random_sample:
        meta["sampling"] = "random_without_replacement"
        rng = random.Random(sample_seed)
        return rng.sample(indices, n_use), meta
    meta["sampling"] = "ordered_prefix"
    return indices[:n_use], meta


def flow_integrate_output_name(
    n_steps: int,
    schedule: str,
    out_tag: str,
) -> str:
    """Output filename for integrate-mode nMAPE (keeps legacy name for uniform)."""
    sched = resolve_flow_integrate_schedule(schedule)
    if sched == "uniform":
        return f"nmape_flow_integrate_s{n_steps}_{out_tag}.txt"
    return f"nmape_flow_integrate_s{n_steps}_{sched}_{out_tag}.txt"


def resolve_model_integrate_schedule(model: ChgFlowMatchingModule) -> str:
    return resolve_flow_integrate_schedule(
        getattr(model, "flow_inference_integrate_schedule", None)
        or getattr(model, "flow_integrate_schedule", "uniform")
    )


def main(
    ckpt_path,
    data_path,
    split_file,
    sad_coeffs_path=None,
    tag="test",
    output_tag=None,
    max_n_graphs=10000,
    batch_size=4,
    max_n_probe=500000,
    use_last=False,
    use_best=True,
    n_steps=2,
    coeff_mode="integrate",
    integrate_schedule=None,
    integrate_t_hi=None,
    integrate_power=None,
    seed=42,
    random_sample=False,
    sample_seed=42,
    write_nmape_txt=True,
    eval_log=None,
):
    seed_everything(seed)
    ckpt_path = Path(ckpt_path)
    coeff_mode = coeff_mode.strip().lower()
    if coeff_mode not in ("integrate", "readout"):
        raise ValueError(f"coeff_mode must be 'integrate' or 'readout', got {coeff_mode!r}")

    out_tag = output_tag if output_tag is not None else tag
    log_tag = out_tag if coeff_mode != "integrate" else f"{out_tag}_s{n_steps}"
    log_handlers: List[logging.Handler]
    if eval_log is None:
        log_path = ckpt_path / f"eval_flow_{log_tag}.log"
        log_handlers = [logging.FileHandler(log_path, mode="w")]
    elif str(eval_log).strip() in ("-", "stdout"):
        log_handlers = [logging.StreamHandler()]
    else:
        log_path = Path(eval_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handlers = [logging.FileHandler(log_path, mode="a" if log_path.is_file() else "w")]
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        handlers=log_handlers,
        force=True,
    )

    cfg = omegaconf.OmegaConf.load(ckpt_path / "config.yaml")
    ds_cfg = cfg.data.dataset
    sad_path = sad_coeffs_path
    if sad_path is None and ds_cfg.get("sad_coeffs_path"):
        sad_path = ds_cfg.sad_coeffs_path

    dataset_kw = {"path": data_path}
    if sad_path is not None:
        dataset_kw["sad_coeffs_path"] = sad_path
    dataset = LmdbDataset(**dataset_kw)

    with open(split_file, "r") as fp:
        splits = json.load(fp)
    split_indices = list(splits[tag])
    selected_indices, sampling_meta = _select_graph_indices(
        split_indices,
        max_n_graphs,
        random_sample=random_sample,
        sample_seed=sample_seed,
    )
    test_dataset = Subset(dataset, selected_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=worker_init_fn,
    )

    ckpt, ckpt_score = resolve_eval_ckpt(ckpt_path, use_last=use_last, use_best=use_best)
    if ckpt_score is not None:
        pylogger.info(f"checkpoint monitor score: {ckpt_score:.6f}")
    pylogger.info(f"loaded checkpoint: {ckpt}")
    model = ChgFlowMatchingModule.load_from_checkpoint(checkpoint_path=ckpt).to("cuda")
    model.eval()
    model.ema.copy_to(model.parameters())
    triton_on = apply_inference_gto_kernels(model, triton_default=True)
    pylogger.info("gto triton coeff eval (inference): %s", triton_on)
    model.flow_coeff_mode = coeff_mode
    model.flow_inference_n_steps = int(n_steps)

    ckpt_schedule = getattr(model, "flow_integrate_schedule", "uniform")
    effective_schedule = (
        integrate_schedule if integrate_schedule is not None else ckpt_schedule
    )
    effective_schedule = resolve_flow_integrate_schedule(effective_schedule)
    model.flow_inference_integrate_schedule = effective_schedule

    if integrate_t_hi is not None:
        model.flow_inference_integrate_t_hi = float(integrate_t_hi)
    if integrate_power is not None:
        model.flow_inference_integrate_power = float(integrate_power)

    effective_t_hi = float(
        integrate_t_hi
        if integrate_t_hi is not None
        else getattr(model, "flow_integrate_t_hi", 0.99)
    )
    effective_power = float(
        integrate_power
        if integrate_power is not None
        else getattr(model, "flow_integrate_power", 4.0)
    )

    if needs_sad_coeffs(model) and sad_path is None:
        raise ValueError(
            "This checkpoint needs SAD coefficients on each batch "
            "(flow_target_mode=res or flow_prior_mode=sad_coeffs). "
            "Pass --sad_coeffs_path or set data.dataset.sad_coeffs_path in the run config."
        )

    pylogger.info(
        "flow test: split=%s coeff_mode=%s n_steps=%s schedule=%s t_hi=%s power=%s "
        "graphs=%d/%d sampling=%s bridge=%s prior=%s target=%s loss=%s",
        tag,
        coeff_mode,
        n_steps,
        effective_schedule,
        effective_t_hi,
        effective_power,
        sampling_meta["n_graphs_selected"],
        sampling_meta["split_size"],
        sampling_meta["sampling"],
        model.bridge_mode,
        model.flow_prior_mode,
        model.flow_target_mode,
        model.flow_loss_mode,
    )
    if sampling_meta.get("sample_seed") is not None:
        pylogger.info("flow test: random_sample seed=%s", sampling_meta["sample_seed"])

    out_name = f"nmape_flow_{coeff_mode}_{out_tag}.txt"
    if coeff_mode == "integrate":
        out_name = flow_integrate_output_name(n_steps, effective_schedule, out_tag)

    storage_dir = ckpt_path
    nmape_out = storage_dir / out_name
    with torch.no_grad(), contextlib.ExitStack() as context_stack:
        if write_nmape_txt:
            f = context_stack.enter_context(open(nmape_out, "w"))
        else:
            f = None
        prog = context_stack.enter_context(
            tqdm(total=min(max_n_graphs, len(test_dataset)), disable=None)
        )
        display_bar = context_stack.enter_context(
            tqdm(
                bar_format=""
                if prog.disable
                else ("{desc:." + str(prog.ncols) + "}"),
                disable=None,
            )
        )

        curr_time = time.time()
        idx = 0
        all_nmapes = []
        for batch in test_loader:
            batch = batch.to("cuda")
            coeffs, expo_scaling = model.predict_coeffs(batch)
            n_pass, n_per_pass, probes_to_process = get_probe_chunks(
                batch.n_probe, max_n_probe
            )
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
            nmape = get_nmape(
                all_preds,
                batch.chg_labels,
                torch.arange(len(batch), device=all_preds.device).repeat_interleave(
                    batch.n_probe
                ),
            ).cpu().numpy().tolist()
            all_nmapes.extend(nmape)

            if f is not None:
                for item in nmape:
                    f.write(f"{item}\n")
                f.flush()
            prog.update(batch.num_graphs)
            display_bar.set_description_str(
                f"nmape: {np.mean(all_nmapes):.4f} ± {np.std(all_nmapes):.4f}"
            )
            idx += batch.num_graphs
            if idx >= len(test_dataset):
                break

        elapsed_time = time.time() - curr_time
        pylogger.info("elapsed time: %.2f seconds.", elapsed_time)
        if write_nmape_txt:
            pylogger.info("wrote per-graph nMAPE: %s", nmape_out)
        if all_nmapes:
            arr = np.asarray(all_nmapes, dtype=np.float64)
            summary = (
                f"flow test nMAPE split={tag} coeff_mode={coeff_mode} "
                f"n_steps={n_steps} schedule={effective_schedule}: "
                f"mean={arr.mean():.6f} std={arr.std():.6f} "
                f"min={arr.min():.6f} max={arr.max():.6f} n={len(arr)} "
                f"elapsed_s={elapsed_time:.1f} ckpt={ckpt}"
            )
            pylogger.info(summary)
            print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=Path, help="Training run directory.")
    parser.add_argument("--data_path", type=Path, help="Main LMDB path.")
    parser.add_argument("--split_file", type=Path, help="datasplits.json")
    parser.add_argument(
        "--sad_coeffs_path",
        type=Path,
        default=None,
        help="SAD coeffs sidecar LMDB (required for res / sad prior).",
    )
    parser.add_argument("--tag", type=str, default="test", help="Split key in splits JSON.")
    parser.add_argument(
        "--output_tag",
        type=str,
        default=None,
        help="Suffix for output files (default: same as --tag).",
    )
    parser.add_argument("--max_n_graphs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_n_probe", type=int, default=180000)
    parser.add_argument("--use_last", action="store_true")
    parser.add_argument(
        "--use_latest_epoch",
        action="store_true",
        help="Use highest epoch=*.ckpt instead of best val monitor score (default: best).",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=2,
        help="Euler steps for coeff_mode=integrate (default: 2).",
    )
    parser.add_argument(
        "--integrate_schedule",
        type=str,
        default=None,
        choices=FLOW_INTEGRATE_SCHEDULES,
        help=(
            "Euler time grid: uniform (legacy linspace), endpoint_refine (big jump to t_hi "
            "then refine), or power (front-loaded). Default: checkpoint config, else uniform."
        ),
    )
    parser.add_argument(
        "--integrate_t_hi",
        type=float,
        default=None,
        help="endpoint_refine: end of the first big jump (default: checkpoint or 0.99).",
    )
    parser.add_argument(
        "--integrate_power",
        type=float,
        default=None,
        help="power schedule exponent (default: checkpoint or 4.0).",
    )
    parser.add_argument(
        "--coeff_mode",
        type=str,
        default="integrate",
        choices=("integrate", "readout"),
        help="integrate: true flow sampling; readout: one-shot eSCN (val-style).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--random_sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Randomly sample max_n_graphs from the split (default: ordered prefix)",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="RNG seed for --random_sample",
    )
    parser.add_argument(
        "--no_nmape_txt",
        action="store_true",
        help="Do not write nmape_flow_*.txt under the checkpoint dir (log summary only).",
    )
    parser.add_argument(
        "--eval_log",
        type=str,
        default=None,
        help=(
            "Log destination: default writes eval_flow_<tag>.log under ckpt_path; "
            "'-' logs to stdout (e.g. Slurm .out); else append/create this file path."
        ),
    )
    args = parser.parse_args()
    main(
        args.ckpt_path,
        args.data_path,
        args.split_file,
        args.sad_coeffs_path,
        args.tag,
        args.output_tag,
        args.max_n_graphs,
        args.batch_size,
        args.max_n_probe,
        args.use_last,
        not args.use_latest_epoch,
        args.n_steps,
        args.coeff_mode,
        args.integrate_schedule,
        args.integrate_t_hi,
        args.integrate_power,
        args.seed,
        args.random_sample,
        args.sample_seed,
        write_nmape_txt=not args.no_nmape_txt,
        eval_log=args.eval_log,
    )
