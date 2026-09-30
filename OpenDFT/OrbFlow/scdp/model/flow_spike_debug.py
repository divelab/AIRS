"""
Diagnostics printed when flow training loss spikes abnormally.

Used by ``FlowLossSpikeCallback``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import IO, Optional, TextIO, Union

import torch

from scdp.model.coeff_flow_matching import (
    coeff_norm_denominator,
    per_node_coeff_mask,
    sample_flow_prior_c0_raw,
)

TextSink = Union[TextIO, IO[str]]


def _fmt(x: float) -> str:
    if abs(x) >= 1e4 or (abs(x) > 0 and abs(x) < 1e-4):
        return f"{x:.4e}"
    return f"{x:.6f}"


def _tensor_stats(name: str, t: torch.Tensor, mask: Optional[torch.Tensor] = None) -> str:
    if mask is not None:
        m = mask.bool()
        vals = t[m]
        n_active = int(m.sum().item())
    else:
        vals = t.reshape(-1)
        n_active = vals.numel()
    if n_active == 0:
        return f"{name}: (empty)"
    return (
        f"{name}: n={n_active} "
        f"mean={_fmt(vals.mean().item())} "
        f"std={_fmt(vals.std(unbiased=False).item())} "
        f"|.|_mean={_fmt(vals.abs().mean().item())} "
        f"min={_fmt(vals.min().item())} max={_fmt(vals.max().item())}"
    )


def _graph_sizes(batch) -> str:
    bsz = int(batch.batch.max().item()) + 1
    parts = [f"g{g}:nodes={int((batch.batch == g).sum())}" for g in range(bsz)]
    return "graphs {" + ", ".join(parts) + "}"


def _batch_metadata(batch, out: TextSink) -> None:
    meta = getattr(batch, "metadata", None)
    if meta is not None:
        print(f"  metadata={meta}", file=out)
    if hasattr(batch, "n_atom"):
        print(
            f"  n_atom={batch.n_atom.tolist()} n_probe={batch.n_probe.tolist()}",
            file=out,
        )


def _prior_debug(model, batch, out: TextSink) -> None:
    norm = coeff_norm_denominator(batch, model.n_orbitals, model.unique_atom_types)
    c0_raw = sample_flow_prior_c0_raw(
        batch,
        int(model.max_outdim),
        prior_mode=model.flow_prior_mode,
        orb_index=model.orb_index,
        orbital_irreps=model._orbital_irreps,
        sigma=model.flow_prior_sigma,
        prior_mu=getattr(model, "flow_prior_mu", 0.0),
        n_orbitals=model.n_orbitals,
        unique_atom_types=model.unique_atom_types,
    )
    print(f"  {_tensor_stats('c0_raw (fresh prior draw)', c0_raw)}", file=out)

    if (
        model.flow_prior_nelectron_normalize
        and model.flow_prior_mode in ("gaussian", "irrep_gaussian")
    ):
        from scdp.model.coeff_flow_matching import target_nelectrons_per_graph

        q_tgt = target_nelectrons_per_graph(batch)
        q_est = model._integrated_charge_from_c0_raw(batch, c0_raw).clamp(min=1e-8)
        scale = (q_tgt / q_est).view(-1, 1)
        c0_scaled = c0_raw * scale[batch.batch]
        print(
            "  nelectron normalize: ON "
            f"q_tgt_mean={_fmt(q_tgt.mean().item())} "
            f"q_est_mean={_fmt(q_est.mean().item())} "
            f"scale_mean={_fmt(scale.mean().item())} "
            f"min={_fmt(scale.min().item())} max={_fmt(scale.max().item())}",
            file=out,
        )
        print(f"  {_tensor_stats('c0_raw after nelectron scale', c0_scaled)}", file=out)
    else:
        print("  nelectron normalize: OFF", file=out)

    c_gt_raw = batch["gt_coeffs"].float()
    ratio = c0_raw.abs().mean() / c_gt_raw.abs().mean().clamp(min=1e-12)
    print(f"  |c0|/|c_gt| (raw, global mean ratio) ≈ {_fmt(ratio.item())}", file=out)


def _bridge_sanity(c0, c_gt, x_t, t, mask, out: TextSink) -> None:
    x_recon = (1.0 - t) * c0 + t * c_gt
    err = ((x_recon - x_t) * mask).abs()
    denom = mask.sum().clamp(min=1.0)
    print(
        "  bridge check x_t vs (1-t)*c0+t*c_gt: "
        f"MAE={_fmt((err.sum() / denom).item())} max_err={_fmt(err.max().item())}",
        file=out,
    )


def _per_graph_loss(pred, target, mask, t, batch, out: TextSink) -> None:
    bsz = int(batch.batch.max().item()) + 1
    print("  --- per-graph masked MSE ---", file=out)
    for g in range(bsz):
        m = mask[batch.batch == g]
        if m.sum() == 0:
            continue
        diff = (pred - target)[batch.batch == g]
        sq = (diff.pow(2) * m).sum() / m.sum().clamp(min=1.0)
        t_g = float(t[batch.batch == g][0].item())
        print(
            f"    graph {g}: loss={_fmt(sq.item())} t={_fmt(t_g)} "
            f"n_coeffs={int(m.sum())}",
            file=out,
        )


@torch.no_grad()
def dump_flow_spike_diagnostics(
    model,
    batch,
    *,
    loss_value: float,
    global_step: int,
    batch_idx: int,
    recent_median: Optional[float] = None,
    spike_factor: Optional[float] = None,
    log_path: Optional[Path] = None,
) -> Path:
    """
    Print full flow batch diagnostics after an abnormal loss spike.

    Writes to stdout and ``log_path`` (created under checkpoint dir).
    """
    if log_path is None:
        log_path = Path(f"flow_spike_step{global_step}_b{batch_idx}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    sinks: list[TextSink] = [sys.stdout]
    file_handle = open(log_path, "w", encoding="utf-8")
    sinks.append(file_handle)

    try:
        for out in sinks:
            header = "=" * 72
            print(f"\n{header}", file=out)
            print("FLOW LOSS SPIKE DIAGNOSTICS", file=out)
            print(header, file=out)
            print(f"  global_step={global_step} batch_idx={batch_idx}", file=out)
            print(f"  reported_loss={_fmt(loss_value)}", file=out)
            if recent_median is not None:
                print(f"  recent_loss_median={_fmt(recent_median)}", file=out)
            if spike_factor is not None and recent_median and recent_median > 0:
                print(f"  loss/median={_fmt(loss_value / recent_median)}", file=out)
            print(f"  bridge_mode={model.bridge_mode}", file=out)
            print(f"  flow_loss_mode={model.flow_loss_mode}", file=out)
            if model.flow_loss_mode == "endpoint":
                print(f"  flow_endpoint_loss={model.flow_endpoint_loss}", file=out)
            print(f"  flow_target_mode={model.flow_target_mode}", file=out)
            print(f"  flow_prior_mode={model.flow_prior_mode}", file=out)
            print(f"  flow_prior_sigma={model.flow_prior_sigma}", file=out)
            print(f"  flow_prior_mu={getattr(model, 'flow_prior_mu', 0.0)}", file=out)
            print(
                f"  flow_prior_nelectron_normalize={model.flow_prior_nelectron_normalize}",
                file=out,
            )
            print(f"  flow_use_t_scale={model.flow_use_t_scale}", file=out)
            ent = getattr(model.model, "enable_flow_coeff_entangle", False)
            print(f"  enable_flow_coeff_entangle={ent}", file=out)
            print("-" * 72, file=out)
            print(f"  {_graph_sizes(batch)}", file=out)
            _batch_metadata(batch, out)

            mask = per_node_coeff_mask(batch.atom_types, model.orb_index)
            gt_raw = batch["gt_coeffs"].float()
            norm = coeff_norm_denominator(
                batch, model.n_orbitals, model.unique_atom_types
            )
            c_gt = gt_raw / norm
            print(f"  {_tensor_stats('gt_coeffs raw', gt_raw, mask)}", file=out)
            print(f"  {_tensor_stats('c_gt normalized', c_gt, mask)}", file=out)
            print(
                f"  norm per node: mean={_fmt(norm.mean().item())} "
                f"min={_fmt(norm.min().item())} max={_fmt(norm.max().item())}",
                file=out,
            )

            print("  --- prior (extra draw; step used another RNG draw) ---", file=out)
            _prior_debug(model, batch, out)

            c0, c_gt2, x_t, t, mask2, target = model._prepare_flow_batch(batch)
            print("  --- bridge state (recomputed for this dump) ---", file=out)
            bsz = int(batch.batch.max().item()) + 1
            t_graph = [float(t[batch.batch == g][0].item()) for g in range(bsz)]
            print(f"  t per graph: {', '.join(_fmt(x) for x in t_graph)}", file=out)
            print(f"  {_tensor_stats('c0', c0, mask2)}", file=out)
            print(f"  {_tensor_stats('x_t', x_t, mask2)}", file=out)
            if model.bridge_mode == "cfm":
                _bridge_sanity(c0, c_gt2, x_t, t, mask2, out)
            print(f"  {_tensor_stats('target', target, mask2)}", file=out)

            fm_kw = model._flow_forward_kwargs(x_t, t) if model._flow_needs_xt_forward() else {}
            _, expo_scaling, latent = model.model(batch, return_latent=True, **fm_kw)
            pred = model._predict_flow_field(batch, latent, x_t, t)
            print(f"  {_tensor_stats('latent', latent)}", file=out)
            print(f"  {_tensor_stats('head pred', pred, mask2)}", file=out)

            diff = (pred - target) * mask2
            mse = diff.pow(2).sum() / mask2.sum().clamp(min=1.0)
            print(f"  recomputed MSE(pred,target)={_fmt(mse.item())}", file=out)
            _per_graph_loss(pred, target, mask2, t, batch, out)
            print(f"{header}\n", file=out)
    finally:
        file_handle.close()

    return log_path
