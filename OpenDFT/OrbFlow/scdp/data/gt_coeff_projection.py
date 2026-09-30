"""
Least-squares projection of voxel / probe charge density onto the atom-centered GTO basis.

Produces ``gt_coeffs`` in the **same raw tensor space as the eSCN readout before**
``predict_coeffs`` divides by ``sqrt(total_orbitals_in_graph))``:

    linear:   rho_labels ≈ scale * Phi  @ (gt_coeffs / sqrt(N_orb_sum))   [coeff c on phi]
    squared:  rho_labels ≈ scale * Phi_sq @ (gt_coeffs / sqrt(N_orb_sum))   [coeff d on phi^2, d >= 0]

``gt_coeffs`` stores the active basis coefficients (c or d). Squared mode fits d on Phi_sq
with ridge LS; optional ``coeff_nonneg=True`` uses GPU PGD NNLS (``d >= 0``). Default
unconstrained path uses ``torch.linalg.solve`` / cuSOLVER on CUDA. Stored as
per atom row (graph-wise scalar).
Inactive padded dimensions are set to zero using ``orb_index`` (same mask as training).

**Memory:** pass ``compact_active_orbitals=True`` to accumulate/solve on active orbital
columns only (``d_active = sum of per-atom outdims``) instead of the padded
``n_atom * max_outdim`` layout. Default is ``False`` (legacy padded path; QM9/MD unchanged).
MP cubic jobs opt in via ``GT_COMPACT_ACTIVE_ORBITALS=1`` / ``--compact_active_orbitals``.

**Uniqueness:** plain least squares is non-unique if ``Phi`` is rank-deficient. This module solves a
**regularized normal system** ``(Phi^T Phi + ridge I + ridge_diag_frac diag(Phi^T Phi)) c = Phi^T y``,
which is strictly positive definite for ``ridge > 0`` (and stabilizes ill-conditioned columns via
data-adaptive LM-style damping when ``ridge_diag_frac > 0``). That yields a **single** coefficient
vector minimizing the penalized objective (not uniqueness in the strict physical identifiability
sense if the basis cannot represent ``rho`` exactly).

**Accuracy:** use ``max_probe_samples=-1`` to use every probe, optional ``probe_weighting='abs_rho'``
for row weights that emphasize larger |density| voxels, and tune ``ridge`` / ``ridge_diag_frac``.

**Large graphs:** ``probe_subsample_above`` with ``max_probe_samples`` keeps the full grid when
``n_probe`` is at or below the threshold and randomly subsamples to the cap only above it.

**Iterative solve:** ``solve_method='iterative_cg'`` uses matrix-free CG on ``(Phi^T W Phi + reg) c = Phi^T W y``
without forming the ``d×d`` normal matrix (GPU-friendly for large ``d_active``).
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from scdp.common.constants import GTO_VEC_LENGTH_SCALE_TO_AU
from scdp.model.basis_set import aug_etb_for_basis, build_transformed_basis_set
from scdp.model.density_synthesis import (
    apply_gto_exponent_scale_for_mode,
    is_squared_density_synthesis,
    projection_weights_to_stored_coeffs,
    resolve_density_synthesis_mode,
    synthesis_basis_from_phi,
)
from scdp.model.gtos import GTOs


@dataclass
class GTOBasisPack:
    """Mirrors ChgLightningModule.construct_orbitals bookkeeping for projection."""

    gto_dict: nn.ModuleDict
    unique_atom_types: torch.Tensor  # (U,)
    n_orbitals: torch.Tensor  # (U,) outdim per entry in unique_atom_types
    orb_index: torch.Tensor  # (Zmax+1, max_outdim) bool
    max_outdim: int
    Lmax: int
    col_maps: Dict[int, torch.Tensor]  # Z -> (outdim_Z,) global column indices


def make_gto_basis_pack(
    unique_atom_types: Sequence[int],
    dft_basis_set: str = "def2-qzvppd",
    dft_wt_aug: bool = True,
    beta: float = 2.0,
    lmax_restriction: bool = True,
    lmax_relax: int = 0,
    uncontracted: bool = True,
    orb_cutoff: float = 5.0,
    vnode_elem: int = 8,
    device: str = "cpu",
    density_synthesis_mode: str = "linear",
) -> GTOBasisPack:
    synth_mode = resolve_density_synthesis_mode(density_synthesis_mode)
    basis_set = build_transformed_basis_set(
        dft_basis_set,
        required_atom_types=unique_atom_types,
    )
    if dft_wt_aug:
        basis_set = aug_etb_for_basis(
            basis_set,
            beta=beta,
            lmax_restriction=lmax_restriction,
            lmax_relax=lmax_relax,
        )
    basis_set = apply_gto_exponent_scale_for_mode(basis_set, synth_mode)
    vbasis = basis_set[vnode_elem]
    if uncontracted:
        for v in basis_set.values():
            v["contraction"] = None
        vbasis["contraction"] = None

    uat = sorted(int(x) for x in unique_atom_types)
    gto_dict: Dict[str, GTOs] = {}
    for elem in uat:
        if elem == 0:
            gto_dict["0"] = GTOs(**vbasis, cutoff=orb_cutoff)
        else:
            gto_dict[str(elem)] = GTOs(**basis_set[elem], cutoff=orb_cutoff)

    module_dict = nn.ModuleDict(gto_dict).to(device)
    dev = torch.device(device)
    Lmax = max(g.Lmax for g in gto_dict.values())
    max_n_orbitals_per_L = torch.stack(
        [x.n_orbitals_per_L.to(dev) for x in gto_dict.values()]
    ).max(dim=0)[0]
    # n_orbitals_per_L is length MAX_L+1 (see GTOs); must match ChgLightningModule.construct_orbitals.
    max_outdim_per_L = max_n_orbitals_per_L * (
        2 * torch.arange(len(max_n_orbitals_per_L), device=device) + 1
    )
    max_outdim = int(max_outdim_per_L.sum().item())

    zmax = max(uat)
    orb_index = torch.zeros(zmax + 1, max_outdim, dtype=torch.bool, device=device)
    offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(max_outdim_per_L.long(), dim=0)]
    )
    col_maps: Dict[int, torch.Tensor] = {}
    for k, v in gto_dict.items():
        zi = int(k)
        index = torch.cat(
            [torch.arange(offsets[l], offsets[l] + v.outdim_per_L[l], device=device) for l in range(Lmax + 1)]
        )
        orb_index[zi, index] = True
        col_maps[zi] = index.clone()

    ua_tensor = torch.tensor(uat, dtype=torch.long, device=device)
    n_orb_list = [gto_dict[str(int(z))].outdim for z in uat]
    n_orbitals = torch.tensor(n_orb_list, dtype=torch.long, device=device)

    return GTOBasisPack(
        gto_dict=module_dict,
        unique_atom_types=ua_tensor,
        n_orbitals=n_orbitals,
        orb_index=orb_index,
        max_outdim=max_outdim,
        Lmax=Lmax,
        col_maps=col_maps,
    )


def _total_orbitals(atom_types: torch.Tensor, pack: GTOBasisPack) -> int:
    """Sum of outdims over atoms (same as scatter(n_orbs, batch) for one graph)."""
    n = 0
    for z in atom_types.long().tolist():
        row = (pack.unique_atom_types == z).nonzero(as_tuple=False)
        n += int(pack.n_orbitals[row[0, 0]].item())
    return n


def build_active_column_indices(
    atom_types: torch.Tensor,
    pack: GTOBasisPack,
) -> Tuple[torch.Tensor, int, int]:
    """
    Padded flat column indices for every active orbital in ``atom_types``.

    Returns ``(active_cols, d_active, d_padded)`` where ``d_padded = n_atom * max_outdim``.
    """
    device = atom_types.device
    n_atom = int(atom_types.shape[0])
    d_padded = n_atom * pack.max_outdim
    if n_atom == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, 0, d_padded

    parts: List[torch.Tensor] = []
    for a, z in enumerate(atom_types.long().tolist()):
        base = a * pack.max_outdim
        parts.append(base + pack.col_maps[int(z)])
    active_cols = torch.cat(parts)
    return active_cols, int(active_cols.numel()), d_padded


def _padded_to_compact_map(active_cols: torch.Tensor, d_padded: int) -> torch.Tensor:
    """Map padded flat column index -> compact column index (-1 for inactive)."""
    device = active_cols.device
    padded_to_compact = torch.full((d_padded,), -1, dtype=torch.long, device=device)
    padded_to_compact[active_cols] = torch.arange(
        active_cols.numel(), device=device, dtype=torch.long
    )
    return padded_to_compact


def _expand_compact_weights(
    weights_compact: torch.Tensor,
    active_cols: torch.Tensor,
    d_padded: int,
) -> torch.Tensor:
    """Scatter compact normal-equation solution back to padded flat layout."""
    weights_flat = torch.zeros(d_padded, device=weights_compact.device, dtype=weights_compact.dtype)
    weights_flat[active_cols] = weights_compact
    return weights_flat


def _pbc_cartesian_offsets(
    cell: torch.Tensor,
    cutoff: float,
    device: torch.device,
) -> torch.Tensor:
    """Periodic image offsets in Angstrom (same convention as ``GTOs.forward``)."""
    cell_b = cell.unsqueeze(0) if cell.dim() == 2 else cell
    crossproducts = torch.cross(cell_b[:, [1, 2, 0]], cell_b[:, [2, 0, 1]], dim=-1)
    cell_vol = torch.sum(cell_b[:, 0] * crossproducts[:, 0], dim=-1, keepdim=True)
    n_rep = torch.ceil(
        cutoff * torch.norm(crossproducts / cell_vol[:, None], p=2, dim=-1)
    ).max(dim=0)[0].long()
    reps = [
        torch.arange(-int(n_rep[d]), int(n_rep[d]) + 1, device=device) for d in range(3)
    ]
    unit_cell = torch.tensor(
        [(x, y, z) for x in reps[0] for y in reps[1] for z in reps[2]],
        device=device,
        dtype=cell_b.dtype,
    )
    unit_cell_batch = unit_cell.transpose(0, 1).unsqueeze(0).expand(len(cell_b), -1, -1)
    data_cell = torch.transpose(cell_b, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    return pbc_offsets[0].transpose(0, 1)


def build_design_matrix_chunked(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    probe_coords: torch.Tensor,
    probe_idx: torch.Tensor,
    pack: GTOBasisPack,
    pbc: bool,
    cell: Optional[torch.Tensor],
    chunk_size: int = 2048,
    density_synthesis_mode: str = "linear",
    active_cols: Optional[torch.Tensor] = None,
    padded_to_compact: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Dense design matrix Phi with inactive columns zero (padded) or omitted (compact).

    Padded shape: ``(P, n_atom * max_outdim)``. Compact shape when ``active_cols`` is set:
    ``(P, d_active)`` using only active orbital columns (same values as padded layout).

    Matches GTO evaluation order and e3nn vecs reorder used in GTOs.forward.

    Evaluates GTOs in one batched ``compute`` per (element type, probe chunk) instead of
    one call per atom, which greatly reduces kernel launch overhead on large probe grids.

    When ``pbc=True``, sums GTO values over all periodic images within ``cutoff`` (matching
    ``GTOs.forward`` / ``orbital_inference``). Multiple images use ``scatter_add`` on ``Phi``.
    """
    device = coords.device
    dtype = coords.dtype
    synth_mode = resolve_density_synthesis_mode(density_synthesis_mode)
    n_atom = atom_types.shape[0]
    p = probe_idx.shape[0]
    d_padded = n_atom * pack.max_outdim
    if active_cols is not None:
        d_cols = int(active_cols.numel())
        if padded_to_compact is None:
            padded_to_compact = _padded_to_compact_map(active_cols, d_padded)
    else:
        d_cols = d_padded
        padded_to_compact = None
    Phi = torch.zeros(p, d_cols, device=device, dtype=dtype)
    if n_atom == 0 or p == 0:
        return Phi

    probes = probe_coords[probe_idx]
    chunk_size = max(1, int(chunk_size))
    vec_scale = GTO_VEC_LENGTH_SCALE_TO_AU

    for z_t in atom_types.unique():
        z = int(z_t.item())
        atom_inds = (atom_types == z).nonzero(as_tuple=False).view(-1)
        if atom_inds.numel() == 0:
            continue
        gto = pack.gto_dict[str(z)]
        col_map = pack.col_maps[z]
        if col_map.numel() == 0:
            continue
        atom_coords_z = coords[atom_inds]
        nz = int(atom_inds.shape[0])

        for start in range(0, p, chunk_size):
            end = min(start + chunk_size, p)
            probe_blk = probes[start:end]

            if pbc:
                if cell is None:
                    raise ValueError("cell is required when pbc=True")
                offsets = _pbc_cartesian_offsets(
                    cell.to(device=device, dtype=dtype),
                    float(gto.cutoff if gto.cutoff is not None else 0.0),
                    device,
                )
                vecs_ang = (
                    probe_blk[None, :, None, :]
                    + offsets[None, None, :, :]
                    - atom_coords_z[:, None, None, :]
                )
            else:
                vecs_ang = probe_blk[None, :, :] - atom_coords_z[:, None, :]

            if gto.cutoff is not None:
                mask = vecs_ang.norm(dim=-1) < gto.cutoff
            else:
                mask = torch.ones(vecs_ang.shape[:-1], dtype=torch.bool, device=device)
            if not mask.any():
                continue

            ai_flat, pi_flat = mask.nonzero(as_tuple=True)[:2]
            vecs = vecs_ang[mask][..., [1, 2, 0]] / vec_scale
            index_atom = torch.zeros(vecs.shape[0], dtype=torch.long, device=device)
            b = gto.compute(vecs, expo_scaling=None, index_atom=index_atom)
            b = synthesis_basis_from_phi(b, synth_mode)

            rows_local = pi_flat + int(start)
            atom_global = atom_inds[ai_flat]
            cols_padded = atom_global[:, None] * pack.max_outdim + col_map[None, :]
            if padded_to_compact is not None:
                cols = padded_to_compact[cols_padded]
            else:
                cols = cols_padded
            flat_idx = (rows_local[:, None] * d_cols + cols).reshape(-1)
            Phi.view(-1).scatter_add_(0, flat_idx, b.reshape(-1))
    return Phi


def accumulate_weighted_normal_equations(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    probe_coords: torch.Tensor,
    probe_idx: torch.Tensor,
    chg_labels: torch.Tensor,
    pack: GTOBasisPack,
    scale_t: torch.Tensor,
    pbc: bool,
    cell: Optional[torch.Tensor],
    probe_weighting: Optional[str],
    chunk_size: int = 2048,
    probe_chunk: int = 2048,
    accumulate_on_cpu: Optional[bool] = None,
    density_synthesis_mode: str = "linear",
    compact_active_orbitals: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Stream ``Phi`` in probe blocks; return ``(Phi^T W Phi, Phi^T W y, active_cols)``.

    When ``compact_active_orbitals=True``, accumulates only the active orbital
    subspace (``d_active = sum of per-atom outdims``) instead of the padded
    ``n_atom * max_outdim`` layout. ``active_cols`` maps compact indices back to padded
    flat layout for scatter after the solve; it is ``None`` when compact mode is off.

    By default accumulates on the same device as coords (GPU when ``device=cuda``) so
    ``Phi^T W Phi`` stays off the CPU. Pass ``accumulate_on_cpu=True`` if ``d_tot`` is huge
    and GPU memory is tight.
    """
    device = coords.device
    dtype = coords.dtype
    active_cols: Optional[torch.Tensor] = None
    padded_to_compact: Optional[torch.Tensor] = None
    if compact_active_orbitals:
        active_cols, d_sys, d_padded = build_active_column_indices(atom_types, pack)
        padded_to_compact = _padded_to_compact_map(active_cols, d_padded)
    else:
        d_sys = atom_types.shape[0] * pack.max_outdim
    if accumulate_on_cpu is None:
        accumulate_on_cpu = device.type != "cuda"
    acc_dev = torch.device("cpu") if accumulate_on_cpu else device
    ata = torch.zeros(d_sys, d_sys, device=acc_dev, dtype=dtype)
    atb = torch.zeros(d_sys, 1, device=acc_dev, dtype=dtype)
    p = probe_idx.shape[0]
    pc = max(1, int(probe_chunk))

    for start in range(0, p, pc):
        end = min(start + pc, p)
        idx_blk = probe_idx[start:end]
        Phi_blk = build_design_matrix_chunked(
            atom_types,
            coords,
            probe_coords,
            idx_blk,
            pack,
            pbc=pbc,
            cell=cell,
            chunk_size=chunk_size,
            density_synthesis_mode=density_synthesis_mode,
            active_cols=active_cols,
            padded_to_compact=padded_to_compact,
        )
        rho_blk = chg_labels[idx_blk].view(-1, 1)
        y_blk = rho_blk / scale_t.clamp(min=1e-12)
        w_sqrt = _probe_row_weights(rho_blk.view(-1), probe_weighting, dtype, device).view(-1, 1)
        Phi_w = Phi_blk * w_sqrt
        y_w = y_blk * w_sqrt
        if acc_dev == device:
            ata.addmm_(Phi_w.T, Phi_w)
            atb.addmm_(Phi_w.T, y_w)
        else:
            ata += (Phi_w.T @ Phi_w).to(device=acc_dev, dtype=dtype)
            atb += (Phi_w.T @ y_w).to(device=acc_dev, dtype=dtype)
        del Phi_blk, Phi_w, y_w, y_blk, rho_blk, w_sqrt

    return ata, atb, active_cols


@dataclass
class _WeightedNormalStream:
    """Shared probe-chunk context for matrix-free ``Phi^T W Phi`` matvecs."""

    atom_types: torch.Tensor
    coords: torch.Tensor
    probe_coords: torch.Tensor
    probe_idx: torch.Tensor
    chg_labels: torch.Tensor
    pack: GTOBasisPack
    scale_t: torch.Tensor
    pbc: bool
    cell: Optional[torch.Tensor]
    probe_weighting: Optional[str]
    chunk_size: int
    probe_chunk: int
    density_synthesis_mode: str
    active_cols: Optional[torch.Tensor]
    padded_to_compact: Optional[torch.Tensor]
    d_sys: int
    device: torch.device
    dtype: torch.dtype


def _build_weighted_normal_stream(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    probe_coords: torch.Tensor,
    probe_idx: torch.Tensor,
    chg_labels: torch.Tensor,
    pack: GTOBasisPack,
    scale_t: torch.Tensor,
    pbc: bool,
    cell: Optional[torch.Tensor],
    probe_weighting: Optional[str],
    chunk_size: int,
    probe_chunk: int,
    density_synthesis_mode: str,
    compact_active_orbitals: bool,
) -> _WeightedNormalStream:
    device = coords.device
    dtype = coords.dtype
    active_cols: Optional[torch.Tensor] = None
    padded_to_compact: Optional[torch.Tensor] = None
    if compact_active_orbitals:
        active_cols, d_sys, d_padded = build_active_column_indices(atom_types, pack)
        padded_to_compact = _padded_to_compact_map(active_cols, d_padded)
    else:
        d_sys = atom_types.shape[0] * pack.max_outdim
    return _WeightedNormalStream(
        atom_types=atom_types,
        coords=coords,
        probe_coords=probe_coords,
        probe_idx=probe_idx,
        chg_labels=chg_labels,
        pack=pack,
        scale_t=scale_t,
        pbc=pbc,
        cell=cell,
        probe_weighting=probe_weighting,
        chunk_size=int(chunk_size),
        probe_chunk=max(1, int(probe_chunk)),
        density_synthesis_mode=density_synthesis_mode,
        active_cols=active_cols,
        padded_to_compact=padded_to_compact,
        d_sys=int(d_sys),
        device=device,
        dtype=dtype,
    )


def _iter_weighted_phi_blocks(
    stream: _WeightedNormalStream,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Yield ``(Phi_w, y_w)`` probe blocks with sqrt row weights applied."""
    p = stream.probe_idx.shape[0]
    pc = stream.probe_chunk
    for start in range(0, p, pc):
        end = min(start + pc, p)
        idx_blk = stream.probe_idx[start:end]
        Phi_blk = build_design_matrix_chunked(
            stream.atom_types,
            stream.coords,
            stream.probe_coords,
            idx_blk,
            stream.pack,
            pbc=stream.pbc,
            cell=stream.cell,
            chunk_size=stream.chunk_size,
            density_synthesis_mode=stream.density_synthesis_mode,
            active_cols=stream.active_cols,
            padded_to_compact=stream.padded_to_compact,
        )
        rho_blk = stream.chg_labels[idx_blk].view(-1, 1)
        y_blk = rho_blk / stream.scale_t.clamp(min=1e-12)
        w_sqrt = _probe_row_weights(
            rho_blk.view(-1), stream.probe_weighting, stream.dtype, stream.device
        ).view(-1, 1)
        yield Phi_blk * w_sqrt, y_blk * w_sqrt
        del Phi_blk, y_blk, rho_blk, w_sqrt


def stream_weighted_rhs(
    stream: _WeightedNormalStream,
    *,
    out_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """``Phi^T W y`` without forming ``Phi^T W Phi``."""
    out_dev = out_device or stream.device
    atb = torch.zeros(stream.d_sys, 1, device=out_dev, dtype=stream.dtype)
    for Phi_w, y_w in _iter_weighted_phi_blocks(stream):
        blk = Phi_w.T @ y_w
        if blk.device != out_dev:
            blk = blk.to(device=out_dev, dtype=stream.dtype)
        atb.add_(blk)
    return atb.view(-1)


def stream_weighted_hessian_diag(
    stream: _WeightedNormalStream,
    *,
    out_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """``diag(Phi^T W Phi)`` via column-wise streaming."""
    out_dev = out_device or stream.device
    diag = torch.zeros(stream.d_sys, device=out_dev, dtype=stream.dtype)
    for Phi_w, _y_w in _iter_weighted_phi_blocks(stream):
        col = Phi_w.pow(2).sum(dim=0)
        if col.device != out_dev:
            col = col.to(device=out_dev, dtype=stream.dtype)
        diag.add_(col)
    return diag


def stream_weighted_normal_matvec(
    stream: _WeightedNormalStream,
    vec: torch.Tensor,
) -> torch.Tensor:
    """Matrix-free ``(Phi^T W Phi) @ vec``."""
    v = vec.view(-1, 1).to(device=stream.device, dtype=stream.dtype)
    out = torch.zeros(stream.d_sys, 1, device=stream.device, dtype=stream.dtype)
    for Phi_w, _y_w in _iter_weighted_phi_blocks(stream):
        t = Phi_w @ v
        out.addmm_(Phi_w.T, t)
    return out.view(-1)


@torch.no_grad()
def _solve_symmetric_cg(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    maxiter: int,
    tol: float,
) -> Tuple[torch.Tensor, dict]:
    """CG for SPD ``A x = rhs`` where ``A`` is provided via ``matvec``."""
    b = rhs.view(-1).to(dtype=rhs.dtype)
    x = torch.zeros_like(b)
    r = b - matvec(x)
    b_norm = b.norm().clamp(min=1e-30)
    if r.norm() <= tol * b_norm:
        return x, {"iterations": 0, "residual_norm": float(r.norm().item()), "converged": True}

    p = r.clone()
    rs_old = r.dot(r)
    for it in range(1, int(maxiter) + 1):
        Ap = matvec(p)
        denom = p.dot(Ap).clamp(min=1e-30)
        alpha = rs_old / denom
        x.add_(alpha * p)
        r.sub_(alpha * Ap)
        rs_new = r.dot(r)
        if rs_new.sqrt() <= tol * b_norm:
            return x, {
                "iterations": it,
                "residual_norm": float(rs_new.sqrt().item()),
                "converged": True,
            }
        beta = rs_new / rs_old.clamp(min=1e-30)
        p = r + beta * p
        rs_old = rs_new

    return x, {
        "iterations": int(maxiter),
        "residual_norm": float(rs_new.sqrt().item()),
        "converged": False,
    }


def _solve_normal_equations_cg(
    stream: _WeightedNormalStream,
    ridge: float,
    ridge_diag_frac: float,
    return_device: torch.device,
    *,
    cg_maxiter: Optional[int] = None,
    cg_tol: float = 1e-8,
) -> Tuple[torch.Tensor, dict]:
    """Matrix-free ridge normal equations via CG (no ``d×d`` allocation)."""
    rhs = stream_weighted_rhs(stream, out_device=stream.device)
    hessian_diag: Optional[torch.Tensor] = None
    if float(ridge_diag_frac) > 0.0:
        hessian_diag = stream_weighted_hessian_diag(stream, out_device=stream.device)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        out = stream_weighted_normal_matvec(stream, v)
        if hessian_diag is not None:
            out = out + float(ridge_diag_frac) * hessian_diag * v
        return out + float(ridge) * v

    maxiter = int(cg_maxiter) if cg_maxiter is not None else max(500, min(5000, 10 * stream.d_sys))
    weights, info = _solve_symmetric_cg(matvec, rhs, maxiter=maxiter, tol=float(cg_tol))
    if not info["converged"]:
        print(
            f"gt_coeff_projection: CG did not converge in {info['iterations']} iters "
            f"(residual={info['residual_norm']:.3e}, d={stream.d_sys})",
            file=sys.stderr,
            flush=True,
        )
    return weights.to(device=return_device, dtype=stream.dtype), info


def _is_cuda_oom(err: BaseException) -> bool:
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(err, RuntimeError):
        msg = str(err).lower()
        return "out of memory" in msg
    return False


class GtCoeffCudaOomSkip(RuntimeError):
    """Raised when GPU OOM occurs and ``oom_policy='skip'`` (no CPU fallback)."""


def _normalize_oom_policy(oom_policy: str) -> str:
    policy = str(oom_policy or "cpu_fallback").strip().lower()
    if policy in ("cpu", "cpu_fallback", "fallback"):
        return "cpu_fallback"
    if policy in ("skip", "skip_graph", "raise"):
        return "skip"
    raise ValueError(
        f"oom_policy must be 'cpu_fallback' or 'skip', got {oom_policy!r}"
    )


def _solve_normal_equations_on_device(
    ata: torch.Tensor,
    atb: torch.Tensor,
    ridge: float,
    ridge_diag_frac: float,
) -> torch.Tensor:
    """Factor ``(ata + reg) c = atb`` on the device where ``ata`` already lives."""
    diag_ata = torch.diag(ata).clamp(min=1e-18)
    trace_scale = float(diag_ata.mean().clamp(min=1e-18).item())
    rhs = atb

    # In-place diagonal regularization — never materialize ``torch.eye(d)`` (OOM when d≈30k).
    h = ata.clone()
    h.diagonal().add_(float(ridge) + float(ridge_diag_frac) * diag_ata)

    jitter_fracs = (0.0, 1e-12, 1e-10, 1e-8, 1e-6)
    for jitter in jitter_fracs:
        if jitter == 0.0:
            hj = h
        else:
            hj = h.clone()
            hj.diagonal().add_(float(jitter) * trace_scale)
        try:
            return torch.linalg.solve(hj, rhs).view(-1)
        except RuntimeError as e:
            if _is_cuda_oom(e):
                raise
            continue

    hj = h.clone()
    hj.diagonal().add_(1e-6 * trace_scale)
    return torch.linalg.lstsq(hj, rhs, rcond=1e-10).solution.view(-1)


def _solve_normal_equations(
    ata: torch.Tensor,
    atb: torch.Tensor,
    ridge: float,
    ridge_diag_frac: float,
    return_device: torch.device,
) -> torch.Tensor:
    """GPU cuSOLVER first when ``ata`` is on CUDA; on OOM, retry once on CPU."""
    dtype = ata.dtype
    ata = ata.detach()
    atb = atb.detach()
    dtot = int(ata.shape[0])

    if ata.device.type == "cuda":
        try:
            c = _solve_normal_equations_on_device(ata, atb, ridge, ridge_diag_frac)
            return c if return_device == c.device else c.to(device=return_device, dtype=dtype)
        except RuntimeError as e:
            if not _is_cuda_oom(e):
                raise
            torch.cuda.empty_cache()
            print(
                f"gt_coeff_projection: GPU OOM during solve at d={dtot}, retrying on CPU",
                file=sys.stderr,
                flush=True,
            )

    ata_cpu = ata.to(device="cpu", dtype=dtype)
    atb_cpu = atb.to(device="cpu", dtype=dtype)
    if ata.device.type == "cuda":
        del ata, atb
        torch.cuda.empty_cache()
    c = _solve_normal_equations_on_device(ata_cpu, atb_cpu, ridge, ridge_diag_frac)
    return c.to(device=return_device, dtype=dtype)


def _regularized_hessian(
    ata: torch.Tensor,
    ridge: float,
    ridge_diag_frac: float,
) -> torch.Tensor:
    """Return ``H = ata + ridge I + ridge_diag_frac diag(ata)`` (new tensor)."""
    diag_ata = torch.diag(ata).clamp(min=1e-18)
    h = ata.clone()
    h.diagonal().add_(float(ridge) + float(ridge_diag_frac) * diag_ata)
    return h


def _nnls_pgd_normal_equations(
    h: torch.Tensor,
    f: torch.Tensor,
    tol: float = 1e-9,
    max_iter: int = 10000,
) -> torch.Tensor:
    """
    Projected gradient for ``min_{x>=0} 0.5 x^T H x - f^T x`` (strictly convex when ridge > 0).

    Stays on the same device as ``H`` (CUDA matmuls; no CPU/scipy round-trip). Converges to
    the same unique minimizer as bound-constrained NNLS for ridge-regularized sidecars.
    """
    f = f.view(-1)
    x = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
    row_sum = h.abs().sum(dim=1).max()
    diag = h.diagonal().abs().max()
    lipschitz = torch.maximum(row_sum, diag).clamp(min=1e-12)
    inv_l = 1.0 / lipschitz

    for _ in range(int(max_iter)):
        g = h @ x - f
        x_new = (x - inv_l * g).clamp(min=0.0)
        if torch.linalg.norm(x_new - x) <= tol * (torch.linalg.norm(x) + 1.0):
            return x_new
        x = x_new
    return x


def _solve_nonneg_normal_equations_scipy_cpu(
    h_cpu: torch.Tensor,
    f_cpu: torch.Tensor,
) -> torch.Tensor:
    """CPU L-BFGS-B fallback when GPU active-set NNLS fails."""
    import numpy as np
    from scipy.optimize import minimize

    h = h_cpu.numpy()
    f = f_cpu.numpy().reshape(-1)
    dtot = h.shape[0]

    def objective(d: np.ndarray) -> float:
        return float(0.5 * d @ h @ d - f @ d)

    def gradient(d: np.ndarray) -> np.ndarray:
        return h @ d - f

    res = minimize(
        objective,
        np.zeros(dtot, dtype=h.dtype),
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * dtot,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not res.success:
        print(
            f"gt_coeff_projection: nonneg LS scipy fallback ({res.message}); "
            f"d_min={res.x.min():.3e}",
            file=sys.stderr,
            flush=True,
        )
    return torch.from_numpy(res.x).to(device=h_cpu.device, dtype=h_cpu.dtype)


def _solve_nonneg_normal_equations(
    ata: torch.Tensor,
    atb: torch.Tensor,
    ridge: float,
    ridge_diag_frac: float,
    return_device: torch.device,
) -> torch.Tensor:
    """
    Non-negative Tikhonov WLS via normal equations:

        min_d  0.5 d^T H d - f^T d   s.t. d >= 0
        H = Phi^T W Phi + ridge I + ridge_diag_frac diag(Phi^T W Phi),  f = Phi^T W y.

    Primary path: projected gradient on the same device as ``ata`` (CUDA matmuls; no CPU
    round-trip). Falls back to CPU scipy L-BFGS-B on failure / CUDA OOM.
    """
    dtype = ata.dtype
    ata = ata.detach()
    atb = atb.detach().view(-1)
    dtot = int(ata.shape[0])

    def _run_on_device(dev: torch.device) -> torch.Tensor:
        h = _regularized_hessian(ata.to(device=dev, dtype=dtype), ridge, ridge_diag_frac)
        f = atb.to(device=dev, dtype=dtype)
        return _nnls_pgd_normal_equations(h, f)

    if ata.device.type == "cuda":
        try:
            d = _run_on_device(ata.device)
            solver = "cuda_pgd"
        except RuntimeError as e:
            if not _is_cuda_oom(e):
                print(
                    f"gt_coeff_projection: GPU nonneg LS failed ({e!r}); scipy CPU fallback d={dtot}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                torch.cuda.empty_cache()
                print(
                    f"gt_coeff_projection: GPU OOM during nonneg LS d={dtot}, scipy CPU fallback",
                    file=sys.stderr,
                    flush=True,
                )
            h_cpu = _regularized_hessian(ata.to(device="cpu", dtype=dtype), ridge, ridge_diag_frac)
            f_cpu = atb.to(device="cpu", dtype=dtype)
            del ata, atb
            d = _solve_nonneg_normal_equations_scipy_cpu(h_cpu, f_cpu)
            solver = "scipy_lbfgsb_cpu"
    else:
        try:
            d = _run_on_device(ata.device)
            solver = "cpu_pgd"
        except RuntimeError as e:
            print(
                f"gt_coeff_projection: CPU PGD nonneg LS failed ({e!r}); scipy fallback d={dtot}",
                file=sys.stderr,
                flush=True,
            )
            h_cpu = _regularized_hessian(ata, ridge, ridge_diag_frac)
            d = _solve_nonneg_normal_equations_scipy_cpu(h_cpu, atb.to(device=ata.device, dtype=dtype))
            solver = "scipy_lbfgsb_cpu"

    if solver != "cuda_pgd":
        print(
            f"gt_coeff_projection: nonneg LS solver={solver} d={dtot} device={return_device.type}",
            file=sys.stderr,
            flush=True,
        )

    return d.to(device=return_device, dtype=dtype)


def _solve_regularized_ls(
    Phi: torch.Tensor,
    y: torch.Tensor,
    ridge: float,
    ridge_diag_frac: float,
) -> torch.Tensor:
    """
    Minimize ||Phi c - y||^2 + ridge||c||^2 + ridge_diag_frac * c^T diag(Phi^T Phi) c
    via normal equations.

    Prefer ``accumulate_weighted_normal_equations`` when P is large (avoids O(P·d) ``Phi``).
    """
    orig_device, dtype = Phi.device, Phi.dtype
    ata = Phi.T @ Phi
    atb = Phi.T @ y
    del Phi
    if orig_device.type == "cuda":
        torch.cuda.empty_cache()
    return _solve_normal_equations(ata, atb, ridge, ridge_diag_frac, orig_device)


def _probe_row_weights(
    rho_sel: torch.Tensor, mode: Optional[str], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Per-probe sqrt-weights for WLS; mean 1 for scale stability."""
    p = rho_sel.shape[0]
    if mode is None or mode == "uniform":
        return torch.ones(p, device=device, dtype=dtype)
    if mode == "abs_rho":
        w = torch.sqrt(1.0 + rho_sel.abs() / (rho_sel.abs().mean().clamp(min=1e-12)))
        return w / w.mean().clamp(min=1e-12)
    raise ValueError(f"unknown probe_weighting {mode!r}; use None, 'uniform', or 'abs_rho'")



def configure_linear_algebra_threads(num_threads: int) -> None:
    """Pin OpenMP/MKL and torch CPU threads (call once per process / worker)."""
    import os

    n = max(1, int(num_threads))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)


def _compact_weights_to_gt_raw(
    weights_compact: torch.Tensor,
    atom_types: torch.Tensor,
    pack: GTOBasisPack,
    active_cols: Optional[torch.Tensor],
    norm: float,
    dtype_name: str,
    density_synthesis_mode: str = "linear",
) -> torch.Tensor:
    d_padded = atom_types.shape[0] * pack.max_outdim
    if active_cols is not None:
        weights_flat = _expand_compact_weights(weights_compact, active_cols, d_padded)
    else:
        weights_flat = weights_compact

    c_norm = projection_weights_to_stored_coeffs(
        weights_flat.view(atom_types.shape[0], pack.max_outdim),
        density_synthesis_mode,
    )
    mask = pack.orb_index[atom_types.long()].to(dtype=weights_compact.dtype)
    c_norm = c_norm * mask
    store_dtype = torch.float32 if dtype_name == "float32" else torch.float64
    gt_raw = (c_norm * norm).to(dtype=store_dtype)
    return gt_raw.cpu() if gt_raw.device.type != "cpu" else gt_raw


def _project_labels_to_gt_coeffs_once(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    cell: torch.Tensor,
    pbc: bool,
    probe_coords: torch.Tensor,
    chg_labels: torch.Tensor,
    pack: GTOBasisPack,
    scale_t: torch.Tensor,
    perm: torch.Tensor,
    norm: float,
    ridge: float,
    ridge_diag_frac: float,
    probe_weighting: Optional[str],
    chunk_size: int,
    accumulate_on_cpu: bool,
    dtype_name: str,
    dev: torch.device,
    dtype: torch.dtype,
    density_synthesis_mode: str = "linear",
    coeff_nonneg: bool = False,
    compact_active_orbitals: bool = False,
) -> torch.Tensor:
    probe_chunk = min(int(chunk_size), 4096)
    ata, atb, active_cols = accumulate_weighted_normal_equations(
        atom_types,
        coords,
        probe_coords,
        perm,
        chg_labels,
        pack,
        scale_t,
        pbc=pbc,
        cell=cell if cell is not None else None,
        probe_weighting=probe_weighting,
        chunk_size=int(chunk_size),
        probe_chunk=probe_chunk,
        accumulate_on_cpu=accumulate_on_cpu,
        density_synthesis_mode=density_synthesis_mode,
        compact_active_orbitals=compact_active_orbitals,
    )
    if is_squared_density_synthesis(density_synthesis_mode) and coeff_nonneg:
        weights_compact = _solve_nonneg_normal_equations(
            ata,
            atb,
            ridge=float(ridge),
            ridge_diag_frac=float(ridge_diag_frac),
            return_device=dev,
        )
    else:
        weights_compact = _solve_normal_equations(
            ata,
            atb,
            ridge=float(ridge),
            ridge_diag_frac=float(ridge_diag_frac),
            return_device=dev,
        )
    del ata, atb
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    return _compact_weights_to_gt_raw(
        weights_compact,
        atom_types,
        pack,
        active_cols,
        norm,
        dtype_name,
        density_synthesis_mode,
    )


def _project_labels_to_gt_coeffs_iterative(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    cell: torch.Tensor,
    pbc: bool,
    probe_coords: torch.Tensor,
    chg_labels: torch.Tensor,
    pack: GTOBasisPack,
    scale_t: torch.Tensor,
    perm: torch.Tensor,
    norm: float,
    ridge: float,
    ridge_diag_frac: float,
    probe_weighting: Optional[str],
    chunk_size: int,
    dtype_name: str,
    dev: torch.device,
    dtype: torch.dtype,
    density_synthesis_mode: str = "linear",
    coeff_nonneg: bool = False,
    compact_active_orbitals: bool = False,
    cg_maxiter: Optional[int] = None,
    cg_tol: float = 1e-8,
) -> torch.Tensor:
    """Matrix-free ridge LS via CG (no ``d×d`` normal matrix)."""
    if is_squared_density_synthesis(density_synthesis_mode) and coeff_nonneg:
        raise ValueError("solve_method='iterative_cg' does not support coeff_nonneg NNLS")
    probe_chunk = min(int(chunk_size), 4096)
    stream = _build_weighted_normal_stream(
        atom_types,
        coords,
        probe_coords,
        perm,
        chg_labels,
        pack,
        scale_t,
        pbc=pbc,
        cell=cell if cell is not None else None,
        probe_weighting=probe_weighting,
        chunk_size=int(chunk_size),
        probe_chunk=probe_chunk,
        density_synthesis_mode=density_synthesis_mode,
        compact_active_orbitals=compact_active_orbitals,
    )
    weights_compact, _cg_info = _solve_normal_equations_cg(
        stream,
        ridge=float(ridge),
        ridge_diag_frac=float(ridge_diag_frac),
        return_device=dev,
        cg_maxiter=cg_maxiter,
        cg_tol=float(cg_tol),
    )
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return _compact_weights_to_gt_raw(
        weights_compact,
        atom_types,
        pack,
        stream.active_cols,
        norm,
        dtype_name,
        density_synthesis_mode,
    )


def _format_projection_dims(d_active: int, d_padded: int) -> str:
    if d_active == d_padded:
        return f"d={d_active}"
    return f"d_active={d_active} d_padded={d_padded}"


def _project_gpu_first_with_oom_policy(
    core_kwargs: dict,
    d_active: int,
    d_padded: int,
    dev: torch.device,
    oom_policy: str = "cpu_fallback",
) -> torch.Tensor:
    """Try GPU ``Phi^T W Phi``; on CUDA OOM, CPU-fallback or skip per ``oom_policy``."""
    policy = _normalize_oom_policy(oom_policy)
    dim_msg = _format_projection_dims(d_active, d_padded)
    try:
        return _project_labels_to_gt_coeffs_once(**core_kwargs, accumulate_on_cpu=False)
    except RuntimeError as e:
        if not (dev.type == "cuda" and _is_cuda_oom(e)):
            raise
        torch.cuda.empty_cache()
        if policy == "skip":
            print(
                f"gt_coeff_projection: GPU OOM at {dim_msg}, skipping graph "
                f"(oom_policy=skip; no CPU fallback)",
                file=sys.stderr,
                flush=True,
            )
            raise GtCoeffCudaOomSkip(
                f"GPU OOM at {dim_msg}; skipped (oom_policy=skip)"
            ) from e
        print(
            f"gt_coeff_projection: GPU OOM at {dim_msg}, retrying with CPU accumulation",
            file=sys.stderr,
            flush=True,
        )
        try:
            return _project_labels_to_gt_coeffs_once(
                **core_kwargs, accumulate_on_cpu=True
            )
        except RuntimeError as e2:
            # CPU path still builds Phi on CUDA today; clear and re-raise cleanly.
            if _is_cuda_oom(e2):
                torch.cuda.empty_cache()
            raise


def resolve_effective_max_probe_samples(
    p_total: int,
    max_probe_samples: int,
    probe_subsample_above: int = -1,
) -> int:
    """
    Per-graph probe cap for ridge fit.

    When ``probe_subsample_above >= 0``: full grid if ``p_total <= threshold``; else cap at
    ``max_probe_samples`` (must be ``>= 0``). When ``probe_subsample_above < 0``: legacy behavior
    — ``max_probe_samples`` applies to every graph when ``>= 0``.
    """
    p_total = int(p_total)
    threshold = int(probe_subsample_above)
    cap = int(max_probe_samples)
    if threshold >= 0:
        if p_total <= threshold:
            return -1
        if cap < 0:
            raise ValueError(
                f"probe_subsample_above={threshold} requires max_probe_samples >= 0, got {cap}"
            )
        return cap
    return cap


def project_labels_to_gt_coeffs(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    cell: torch.Tensor,
    pbc: bool,
    probe_coords: torch.Tensor,
    chg_labels: torch.Tensor,
    pack: GTOBasisPack,
    scale: float,
    ridge: float = 1e-3,
    ridge_diag_frac: float = 0.0,
    probe_weighting: Optional[str] = None,
    max_probe_samples: int = -1,
    seed: int = 0,
    device: str = "cpu",
    dtype_name: str = "float64",
    chunk_size: int = 4096,
    accumulate_on_cpu: Optional[bool] = None,
    density_synthesis_mode: str = "linear",
    coeff_nonneg: bool = False,
    compact_active_orbitals: bool = False,
    solve_method: str = "dense",
    cg_maxiter: Optional[int] = None,
    cg_tol: float = 1e-8,
    oom_policy: str = "cpu_fallback",
) -> torch.Tensor:
    """
    Regularized least-squares fit in **normalized** coefficient space, returned as **raw** gt_coeffs.

    ``solve_method='dense'`` (default): form ``Phi^T W Phi`` and dense solve.
    ``solve_method='iterative_cg'``: matrix-free CG (no ``d×d`` matrix; GPU-friendly for large ``d_active``).
    ``oom_policy``: ``cpu_fallback`` (default) or ``skip`` (clear GPU and raise ``GtCoeffCudaOomSkip``).
    """
    if solve_method not in ("dense", "iterative_cg"):
        raise ValueError(f"solve_method must be 'dense' or 'iterative_cg', got {solve_method!r}")
    oom_policy = _normalize_oom_policy(oom_policy)
    synth_mode = resolve_density_synthesis_mode(density_synthesis_mode)
    dev = torch.device(device)
    if dtype_name not in ("float32", "float64"):
        raise ValueError(f"dtype_name must be 'float32' or 'float64', got {dtype_name!r}")
    dtype = torch.float32 if dtype_name == "float32" else torch.float64
    atom_types = atom_types.to(device=dev, dtype=torch.long)
    coords = coords.to(device=dev, dtype=dtype)
    probe_coords = probe_coords.to(device=dev, dtype=dtype)
    chg_labels = chg_labels.to(device=dev, dtype=dtype)
    cell = cell.to(device=dev, dtype=dtype)
    pack.gto_dict.to(dev)
    pack.orb_index = pack.orb_index.to(dev)
    pack.unique_atom_types = pack.unique_atom_types.to(dev)
    pack.n_orbitals = pack.n_orbitals.to(dev)
    pack.col_maps = {k: v.to(dev) for k, v in pack.col_maps.items()}

    scale_t = torch.tensor(float(scale), device=dev, dtype=dtype)
    n_orb_sum = _total_orbitals(atom_types, pack)
    norm = math.sqrt(float(n_orb_sum))
    d_padded = atom_types.shape[0] * pack.max_outdim
    if compact_active_orbitals:
        _active_cols, d_active, _ = build_active_column_indices(atom_types, pack)
    else:
        d_active = d_padded

    p_total = probe_coords.shape[0]
    g = torch.Generator(device=dev)
    g.manual_seed(int(seed))
    if max_probe_samples is not None and max_probe_samples >= 0 and max_probe_samples < p_total:
        perm = torch.randperm(p_total, device=dev, generator=g)[: int(max_probe_samples)]
    else:
        perm = torch.arange(p_total, device=dev)

    core_kwargs = dict(
        atom_types=atom_types,
        coords=coords,
        cell=cell,
        pbc=pbc,
        probe_coords=probe_coords,
        chg_labels=chg_labels,
        pack=pack,
        scale_t=scale_t,
        perm=perm,
        norm=norm,
        ridge=ridge,
        ridge_diag_frac=ridge_diag_frac,
        probe_weighting=probe_weighting,
        chunk_size=chunk_size,
        dtype_name=dtype_name,
        dev=dev,
        dtype=dtype,
        density_synthesis_mode=synth_mode,
        coeff_nonneg=bool(coeff_nonneg),
        compact_active_orbitals=compact_active_orbitals,
    )

    if solve_method == "iterative_cg":
        return _project_labels_to_gt_coeffs_iterative(
            **core_kwargs,
            cg_maxiter=cg_maxiter,
            cg_tol=cg_tol,
        )

    if accumulate_on_cpu is True or dev.type != "cuda":
        return _project_labels_to_gt_coeffs_once(**core_kwargs, accumulate_on_cpu=True)

    return _project_gpu_first_with_oom_policy(
        core_kwargs, d_active, d_padded, dev, oom_policy=oom_policy
    )


@torch.no_grad()
def max_abs_diff_phi_vs_gto_forward(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    probe_coords: torch.Tensor,
    probe_idx: torch.Tensor,
    pack: GTOBasisPack,
    pbc: bool = False,
    cell: Optional[torch.Tensor] = None,
    chunk_size: int = 2048,
    n_atom_col_samples: int = 8,
    seed: int = 0,
) -> float:
    """
    Spot-check that ``Phi`` columns match ``GTOs.forward`` with unit coefficients.
    Returns max |Phi_ij - rho_unit_j| / scale-free basis value (max abs over samples).
    """
    dev = coords.device
    g = torch.Generator(device=dev)
    g.manual_seed(int(seed))
    Phi = build_design_matrix_chunked(
        atom_types, coords, probe_coords, probe_idx, pack, pbc=pbc, cell=cell, chunk_size=chunk_size
    )
    n_atom = atom_types.shape[0]
    if n_atom == 0:
        return 0.0
    atom_ids = torch.randperm(n_atom, device=dev, generator=g)[: min(n_atom_col_samples, n_atom)]
    max_diff = 0.0
    n_probe = torch.tensor([int(probe_idx.shape[0])], device=dev, dtype=torch.long)
    probes = probe_coords[probe_idx]
    cell_b = cell if cell is not None and cell.dim() == 3 else cell.unsqueeze(0) if cell is not None else None

    for a in atom_ids.tolist():
        z = int(atom_types[a].item())
        gto = pack.gto_dict[str(z)]
        col_map = pack.col_maps[z]
        base = a * pack.max_outdim
        atom_coords_z = coords[a : a + 1]
        n_atoms = torch.tensor([1], device=dev, dtype=torch.long)
        for j, col_i in enumerate(col_map.tolist()):
            coeffs_z = torch.zeros(1, gto.outdim, device=dev, dtype=coords.dtype)
            coeffs_z[0, j] = 1.0
            rho_u = gto(
                probe_coords=probes,
                atom_coords=atom_coords_z,
                n_probes=n_probe,
                n_atoms=n_atoms,
                coeffs=coeffs_z,
                expo_scaling=None,
                pbc=pbc,
                cell=cell_b,
            )
            phi_col = Phi[:, base + col_i]
            max_diff = max(max_diff, float((phi_col - rho_u).abs().max().item()))
    return max_diff


def attach_gt_coeffs_to_data(
    data,
    pack: GTOBasisPack,
    scale: float,
    ridge: float = 1e-3,
    ridge_diag_frac: float = 0.0,
    probe_weighting: Optional[str] = None,
    max_probe_samples: int = -1,
    seed: int = 0,
    device: str = "cpu",
    dtype_name: str = "float64",
    chunk_size: int = 4096,
    accumulate_on_cpu: Optional[bool] = None,
    density_synthesis_mode: str = "linear",
    coeff_nonneg: bool = False,
    compact_active_orbitals: bool = False,
    lmdb_pbc: bool = False,
    oom_policy: str = "cpu_fallback",
) -> None:
    """In-place: sets ``data['gt_coeffs']`` tensor (num_nodes, max_outdim)."""
    pbc = resolve_pbc_for_graph(data, lmdb_pbc)
    cell = data.cell
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
    gt = project_labels_to_gt_coeffs(
        data.atom_types,
        data.coords,
        cell,
        pbc,
        data.probe_coords,
        data.chg_labels,
        pack,
        scale=scale,
        ridge=ridge,
        ridge_diag_frac=ridge_diag_frac,
        probe_weighting=probe_weighting,
        max_probe_samples=max_probe_samples,
        seed=seed,
        device=device,
        dtype_name=dtype_name,
        chunk_size=chunk_size,
        accumulate_on_cpu=accumulate_on_cpu,
        density_synthesis_mode=density_synthesis_mode,
        coeff_nonneg=coeff_nonneg,
        compact_active_orbitals=compact_active_orbitals,
        oom_policy=oom_policy,
    )
    data["gt_coeffs"] = gt.to(device=data.coords.device, dtype=data.coords.dtype)


def encode_gt_coeffs_tensor(gt: torch.Tensor) -> bytes:
    return pickle.dumps(gt, protocol=-1)


def compute_gt_coeffs_payload(
    raw: bytes,
    pack: GTOBasisPack,
    scale: float,
    ridge: float,
    ridge_diag_frac: float,
    probe_weighting: Optional[str],
    max_probe_samples: int,
    seed: int,
    device: str,
    dtype_name: str,
    chunk_size: int,
    accumulate_on_cpu: Optional[bool] = None,
    density_synthesis_mode: str = "linear",
    coeff_nonneg: bool = False,
    compact_active_orbitals: bool = False,
    lmdb_pbc: bool = False,
    probe_subsample_above: int = -1,
    oom_policy: str = "cpu_fallback",
) -> bytes:
    """Unpickle one LMDB graph, project labels to gt_coeffs, return pickled tensor bytes."""
    data = pickle.loads(raw)
    p_total = int(data.probe_coords.shape[0])
    effective_max = resolve_effective_max_probe_samples(
        p_total, max_probe_samples, probe_subsample_above
    )
    if (
        int(probe_subsample_above) >= 0
        and p_total > int(probe_subsample_above)
        and effective_max >= 0
        and effective_max < p_total
    ):
        print(
            f"gt_coeff probe subsample: n_probe={p_total} -> {effective_max} "
            f"(threshold={int(probe_subsample_above)})",
            flush=True,
        )
    attach_gt_coeffs_to_data(
        data,
        pack,
        scale=scale,
        ridge=ridge,
        ridge_diag_frac=ridge_diag_frac,
        probe_weighting=probe_weighting,
        max_probe_samples=effective_max,
        seed=seed,
        device=device,
        dtype_name=dtype_name,
        chunk_size=chunk_size,
        accumulate_on_cpu=accumulate_on_cpu,
        density_synthesis_mode=density_synthesis_mode,
        coeff_nonneg=coeff_nonneg,
        compact_active_orbitals=compact_active_orbitals,
        lmdb_pbc=lmdb_pbc,
        oom_policy=oom_policy,
    )
    return encode_gt_coeffs_tensor(data.gt_coeffs)


def load_target_var_from_metadata(path: Path) -> float:
    with open(path, "r") as fp:
        meta = json.load(fp)
    return float(meta["target_var"])


def resolve_pbc_for_graph(data, lmdb_pbc: bool = False) -> bool:
    """Per-graph ``data.pbc`` when stored; else LMDB-wide default (MP ``units.json``)."""
    if hasattr(data, "pbc") and data.pbc is not None:
        return bool(data.pbc)
    return bool(lmdb_pbc)


def load_pbc_from_lmdb_dir(lmdb_dir: Path) -> bool:
    """Read ``pbc`` from ``units.json`` (MP preprocess only). Returns False if absent."""
    units_path = Path(lmdb_dir) / "units.json"
    if not units_path.is_file():
        return False
    with open(units_path, encoding="utf-8") as fp:
        return bool(json.load(fp).get("pbc", False))


def infer_unique_atom_types_from_lmdb_sample(dataset, max_samples: int = 5000) -> List[int]:
    """Scan first graphs in an LMDB dataset for atomic numbers present (including 0 vnode)."""
    zs = set()
    n = min(len(dataset), max_samples)
    for i in range(n):
        d = dataset[i]
        zs.update(int(x) for x in d.atom_types.numpy().tolist())
    return sorted(zs)


def make_pack_for_lmdb(
    unique_atom_types: Sequence[int],
    dft_basis_set: str,
    dft_wt_aug: bool,
    beta: float,
    lmax_restriction: bool,
    uncontracted: bool,
    orb_cutoff: float,
    vnode_elem: int,
    device: str = "cpu",
    density_synthesis_mode: str = "linear",
) -> GTOBasisPack:
    return make_gto_basis_pack(
        unique_atom_types=sorted(set(int(z) for z in unique_atom_types)),
        dft_basis_set=dft_basis_set,
        dft_wt_aug=dft_wt_aug,
        beta=beta,
        lmax_restriction=lmax_restriction,
        lmax_relax=0,
        uncontracted=uncontracted,
        orb_cutoff=orb_cutoff,
        vnode_elem=vnode_elem,
        device=device,
        density_synthesis_mode=density_synthesis_mode,
    )
