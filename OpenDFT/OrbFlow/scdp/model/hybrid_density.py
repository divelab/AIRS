"""Hybrid density synthesis: soft GTOs via FT+IFFT, sharp GTOs via real-space.

Opt-in only (``density_eval_mode=hybrid_fft``). Default training stays real-space.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch

from scdp.model.gto_fourier import density_from_gto_fft, orbital_zeta_mask

DensityEvalMode = str  # "realspace" | "hybrid_fft"


def resolve_density_eval_mode(mode: Optional[str] = None) -> str:
    if mode is None:
        return "realspace"
    m = str(mode).strip().lower()
    aliases = {
        "realspace": "realspace",
        "real": "realspace",
        "rs": "realspace",
        "gto": "realspace",
        "hybrid_fft": "hybrid_fft",
        "hybrid": "hybrid_fft",
        "fft_hybrid": "hybrid_fft",
    }
    if m not in aliases:
        raise ValueError(
            f"unknown density_eval_mode={mode!r}; use 'realspace' (default) or 'hybrid_fft'"
        )
    return aliases[m]


def parse_grid_shape(grid_size) -> Tuple[int, int, int]:
    if torch.is_tensor(grid_size):
        vals = grid_size.detach().cpu().view(-1).tolist()
    else:
        vals = list(grid_size)
    if len(vals) < 3:
        raise ValueError(f"grid_size must have 3 entries, got {vals}")
    return int(vals[0]), int(vals[1]), int(vals[2])


def probe_flat_indices_from_coords(
    probe_coords: torch.Tensor,
    cell: torch.Tensor,
    grid_shape: Sequence[int],
    grid_convention: str = "corner",
) -> torch.Tensor:
    """Map Cartesian probe coords on a regular crystal grid to flat C-order indices.

    Assumes probes lie on the same grid as ``calculate_grid_pos`` (within roundoff).
    """
    cell = cell.reshape(3, 3)
    nx, ny, nz = (int(x) for x in grid_shape)
    # r = frac @ cell  ⇒  frac = r @ inv(cell)
    frac = probe_coords @ torch.linalg.inv(cell)
    frac = frac - torch.floor(frac)  # wrap to [0, 1)

    if grid_convention == "corner":
        # frac = k / n, k = 0..n-1
        ix = torch.remainder(torch.round(frac[:, 0] * nx), nx).long()
        iy = torch.remainder(torch.round(frac[:, 1] * ny), ny).long()
        iz = torch.remainder(torch.round(frac[:, 2] * nz), nz).long()
    elif grid_convention == "gpwno":
        # frac = (k+1)/n, k = 0..n-1
        ix = torch.remainder(torch.round(frac[:, 0] * nx) - 1, nx).long()
        iy = torch.remainder(torch.round(frac[:, 1] * ny) - 1, ny).long()
        iz = torch.remainder(torch.round(frac[:, 2] * nz) - 1, nz).long()
    else:
        raise ValueError(
            f"unknown grid_convention={grid_convention!r}; use 'corner' or 'gpwno'"
        )
    return ix * (ny * nz) + iy * nz + iz


def _zero_padded_outside_zeta(
    coeffs: torch.Tensor,
    atom_types: torch.Tensor,
    gto_dict: torch.nn.ModuleDict,
    orb_index: torch.Tensor,
    *,
    zeta_max: Optional[float] = None,
    zeta_min: Optional[float] = None,
) -> torch.Tensor:
    out = coeffs.clone()
    for t in torch.unique(atom_types):
        ti = int(t.item())
        gto: GTOs = gto_dict[str(ti)]
        zmask = orbital_zeta_mask(gto, zeta_max=zeta_max, zeta_min=zeta_min).to(
            device=coeffs.device
        )
        cols = orb_index[ti].nonzero(as_tuple=False).view(-1)
        if cols.numel() != zmask.numel():
            raise RuntimeError(
                f"orb layout mismatch type={ti}: cols={cols.numel()} orbs={zmask.numel()}"
            )
        drop = cols[~zmask]
        if drop.numel() == 0:
            continue
        rows = (atom_types == ti).nonzero(as_tuple=False).view(-1)
        out[rows[:, None], drop[None, :]] = 0
    return out


def any_shells_in_zeta_range(
    atom_types: torch.Tensor,
    gto_dict: torch.nn.ModuleDict,
    *,
    zeta_max: Optional[float] = None,
    zeta_min: Optional[float] = None,
) -> bool:
    for t in torch.unique(atom_types):
        gto = gto_dict[str(int(t.item()))]
        if bool(orbital_zeta_mask(gto, zeta_max=zeta_max, zeta_min=zeta_min).any()):
            return True
    return False


def hybrid_density_at_probes(
    *,
    cell: torch.Tensor,
    grid_shape: Sequence[int],
    grid_convention: str,
    atom_coords: torch.Tensor,
    atom_types: torch.Tensor,
    coeffs: torch.Tensor,
    probe_coords: torch.Tensor,
    n_probe: Union[int, torch.Tensor],
    gto_dict: torch.nn.ModuleDict,
    orb_index: torch.Tensor,
    zeta_soft_max: float,
    phase_sign: float = -1.0,
    chunk_G: int = 65536,
    pbc: bool = True,
    density_synthesis_mode: str = "linear",
) -> torch.Tensor:
    """Soft (ζ≤ζ_soft_max) via FT+IFFT + sharp (ζ>ζ_soft_max) via real-space GTO.

    Returns unscaled density at ``probe_coords`` (same units as ``GTOs.forward``).
    """
    if density_synthesis_mode != "linear":
        raise NotImplementedError(
            "hybrid_fft currently supports density_synthesis_mode='linear' only"
        )

    cell2 = cell.reshape(3, 3)
    nx, ny, nz = parse_grid_shape(grid_shape)
    shape = (nx, ny, nz)
    n_probe_t = (
        torch.tensor([int(n_probe)], device=probe_coords.device, dtype=torch.long)
        if not torch.is_tensor(n_probe)
        else n_probe.reshape(-1).to(device=probe_coords.device, dtype=torch.long)
    )
    if n_probe_t.numel() != 1:
        raise ValueError("hybrid_density_at_probes expects a single graph (n_probe scalar)")

    pred = torch.zeros(probe_coords.shape[0], device=coeffs.device, dtype=coeffs.dtype)

    # --- soft: FT + IFFT on full grid, gather probes ---
    if any_shells_in_zeta_range(atom_types, gto_dict, zeta_max=float(zeta_soft_max)):
        rho_full = density_from_gto_fft(
            cell=cell2,
            grid_shape=shape,
            atom_coords=atom_coords,
            atom_types=atom_types,
            coeffs=coeffs,
            gto_dict=gto_dict,
            orb_index=orb_index,
            scale=1.0,
            phase_sign=phase_sign,
            chunk_G=chunk_G,
            zeta_max=float(zeta_soft_max),
            zeta_min=None,
        )
        flat_idx = probe_flat_indices_from_coords(
            probe_coords, cell2, shape, grid_convention=grid_convention
        )
        pred = pred + rho_full.to(dtype=pred.dtype)[flat_idx]

    # --- sharp: real-space GTO with soft coeffs zeroed ---
    if any_shells_in_zeta_range(
        atom_types, gto_dict, zeta_min=float(zeta_soft_max)
    ):
        coeffs_sharp = _zero_padded_outside_zeta(
            coeffs,
            atom_types,
            gto_dict,
            orb_index,
            zeta_min=float(zeta_soft_max),
            zeta_max=None,
        )
        # Per-element loop mirrors ChgLightningModule.orbital_inference (unscaled).
        from scdp.common.utils import scatter

        unique_atom_types = torch.unique(atom_types)
        n_atoms_total = atom_types.shape[0]
        # batch index all zeros (single graph)
        batch_idx = torch.zeros(n_atoms_total, dtype=torch.long, device=atom_types.device)
        for i in unique_atom_types:
            ti = int(i.item())
            atom_mask = atom_types == i
            n_atom_i = scatter(atom_mask.long(), batch_idx, 1)
            gto = gto_dict[str(ti)]
            orb_i = orb_index[ti]
            pred = pred + gto(
                probe_coords=probe_coords,
                atom_coords=atom_coords[atom_mask],
                n_probes=n_probe_t,
                n_atoms=n_atom_i,
                coeffs=coeffs_sharp[atom_mask][:, orb_i],
                expo_scaling=None,
                pbc=pbc,
                cell=cell2.unsqueeze(0),
                density_synthesis_mode=density_synthesis_mode,
            )
    return pred
