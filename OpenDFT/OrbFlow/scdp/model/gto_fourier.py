"""Analytical Fourier transform of OrbFlow GTOs → plane-wave coeffs → density.

Matches ``GTOs.compute`` conventions:
  - length: Å vectors are converted to a.u. via ``/ GTO_VEC_LENGTH_SCALE_TO_AU``
  - angular: e3nn real SH with ``normalization='norm'`` on coords reordered ``[y, z, x]``
  - radial: ``N * r^L * exp(-ζ r^2)`` with the same ``lognorm`` as ``GTOs``

For a crystal cell, reciprocal lattice samples of the continuous FT are the structure
factors of the *periodized* density (Poisson summation). With
``c_DFT = F(G) / dv`` and ``torch.fft.irfftn(..., norm='backward')`` one recovers
the real-space density on the ``corner`` grid (frac = k/n).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
from e3nn import o3

from scdp.common.constants import GTO_VEC_LENGTH_SCALE_TO_AU
from scdp.model.gtos import EPSILON, GTOs


def reciprocal_lattice_G_rfft(
    cell: torch.Tensor,
    grid_shape: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return G (Å^-1) on the rfft grid, |G|, and Miller (H,K,L).

    ``cell`` is (3, 3) with rows = lattice vectors (Å), same as LMDB / ASE.
    """
    nx, ny, nz = (int(x) for x in grid_shape)
    device, dtype = cell.device, cell.dtype
    hx = torch.fft.fftfreq(nx, d=1.0, device=device, dtype=dtype) * nx
    hy = torch.fft.fftfreq(ny, d=1.0, device=device, dtype=dtype) * ny
    hz = torch.fft.rfftfreq(nz, d=1.0, device=device, dtype=dtype) * nz
    H, K, L = torch.meshgrid(hx, hy, hz, indexing="ij")
    # G = 2π (h,k,l) @ inv(cell).T  so that G · (frac @ cell) = 2π h·frac
    B = 2.0 * math.pi * torch.linalg.inv(cell).transpose(0, 1)
    HKL = torch.stack([H, K, L], dim=-1)
    G = HKL @ B
    Gnorm = torch.linalg.norm(G, dim=-1)
    return G, Gnorm, HKL


def _gto_lognorm(Ls: torch.Tensor, expos: torch.Tensor) -> torch.Tensor:
    power = Ls + 1.5
    numerator = power * torch.log(2 * expos) + math.log(2)
    denominator = torch.special.gammaln(power)
    return (numerator - denominator) / 2


def primitive_gto_ft_at_G(
    G_ang: torch.Tensor,
    Ls: torch.Tensor,
    expos: torch.Tensor,
    *,
    normalize: bool = True,
    coeffs_radial: Optional[torch.Tensor] = None,
    phase_sign: float = -1.0,
) -> torch.Tensor:
    """Continuous FT F(G)=∫ φ(r) e^{-i G·r} d³r for each uncontracted orbital.

    Args:
        G_ang: (..., 3) reciprocal vectors in Å^-1
        Ls, expos: (n_shell,) basis (same as ``GTOs``)
        phase_sign: angular factor ``(phase_sign * i)^L``. Empirically ``-1``
            (i.e. ``(-i)^L``) matches ``GTOs.compute`` + e3nn real SH for odd L;
            even L is unchanged vs ``+1``.

    Returns:
        complex tensor (..., n_orb) with n_orb = sum(2L+1)
    """
    device = G_ang.device
    dtype = G_ang.dtype
    Ls = Ls.to(device=device)
    expos = expos.to(device=device, dtype=dtype)
    if coeffs_radial is not None:
        coeffs_radial = coeffs_radial.to(device=device, dtype=dtype)

    s = float(GTO_VEC_LENGTH_SCALE_TO_AU)
    # Work in a.u. for radial FT; Y_lm depends only on direction.
    G_au = G_ang * s
    Gnorm_au = torch.linalg.norm(G_au, dim=-1)
    # e3nn reorder: (x,y,z) -> (y,z,x), same as GTOs.forward
    G_dir = G_au[..., [1, 2, 0]]
    # Avoid NaN Y at G=0: use a tiny dummy direction; |G|^L kills L>0.
    G_safe = torch.where(
        Gnorm_au[..., None] > 1e-12,
        G_dir,
        torch.zeros_like(G_dir) + torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype),
    )

    Lmax = int(Ls.max().item()) if Ls.numel() else 0
    sph = o3.spherical_harmonics(
        list(range(Lmax + 1)),
        G_safe,
        normalize=True,
        normalization="norm",
    )  # (..., (Lmax+1)^2)

    rad_idx = torch.repeat_interleave(torch.arange(len(Ls), device=device), 2 * Ls + 1)
    sph_idx = torch.cat(
        [torch.arange(int(l) ** 2, int(l) ** 2 + 2 * int(l) + 1, device=device) for l in Ls]
    )

    if normalize:
        logN = _gto_lognorm(Ls, expos)
        N = torch.exp(logN)
    else:
        assert coeffs_radial is not None
        N = coeffs_radial

    # F0 = N (π/ζ)^{3/2} (phase_sign * i * |G| / (2ζ))^L exp(-|G|^2/(4ζ)) Y
    # Length Jacobian: d³r_Å = s^3 d³r_au ⇒ F_Å = s^3 F_au
    pi_over_z = math.pi / expos
    gauss_vol = pi_over_z.pow(1.5)  # (n_shell,)
    exp_g = torch.exp(-(Gnorm_au[..., None] ** 2) / (4.0 * expos))  # (..., n_shell)

    L_shell = Ls.to(dtype=dtype)
    # (|G|/(2ζ))^L
    ratio = Gnorm_au[..., None] / (2.0 * expos)
    # stable: 0^0 = 1
    pow_term = torch.where(
        L_shell == 0,
        torch.ones_like(ratio),
        ratio.clamp_min(0.0).pow(L_shell),
    )

    # complex phase (phase_sign * i)^L
    # i^L cycles: 1, i, -1, -i
    phase = torch.ones(len(Ls), dtype=torch.complex64, device=device)
    for idx, L in enumerate(Ls.tolist()):
        phase[idx] = (1j * float(phase_sign)) ** int(L)

    # F_au per shell (..., n_shell) complex
    amp = (N * gauss_vol).to(dtype=Gnorm_au.dtype) * pow_term * exp_g
    F_shell = amp.to(torch.complex64) * phase.to(torch.complex64)

    Y = sph[..., sph_idx].to(torch.complex64)
    F_orb_au = F_shell[..., rad_idx] * Y
    F_orb_ang = F_orb_au * (s ** 3)
    return F_orb_ang


def orbital_zeta_mask(
    gto: GTOs,
    zeta_max: Optional[float] = None,
    zeta_min: Optional[float] = None,
) -> torch.Tensor:
    """Boolean mask over uncontracted orbitals with ``zeta_min < ζ <= zeta_max`` (a.u.).

    ``None`` means no bound on that side. Soft hybrid shells use ``zeta_max=τ`` (and
    ``zeta_min=None``); sharp shells use ``zeta_min=τ``.
    """
    device = gto.Ls.device
    keep_shell = torch.ones(gto.expos.shape[0], dtype=torch.bool, device=device)
    if zeta_max is not None:
        keep_shell = keep_shell & (gto.expos <= float(zeta_max))
    if zeta_min is not None:
        keep_shell = keep_shell & (gto.expos > float(zeta_min))
    return torch.repeat_interleave(keep_shell, 2 * gto.Ls + 1)


def structure_factor_from_gtos(
    cell: torch.Tensor,
    grid_shape: Sequence[int],
    atom_coords: torch.Tensor,
    atom_types: torch.Tensor,
    coeffs: torch.Tensor,
    gto_dict: torch.nn.ModuleDict,
    orb_index: torch.Tensor,
    *,
    phase_sign: float = -1.0,
    chunk_G: int = 65536,
    zeta_max: Optional[float] = None,
    zeta_min: Optional[float] = None,
) -> torch.Tensor:
    """Assemble rfft structure factors c_DFT = F(G)/dv for a batch of atoms.

    ``coeffs`` must already be in the same space as ``GTOs.forward`` (normalized
    active coeffs; caller applies ``orb_index`` slicing per element).
    Gradients flow to ``coeffs`` (and positions if needed later).
    """
    cell = cell.reshape(3, 3)
    G, _, _ = reciprocal_lattice_G_rfft(cell, grid_shape)
    nx, ny, nz = (int(x) for x in grid_shape)
    nzr = nz // 2 + 1
    device = cell.device

    V = torch.abs(torch.det(cell))
    N = nx * ny * nz
    dv = V / N

    F_flat = torch.zeros(nx * ny * nzr, dtype=torch.complex64, device=device)
    G_flat = G.reshape(-1, 3)
    nG = G_flat.shape[0]

    unique_types = torch.unique(atom_types)
    for t in unique_types:
        t_i = int(t.item())
        gto: GTOs = gto_dict[str(t_i)]
        mask = atom_types == t
        R = atom_coords[mask]  # (n_a, 3) Å
        c_full = coeffs[mask]  # (n_a, max_outdim)
        c = c_full[:, orb_index[t_i]].to(torch.complex64)  # (n_a, n_orb)
        zmask = orbital_zeta_mask(gto, zeta_max=zeta_max, zeta_min=zeta_min).to(
            device=c.device
        )
        if not bool(zmask.any()):
            continue
        if not bool(zmask.all()):
            c = c * zmask.to(dtype=c.dtype)[None, :]

        for start in range(0, nG, chunk_G):
            stop = min(start + chunk_G, nG)
            G_chunk = G_flat[start:stop]
            # (nG_c, n_orb)
            F0 = primitive_gto_ft_at_G(
                G_chunk,
                gto.Ls,
                gto.expos,
                normalize=gto.normalize,
                coeffs_radial=None if gto.normalize else gto.coeffs,
                phase_sign=phase_sign,
            )
            if not bool(zmask.all()):
                F0 = F0 * zmask.to(dtype=F0.dtype)[None, :]
            # phase e^{-i G·R}: (nG_c, n_a)
            phase = torch.exp(-1j * torch.matmul(G_chunk, R.T).to(torch.complex64))
            # sum_a phase_a * sum_orb F0_orb * c_a,orb
            # (nG_c, n_a, n_orb)
            contrib = phase[:, :, None] * F0[:, None, :] * c[None, :, :]
            F_flat[start:stop] += contrib.sum(dim=(1, 2))

    cG = (F_flat / dv.to(torch.float32)).reshape(nx, ny, nzr)
    return cG


def density_from_gto_fft(
    cell: torch.Tensor,
    grid_shape: Sequence[int],
    atom_coords: torch.Tensor,
    atom_types: torch.Tensor,
    coeffs: torch.Tensor,
    gto_dict: torch.nn.ModuleDict,
    orb_index: torch.Tensor,
    *,
    scale: float = 1.0,
    phase_sign: float = -1.0,
    chunk_G: int = 65536,
    fft_norm: str = "backward",
    zeta_max: Optional[float] = None,
    zeta_min: Optional[float] = None,
) -> torch.Tensor:
    """Full-grid density via analytical GTO FT + ``irfftn`` (flat, length nx*ny*nz).

    ``zeta_max`` / ``zeta_min`` restrict which shells contribute (inclusive bounds on ζ).
    """
    cG = structure_factor_from_gtos(
        cell,
        grid_shape,
        atom_coords,
        atom_types,
        coeffs,
        gto_dict,
        orb_index,
        phase_sign=phase_sign,
        chunk_G=chunk_G,
        zeta_max=zeta_max,
        zeta_min=zeta_min,
    )
    nx, ny, nz = (int(x) for x in grid_shape)
    rho = torch.fft.irfftn(cG, s=(nx, ny, nz), norm=fft_norm).real
    return (rho * float(scale)).reshape(-1)


@torch.no_grad()
def nmape(pred: torch.Tensor, target: torch.Tensor) -> float:
    denom = target.abs().sum().clamp_min(1e-12)
    return float((100.0 * (pred - target).abs().sum() / denom).item())
