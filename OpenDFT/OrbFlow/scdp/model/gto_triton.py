"""Triton GTO pair density: same math as ``(R[:,rad] * Y[:,sph] * c).sum(-1)``.

Spherical harmonics stay in e3nn so Y matches the original kernel exactly.
The kernel never materializes ``phi[N, n_orb]``.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_OK = True
except Exception:
    _TRITON_OK = False


if _TRITON_OK:

    @triton.jit
    def _pair_rho_kernel(
        sph_ptr,
        coeff_ptr,
        rad_idx_ptr,
        sph_idx_ptr,
        expos_ptr,
        ls_ptr,
        lognorm_ptr,
        prefac_ptr,
        r2_ptr,
        logr_ptr,
        out_ptr,
        n_orb,
        sph_stride,
        coeff_stride,
        BLOCK_K: tl.constexpr,
    ):
        # int64: pid * stride exceeds 2**31 for large probe chunks (e.g. QM9 eval).
        pid = tl.program_id(0).to(tl.int64)
        r2 = tl.load(r2_ptr + pid)
        logr = tl.load(logr_ptr + pid)
        acc = 0.0
        k0 = 0
        while k0 < n_orb:
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < n_orb
            rid = tl.load(rad_idx_ptr + offs, mask=mask, other=0)
            sid = tl.load(sph_idx_ptr + offs, mask=mask, other=0)
            expos = tl.load(expos_ptr + rid, mask=mask, other=0.0)
            ls = tl.load(ls_ptr + rid, mask=mask, other=0.0)
            lnm = tl.load(lognorm_ptr + rid, mask=mask, other=0.0)
            pref = tl.load(prefac_ptr + rid, mask=mask, other=0.0)
            y = tl.load(sph_ptr + pid * sph_stride + sid, mask=mask, other=0.0)
            c = tl.load(coeff_ptr + pid * coeff_stride + offs, mask=mask, other=0.0)
            radial = pref * tl.exp(-expos * r2 + ls * logr + lnm)
            acc += tl.sum(radial * y * c)
            k0 += BLOCK_K
        tl.store(out_ptr + pid, acc)


def pair_density_triton(gto, vecs, spherical, pair_coeffs) -> torch.Tensor:
    """``rho_pair = (R * Y * c).sum`` without a ``phi`` tensor. CUDA float32 only."""
    if not _TRITON_OK:
        raise RuntimeError("triton is not available")
    if vecs.numel() == 0:
        return vecs.new_zeros(0)
    if not vecs.is_cuda:
        raise RuntimeError("Triton GTO kernel requires CUDA")
    if pair_coeffs.dtype != torch.float32 or spherical.dtype != torch.float32:
        raise RuntimeError("Triton GTO kernel requires float32")

    n_pair = int(vecs.shape[0])
    r = torch.linalg.norm(vecs, dim=-1) + 1e-8
    r2 = (r * r).contiguous()
    logr = torch.log(r).contiguous()
    sph = spherical.contiguous()
    coeffs = pair_coeffs.contiguous()
    rad_idx = gto.rad_idx.to(dtype=torch.int32, device=vecs.device).contiguous()
    sph_idx = gto.sph_idx.to(dtype=torch.int32, device=vecs.device).contiguous()
    expos = gto.expos.to(dtype=torch.float32, device=vecs.device).contiguous()
    ls = gto.Ls.to(dtype=torch.float32, device=vecs.device).contiguous()
    if gto.normalize:
        lognorm = gto.lognorm.to(dtype=torch.float32, device=vecs.device).contiguous()
        prefac = torch.ones_like(expos)
    else:
        lognorm = torch.zeros_like(expos)
        prefac = gto.coeffs.to(dtype=torch.float32, device=vecs.device).contiguous()

    out = torch.empty(n_pair, device=vecs.device, dtype=torch.float32)
    _pair_rho_kernel[(n_pair,)](
        sph,
        coeffs,
        rad_idx,
        sph_idx,
        expos,
        ls,
        lognorm,
        prefac,
        r2,
        logr,
        out,
        int(coeffs.shape[1]),
        sph.stride(0),
        coeffs.stride(0),
        BLOCK_K=64,
    )
    return out
