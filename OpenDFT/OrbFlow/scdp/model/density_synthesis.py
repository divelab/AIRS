"""
Density synthesis: map GTO values and per-orbital coefficients to probe density.

``linear`` (default):
  Basis psi_j = phi_j,   coefficient c_j,   rho = sum_j c_j psi_j(r)

``squared``:
  Basis psi_j = phi_j^2, coefficient d_j (d >= 0),   rho = sum_j d_j psi_j(r)

  Underlying GTO exponents are scaled by ``SQUARED_GTO_EXPONENT_SCALE`` (default 0.5) so
  psi_j = phi_j^2 has Gaussian decay exp(-zeta r^2) rather than exp(-2 zeta r^2). ``GTOs``
  recompute ``lognorm`` from the scaled exponents; int phi_j^2 dr = 1 is preserved.

Squared mode uses d directly everywhere (``gt_coeffs``, network readout, flow bridge,
Euler integration). GT / SAD sidecars fit d on Phi_sq with ridge LS. Set ``coeff_nonneg=True`` for ``d >= 0`` NNLS;
default unconstrained path uses cuSOLVER (same as linear mode).

GTO ``normalize=True`` is unchanged so int phi_j^2 dr = 1 per basis function.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

import torch

VALID_DENSITY_SYNTHESIS_MODES = ("linear", "squared")

# Halve primitive zeta so psi=phi^2 has exp(-zeta r^2) not exp(-2 zeta r^2).
SQUARED_GTO_EXPONENT_SCALE = 0.5


def resolve_density_synthesis_mode(mode: Optional[str] = None) -> str:
    if mode is None or not str(mode).strip():
        return "linear"
    name = str(mode).strip().lower()
    if name not in VALID_DENSITY_SYNTHESIS_MODES:
        raise ValueError(
            f"density_synthesis_mode must be one of {VALID_DENSITY_SYNTHESIS_MODES}, got {mode!r}"
        )
    return name


def is_squared_density_synthesis(mode: Optional[str]) -> bool:
    return resolve_density_synthesis_mode(mode) == "squared"


def gto_exponent_scale(mode: Optional[str]) -> float:
    """Scale applied to all GTO ``expos`` when building primitives (1.0 = linear)."""
    return SQUARED_GTO_EXPONENT_SCALE if is_squared_density_synthesis(mode) else 1.0


def scale_basis_set_exponents(
    basis_set: Mapping[Union[int, str], Dict[str, Any]],
    scale: float,
) -> Dict[Union[int, str], Dict[str, Any]]:
    """Return a copy with every primitive exponent multiplied by ``scale``."""
    if scale == 1.0:
        return dict(basis_set)
    out: Dict[Union[int, str], Dict[str, Any]] = {}
    for z, entry in basis_set.items():
        scaled = dict(entry)
        scaled["expos"] = [float(e) * scale for e in entry["expos"]]
        out[z] = scaled
    return out


def apply_gto_exponent_scale_for_mode(
    basis_set: Mapping[Union[int, str], Dict[str, Any]],
    mode: Optional[str],
) -> Dict[Union[int, str], Dict[str, Any]]:
    """Apply squared-only exponent scaling after ETB augmentation."""
    return scale_basis_set_exponents(basis_set, gto_exponent_scale(mode))


def synthesis_basis_from_phi(phi: torch.Tensor, mode: str) -> torch.Tensor:
    """psi_j(r): phi_j (linear) or phi_j^2 (squared)."""
    if is_squared_density_synthesis(mode):
        return phi * phi
    return phi


def synthesis_weights_from_coeffs(coeffs: torch.Tensor, mode: str) -> torch.Tensor:
    """Bridge / readout coefficients paired with ``synthesis_basis_from_phi`` (c or d)."""
    return coeffs


def projection_weights_to_stored_coeffs(weights: torch.Tensor, mode: str) -> torch.Tensor:
    """LS / NNLS solution vector -> ``gt_coeffs`` row layout (already c or d)."""
    return weights
