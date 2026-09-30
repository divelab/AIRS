"""
Flow matching and Schrödinger-bridge-style training on per-atom GTO expansion coefficients.

**CFM** uses x_t = (1-t) x0 + t x1. Training supports two CFM objectives:

- ``flow_loss_mode=velocity``: target velocity v* = x1 - x0. Separate heads (``e3nn``/``mlp``)
  predict ``v`` directly; QHFlow unified (``flow_head=escn``) predicts the endpoint and
  uses v = (pred - x_t) / (1-t) for both training FM loss and Euler integration.
- ``flow_loss_mode=endpoint`` (QHFlow-style): head predicts bridge endpoint (``c_gt`` or
  ``c_gt - sad_coeffs`` per ``flow_target_mode``); inference uses the same
  v = (pred - x_t) / (1-t) conversion at each Euler step.
  Default training loss (``flow_endpoint_loss=density``): MAE/MSE on charge density from
  ``orbital_inference(head pred)``. ``flow_endpoint_loss=integrated_density``: same density
  MAE/MSE but on coeffs after ``n``-step Euler CFM integration (matches val ``nmape/val_flow``).
  Legacy ``flow_endpoint_loss=coeff``: masked MSE on coefficients.

**SB (Gaussian bridge)** uses the same mean path mu_t but samples
x_t = mu_t + sqrt(sigma^2 t(1-t)) * eta (masked noise) and trains the same MLP to match
the closed-form conditional score (mu_t - x_t) / var — a diagonal Brownian bridge
between coefficient endpoints, not the full iterative Schrödinger solver.

**SB noise shaping:** ``sb_noise_mode=isotropic`` uses one variance per node (broadcast to all
dims). ``sb_noise_mode=per_irrep`` applies a **per-angular-momentum block** scalar scale on
``max_n_orbitals_per_L`` layout (same slices as ``max_outdim_per_L``), so each coefficient
dimension can have a different ``Var ∝ scale_l^2``; see ``make_sb_per_dim_noise_scale``.

x0 is set by ``flow_prior_mode`` (direct d0 or c0 draw per ``density_synthesis_mode``,
or ``batch['sad_coeffs']``); x1 is ``batch.gt_coeffs``.

**Flow field:** ``flow_head=e3nn`` uses ``CoeffEquivariantVelocityHead`` (separate head on eSCN
latent + ``x_t`` + ``t``). ``flow_head=mlp`` is the flat MLP baseline. ``flow_head=escn``
(QHFlow unified) has no separate head: ``eSCN`` readout predicts the endpoint from
``(geometry, coeff_xt, bridge_t)`` with ``enable_flow_coeff_entangle`` and
``flow_bridge_time_dim``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from e3nn import o3

from scdp.common.utils import scatter

VALID_FLOW_TARGET_MODES = ("gt", "res")
VALID_FLOW_ENDPOINT_LOSSES = (
    "density",
    "coeff",
    "integrated_density",
    # One-stage joint: endpoint density + Euler-integrated density (see flow_combined_integrated_weight).
    "density_and_integrated",
)


def resolve_flow_target_mode(flow_target_mode: Optional[str] = None) -> str:
    """
    Resolve CFM endpoint / bridge target mode.

    - ``gt``: predict full ``c_gt``; bridge ends at ``c_gt``.
    - ``res``: predict ``c_gt - c_sad``; bridge ends at ``c_gt - c_sad``; sampling adds ``c_sad``.
    """
    if flow_target_mode is not None and str(flow_target_mode).strip():
        mode = str(flow_target_mode).strip()
    else:
        mode = "gt"
    if mode not in VALID_FLOW_TARGET_MODES:
        raise ValueError(
            f"flow_target_mode must be one of {VALID_FLOW_TARGET_MODES}, got {mode!r}"
        )
    return mode


def escn_flat_latent_dim(lmax_list, sphere_channels: int) -> int:
    """Match x_pt = x.embedding.flatten(1,2) in eSCN (single-resolution layout)."""
    num_coeff = sum((L + 1) ** 2 for L in lmax_list)
    return num_coeff * sphere_channels


def escn_spherical_irreps(lmax: int, sphere_channels_all: int) -> o3.Irreps:
    """Same layout as ``eSCN.orbit_readout`` input ``irreps_in`` (parity alternates with l)."""
    return o3.Irreps(
        [
            (sphere_channels_all, (ell, 1 if ell % 2 == 0 else -1))
            for ell in range(lmax + 1)
        ]
    )


def escn_orbital_irreps(max_n_orbitals_per_L: torch.Tensor) -> o3.Irreps:
    """Same layout as ``eSCN.orbit_readout`` output ``irreps_out`` (even parity, l from basis)."""
    pairs = [
        (int(x.item()), (ell, 1))
        for ell, x in enumerate(max_n_orbitals_per_L)
        if int(x.item()) > 0
    ]
    return o3.Irreps(pairs)


FLOW_INTEGRATE_SCHEDULES = ("uniform", "endpoint_refine", "power")
# Minimum Euler segments per schedule (``power`` needs ≥2 interior knots near ``t→1``).
FLOW_INTEGRATE_MIN_STEPS: dict[str, int] = {
    "uniform": 1,
    "endpoint_refine": 1,
    "power": 3,
}


def resolve_flow_integrate_schedule(schedule: Optional[str] = None) -> str:
    """Normalize ``flow_integrate_schedule`` (default ``uniform`` for backward compatibility)."""
    if schedule is None:
        return "uniform"
    name = str(schedule).strip().lower()
    if name not in FLOW_INTEGRATE_SCHEDULES:
        raise ValueError(
            f"flow_integrate_schedule must be one of {FLOW_INTEGRATE_SCHEDULES}, got {schedule!r}"
        )
    return name


def min_flow_integrate_steps(schedule: Optional[str] = None) -> int:
    """Minimum valid ``n_steps`` for ``build_flow_integrate_t_grid``."""
    return FLOW_INTEGRATE_MIN_STEPS[resolve_flow_integrate_schedule(schedule)]


def iter_flow_integrate_sweep_pairs(
    schedules: Sequence[str],
    step_counts: Sequence[int],
) -> list[tuple[str, int]]:
    """(schedule, n_steps) pairs that satisfy each schedule's minimum step count."""
    pairs: list[tuple[str, int]] = []
    for sched in schedules:
        sched_name = resolve_flow_integrate_schedule(sched)
        min_steps = min_flow_integrate_steps(sched_name)
        for n_steps in step_counts:
            n = int(n_steps)
            if n >= min_steps:
                pairs.append((sched_name, n))
    return pairs


def build_flow_integrate_t_grid(
    n_steps: int,
    min_t: float,
    *,
    schedule: str = "uniform",
    t_hi: float = 0.99,
    power: float = 4.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Time knots ``t_0, …, t_n`` for Euler CFM integration from ``min_t`` to ``1``.

    Schedules (``n_steps`` Euler segments → ``n_steps + 1`` knots):

    - ``uniform``: ``linspace(min_t, 1, n+1)`` (legacy default).
    - ``endpoint_refine``: one jump ``min_t → t_hi`` (default ``0.99``, top of training
      support), then ``n-1`` uniform refinements on ``[t_hi, 1]``.
    - ``power``: ``t_k = 1 - (1-min_t) * (1 - k/n)^α`` with ``α=power`` (front-loaded;
      ``α=1`` equals uniform).
    """
    n = int(n_steps)
    if n <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    min_t_val = float(min_t)
    if not (0.0 < min_t_val < 0.5):
        raise ValueError(f"min_t must be in (0, 0.5), got {min_t_val}")

    sched = resolve_flow_integrate_schedule(schedule)
    min_steps = min_flow_integrate_steps(sched)
    if n < min_steps:
        raise ValueError(
            f"flow_integrate_schedule={sched!r} requires n_steps >= {min_steps}, got {n}"
        )

    if sched == "uniform":
        return torch.linspace(min_t_val, 1.0, n + 1, device=device, dtype=dtype)

    if sched == "endpoint_refine":
        if n == 1:
            return torch.tensor([min_t_val, 1.0], device=device, dtype=dtype)
        t_hi_val = float(t_hi)
        if not (min_t_val < t_hi_val < 1.0):
            raise ValueError(
                f"flow_integrate_t_hi must satisfy min_t < t_hi < 1 for endpoint_refine "
                f"with n_steps>1; got min_t={min_t_val}, t_hi={t_hi_val}"
            )
        knots = [min_t_val, t_hi_val]
        n_refine = n - 1
        if n_refine == 1:
            knots.append(1.0)
        else:
            refine = torch.linspace(t_hi_val, 1.0, n_refine + 1, device=device, dtype=dtype)
            knots.extend(refine[1:].tolist())
        return torch.tensor(knots, device=device, dtype=dtype)

    alpha = float(power)
    if alpha <= 0.0:
        raise ValueError(f"flow_integrate_power must be positive, got {power!r}")
    k = torch.arange(n + 1, device=device, dtype=dtype)
    frac = (1.0 - k / float(n)).clamp(min=0.0).pow(alpha)
    return 1.0 - (1.0 - min_t_val) * frac


def sample_flow_bridge_t(
    n_graphs: int,
    device: torch.device,
    min_t: float = 0.01,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample one bridge time per graph, uniform on ``[min_t, 1 - min_t]``.

    Matches QHFlow ``sample_t`` (``DEFAULT_MIN_T=0.01`` → ``[0.01, 0.99]``).
    """
    min_t = float(min_t)
    if not (0.0 < min_t < 0.5):
        raise ValueError(f"min_t must be in (0, 0.5), got {min_t}")
    t = torch.rand(int(n_graphs), device=device, dtype=dtype)
    return t * (1.0 - 2.0 * min_t) + min_t


def linear_flow_path(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x0, x1: (N, D) same shape
        t: (N, 1) or broadcastable to (N, D)
    Returns:
        x_t, v_star where dx/dt = v_star = x1 - x0 along the path x_t = (1-t)x0 + t x1
    """
    x_t = (1.0 - t) * x0 + t * x1
    v_star = x1 - x0
    return x_t, v_star


def gaussian_bridge_mean(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Mean path (1-t) x0 + t x1 (same as linear CFM path)."""
    return (1.0 - t) * x0 + t * x1


def make_sb_per_dim_noise_scale(
    max_n_orbitals_per_L: torch.Tensor,
    noise_mode: str,
    per_l_sigma_mode: str = "inv_sqrt_2lp1",
    per_l_scales: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """
    Build ``(max_outdim,)`` positive scales ``s_d`` aligned with ``max_outdim_per_L`` blocks:
    for each ``l`` with ``n_l > 0``, append ``n_l * (2l+1)`` entries.

    Variance per coefficient (diagonal bridge): ``Var_d = (sigma * s_d)^2 * t * (1-t)``.

    - ``noise_mode == "isotropic"``: all ``s_d = 1`` (same as legacy single ``var`` per row).
    - ``noise_mode == "per_irrep"``: set ``s_d`` constant within each ``l`` block from
      ``per_l_scales[l]`` if given, else ``per_l_sigma_mode``:

        - ``uniform``: ``s = 1`` per block (same magnitudes as isotropic; only bookkeeping).
        - ``inv_sqrt_2lp1``: ``s = 1/sqrt(2l+1)`` within block ``l`` (down-weight higher ``l``).
    """
    if noise_mode not in ("isotropic", "per_irrep"):
        raise ValueError(f"noise_mode must be 'isotropic' or 'per_irrep', got {noise_mode!r}")
    max_n = max_n_orbitals_per_L.detach().cpu().long()
    parts: List[torch.Tensor] = []
    for l in range(max_n.numel()):
        n = int(max_n[l].item())
        if n <= 0:
            continue
        block = n * (2 * l + 1)
        if noise_mode == "isotropic":
            s = 1.0
        elif per_l_scales is not None:
            s = float(per_l_scales[l]) if l < len(per_l_scales) else 1.0
            if s <= 0:
                raise ValueError(f"sb per-l scale at l={l} must be positive, got {s}")
        elif per_l_sigma_mode == "uniform":
            s = 1.0
        elif per_l_sigma_mode == "inv_sqrt_2lp1":
            s = 1.0 / math.sqrt(2 * l + 1)
        else:
            raise ValueError(
                f"per_l_sigma_mode must be 'uniform' or 'inv_sqrt_2lp1', got {per_l_sigma_mode!r}"
            )
        parts.append(torch.full((block,), s, dtype=torch.float32))
    if not parts:
        raise ValueError("max_n_orbitals_per_L has no positive shells; cannot build SB scales")
    return torch.cat(parts, dim=0)


def schrodinger_bridge_gaussian_sample(
    mu_t: torch.Tensor,
    t: torch.Tensor,
    sigma: float,
    mask: torch.Tensor,
    per_dim_noise_scale: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Brownian-bridge marginal with diagonal Gaussian on coefficient indices.

    ``per_dim_noise_scale``: shape ``(D,)``, positive; ``Var_d = (sigma * s_d)^2 * t * (1-t)``.

    Returns:
        x_t, var with ``var`` shape ``(N, D)`` (per-dimension variance; broadcast-safe for score).
    """
    if per_dim_noise_scale.shape[0] != mu_t.shape[1]:
        raise ValueError(
            f"per_dim_noise_scale length {per_dim_noise_scale.shape[0]} != coeff dim {mu_t.shape[1]}"
        )
    device, dtype = mu_t.device, mu_t.dtype
    s = per_dim_noise_scale.to(device=device, dtype=dtype).clamp(min=1e-12).view(1, -1)
    base = (float(sigma) ** 2) * t * (1.0 - t)
    var = base * (s**2)
    var = var.clamp(min=1e-8)
    noise = torch.randn(mu_t.shape, device=device, dtype=dtype, generator=generator)
    noise = noise * mask
    x_t = mu_t + torch.sqrt(var) * noise
    return x_t, var


def gaussian_bridge_score_target(
    x_t: torch.Tensor,
    mu_t: torch.Tensor,
    var: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Conditional score (mu_t - x_t) / var on active dims (diagonal Gaussian; var (N,D) or (N,1))."""
    return (mu_t - x_t) / var.clamp(min=1e-8) * mask


def per_node_coeff_mask(atom_types: torch.Tensor, orb_index: torch.Tensor) -> torch.Tensor:
    """
    orb_index[z, d] is True if coefficient index d is active for atom type z.
    Returns (N, D) float mask.
    """
    return orb_index[atom_types.long()].to(dtype=torch.float32)


def num_graphs_in_batch(batch) -> int:
    """Number of graphs in a PyG batch (never use ``len(batch)`` — that can be num_nodes)."""
    if hasattr(batch, "batch") and batch.batch is not None and batch.batch.numel() > 0:
        return int(batch.batch.max().item()) + 1
    return 1


def coeff_norm_denominator(batch, n_orbitals: torch.Tensor, unique_atom_types: torch.Tensor) -> torch.Tensor:
    """Same sqrt(sum of orbital dims per graph) as ChgLightningModule.predict_coeffs."""
    dev = batch.coords.device
    atom_types = batch.atom_types.to(dev)
    n_orbitals = n_orbitals.to(dev)
    unique_atom_types = unique_atom_types.to(dev)
    n_orbs = n_orbitals[
        (atom_types.repeat(len(unique_atom_types), 1).T == unique_atom_types).nonzero()[
            :, 1
        ]
    ]
    ng = num_graphs_in_batch(batch)
    batch_n_orbs = scatter(n_orbs, batch.batch.to(dev), ng)
    per_graph = batch_n_orbs.sqrt().clamp(min=1e-8)
    return per_graph[batch.batch.to(dev)].view(-1, 1)


class FlowPriorStatsAccumulator:
    """Online mean / std / RMS over masked normalized coefficient values."""

    __slots__ = ("count", "sum", "sum_sq", "sum_abs")

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.sum_abs = 0.0

    def update(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        """Accumulate stats for ``values * mask`` (same shape, typically (N, D))."""
        m = mask.reshape(-1) > 0
        if not m.any():
            return
        x = (values * mask).reshape(-1)[m].double()
        self.count += int(x.numel())
        self.sum += float(x.sum().item())
        self.sum_sq += float(x.pow(2).sum().item())
        self.sum_abs += float(x.abs().sum().item())

    def finalize(self) -> dict:
        if self.count == 0:
            return {}
        mean = self.sum / self.count
        rms = (self.sum_sq / self.count) ** 0.5
        var = max(self.sum_sq / self.count - mean**2, 0.0)
        std = var**0.5
        return {
            "count": int(self.count),
            "mean": float(mean),
            "std": float(std),
            "rms": float(rms),
            "abs_mean": float(self.sum_abs / self.count),
            "suggested_flow_prior_mu": float(mean),
            "suggested_flow_prior_sigma": float(std),
        }


def accumulate_flow_prior_stats_batch(
    batch,
    accum_gt: FlowPriorStatsAccumulator,
    accum_res: Optional[FlowPriorStatsAccumulator],
    n_orbitals: torch.Tensor,
    unique_atom_types: torch.Tensor,
    orb_index: torch.Tensor,
) -> None:
    """
    Update accumulators with normalized ``gt_coeffs`` and optional residual target.

    Statistics match flow bridge space: c (linear) or d (squared) after ``gt_raw / norm``.
    """
    if "gt_coeffs" not in batch:
        return
    gt_raw = batch["gt_coeffs"].to(device=batch.coords.device, dtype=batch.coords.dtype)
    norm = coeff_norm_denominator(batch, n_orbitals, unique_atom_types)
    gt_norm = gt_raw / norm
    mask = per_node_coeff_mask(batch.atom_types, orb_index)
    accum_gt.update(gt_norm, mask)
    if accum_res is not None and "sad_coeffs" in batch:
        sad_raw = batch["sad_coeffs"].to(device=batch.coords.device, dtype=batch.coords.dtype)
        accum_res.update((gt_raw - sad_raw) / norm, mask)


def sample_flow_prior_c0_raw(
    batch,
    max_outdim: int,
    prior_mode: str,
    orb_index: torch.Tensor,
    orbital_irreps: o3.Irreps,
    sigma: float = 1.0,
    prior_mu: float = 0.0,
    n_orbitals: Optional[torch.Tensor] = None,
    unique_atom_types: Optional[torch.Tensor] = None,
    is_vnode: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    density_synthesis_mode: str = "linear",
) -> torch.Tensor:
    """
    Sample bridge start field in **raw** storage space (before graph norm in callers).

    ``linear``: c0 ~ sigma * noise + prior_mu (normalized center).
    ``squared``: d0 >= 0 sampled directly — (sigma * noise / norm + prior_mu).clamp(min=0) * norm
    on active GTO slots (same irrep / isotropic structure as linear, no c_aux^2 map).

    ``prior_mode``:
      - ``gaussian``: masked isotropic noise on active GTO slots.
      - ``irrep_gaussian``: ``orbital_irreps.randn`` per node (structured prior).
      - ``sad_coeffs``: ``batch['sad_coeffs']`` (c or d per ``density_synthesis_mode``).

    ``prior_mu``: center in **normalized** coefficient space.
    """
    from scdp.model.density_synthesis import is_squared_density_synthesis

    if prior_mode not in ("gaussian", "irrep_gaussian", "sad_coeffs"):
        raise ValueError(
            f"prior_mode must be 'gaussian', 'irrep_gaussian', or 'sad_coeffs', "
            f"got {prior_mode!r}"
        )
    device = batch.coords.device
    dtype = batch.coords.dtype
    n = batch.atom_types.shape[0]
    is_vnode = batch["is_vnode"] if is_vnode is None and "is_vnode" in batch else is_vnode
    mask = orb_index[batch.atom_types.long()].to(dtype=dtype)
    squared = is_squared_density_synthesis(density_synthesis_mode)

    if prior_mode == "sad_coeffs":
        if "sad_coeffs" not in batch:
            raise KeyError(
                "flow_prior_mode='sad_coeffs' requires batch['sad_coeffs'] on each graph "
                "(precompute SAD projections into the LMDB)."
            )
        return batch["sad_coeffs"].to(device=device, dtype=dtype)

    if prior_mode == "gaussian":
        noise = torch.randn(
            n,
            max_outdim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        z = noise * (float(sigma)) * mask
    else:
        irrep_dim = int(orbital_irreps.dim)
        if irrep_dim != int(max_outdim):
            raise ValueError(
                f"orbital_irreps.dim={irrep_dim} != max_outdim={max_outdim}; "
                "cannot sample irrep_gaussian prior"
            )
        z = orbital_irreps.randn(n, -1, device=device, dtype=torch.float32)
        if generator is not None:
            pass
        z = z.to(dtype=dtype) * float(sigma) * mask

    if is_vnode is not None:
        z = z * (~is_vnode).to(dtype).unsqueeze(1)

    if squared:
        if n_orbitals is None or unique_atom_types is None:
            raise ValueError(
                "density_synthesis_mode='squared' requires n_orbitals and unique_atom_types"
            )
        norm = coeff_norm_denominator(batch, n_orbitals, unique_atom_types)
        z_norm = z / norm.clamp(min=1e-8)
        d_norm = (z_norm + float(prior_mu)).clamp(min=0.0)
        return d_norm * norm

    if float(prior_mu) != 0.0:
        if n_orbitals is None or unique_atom_types is None:
            raise ValueError(
                "prior_mu != 0 requires n_orbitals and unique_atom_types to map normalized "
                "center into raw coefficient space"
            )
        norm = coeff_norm_denominator(batch, n_orbitals, unique_atom_types)
        z = z + float(prior_mu) * norm

    return z


def target_nelectrons_per_graph(batch) -> torch.Tensor:
    """
    Target valence electron count per graph for neutral PAW/VASP molecules
    (baseline ``chg_labels`` integrate to this; excludes virtual nodes).
    """
    from scdp.data.electron_count import valence_electron_count_per_graph

    return valence_electron_count_per_graph(batch)


def monte_carlo_integrated_charge(
    probe_density: torch.Tensor,
    probe_graph: torch.Tensor,
    cell_volume: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """
    Estimate ``∫ ρ dV`` from probe samples: ``V * mean(ρ)`` per graph (uniform MC over the cell).
    ``probe_density`` and ``cell_volume`` use the same units as ``chg_labels`` / ``orbital_inference``.
    """
    sum_rho = scatter(probe_density, probe_graph, num_graphs)
    counts = scatter(probe_density.new_ones(probe_density.shape[0]), probe_graph, num_graphs)
    mean_rho = sum_rho / counts.clamp(min=1.0)
    return mean_rho * cell_volume.view(-1)


class SinusoidalTimeEmbedding(nn.Module):
    """DDPM-style sinusoidal embedding for continuous t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (N, 1) in [0, 1]
        returns (N, dim)
        """
        out_dtype = t.dtype
        half = self.dim // 2
        t_flat = t.view(-1).clamp(0.0, 1.0).float()
        device = t_flat.device
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, device=device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t_flat.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[1]))
        elif emb.shape[1] > self.dim:
            emb = emb[:, : self.dim]
        return emb.to(dtype=out_dtype)


class CoeffEquivariantVelocityHead(nn.Module):
    """
    SO(3)-equivariant velocity / score on coefficients, using the same ``Irreps`` layout as ``eSCN``:

    - ``latent`` is flattened ``x_pt`` with ``irreps_in`` (parity alternates by ``l``).
    - ``x_t`` uses ``irreps_out`` (even parity, multiplicities from ``max_n_orbitals_per_L``).

    Time is sinusoidal scalars injected as ``time_dim x 0e`` via ``FullyConnectedTensorProduct``.
    Intermediate maps use ``o3.Linear`` only (no nonlinearity on mixed irreps).
    """

    def __init__(
        self,
        lmax_list: List[int],
        sphere_channels_all: int,
        max_n_orbitals_per_L: torch.Tensor,
        time_dim: int = 64,
        hidden_multiplier: int = 2,
    ):
        super().__init__()
        if len(lmax_list) != 1:
            raise ValueError(
                "CoeffEquivariantVelocityHead requires len(lmax_list)==1 (same as eSCN readout)."
            )
        lmax = int(lmax_list[0])
        self.irreps_latent = escn_spherical_irreps(lmax, sphere_channels_all)
        self.irreps_coeff = escn_orbital_irreps(max_n_orbitals_per_L)
        self.irreps_cat = self.irreps_latent + self.irreps_coeff

        hid = []
        for mul_ir in self.irreps_coeff:
            mul, ir = mul_ir
            hid.append((max(int(mul) * int(hidden_multiplier), int(mul)), ir))
        self.irreps_hidden = o3.Irreps(hid)

        self.lin_mix = o3.Linear(self.irreps_cat, self.irreps_hidden)
        self.lin_out = o3.Linear(self.irreps_hidden, self.irreps_coeff)
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.irreps_time = o3.Irreps(f"{int(time_dim)}x0e")
        self.time_tp = o3.FullyConnectedTensorProduct(
            self.irreps_coeff,
            self.irreps_time,
            self.irreps_coeff,
        )

    def forward(self, latent: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out_dtype = latent.dtype
        if latent.shape[-1] != self.irreps_latent.dim:
            raise ValueError(
                f"latent dim {latent.shape[-1]} != irreps_latent.dim {self.irreps_latent.dim}"
            )
        if x_t.shape[-1] != self.irreps_coeff.dim:
            raise ValueError(
                f"x_t dim {x_t.shape[-1]} != irreps_coeff.dim {self.irreps_coeff.dim}"
            )

        latent32 = latent.to(dtype=torch.float32)
        x32 = x_t.to(dtype=torch.float32)
        cat = torch.cat([latent32, x32], dim=-1)
        h = self.lin_mix(cat)
        h = self.lin_out(h)
        te = self.time_emb(t).to(dtype=torch.float32)
        out = self.time_tp(h, te)
        return out.to(dtype=out_dtype)


class CoeffVelocityMLP(nn.Module):
    """
    Field head on (latent, x_t, t): predicts either CFM **velocity** v_theta or SB **score** s_theta,
    same output shape (max_outdim); training objective chooses the target.
    """

    def __init__(
        self,
        latent_dim: int,
        coeff_dim: int,
        time_dim: int = 64,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.coeff_in = nn.Linear(coeff_dim, hidden_dim // 2)
        in_dim = latent_dim + hidden_dim // 2 + time_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, coeff_dim),
        )

    def forward(self, latent: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        latent: (N, latent_dim)
        x_t: (N, coeff_dim)
        t: (N, 1)
        """
        te = self.time_emb(t)
        xc = self.coeff_in(x_t)
        h = torch.cat([latent, xc, te], dim=-1)
        return self.net(h)


def flow_matching_loss(
    v_pred: torch.Tensor,
    v_star: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Masked MSE between predicted and target **vector fields** on coefficients (velocity or score)."""
    diff = (v_pred - v_star) ** 2 * mask
    denom = mask.sum().clamp(min=eps)
    return diff.sum() / denom


def resolve_flow_endpoint_loss(
    flow_loss_mode: str,
    flow_endpoint_loss: Optional[str] = None,
) -> str:
    """Default ``density`` when ``flow_loss_mode=endpoint``; unused for velocity mode."""
    if flow_loss_mode != "endpoint":
        return "coeff"
    if flow_endpoint_loss is None or not str(flow_endpoint_loss).strip():
        return "density"
    mode = str(flow_endpoint_loss).strip()
    if mode not in VALID_FLOW_ENDPOINT_LOSSES:
        raise ValueError(
            f"flow_endpoint_loss must be one of {VALID_FLOW_ENDPOINT_LOSSES}, got {mode!r}"
        )
    return mode


def t_per_graph(t: torch.Tensor, node_batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    """Extract one scalar ``t`` per graph from per-node ``t`` (constant within each graph)."""
    t_flat = t.reshape(-1)
    t_sum = scatter(t_flat, node_batch, n_graphs)
    counts = scatter(
        torch.ones_like(t_flat, dtype=t_flat.dtype),
        node_batch,
        n_graphs,
    )
    return t_sum / counts.clamp(min=1.0)


def endpoint_pred_to_norm_coeffs(
    pred: torch.Tensor,
    c_sad_norm: Optional[torch.Tensor],
    flow_target_mode: str,
) -> torch.Tensor:
    """
    Convert head endpoint prediction to normalized coefficients for density evaluation.

    ``flow_target_mode=res``: head predicts ``c_gt - c_sad``; add ``c_sad`` back before inference.
    """
    if flow_target_mode == "res":
        if c_sad_norm is None:
            raise ValueError("flow_target_mode='res' requires c_sad_norm for density loss")
        return pred + c_sad_norm
    return pred


def endpoint_density_per_probe_loss(
    pred_density: torch.Tensor,
    target_density: torch.Tensor,
    density_scale: torch.Tensor,
    *,
    criterion: str = "mae",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-probe MAE/MSE on scaled charge density (before graph reduction)."""
    scale = density_scale.reshape(()).clamp(min=eps)
    diff = pred_density / scale - target_density / scale
    if criterion == "mse":
        return diff.pow(2)
    if criterion == "mae":
        return diff.abs()
    raise ValueError(f"criterion must be 'mae' or 'mse', got {criterion!r}")


def accumulate_endpoint_density_flow_loss(
    sum_per_graph: Optional[torch.Tensor],
    count_per_graph: Optional[torch.Tensor],
    pred_density: torch.Tensor,
    target_density: torch.Tensor,
    probe_graph: torch.Tensor,
    density_scale: torch.Tensor,
    *,
    n_graphs: Optional[int] = None,
    criterion: str = "mae",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accumulate per-graph probe-loss sum/count (for chunked ``orbital_inference``)."""
    if n_graphs is None:
        n_graphs = int(probe_graph.max().item()) + 1 if probe_graph.numel() else 0
    else:
        n_graphs = int(n_graphs)
    per_probe = endpoint_density_per_probe_loss(
        pred_density,
        target_density,
        density_scale,
        criterion=criterion,
        eps=eps,
    )
    chunk_sum = scatter(per_probe, probe_graph, n_graphs)
    chunk_count = scatter(
        torch.ones_like(per_probe, dtype=per_probe.dtype),
        probe_graph,
        n_graphs,
    )
    if sum_per_graph is None:
        return chunk_sum, chunk_count
    return sum_per_graph + chunk_sum, count_per_graph + chunk_count


def finalize_endpoint_density_flow_loss(
    sum_per_graph: torch.Tensor,
    count_per_graph: torch.Tensor,
    node_batch: torch.Tensor,
    t: torch.Tensor,
    *,
    use_t_scale: bool = False,
    t_scale_max: float = 0.9,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Reduce accumulated per-graph sums to a scalar training loss."""
    n_graphs = int(sum_per_graph.numel())
    per_graph_loss = sum_per_graph / count_per_graph.clamp(min=1)

    if use_t_scale:
        t_graph = t_per_graph(t, node_batch, n_graphs)
        t_cap = torch.min(
            t_graph,
            torch.tensor(float(t_scale_max), device=t.device, dtype=t.dtype),
        )
        weights = 1.0 / (1.0 - t_cap).clamp(min=1e-4).pow(2)
        return (per_graph_loss * weights).sum() / weights.sum().clamp(min=eps)
    return per_graph_loss.mean()


def endpoint_density_flow_loss(
    pred_density: torch.Tensor,
    target_density: torch.Tensor,
    probe_graph: torch.Tensor,
    node_batch: torch.Tensor,
    t: torch.Tensor,
    density_scale: torch.Tensor,
    *,
    criterion: str = "mae",
    use_t_scale: bool = False,
    t_scale_max: float = 0.9,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Endpoint loss on probe charge density: head predicts ``c_end``, map through GTOs, compare
    to ``chg_labels`` (same scaling as ``ChgLightningModule.forward``).
    """
    n_graphs = int(probe_graph.max().item()) + 1 if probe_graph.numel() else 0
    sum_per_graph, count_per_graph = accumulate_endpoint_density_flow_loss(
        None,
        None,
        pred_density,
        target_density,
        probe_graph,
        density_scale,
        n_graphs=n_graphs,
        criterion=criterion,
        eps=eps,
    )
    return finalize_endpoint_density_flow_loss(
        sum_per_graph,
        count_per_graph,
        node_batch,
        t,
        use_t_scale=use_t_scale,
        t_scale_max=t_scale_max,
        eps=eps,
    )


def endpoint_flow_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    t: torch.Tensor,
    use_t_scale: bool = False,
    t_scale_max: float = 0.9,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    QHFlow-style endpoint regression on the linear bridge: given ``(latent, x_t, t)``, the head
    predicts the endpoint (or residual) and we minimize masked MSE to ``target``.

    Optional ``use_t_scale`` applies QHFlow's ``1 / (1 - min(t, t_scale_max))^2`` weighting using
    per-node ``t`` (typically constant within each graph).
    """
    diff = (pred - target) ** 2 * mask
    if use_t_scale:
        t_cap = torch.min(t, torch.tensor(float(t_scale_max), device=t.device, dtype=t.dtype))
        scale = 1.0 / (1.0 - t_cap).clamp(min=1e-4).pow(2)
        diff = diff * scale
        denom = (mask * scale).sum().clamp(min=eps)
    else:
        denom = mask.sum().clamp(min=eps)
    return diff.sum() / denom


def endpoint_prediction_to_velocity(
    pred: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Derive Euler velocity from an endpoint prediction: ``v = (pred - x_t) / (1 - t)``."""
    return (pred - x_t) / (1.0 - t).clamp(min=eps)
