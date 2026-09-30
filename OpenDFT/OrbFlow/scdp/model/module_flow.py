"""
Lightning module: conditional flow matching or Schrödinger-bridge (Gaussian) training on
GTO coefficients (SAD -> ground truth).

Requires ``gt_coeffs`` on the batch. ``bridge_mode`` selects ``cfm`` (linear bridge) or ``sb``
(Gaussian bridge score). On CFM, ``flow_loss_mode`` selects direct velocity matching or
QHFlow-style endpoint regression.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint
from torch_ema import ExponentialMovingAverage

from omegaconf import OmegaConf

from scdp.model.coeff_flow_matching import (
    CoeffEquivariantVelocityHead,
    CoeffVelocityMLP,
    accumulate_endpoint_density_flow_loss,
    build_flow_integrate_t_grid,
    coeff_norm_denominator,
    endpoint_flow_loss,
    finalize_endpoint_density_flow_loss,
    endpoint_pred_to_norm_coeffs,
    endpoint_prediction_to_velocity,
    escn_flat_latent_dim,
    escn_orbital_irreps,
    flow_matching_loss,
    gaussian_bridge_mean,
    gaussian_bridge_score_target,
    linear_flow_path,
    make_sb_per_dim_noise_scale,
    per_node_coeff_mask,
    resolve_flow_endpoint_loss,
    resolve_flow_integrate_schedule,
    resolve_flow_target_mode,
    sample_flow_prior_c0_raw,
    sample_flow_bridge_t,
    schrodinger_bridge_gaussian_sample,
    target_nelectrons_per_graph,
    monte_carlo_integrated_charge,
)
from scdp.model.module import ChgLightningModule
from scdp.model.utils import get_nmape, get_probe_chunks


class ChgFlowMatchingModule(ChgLightningModule):
    """
    Train with CFM (velocity or QHFlow-style endpoint loss) or SB score loss on coefficients.

    ``flow_loss_mode`` (CFM only): ``velocity`` matches ``v* = field_end - field_0``;
    ``endpoint`` matches the predicted field to the bridge target (c or d per
    ``density_synthesis_mode``), or to probe charge density when ``flow_endpoint_loss=density``.

    ``flow_head='e3nn'`` (default) uses ``CoeffEquivariantVelocityHead`` — SO(3)-equivariant maps
    in the same ``Irreps`` layout as ``eSCN`` (QHFlow-style linear/tensor-product head, no SiLU on
    mixed irreps). ``flow_head='mlp'`` keeps the original flat ``CoeffVelocityMLP``.
    ``flow_head='escn'`` (QHFlow unified): no separate head — ``eSCN`` readout predicts the bridge
    endpoint from ``(geometry, coeff_xt, bridge_t)``; requires ``enable_flow_coeff_entangle`` and
    ``flow_bridge_time_dim > 0``. With ``flow_loss_mode=velocity``, CFM loss uses
    ``v = (endpoint_pred - x_t) / (1 - t)`` (same as Euler integration).

    **SB noise:** ``sb_noise_mode=isotropic`` (default) matches the original per-row scalar
    variance. ``sb_noise_mode=per_irrep`` uses per-coefficient variance from
    ``make_sb_per_dim_noise_scale`` (``sb_per_l_sigma_mode`` or ``sb_per_l_sigma_scales``).

    **Geometry entanglement (optional):** when ``eSCN.enable_flow_coeff_entangle`` is true, each
    flow step passes ``coeff_xt=x_t`` into ``eSCN.forward`` so bridge coefficients are fused into
    every message-passing layer.
    """

    def __init__(
        self,
        flow_velocity_mlp_hidden: int = 512,
        flow_time_dim: int = 64,
        flow_head: str = "e3nn",
        flow_e3nn_hidden_multiplier: int = 2,
        flow_loss_mode: str = "velocity",
        flow_endpoint_loss: Optional[str] = None,
        flow_target_mode: Optional[str] = None,
        flow_use_t_scale: bool = False,
        flow_t_scale_max: float = 0.9,
        flow_prior_mode: str = "gaussian",
        flow_prior_sigma: float = 1.0,
        flow_prior_mu: float = 0.0,
        flow_prior_nelectron_normalize: bool = False,
        flow_nelectron_mc_max_probes: int = 0,
        flow_val_integrate_steps: int = 3,
        flow_val_integrate_sweep_steps: Optional[List[int]] = None,
        flow_integrate_train_steps: Optional[int] = None,
        flow_integrate_min_t: float = 0.01,
        flow_integrate_schedule: Optional[str] = None,
        flow_integrate_t_hi: float = 0.99,
        flow_integrate_power: float = 4.0,
        # Weight on integrated_density term when flow_endpoint_loss=density_and_integrated
        # (endpoint density term has implicit weight 1.0).
        flow_combined_integrated_weight: float = 1.0,
        flow_val_max_n_probe_per_pass: int = 400000,
        flow_val_log_readout: bool = False,
        bridge_mode: str = "cfm",
        sb_sigma: float = 0.1,
        sb_noise_mode: str = "isotropic",
        sb_per_l_sigma_mode: str = "inv_sqrt_2lp1",
        sb_per_l_sigma_scales: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if flow_loss_mode not in ("velocity", "endpoint"):
            raise ValueError(
                f"flow_loss_mode must be 'velocity' or 'endpoint', got {flow_loss_mode!r}"
            )
        self.flow_loss_mode = flow_loss_mode
        self.flow_endpoint_loss = resolve_flow_endpoint_loss(
            flow_loss_mode, flow_endpoint_loss
        )
        self.flow_target_mode = resolve_flow_target_mode(flow_target_mode)
        self.flow_use_t_scale = bool(flow_use_t_scale)
        self.flow_t_scale_max = float(flow_t_scale_max)
        if flow_prior_mode not in ("gaussian", "irrep_gaussian", "sad_coeffs"):
            raise ValueError(
                f"flow_prior_mode must be 'gaussian', 'irrep_gaussian', or 'sad_coeffs', "
                f"got {flow_prior_mode!r}"
            )
        self.flow_prior_mode = flow_prior_mode
        self.flow_prior_sigma = float(flow_prior_sigma)
        self.flow_prior_mu = float(flow_prior_mu)
        self.flow_prior_nelectron_normalize = bool(flow_prior_nelectron_normalize)
        self.flow_nelectron_mc_max_probes = int(flow_nelectron_mc_max_probes)
        self.flow_val_integrate_steps = int(flow_val_integrate_steps)
        sweep_raw = flow_val_integrate_sweep_steps or []
        self.flow_val_integrate_sweep_steps = sorted(
            {int(s) for s in sweep_raw if int(s) > 0}
        )
        self.flow_integrate_train_steps = int(
            flow_val_integrate_steps
            if flow_integrate_train_steps is None
            else flow_integrate_train_steps
        )
        if self.flow_integrate_train_steps <= 0:
            raise ValueError(
                f"flow_integrate_train_steps must be positive, got {self.flow_integrate_train_steps}"
            )
        min_t = float(flow_integrate_min_t)
        if not (0.0 < min_t < 0.5):
            raise ValueError(
                f"flow_integrate_min_t must be in (0, 0.5) so t samples lie in "
                f"[min_t, 1-min_t]; got {flow_integrate_min_t!r}"
            )
        self.flow_integrate_min_t = min_t
        self.flow_integrate_schedule = resolve_flow_integrate_schedule(flow_integrate_schedule)
        t_hi = float(flow_integrate_t_hi)
        if not (min_t < t_hi < 1.0):
            raise ValueError(
                f"flow_integrate_t_hi must satisfy min_t < t_hi < 1; got min_t={min_t}, t_hi={t_hi}"
            )
        self.flow_integrate_t_hi = t_hi
        power = float(flow_integrate_power)
        if power <= 0.0:
            raise ValueError(f"flow_integrate_power must be positive, got {flow_integrate_power!r}")
        self.flow_integrate_power = power
        self.flow_combined_integrated_weight = float(flow_combined_integrated_weight)
        if self.flow_combined_integrated_weight < 0.0:
            raise ValueError(
                "flow_combined_integrated_weight must be >= 0, "
                f"got {flow_combined_integrated_weight!r}"
            )
        self.flow_val_max_n_probe_per_pass = int(flow_val_max_n_probe_per_pass)
        self.flow_val_log_readout = bool(flow_val_log_readout)
        if flow_prior_mode == "sad_coeffs" and self.flow_target_mode == "res":
            raise ValueError(
                "flow_prior_mode='sad_coeffs' with flow_target_mode='res' is invalid: "
                "SAD is already the start field; use flow_target_mode='gt' or a noise prior with res."
            )
        if bridge_mode not in ("cfm", "sb"):
            raise ValueError(f"bridge_mode must be 'cfm' or 'sb', got {bridge_mode!r}")
        if flow_loss_mode == "endpoint" and bridge_mode != "cfm":
            raise ValueError("flow_loss_mode='endpoint' requires bridge_mode='cfm'")
        if self.flow_endpoint_loss in ("integrated_density", "density_and_integrated"):
            if flow_loss_mode != "endpoint":
                raise ValueError(
                    f"flow_endpoint_loss={self.flow_endpoint_loss!r} requires "
                    "flow_loss_mode='endpoint'"
                )
            if bridge_mode != "cfm":
                raise ValueError(
                    f"flow_endpoint_loss={self.flow_endpoint_loss!r} requires "
                    "bridge_mode='cfm'"
                )
        self.bridge_mode = bridge_mode
        self.sb_sigma = float(sb_sigma)

        if sb_noise_mode not in ("isotropic", "per_irrep"):
            raise ValueError(f"sb_noise_mode must be 'isotropic' or 'per_irrep', got {sb_noise_mode!r}")
        self.sb_noise_mode = sb_noise_mode
        self.sb_per_l_sigma_mode = str(sb_per_l_sigma_mode)

        scales_list = None
        if sb_per_l_sigma_scales is not None:
            scales_list = [float(x) for x in OmegaConf.to_container(sb_per_l_sigma_scales, resolve=True)]

        sb_scale_vec = make_sb_per_dim_noise_scale(
            self.max_n_orbitals_per_L,
            noise_mode=self.sb_noise_mode,
            per_l_sigma_mode=self.sb_per_l_sigma_mode,
            per_l_scales=scales_list,
        )
        if int(sb_scale_vec.numel()) != int(self.max_outdim):
            raise RuntimeError(
                f"SB scale vector length {sb_scale_vec.numel()} != max_outdim {int(self.max_outdim)}"
            )
        self.register_buffer("sb_per_dim_noise_scale", sb_scale_vec)

        if flow_head not in ("e3nn", "mlp", "escn"):
            raise ValueError(
                f"flow_head must be 'e3nn', 'mlp', or 'escn', got {flow_head!r}"
            )
        self.flow_head = flow_head

        if flow_head == "escn":
            if bridge_mode != "cfm":
                raise ValueError("flow_head='escn' requires bridge_mode='cfm'")
            if not getattr(self.model, "enable_flow_coeff_entangle", False):
                raise ValueError(
                    "flow_head='escn' requires model.enable_flow_coeff_entangle=true"
                )
            if int(getattr(self.model, "flow_bridge_time_dim", 0)) <= 0:
                raise ValueError(
                    "flow_head='escn' requires model.flow_bridge_time_dim > 0"
                )
            self.velocity_head = None
        elif flow_head == "e3nn":
            self.velocity_head = CoeffEquivariantVelocityHead(
                lmax_list=list(self.model.lmax_list),
                sphere_channels_all=int(self.model.sphere_channels_all),
                max_n_orbitals_per_L=self.model.max_n_orbitals_per_L,
                time_dim=flow_time_dim,
                hidden_multiplier=int(flow_e3nn_hidden_multiplier),
            )
            if int(self.max_outdim) != int(self.velocity_head.irreps_coeff.dim):
                raise ValueError(
                    f"max_outdim={int(self.max_outdim)} does not match e3nn orbital irrep dim "
                    f"{int(self.velocity_head.irreps_coeff.dim)}; basis and eSCN readout must agree."
                )
            latent_dim = escn_flat_latent_dim(
                self.model.lmax_list, self.model.sphere_channels
            )
            if latent_dim != int(self.velocity_head.irreps_latent.dim):
                raise ValueError(
                    f"escn_flat_latent_dim={latent_dim} vs irreps_latent.dim="
                    f"{int(self.velocity_head.irreps_latent.dim)}; check lmax_list / sphere_channels."
                )
        else:
            latent_dim = escn_flat_latent_dim(
                self.model.lmax_list, self.model.sphere_channels
            )
            self.velocity_head = CoeffVelocityMLP(
                latent_dim=latent_dim,
                coeff_dim=int(self.max_outdim),
                time_dim=flow_time_dim,
                hidden_dim=flow_velocity_mlp_hidden,
            )

        self._orbital_irreps = escn_orbital_irreps(self.model.max_n_orbitals_per_L)

        # Parent __init__ builds EMA before velocity_head exists; re-register all parameters.
        self.ema = ExponentialMovingAverage(
            self.parameters(), decay=self.hparams.train.ema.decay
        )

    def _resolve_integrate_schedule_kwargs(
        self,
        *,
        schedule: Optional[str] = None,
        t_hi: Optional[float] = None,
        power: Optional[float] = None,
    ) -> dict:
        """Checkpoint hparams with optional test-time overrides (``flow_inference_*`` attrs)."""
        sched = schedule
        if sched is None:
            sched = getattr(self, "flow_inference_integrate_schedule", None)
        if sched is None:
            sched = getattr(self, "flow_integrate_schedule", "uniform")

        t_hi_val = t_hi
        if t_hi_val is None:
            t_hi_val = getattr(self, "flow_inference_integrate_t_hi", None)
        if t_hi_val is None:
            t_hi_val = getattr(self, "flow_integrate_t_hi", 0.99)

        power_val = power
        if power_val is None:
            power_val = getattr(self, "flow_inference_integrate_power", None)
        if power_val is None:
            power_val = getattr(self, "flow_integrate_power", 4.0)

        return {
            "schedule": resolve_flow_integrate_schedule(sched),
            "t_hi": float(t_hi_val),
            "power": float(power_val),
        }

    def _flow_needs_xt_forward(self) -> bool:
        """Re-run eSCN each flow step with bridge state (entanglement or unified escn head)."""
        return self.flow_head == "escn" or getattr(
            self.model, "enable_flow_coeff_entangle", False
        )

    def _flow_forward_kwargs(self, x_t: torch.Tensor, t: torch.Tensor) -> dict:
        kw: dict = {}
        if getattr(self.model, "enable_flow_coeff_entangle", False):
            kw["coeff_xt"] = x_t
        if self.flow_head == "escn":
            kw["bridge_t"] = t
        return kw

    def _unified_endpoint_pred(
        self,
        batch,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """QHFlow unified: eSCN readout as endpoint prediction in normalized coeff space."""
        coeffs_raw, expo_scaling, _ = self.model(
            batch,
            return_latent=True,
            **self._flow_forward_kwargs(x_t, t),
        )
        coeffs_norm, _ = self._normalize_readout_coeffs(batch, coeffs_raw, expo_scaling)
        return coeffs_norm

    def _predict_flow_field(
        self,
        batch,
        latent: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if self.flow_head == "escn":
            return self._unified_endpoint_pred(batch, x_t, t)
        return self.velocity_head(latent, x_t, t)

    def _charge_integration_probes(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full LMDB probe grid for ``∫ ρ dV`` (same points as ``chg_labels`` / val nMAPE)."""
        device = batch.probe_coords.device
        bsz = int(batch.batch.max().item()) + 1
        n_probe = batch.n_probe
        if not torch.is_tensor(n_probe):
            n_probe = torch.full((bsz,), int(n_probe), device=device, dtype=torch.long)
        else:
            n_probe = n_probe.reshape(-1).to(device=device, dtype=torch.long)
            if n_probe.numel() == 1:
                n_probe = n_probe.expand(bsz)
        return batch.probe_coords, n_probe

    def _cell_volume_per_graph(self, batch, bsz: int) -> torch.Tensor:
        cell = batch.cell
        if cell.dim() == 2:
            cell = cell.unsqueeze(0)
        cell = cell.reshape(-1, 3, 3)
        if cell.shape[0] == 1 and bsz > 1:
            cell = cell.expand(bsz, -1, -1)
        vol = torch.linalg.det(cell.double().cpu()).abs()
        return vol.to(device=batch.coords.device, dtype=batch.coords.dtype).view(-1)

    def _integrated_charge_from_c0_raw(self, batch, c0_raw: torch.Tensor) -> torch.Tensor:
        """
        ``∫ ρ(r; c) dV`` for coefficient field ``c`` via synthesis on the full probe grid.

        Used to measure how much charge a **prior** (or any coeffs) carries before scaling.
        """
        norm = coeff_norm_denominator(batch, self.n_orbitals, self.unique_atom_types)
        c_norm = c0_raw / norm
        probe_coords, n_probe = self._charge_integration_probes(batch)
        pred = self.orbital_inference(batch, c_norm, None, n_probe, probe_coords)
        bsz = int(batch.batch.max().item()) + 1
        probe_graph = torch.arange(bsz, device=pred.device).repeat_interleave(n_probe)
        cell_volume = self._cell_volume_per_graph(batch, bsz)
        return monte_carlo_integrated_charge(pred, probe_graph, cell_volume, bsz)

    def _enforce_nelectron_constraint(self, batch, c0_raw: torch.Tensor) -> torch.Tensor:
        """
        Scale random priors: ``c0 ← c0 * (N_valence / Q(c0))``.

        - ``N_valence``: PAW valence electron count from **atom types only**
          (``valence_electron_count_per_graph`` / SAD convention) — no GT, no labels.
        - ``Q(c0)``: integrate synthesized density from the prior coeffs on the full probe grid.
        """
        if not self.flow_prior_nelectron_normalize:
            return c0_raw
        if self.flow_prior_mode not in ("gaussian", "irrep_gaussian"):
            return c0_raw
        q_target = target_nelectrons_per_graph(batch)
        q_est = self._integrated_charge_from_c0_raw(batch, c0_raw)
        eps = 1e-8
        q_safe = torch.where(
            q_est.abs() < eps,
            torch.sign(q_est) * eps + (q_est == 0).to(dtype=q_est.dtype) * eps,
            q_est,
        )
        scale = (q_target / q_safe).view(-1, 1)
        return c0_raw * scale[batch.batch]

    def _sample_c0_raw(
        self,
        batch,
        sad_coeffs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Raw (unnormalized) bridge start: c0 (linear) or d0 (squared, non-negative prior).
        """
        if sad_coeffs is not None:
            return sad_coeffs.to(device=batch.coords.device, dtype=batch.coords.dtype)
        bridge_raw = sample_flow_prior_c0_raw(
            batch,
            int(self.max_outdim),
            prior_mode=self.flow_prior_mode,
            orb_index=self.orb_index,
            orbital_irreps=self._orbital_irreps,
            sigma=self.flow_prior_sigma,
            prior_mu=self.flow_prior_mu,
            n_orbitals=self.n_orbitals,
            unique_atom_types=self.unique_atom_types,
            density_synthesis_mode=self.density_synthesis_mode,
        )
        return self._enforce_nelectron_constraint(batch, bridge_raw)

    def _normalized_sad(self, batch, norm: torch.Tensor) -> torch.Tensor:
        """``batch['sad_coeffs']`` in the same normalized space as ``c_gt``."""
        if "sad_coeffs" not in batch:
            raise KeyError(
                "flow_target_mode='res' requires batch['sad_coeffs'] on each graph "
                "(precompute SAD projections into the LMDB)."
            )
        device = batch.coords.device
        dtype = batch.coords.dtype
        sad_raw = batch["sad_coeffs"].to(device=device, dtype=dtype)
        if sad_raw.shape[1] != int(self.max_outdim):
            raise ValueError(
                f"sad_coeffs.shape[1]={sad_raw.shape[1]} but max_outdim={int(self.max_outdim)}"
            )
        return sad_raw / norm

    def _cfm_bridge_end(self, c_gt: torch.Tensor, c_sad: Optional[torch.Tensor]):
        """Bridge endpoint and endpoint-regression target for the active ``flow_target_mode``."""
        if self.flow_target_mode == "res":
            assert c_sad is not None
            c_end = c_gt - c_sad
        else:
            c_end = c_gt
        return c_end, c_end

    def _uses_integrated_density_loss(self) -> bool:
        return self.flow_loss_mode == "endpoint" and self.flow_endpoint_loss in (
            "integrated_density",
            "density_and_integrated",
        )

    def _bridge_loss_log_key(self, split: str) -> str:
        if self.bridge_mode == "sb":
            return f"loss/{split}_sb"
        if self.flow_loss_mode == "endpoint":
            if self.flow_endpoint_loss == "coeff":
                return f"loss/{split}_endpoint_coeff"
            if self.flow_endpoint_loss == "integrated_density":
                return f"loss/{split}_endpoint_integrated"
            if self.flow_endpoint_loss == "density_and_integrated":
                return f"loss/{split}_endpoint_joint"
            return f"loss/{split}_endpoint"
        return f"loss/{split}_fm"

    def _density_flow_loss_from_norm_coeffs(
        self,
        batch,
        c_norm: torch.Tensor,
        t: torch.Tensor,
        *,
        use_t_scale: Optional[bool] = None,
    ) -> torch.Tensor:
        """MAE/MSE on probe density from normalized coefficients (training scale)."""
        bsz = int(batch.batch.max().item()) + 1
        probe_graph_full = torch.arange(bsz, device=c_norm.device).repeat_interleave(
            batch.n_probe
        )
        n_pass, n_per_pass, probes_to_process = get_probe_chunks(
            batch.n_probe, self.flow_val_max_n_probe_per_pass
        )
        sum_per_graph = None
        count_per_graph = None
        for i_pass in range(n_pass):
            n_probe = n_per_pass[i_pass]
            probe_idx = probes_to_process[i_pass]
            probe_coords = batch.probe_coords[probe_idx]
            pred_rho = self.orbital_inference(
                batch, c_norm, None, n_probe, probe_coords
            )
            sum_per_graph, count_per_graph = accumulate_endpoint_density_flow_loss(
                sum_per_graph,
                count_per_graph,
                pred_rho,
                batch.chg_labels[probe_idx],
                probe_graph_full[probe_idx],
                self.scale,
                n_graphs=bsz,
                criterion=str(self.hparams.criterion),
            )
        if use_t_scale is None:
            use_t_scale = self.flow_use_t_scale
        return finalize_endpoint_density_flow_loss(
            sum_per_graph,
            count_per_graph,
            batch.batch,
            t,
            use_t_scale=use_t_scale,
            t_scale_max=self.flow_t_scale_max,
        )

    def _endpoint_density_flow_loss(
        self,
        batch,
        pred: torch.Tensor,
        t: torch.Tensor,
        c_sad_norm: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """MAE/MSE on probe density from predicted endpoint coefficients."""
        c_norm = endpoint_pred_to_norm_coeffs(
            pred, c_sad_norm, self.flow_target_mode
        )
        return self._density_flow_loss_from_norm_coeffs(batch, c_norm, t)

    def _integrated_density_flow_loss(self, batch) -> torch.Tensor:
        """
        Density MAE/MSE on coeffs after differentiable Euler CFM integration.

        Matches the val/test ``nmape/val_flow`` sampling path (same step count / time grid).
        ``flow_use_t_scale`` is ignored (loss is on the final integrated state at ``t=1``).
        """
        coeffs_norm = self._integrate_coeffs_euler_unrolled(
            batch,
            n_steps=self.flow_integrate_train_steps,
            min_t=self.flow_integrate_min_t,
        )
        t = torch.ones(
            (batch.batch.shape[0], 1),
            device=coeffs_norm.device,
            dtype=coeffs_norm.dtype,
        )
        return self._density_flow_loss_from_norm_coeffs(
            batch, coeffs_norm, t, use_t_scale=False
        )

    def _physical_coeffs_from_state(
        self,
        x: torch.Tensor,
        c_sad_norm: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Map integrator state to coefficients used by ``orbital_inference``."""
        if self.flow_target_mode == "res":
            assert c_sad_norm is not None
            return x + c_sad_norm
        return x

    def _integrate_coeffs_euler_unrolled(
        self,
        batch,
        *,
        n_steps: int,
        min_t: float,
        sad_coeffs: Optional[torch.Tensor] = None,
        schedule: Optional[str] = None,
        t_hi: Optional[float] = None,
        power: Optional[float] = None,
        c0_norm: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ):
        """
        Differentiable Euler CFM integration from sampled ``c_0`` to ``t=1``.

        Time knots follow ``flow_integrate_schedule`` (see ``build_flow_integrate_t_grid``).
        Shared by training (``integrated_density`` loss) and inference (``@torch.no_grad`` wrapper).

        Optional ``c0_norm`` skips ``_sample_c0_raw`` and starts from the given normalized
        coefficients (for fair multi-scheduler / multi-step comparisons on one prior).

        If ``return_trajectory`` is True, returns
        ``(final_coeffs, t_knots [T], coeffs_traj [T, N, C])`` where ``T = n_steps + 1``
        (state at each time knot, including the start). Trajectory coefficients are the
        physical coeffs passed to density evaluation (SAD residual added when applicable).
        """
        if self.bridge_mode != "cfm":
            raise RuntimeError(
                "Euler integration only applies when bridge_mode='cfm' (learned velocity)."
            )
        if int(n_steps) <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        min_t_val = float(min_t)
        if not (0.0 < min_t_val < 0.5):
            raise ValueError(f"min_t must be in (0, 0.5), got {min_t_val}")

        device = batch.coords.device
        dtype = batch.coords.dtype
        norm = coeff_norm_denominator(batch, self.n_orbitals, self.unique_atom_types)

        if c0_norm is not None:
            x = c0_norm
        else:
            c0_raw = self._sample_c0_raw(batch, sad_coeffs=sad_coeffs)
            x = c0_raw / norm
        c_sad_norm = self._normalized_sad(batch, norm) if self.flow_target_mode == "res" else None

        sched_kw = self._resolve_integrate_schedule_kwargs(
            schedule=schedule, t_hi=t_hi, power=power
        )
        lin_t = build_flow_integrate_t_grid(
            int(n_steps),
            min_t_val,
            device=device,
            dtype=dtype,
            **sched_kw,
        )
        cur_t = lin_t[0]

        needs_xt = self._flow_needs_xt_forward()
        latent = None
        if not needs_xt:
            _, _, latent = self.model(batch, return_latent=True)

        # QHFlow re-runs eSCN each step; checkpoint steps during training to avoid OOM
        # from storing 3 full forward graphs before probe density evaluation.
        use_step_checkpoint = (
            self.training and torch.is_grad_enabled() and needs_xt and int(n_steps) > 1
        )

        traj_coeffs: Optional[List[torch.Tensor]] = [] if return_trajectory else None
        if traj_coeffs is not None:
            traj_coeffs.append(self._physical_coeffs_from_state(x, c_sad_norm).detach())

        for next_t in lin_t[1:]:
            dt = next_t - cur_t
            cur_t_float = float(cur_t)
            if use_step_checkpoint:

                def euler_step(x_in: torch.Tensor) -> torch.Tensor:
                    t_node = torch.full(
                        (x_in.shape[0], 1), cur_t_float, device=device, dtype=dtype
                    )
                    _, _, latent_step = self.model(
                        batch,
                        return_latent=True,
                        **self._flow_forward_kwargs(x_in, t_node),
                    )
                    v = self._field_from_head(batch, latent_step, x_in, t_node)
                    return x_in + dt * v

                x = checkpoint(euler_step, x, use_reentrant=False)
            else:
                t = torch.full((x.shape[0], 1), cur_t_float, device=device, dtype=dtype)
                if needs_xt:
                    _, _, latent = self.model(
                        batch, return_latent=True, **self._flow_forward_kwargs(x, t)
                    )
                v = self._field_from_head(batch, latent, x, t)
                x = x + dt * v
            cur_t = next_t
            if traj_coeffs is not None:
                traj_coeffs.append(self._physical_coeffs_from_state(x, c_sad_norm).detach())

        if self.flow_target_mode == "res":
            assert c_sad_norm is not None
            x = x + c_sad_norm

        if return_trajectory:
            assert traj_coeffs is not None
            # Final state after residual fold matches last traj entry for res mode.
            return x, lin_t.detach(), torch.stack(traj_coeffs, dim=0)

        return x

    def _prepare_flow_batch(
        self, batch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return bridge start, bridge target endpoint, x_t, t, mask, target_field.

        ``linear``: fields are c (coefficients on phi).
        ``squared``: fields are d (non-negative coefficients on phi^2).
        """
        device = batch.coords.device
        dtype = batch.coords.dtype
        if "gt_coeffs" not in batch:
            raise KeyError(
                "Flow training requires batch['gt_coeffs'] with shape "
                f"(num_nodes, {int(self.max_outdim)}). "
                "Add this field in preprocessing when you have projected / fitted DFT coefficients."
            )

        gt_raw = batch["gt_coeffs"].to(device=device, dtype=dtype)
        if gt_raw.shape[1] != int(self.max_outdim):
            raise ValueError(
                f"gt_coeffs.shape[1]={gt_raw.shape[1]} but max_outdim={int(self.max_outdim)}"
            )

        norm = coeff_norm_denominator(batch, self.n_orbitals, self.unique_atom_types)
        c_gt = gt_raw / norm

        c0_raw = self._sample_c0_raw(batch)
        c0 = c0_raw / norm

        c_sad = self._normalized_sad(batch, norm) if self.flow_target_mode == "res" else None

        bsz = int(batch.batch.max().item()) + 1
        t_graph = sample_flow_bridge_t(
            bsz, device, min_t=self.flow_integrate_min_t, dtype=dtype
        )
        t = t_graph[batch.batch].unsqueeze(1)
        mask = per_node_coeff_mask(batch.atom_types, self.orb_index)

        if self.bridge_mode == "cfm":
            c_end, endpoint_target = self._cfm_bridge_end(c_gt, c_sad)
            x_t, v_star = linear_flow_path(c0, c_end, t)
            if self.flow_loss_mode == "velocity":
                target = v_star
            else:
                target = endpoint_target
        elif self.bridge_mode == "sb":
            mu_t = gaussian_bridge_mean(c0, c_gt, t)
            x_t, var = schrodinger_bridge_gaussian_sample(
                mu_t,
                t,
                self.sb_sigma,
                mask,
                self.sb_per_dim_noise_scale,
                generator=None,
            )
            target = gaussian_bridge_score_target(x_t, mu_t, var, mask)
        else:
            raise RuntimeError(f"unknown bridge_mode {self.bridge_mode}")

        return c0, c_gt, x_t, t, mask, target

    def _field_from_head(
        self,
        batch,
        latent: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Head output as CFM velocity for integration / velocity FM loss."""
        head_out = self._predict_flow_field(batch, latent, x_t, t)
        if self.bridge_mode == "sb":
            return head_out
        if self.flow_head == "escn":
            # QHFlow unified readout predicts the bridge endpoint; derive v = (pred - x_t)/(1-t).
            return endpoint_prediction_to_velocity(head_out, x_t, t)
        if self.flow_loss_mode == "velocity":
            return head_out
        return endpoint_prediction_to_velocity(head_out, x_t, t)

    def _normalize_readout_coeffs(
        self, batch, coeffs_raw: torch.Tensor, expo_scaling: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Match ``predict_coeffs`` (per-graph sqrt normalization + optional expo sigmoid)."""
        norm = coeff_norm_denominator(batch, self.n_orbitals, self.unique_atom_types)
        coeffs = coeffs_raw / norm
        if expo_scaling is not None:
            expo_scaling = 1.5 / (1 + torch.exp(-expo_scaling + math.log(2))) + 0.5
        return coeffs, expo_scaling

    def flow_matching_loss_batch(
        self,
        batch,
        forward_out: Optional[Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]] = None,
        *,
        return_parts: bool = False,
    ) -> torch.Tensor:
        if self.flow_loss_mode == "endpoint" and self.flow_endpoint_loss == "integrated_density":
            loss = self._integrated_density_flow_loss(batch)
            return (loss, {"integrated": loss}) if return_parts else loss

        c0, c_gt, x_t, t, mask, target = self._prepare_flow_batch(batch)

        needs_xt = self._flow_needs_xt_forward()
        fm_kw = self._flow_forward_kwargs(x_t, t) if needs_xt else {}
        if forward_out is not None and not needs_xt:
            _, expo_scaling, latent = forward_out
        else:
            forward_out = self.model(batch, return_latent=True, **fm_kw)
            _, expo_scaling, latent = forward_out
        _ = expo_scaling

        if self.bridge_mode == "sb":
            field_pred = self._predict_flow_field(batch, latent, x_t, t)
            loss = flow_matching_loss(field_pred, target, mask)
            return (loss, {}) if return_parts else loss
        if self.flow_loss_mode == "velocity":
            v_pred = self._field_from_head(batch, latent, x_t, t)
            loss = flow_matching_loss(v_pred, target, mask)
            return (loss, {}) if return_parts else loss
        field_pred = self._predict_flow_field(batch, latent, x_t, t)
        if self.flow_endpoint_loss in ("density", "density_and_integrated"):
            norm = coeff_norm_denominator(
                batch, self.n_orbitals, self.unique_atom_types
            )
            c_sad_norm = (
                self._normalized_sad(batch, norm)
                if self.flow_target_mode == "res"
                else None
            )
            loss_ep = self._endpoint_density_flow_loss(
                batch, field_pred, t, c_sad_norm
            )
            if self.flow_endpoint_loss == "density":
                return (loss_ep, {"endpoint": loss_ep}) if return_parts else loss_ep
            loss_int = self._integrated_density_flow_loss(batch)
            w = self.flow_combined_integrated_weight
            loss = loss_ep + w * loss_int
            parts = {"endpoint": loss_ep, "integrated": loss_int}
            return (loss, parts) if return_parts else loss
        loss = endpoint_flow_loss(
            field_pred,
            target,
            mask,
            t,
            use_t_scale=self.flow_use_t_scale,
            t_scale_max=self.flow_t_scale_max,
        )
        return (loss, {}) if return_parts else loss

    def training_step(self, batch, batch_idx):
        uses_integrated = self._uses_integrated_density_loss()
        if uses_integrated or self._flow_needs_xt_forward():
            loss_bridge, parts = self.flow_matching_loss_batch(
                batch, forward_out=None, return_parts=True
            )
        else:
            forward_out = self.model(batch, return_latent=True)
            loss_bridge, parts = self.flow_matching_loss_batch(
                batch, forward_out=forward_out, return_parts=True
            )
        logs = {"loss/train": loss_bridge, self._bridge_loss_log_key("train"): loss_bridge}
        if "endpoint" in parts:
            logs["loss/train_endpoint"] = parts["endpoint"]
        if "integrated" in parts:
            logs["loss/train_integrated"] = parts["integrated"]
        self.log_dict(logs, batch_size=len(batch), sync_dist=self.distributed)
        return loss_bridge

    @torch.no_grad()
    def _readout_density_nmape(self, batch) -> torch.Tensor:
        """One-shot eSCN readout density nMAPE (not flow sampling)."""
        forward_out = self.model(batch, return_latent=True)
        coeffs_norm, expo = self._normalize_readout_coeffs(
            batch, forward_out[0], forward_out[1]
        )
        pred = self.orbital_inference(
            batch, coeffs_norm, expo, batch.n_probe, batch.probe_coords
        )
        graph_idx = torch.arange(len(batch), device=batch.chg_labels.device).repeat_interleave(
            batch.n_probe
        )
        return get_nmape(pred, batch.chg_labels, graph_idx).mean()

    @torch.no_grad()
    def _flow_integrated_density_nmape(self, batch, n_steps: int) -> torch.Tensor:
        """
        Density nMAPE from Euler CFM sampling — same path as ``scdp/scripts/test_flow.py``.
        """
        coeffs = self.integrate_coeffs_euler(batch, n_steps=n_steps)
        n_pass, n_per_pass, probes_to_process = get_probe_chunks(
            batch.n_probe, self.flow_val_max_n_probe_per_pass
        )
        all_preds = []
        for i_pass in range(n_pass):
            n_probe = n_per_pass[i_pass]
            probe_idx = probes_to_process[i_pass]
            probe_coords = batch.probe_coords[probe_idx]
            pred = self.orbital_inference(
                batch, coeffs, None, n_probe, probe_coords
            )
            all_preds.append(pred)
        pred = torch.cat(all_preds, dim=0)
        graph_idx = torch.arange(len(batch), device=pred.device).repeat_interleave(
            batch.n_probe
        )
        return get_nmape(pred, batch.chg_labels, graph_idx).mean()

    def _log_density_nmape(self, batch, logdict: dict, split: str) -> None:
        """Attach flow-sampled and/or readout nMAPE keys for val or test."""
        use_flow = self.flow_val_integrate_steps > 0 and self.bridge_mode == "cfm"
        if use_flow:
            primary = int(self.flow_val_integrate_steps)
            extra = list(self.flow_val_integrate_sweep_steps)
            step_counts = sorted({primary, *extra}) if extra else [primary]
            for n_steps in step_counts:
                nmape_flow = self._flow_integrated_density_nmape(batch, n_steps)
                if extra:
                    logdict[f"nmape/{split}_flow_s{n_steps}"] = nmape_flow
                if n_steps == primary:
                    logdict[f"nmape/{split}"] = nmape_flow
                    logdict[f"nmape/{split}_flow"] = nmape_flow
        if self.flow_val_log_readout or not use_flow:
            nmape_readout = self._readout_density_nmape(batch)
            key = f"nmape/{split}_readout" if use_flow else f"nmape/{split}"
            logdict[key] = nmape_readout

    def validation_step(self, batch, batch_idx):
        uses_integrated = self._uses_integrated_density_loss()
        needs_xt = self._flow_needs_xt_forward()
        forward_out = None
        if not uses_integrated and not needs_xt:
            forward_out = self.model(batch, return_latent=True)

        logdict: dict = {}
        self._log_density_nmape(batch, logdict, "val")
        if "gt_coeffs" not in batch:
            self.log_dict(logdict, batch_size=len(batch), sync_dist=self.distributed)
            return None

        loss_bridge = self.flow_matching_loss_batch(
            batch, forward_out=forward_out
        )
        logdict["loss/val"] = loss_bridge
        logdict[self._bridge_loss_log_key("val")] = loss_bridge
        self.log_dict(logdict, batch_size=len(batch), sync_dist=self.distributed)
        return loss_bridge

    def test_step(self, batch, batch_idx):
        uses_integrated = self._uses_integrated_density_loss()
        needs_xt = self._flow_needs_xt_forward()
        forward_out = None
        if not uses_integrated and not needs_xt:
            forward_out = self.model(batch, return_latent=True)

        logdict: dict = {}
        self._log_density_nmape(batch, logdict, "test")
        if "gt_coeffs" not in batch:
            self.log_dict(logdict, batch_size=len(batch), sync_dist=self.distributed)
            return None

        loss_bridge = self.flow_matching_loss_batch(
            batch, forward_out=forward_out
        )
        logdict["loss/test"] = loss_bridge
        logdict[self._bridge_loss_log_key("test")] = loss_bridge
        self.log_dict(logdict, batch_size=len(batch), sync_dist=self.distributed)
        return loss_bridge

    @torch.no_grad()
    def integrate_coeffs_euler(
        self,
        batch,
        n_steps: Optional[int] = None,
        sad_coeffs: Optional[torch.Tensor] = None,
        min_t: Optional[float] = None,
        schedule: Optional[str] = None,
        t_hi: Optional[float] = None,
        power: Optional[float] = None,
        c0_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Euler integrate from sampled ``c_0`` to ``t=1``.

        Time knots are built by ``build_flow_integrate_t_grid`` from ``flow_integrate_schedule``
        (``uniform`` | ``endpoint_refine`` | ``power``). State starts at ``c_0``; the first head
        evaluation is at ``t=min_t`` (default 0.01), avoiding the ``1/(1-t)`` singularity at
        ``t=0`` for endpoint mode.

        ``flow_loss_mode=velocity``: head output is integrated directly.
        ``flow_loss_mode=endpoint``: head predicts the endpoint (or residual); we use
        QHFlow's ``v = (endpoint - x_t) / (1 - t)`` at each step.
        """
        if self.bridge_mode == "sb":
            raise RuntimeError(
                "integrate_coeffs_euler only applies when bridge_mode='cfm' (learned velocity). "
                "For bridge_mode='sb' the head matches a conditional score; sampling requires an "
                "SDE / score-based integrator (not implemented here)."
            )
        n_steps_val = int(n_steps if n_steps is not None else self.flow_val_integrate_steps)
        min_t_val = float(min_t if min_t is not None else self.flow_integrate_min_t)

        was_training = self.training
        self.eval()
        try:
            return self._integrate_coeffs_euler_unrolled(
                batch,
                n_steps=n_steps_val,
                min_t=min_t_val,
                sad_coeffs=sad_coeffs,
                schedule=schedule,
                t_hi=t_hi,
                power=power,
                c0_norm=c0_norm,
            )
        finally:
            self.train(was_training)

    @torch.no_grad()
    def integrate_coeffs_euler_trajectory(
        self,
        batch,
        n_steps: Optional[int] = None,
        sad_coeffs: Optional[torch.Tensor] = None,
        min_t: Optional[float] = None,
        schedule: Optional[str] = None,
        t_hi: Optional[float] = None,
        power: Optional[float] = None,
        c0_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same as ``integrate_coeffs_euler``, but also returns the coefficient trajectory.

        Returns
        -------
        final_coeffs :
            Coefficients at ``t=1`` (same as ``integrate_coeffs_euler``).
        t_knots :
            1-D tensor of length ``n_steps + 1`` with Euler time knots.
        coeffs_traj :
            Tensor ``[n_steps + 1, num_nodes, max_outdim]`` of physical coefficients
            at each knot (suitable for ``orbital_inference`` / density viz).
        """
        n_steps_val = int(n_steps if n_steps is not None else self.flow_val_integrate_steps)
        min_t_val = float(min_t if min_t is not None else self.flow_integrate_min_t)
        was_training = self.training
        self.eval()
        try:
            return self._integrate_coeffs_euler_unrolled(
                batch,
                n_steps=n_steps_val,
                min_t=min_t_val,
                sad_coeffs=sad_coeffs,
                schedule=schedule,
                t_hi=t_hi,
                power=power,
                c0_norm=c0_norm,
                return_trajectory=True,
            )
        finally:
            self.train(was_training)

    @torch.no_grad()
    def predict_coeffs(self, batch, n_steps: Optional[int] = None):
        """
        Test-time coefficients for density evaluation.

        Default: Euler CFM integration from sampled ``c_0`` (``integrate_coeffs_euler``).
        Set ``self.flow_coeff_mode = 'readout'`` for one-shot eSCN readout (validation-style).
        """
        mode = getattr(self, "flow_coeff_mode", "integrate")
        if mode == "readout":
            forward_out = self.model(batch, return_latent=True)
            return self._normalize_readout_coeffs(batch, forward_out[0], forward_out[1])

        if self.bridge_mode == "sb":
            raise RuntimeError(
                "predict_coeffs with flow_coeff_mode='integrate' requires bridge_mode='cfm'. "
                "SB checkpoints need a score-based sampler (not implemented)."
            )
        n_steps = n_steps or getattr(
            self, "flow_inference_n_steps", self.flow_val_integrate_steps
        )
        coeffs = self.integrate_coeffs_euler(batch, n_steps=n_steps)
        return coeffs, None
