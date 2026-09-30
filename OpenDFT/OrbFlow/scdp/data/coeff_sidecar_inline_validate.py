"""
Periodic reconstruction QA while writing gt/sad coefficient sidecars.

Every ``interval`` newly computed graphs, reconstruct probe density from stored
coeffs (via ``orbital_inference``) and report nMAPE vs labels / SAD density.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from scdp.model.coeff_flow_matching import coeff_norm_denominator
from scdp.model.density_synthesis import resolve_density_synthesis_mode
from scdp.model.utils import get_nmape


@dataclass
class InlineValidateConfig:
    metadata: dict
    scale: float
    beta: float
    vnode_elem: int = 8
    dft_basis_set: str = "def2-qzvppd"
    dft_wt_aug: bool = True
    lmax_restriction: bool = True
    uncontracted: bool = True
    orb_cutoff: float = 5.0
    density_synthesis_mode: str = "linear"
    coeff_field: str = "gt_coeffs"
    device: str = "cuda:0"
    lmdb_pbc: bool = False  # MP units.json fallback when graph has no data.pbc
    interval: int = 100
    probe_samples: int = 8192
    full_grid_validate: bool = True
    full_grid_chunk: int = 8192
    nmape_warn_threshold: float = 0.02
    sidecar_label: str = "gt_coeffs"


@dataclass
class SidecarInlineValidator:
    cfg: InlineValidateConfig
    atomic_library: Any = None
    _buffer: List[Tuple[str, int, bytes, bytes, float]] = field(default_factory=list)
    _total_computed: int = 0
    _model: Any = None

    @classmethod
    def from_env_and_args(
        cls,
        *,
        metadata: dict,
        scale: float,
        beta: float,
        density_synthesis_mode: str,
        coeff_field: str,
        device: str,
        vnode_elem: int,
        dft_basis_set: str,
        dft_wt_aug: bool,
        lmax_restriction: bool,
        uncontracted: bool,
        orb_cutoff: float,
        interval: int,
        probe_samples: int,
        full_grid_validate: bool = True,
        full_grid_chunk: int = 8192,
        nmape_warn_threshold: float,
        sidecar_label: str,
        lmdb_pbc: bool = False,
        atomic_library: Any = None,
    ) -> Optional["SidecarInlineValidator"]:
        if int(interval) <= 0:
            return None
        cfg = InlineValidateConfig(
            metadata=metadata,
            scale=scale,
            beta=beta,
            vnode_elem=vnode_elem,
            dft_basis_set=dft_basis_set,
            dft_wt_aug=dft_wt_aug,
            lmax_restriction=lmax_restriction,
            uncontracted=uncontracted,
            orb_cutoff=orb_cutoff,
            density_synthesis_mode=resolve_density_synthesis_mode(density_synthesis_mode),
            coeff_field=coeff_field,
            device=device,
            interval=int(interval),
            probe_samples=int(probe_samples),
            full_grid_validate=bool(full_grid_validate),
            full_grid_chunk=int(full_grid_chunk),
            nmape_warn_threshold=float(nmape_warn_threshold),
            sidecar_label=sidecar_label,
            lmdb_pbc=bool(lmdb_pbc),
        )
        return cls(cfg=cfg, atomic_library=atomic_library)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from argparse import Namespace

        from scdp.scripts.validate_gt_coeffs import build_model

        dev = self.cfg.device
        if dev == "cuda":
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        args = Namespace(
            beta=self.cfg.beta,
            vnode_elem=self.cfg.vnode_elem,
            dft_basis_set=self.cfg.dft_basis_set,
            no_dft_wt_aug=not self.cfg.dft_wt_aug,
            no_lmax_restriction=not self.cfg.lmax_restriction,
            contracted=not self.cfg.uncontracted,
            orb_cutoff=self.cfg.orb_cutoff,
            lmax=6,
            density_synthesis_mode=self.cfg.density_synthesis_mode,
            pbc=self.cfg.lmdb_pbc,
        )
        self._model = build_model(self.cfg.metadata, args, dev).eval()

    def _target_labels(
        self,
        data,
        probe_coords: torch.Tensor,
        *,
        perm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from scdp.data.sad_density import superposed_sad_density_at_probes

        if self.cfg.coeff_field == "gt_coeffs":
            labels = data.chg_labels
        else:
            labels = superposed_sad_density_at_probes(
                data.atom_types,
                data.coords,
                probe_coords,
                self.atomic_library,
                is_vnode=getattr(data, "is_vnode", None),
            )
        if perm is not None:
            return labels[perm]
        return labels

    def _nmape_at_probes(
        self,
        model,
        data,
        coeffs: torch.Tensor,
        probe_coords: torch.Tensor,
        target: torch.Tensor,
        *,
        chunk_size: int,
    ) -> float:
        from scdp.scripts.validate_gt_coeffs import predict_density_from_stored_coeffs

        n_probe = int(probe_coords.shape[0])
        data.probe_coords = probe_coords
        data.n_probe = n_probe
        pred = predict_density_from_stored_coeffs(
            model,
            data,
            max_n_probe_per_pass=max(int(chunk_size), 1),
            coeff_field=self.cfg.coeff_field,
        )
        gidx = torch.zeros(pred.shape[0], dtype=torch.long, device=pred.device)
        return float(get_nmape(pred, target.to(pred.device), gidx).item())

    @torch.no_grad()
    def _graph_metrics(self, raw: bytes, coeff_payload: bytes) -> Dict[str, float]:
        self._ensure_model()
        model = self._model
        dev = next(model.parameters()).device
        data = pickle.loads(raw)
        coeffs = pickle.loads(coeff_payload)
        data = data.clone().to(dev)
        data.batch = torch.zeros(data.atom_types.shape[0], dtype=torch.long, device=dev)
        data[self.cfg.coeff_field] = coeffs.to(device=dev, dtype=data.coords.dtype)

        n_probe = int(data.probe_coords.shape[0])
        probe_coords_full = data.probe_coords
        target_full = self._target_labels(data, probe_coords_full)
        use_subsample = (
            self.cfg.probe_samples >= 0 and self.cfg.probe_samples < n_probe
        )

        if use_subsample:
            g = torch.Generator(device=dev)
            g.manual_seed((hash(raw) & 0x7FFFFFFF) ^ self._total_computed)
            perm = torch.randperm(n_probe, device=dev, generator=g)[
                : int(self.cfg.probe_samples)
            ]
            probe_sub = probe_coords_full[perm]
            target_sub = target_full[perm]
            nmape_sub = self._nmape_at_probes(
                model,
                data,
                coeffs,
                probe_sub,
                target_sub,
                chunk_size=self.cfg.probe_samples,
            )
            nmape_full = nmape_sub
            if self.cfg.full_grid_validate:
                nmape_full = self._nmape_at_probes(
                    model,
                    data,
                    coeffs,
                    probe_coords_full,
                    target_full,
                    chunk_size=self.cfg.full_grid_chunk,
                )
        else:
            nmape_sub = self._nmape_at_probes(
                model,
                data,
                coeffs,
                probe_coords_full,
                target_full,
                chunk_size=max(n_probe, self.cfg.full_grid_chunk),
            )
            nmape_full = nmape_sub

        norm = coeff_norm_denominator(data, model.n_orbitals, model.unique_atom_types)
        coeff_norm = coeffs.to(device=dev, dtype=data.coords.dtype) / norm
        mask = model.orb_index[data.atom_types.long()].bool()
        active = coeff_norm[mask]
        return {
            "nmape_pct": nmape_sub,
            "nmape_full_pct": nmape_full,
            "n_probe": float(n_probe),
            "coeff_norm_min": float(active.min().item()) if active.numel() else float("nan"),
        }

    def on_graph(
        self,
        *,
        shard_name: str,
        local_idx: int,
        raw: bytes,
        coeff_payload: bytes,
        elapsed_sec: Optional[float] = None,
    ) -> None:
        self._buffer.append(
            (shard_name, local_idx, raw, coeff_payload, float(elapsed_sec or 0.0))
        )
        self._total_computed += 1
        if len(self._buffer) >= self.cfg.interval:
            self.flush()

    def flush(self, *, force: bool = False) -> None:
        if not self._buffer:
            return
        if not force and len(self._buffer) < self.cfg.interval:
            return

        nmaps: List[float] = []
        nmaps_full: List[float] = []
        mins: List[float] = []
        elapsed_secs: List[float] = []
        n_probe_ref: Optional[int] = None
        for _shard, _idx, raw, payload, elapsed_sec in self._buffer:
            elapsed_secs.append(float(elapsed_sec))
            try:
                m = self._graph_metrics(raw, payload)
                nmaps.append(m["nmape_pct"])
                nmaps_full.append(m["nmape_full_pct"])
                mins.append(m["coeff_norm_min"])
                if n_probe_ref is None:
                    n_probe_ref = int(m["n_probe"])
            except Exception as exc:
                print(
                    f"sidecar QA ERROR [{self.cfg.sidecar_label}]: "
                    f"shard={_shard} idx={_idx} err={exc!r}",
                    file=sys.stderr,
                    flush=True,
                )

        if nmaps:
            arr = np.asarray(nmaps, dtype=np.float64)
            arr_full = np.asarray(nmaps_full, dtype=np.float64)
            mins_arr = np.asarray(mins, dtype=np.float64)
            lo = self._total_computed - len(self._buffer) + 1
            hi = self._total_computed
            shard_hint = self._buffer[-1][0]
            use_subsample = (
                self.cfg.probe_samples >= 0
                and n_probe_ref is not None
                and self.cfg.probe_samples < n_probe_ref
            )
            mean_nmape = float(np.nanmean(arr))
            mean_nmape_full = float(np.nanmean(arr_full))
            coeff_min = float(np.nanmin(mins_arr)) if np.isfinite(mins_arr).any() else float("nan")
            msg = (
                f"sidecar QA [{self.cfg.sidecar_label}/{self.cfg.density_synthesis_mode}] "
                f"graphs {lo}-{hi} (last_shard={shard_hint}): "
                f"n={len(arr)} coeff_norm_min={coeff_min:.3e}"
            )
            if use_subsample:
                msg += (
                    f" | subsample({self.cfg.probe_samples})"
                    f" nMAPE mean={mean_nmape:.4f}%"
                    f" median={float(np.nanmedian(arr)):.4f}%"
                    f" max={float(np.nanmax(arr)):.4f}%"
                    f" p95={float(np.nanpercentile(arr, 95)):.4f}%"
                )
                if self.cfg.full_grid_validate:
                    msg += (
                        f" | full_grid({n_probe_ref})"
                        f" nMAPE mean={mean_nmape_full:.4f}%"
                        f" median={float(np.nanmedian(arr_full)):.4f}%"
                        f" max={float(np.nanmax(arr_full)):.4f}%"
                        f" p95={float(np.nanpercentile(arr_full, 95)):.4f}%"
                    )
                warn_val = mean_nmape_full if self.cfg.full_grid_validate else mean_nmape
            else:
                msg += (
                    f" | full_grid({n_probe_ref})"
                    f" nMAPE mean={mean_nmape_full:.4f}%"
                    f" median={float(np.nanmedian(arr_full)):.4f}%"
                    f" max={float(np.nanmax(arr_full)):.4f}%"
                    f" p95={float(np.nanpercentile(arr_full, 95)):.4f}%"
                )
                warn_val = mean_nmape_full
            n_nan = int(np.sum(~np.isfinite(arr)))
            if n_nan:
                msg += f"  ({n_nan} graph(s) with non-finite nMAPE)"
            if warn_val > self.cfg.nmape_warn_threshold:
                msg += f"  WARNING: mean nMAPE > {self.cfg.nmape_warn_threshold}%"
            if elapsed_secs:
                arr_t = np.asarray(elapsed_secs, dtype=np.float64)
                msg += (
                    f" | project_sec mean={float(arr_t.mean()):.1f}"
                    f" median={float(np.median(arr_t)):.1f}"
                    f" max={float(arr_t.max()):.1f}"
                    f" total={float(arr_t.sum()):.1f}"
                )
            print(msg, flush=True)

        self._buffer.clear()


def inline_validate_interval_from_env(default: int = 100) -> int:
    import os

    for key in ("COEFF_INLINE_VALIDATE_EVERY", "GT_INLINE_VALIDATE_EVERY"):
        val = os.environ.get(key)
        if val is not None and str(val).strip():
            return int(val)
    return int(default)


def inline_validate_probe_samples_from_env(default: int = 8192) -> int:
    import os

    for key in ("COEFF_INLINE_VALIDATE_PROBE_SAMPLES", "GT_INLINE_VALIDATE_PROBE_SAMPLES"):
        val = os.environ.get(key)
        if val is not None and str(val).strip():
            return int(val)
    return int(default)


def inline_validate_full_grid_from_env(default: bool = True) -> bool:
    import os

    for key in ("COEFF_INLINE_VALIDATE_FULL_GRID", "GT_INLINE_VALIDATE_FULL_GRID"):
        val = os.environ.get(key)
        if val is not None and str(val).strip():
            return str(val).strip().lower() not in ("0", "false", "no", "off")
    return bool(default)


def inline_validate_full_grid_chunk_from_env(default: int = 8192) -> int:
    import os

    for key in (
        "COEFF_INLINE_VALIDATE_FULL_GRID_CHUNK",
        "GT_INLINE_VALIDATE_FULL_GRID_CHUNK",
    ):
        val = os.environ.get(key)
        if val is not None and str(val).strip():
            return int(val)
    return int(default)
