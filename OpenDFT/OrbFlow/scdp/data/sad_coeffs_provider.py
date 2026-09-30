"""On-the-fly SAD density superposition + ridge projection to ``sad_coeffs``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from scdp.data.gt_coeff_projection import make_gto_basis_pack
from scdp.data.sad_density import AtomicDensityLibrary, attach_sad_coeffs_to_data


@dataclass
class SadCoeffsProvider:
    """
    Build ``sad_coeffs`` from isolated-atom VASP references (no precomputed sidecar).

    SAD **density** is evaluated on the fly; coefficients come from the same ridge
    projection used in ``compute_sad_coeffs_lmdb.py``.
    """

    pack: Any
    library: AtomicDensityLibrary
    scale: float
    ridge: float = 1e-6
    ridge_diag_frac: float = 0.01
    probe_weighting: Optional[str] = None
    max_probe_samples: int = -1
    projection_device: str = "cpu"
    projection_dtype: str = "float64"
    chunk_size: int = 4096
    accumulate_on_cpu: bool = True
    projection_seed: int = 0
    density_synthesis_mode: str = "linear"

    @classmethod
    def from_model_hparams(cls, hparams, metadata: Dict) -> "SadCoeffsProvider":
        if not getattr(hparams, "atomic_chgcar_dir", None):
            raise ValueError(
                "sad_on_the_fly=true requires model.atomic_chgcar_dir (isolated-atom CHGCAR tree)"
            )
        pack = make_gto_basis_pack(
            unique_atom_types=metadata["unique_atom_types"],
            dft_basis_set=hparams.dft_basis_set,
            dft_wt_aug=bool(hparams.dft_wt_aug),
            beta=float(hparams.beta),
            lmax_restriction=bool(hparams.lmax_restriction),
            lmax_relax=int(getattr(hparams, "lmax_relax", 0) or 0),
            uncontracted=bool(hparams.uncontracted),
            orb_cutoff=float(hparams.orb_cutoff),
            vnode_elem=int(hparams.vnode_elem),
            device="cpu",
            density_synthesis_mode=str(
                getattr(hparams, "density_synthesis_mode", "linear")
            ),
        )
        library = AtomicDensityLibrary.from_directory(
            hparams.atomic_chgcar_dir,
            elements=metadata["unique_atom_types"],
        )
        return cls(
            pack=pack,
            library=library,
            scale=float(math.sqrt(metadata["target_var"])),
            ridge=float(getattr(hparams, "sad_ridge", 1e-6)),
            ridge_diag_frac=float(getattr(hparams, "sad_ridge_diag_frac", 0.01)),
            probe_weighting=getattr(hparams, "sad_probe_weighting", None),
            max_probe_samples=int(getattr(hparams, "sad_max_probe_samples", -1)),
            projection_device=str(getattr(hparams, "sad_projection_device", "cpu")),
            projection_dtype=str(getattr(hparams, "sad_projection_dtype", "float64")),
            chunk_size=int(getattr(hparams, "sad_chunk_size", 4096)),
            accumulate_on_cpu=bool(getattr(hparams, "sad_accumulate_on_cpu", True)),
            projection_seed=int(getattr(hparams, "sad_projection_seed", 0)),
            density_synthesis_mode=str(
                getattr(hparams, "density_synthesis_mode", "linear")
            ),
        )

    def compute_graph(self, data, *, seed_offset: int = 0) -> torch.Tensor:
        """Return raw ``sad_coeffs`` (num_nodes, max_outdim) for one PyG ``Data`` graph."""
        attach_sad_coeffs_to_data(
            data,
            self.pack,
            self.library,
            scale=self.scale,
            ridge=self.ridge,
            ridge_diag_frac=self.ridge_diag_frac,
            probe_weighting=self.probe_weighting,
            max_probe_samples=self.max_probe_samples,
            seed=self.projection_seed + int(seed_offset),
            device=self.projection_device,
            dtype_name=self.projection_dtype,
            chunk_size=self.chunk_size,
            accumulate_on_cpu=self.accumulate_on_cpu,
            density_synthesis_mode=self.density_synthesis_mode,
        )
        return data.sad_coeffs

    def compute_batch(self, batch) -> torch.Tensor:
        """Attach ``sad_coeffs`` for each graph in a PyG batch; return concatenated raw tensor."""
        if not hasattr(batch, "to_data_list"):
            raise TypeError("batch must support to_data_list() for on-the-fly SAD projection")
        parts = []
        for i, data in enumerate(batch.to_data_list()):
            parts.append(self.compute_graph(data, seed_offset=i))
        return torch.cat(parts, dim=0).to(
            device=batch.coords.device, dtype=batch.coords.dtype
        )
