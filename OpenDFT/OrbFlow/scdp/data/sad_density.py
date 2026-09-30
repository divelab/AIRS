"""
Superposition of atomic (VASP) charge densities and projection onto GTO coefficients.

Pipeline
--------
1. Collaborator provides isolated-atom VASP ``CHGCAR`` (or ASE-readable charge files) per element.
2. ``AtomicDensityLibrary`` loads each reference, centers the grid on the atomic nucleus, and
   keeps VASP valence density as-is (same convention as baseline LMDB ``chg_labels``).
3. For each molecule graph, ``superposed_sad_density_at_probes`` evaluates
   ``ρ_SAD(r) = Σ_A ρ_Z(r - R_A)`` on the training probe coordinates (same layout as ``chg_labels``).
4. ``project_labels_to_gt_coeffs`` (in ``gt_coeff_projection``) fits ``sad_coeffs`` with the same
   ridge-regularized least squares as ``gt_coeffs``.

Expected atomic reference layout (``--atomic_chgcar_dir``)::

    atomic_chgcar/
      Z1/CHGCAR
      Z6/CHGCAR
      Z7/CHGCAR
      ...

Also accepts ``H/CHGCAR``, ``C/CHGCAR`` (element symbol folders).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from scdp.data.electron_count import _real_atom_mask
from scdp.data.gt_coeff_projection import GTOBasisPack, project_labels_to_gt_coeffs
from scdp.data.utils import calculate_grid_pos, read_vasp

try:
    from scipy.interpolate import RegularGridInterpolator
except ImportError:  # pragma: no cover
    RegularGridInterpolator = None  # type: ignore


# QM9: DTU VASP references via collaborate/QM9 (H,C,N,O,F).
# MD: GT is not VASP — SAD atomic refs TBD (see collaborate/MD/); do not assume VASP CHGCAR.
QM9_ATOMIC_NUMBERS = (1, 6, 7, 8, 9)
MD_ATOMIC_NUMBERS = (1, 6, 8)
DEFAULT_ATOMIC_NUMBERS = tuple(sorted(set(QM9_ATOMIC_NUMBERS) | set(MD_ATOMIC_NUMBERS)))


@dataclass
class AtomicDensityReference:
    """Isolated-atom density grid centered on the nucleus (local frame origin = atom center)."""

    Z: int
    density: np.ndarray  # (nx, ny, nz), e/Å³
    axes_local: Tuple[np.ndarray, np.ndarray, np.ndarray]  # 1D monotone Cartesian axes (Å)
    integrated_charge: float  # ∫ρ dV (VASP valence charge)


def _load_chgcar_file(path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = path.read_bytes()
    Zs, coords, cell, density, origin, _ = read_vasp(raw)
    return Zs, coords, cell, density, origin if origin is not None else torch.zeros(3)


def _build_local_axes(
    density: torch.Tensor, cell: torch.Tensor, origin: torch.Tensor, center: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid_cart = calculate_grid_pos(density, cell, origin)  # (nx, ny, nz, 3)
    local = grid_cart - center.view(1, 1, 1, 3)
    dens = density.detach().cpu().numpy()
    x_axis = local[:, 0, 0, 0].numpy()
    y_axis = local[0, :, 0, 1].numpy()
    z_axis = local[0, 0, :, 2].numpy()
    return dens, x_axis, y_axis, z_axis


def _integrated_charge(density: np.ndarray, cell: torch.Tensor) -> float:
    vol = float(torch.linalg.det(cell).abs().item())
    return float(density.sum() * vol / density.size)


def load_atomic_density_reference(path: Path, Z: int) -> AtomicDensityReference:
    """Load one element CHGCAR and center on the (single) atom position."""
    Zs, coords, cell, density, origin = _load_chgcar_file(path)
    if Zs.numel() != 1:
        raise ValueError(f"Expected one atom in {path}, found {Zs.numel()}")
    if int(Zs.item()) != int(Z):
        raise ValueError(f"{path}: file Z={int(Zs.item())} does not match expected Z={Z}")
    center = coords[0]
    dens, x_axis, y_axis, z_axis = _build_local_axes(density, cell, origin, center)
    q = _integrated_charge(dens, cell)
    if q <= 0:
        raise ValueError(f"Non-positive integrated charge in {path}: {q}")
    return AtomicDensityReference(
        Z=int(Z),
        density=dens,
        axes_local=(x_axis, y_axis, z_axis),
        integrated_charge=q,
    )


def _resolve_atomic_chgcar_path(root: Path, Z: int) -> Path:
    sym = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl"}.get(int(Z))
    candidates = [
        root / f"Z{Z}" / "CHGCAR",
        root / f"Z{Z}" / "CHGCAR.gz",
        root / f"Z{Z}_{sym}" / "CHGCAR" if sym else None,
        root / f"Z{Z}_{sym}" / "CHGCAR.gz" if sym else None,
        root / f"{sym}" / "CHGCAR" if sym else None,
        root / f"Z{Z}.CHGCAR",
        root / f"{sym}.CHGCAR" if sym else None,
    ]
    for p in candidates:
        if p is not None and p.is_file():
            return p
    raise FileNotFoundError(
        f"No CHGCAR for Z={Z} under {root}. Tried: "
        + ", ".join(str(c) for c in candidates if c is not None)
    )


class AtomicDensityLibrary:
    """Cached per-element interpolators for SAD superposition."""

    def __init__(self, refs: Dict[int, AtomicDensityReference]):
        self.refs = refs
        self._interp: Dict[int, RegularGridInterpolator] = {}
        for Z, ref in refs.items():
            if RegularGridInterpolator is None:
                raise ImportError("scipy is required for SAD density interpolation")
            self._interp[Z] = RegularGridInterpolator(
                ref.axes_local,
                ref.density,
                bounds_error=False,
                fill_value=0.0,
            )

    @classmethod
    def from_directory(
        cls,
        root: Union[str, Path],
        elements: Sequence[int] = DEFAULT_ATOMIC_NUMBERS,
    ) -> "AtomicDensityLibrary":
        root = Path(root)
        refs: Dict[int, AtomicDensityReference] = {}
        for Z in sorted(set(int(z) for z in elements)):
            path = _resolve_atomic_chgcar_path(root, Z)
            refs[Z] = load_atomic_density_reference(path, Z)
        return cls(refs)

    def rho_element_at(self, Z: int, points_local: np.ndarray) -> np.ndarray:
        """Evaluate ρ_Z at points in the atom-centered frame (Å)."""
        if Z not in self._interp:
            raise KeyError(f"Z={Z} not in atomic library (have {sorted(self._interp)})")
        return self._interp[Z](points_local).astype(np.float64, copy=False)


def superposed_sad_density_at_probes(
    atom_types: torch.Tensor,
    coords: torch.Tensor,
    probe_coords: torch.Tensor,
    library: AtomicDensityLibrary,
    is_vnode: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    ρ_SAD at probe points = sum of translated atomic references.

    Returns (n_probe,) float32 tensor in e/Å³ (same units as ``chg_labels`` after dataset prep).
    """
    coords_np = coords.detach().cpu().numpy()
    probes_np = probe_coords.detach().cpu().numpy()
    mask = _real_atom_mask(atom_types, is_vnode).cpu().numpy()
    types = atom_types.cpu().numpy()

    rho = np.zeros(probes_np.shape[0], dtype=np.float64)
    for i in np.where(mask)[0]:
        Z = int(types[i])
        if Z == 0:
            continue
        local = probes_np - coords_np[i]
        rho += library.rho_element_at(Z, local)

    return torch.from_numpy(rho.astype(np.float32))


def attach_sad_coeffs_to_data(
    data,
    pack: GTOBasisPack,
    library: AtomicDensityLibrary,
    scale: float,
    ridge: float = 1e-3,
    ridge_diag_frac: float = 0.01,
    probe_weighting: Optional[str] = None,
    max_probe_samples: int = 16000,
    seed: int = 0,
    device: str = "cpu",
    dtype_name: str = "float64",
    chunk_size: int = 4096,
    accumulate_on_cpu: Optional[bool] = None,
    density_synthesis_mode: str = "linear",
    coeff_nonneg: bool = False,
) -> None:
    """In-place: set ``data['sad_coeffs']`` via SAD superposition + ridge projection."""
    pbc = bool(getattr(data, "pbc", False)) if hasattr(data, "pbc") else False
    cell = data.cell
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
    is_vnode = getattr(data, "is_vnode", None)
    sad_rho = superposed_sad_density_at_probes(
        data.atom_types,
        data.coords,
        data.probe_coords,
        library,
        is_vnode=is_vnode,
    )
    gt = project_labels_to_gt_coeffs(
        data.atom_types,
        data.coords,
        cell,
        pbc,
        data.probe_coords,
        sad_rho,
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
    )
    data["sad_coeffs"] = gt.to(device=data.coords.device, dtype=data.coords.dtype)


def encode_sad_coeffs_tensor(t: torch.Tensor) -> bytes:
    return pickle.dumps(t, protocol=-1)


def compute_sad_coeffs_payload(
    raw: bytes,
    pack: GTOBasisPack,
    library: AtomicDensityLibrary,
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
) -> bytes:
    """Unpickle one LMDB graph, project SAD density to sad_coeffs, return pickled tensor bytes."""
    data = pickle.loads(raw)
    attach_sad_coeffs_to_data(
        data,
        pack,
        library,
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
    )
    return encode_sad_coeffs_tensor(data.sad_coeffs)
