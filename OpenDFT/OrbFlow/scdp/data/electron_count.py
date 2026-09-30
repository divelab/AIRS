"""Valence electron counts for PAW/VASP charge densities (matches baseline LMDB labels)."""

from __future__ import annotations

from typing import Optional

import torch

from scdp.common.utils import scatter

# Valence electrons per element in DTU QM9 / typical PAW setups (not nuclear charge Z).
VALENCE_ELECTRONS = {
    1: 1,  # H
    6: 4,  # C
    7: 5,  # N
    8: 6,  # O
    9: 7,  # F
    16: 6,  # S (MD)
    17: 7,  # Cl (MD)
}


def valence_electrons_for_Z(Z: int) -> int:
    z = int(Z)
    if z not in VALENCE_ELECTRONS:
        raise KeyError(
            f"No valence electron count for Z={z}. "
            f"Known elements: {sorted(VALENCE_ELECTRONS)}"
        )
    return VALENCE_ELECTRONS[z]


def _real_atom_mask(atom_types: torch.Tensor, is_vnode: Optional[torch.Tensor]) -> torch.Tensor:
    if is_vnode is not None:
        return ~is_vnode
    return atom_types > 0


def valence_electron_count(
    atom_types: torch.Tensor,
    is_vnode: Optional[torch.Tensor] = None,
) -> float:
    """Total valence electrons on real atoms (baseline ``chg_labels`` integrate to this)."""
    mask = _real_atom_mask(atom_types, is_vnode)
    types = atom_types[mask].tolist()
    return float(sum(valence_electrons_for_Z(int(z)) for z in types))


def valence_electron_count_per_graph(batch) -> torch.Tensor:
    """Per-graph valence electron counts (excludes virtual nodes)."""
    bsz = int(batch.batch.max().item()) + 1
    device = batch.coords.device
    real = _real_atom_mask(batch.atom_types, getattr(batch, "is_vnode", None))
    types = batch.atom_types.long()[real]
    graph = batch.batch[real]
    valence = torch.tensor(
        [valence_electrons_for_Z(int(z)) for z in types.tolist()],
        device=device,
        dtype=torch.float32,
    )
    return scatter(valence, graph, bsz).to(device=device)
