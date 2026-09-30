"""
Virtual node generation methods.
"""

import numpy as np
import torch
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph

from scdp.common.utils import compute_bonds

CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False
)

VNODE_METHODS = (
    "voronoi",
    "bond",
    "both",
    "none",
)


def get_virtual_nodes(
        atom_coords, cell, pbc, 
        method='both',
        factor=5, 
        min_radius=1.5, 
        resolution=0.6,
        in_dist=0.6,
        atom_types=None, 
        struct=None,
    ):
    """
    Get virtual nodes for the given atom coordinates and cell.
    Args:
        atom_coords : (np.ndarray, N x 3) The atomic coordinates.
        cell        : (np.ndarray, 3 x 3) The cell matrix.
        pbc         : (bool) Whether to use PBC.
        method      : (str) The method to use for virtual node generation.
        factor      : (int) Max void count factor for legacy voronoi knobs.
        min_radius  : (float) The minimum radius for a virtual node.
        in_dist     : (float) The distance cutoff for the distance_mask.
        atom_types  : (np.ndarray, N x 1) The atomic numbers. only used for non-pbc bond method.
        struct      : (pymatgen.Structure) Required for pbc bond.
    Return:
        (np.ndarray) Cartisan coordinates of the virtual nodes.        
    """
    if method == "none":
        return None
    assert method in VNODE_METHODS, f"Unknown vnode method: {method}"

    if isinstance(atom_coords, torch.Tensor):
        atom_coords = atom_coords.cpu().numpy()
    if isinstance(cell, torch.Tensor):
        cell = cell.cpu().numpy()
    if isinstance(atom_types, torch.Tensor):
        atom_types = atom_types.cpu().numpy()
        
    if method == 'bond':
        if pbc:
            return pbc_bond_mid(struct)[0]
        else:
            return molecular_bond_mid(atom_coords, atom_types)[0]

def distance_mask(vertices, atom_coords, cutoff, le=True):
    min_dists = np.sort(
            ((vertices[:, None] - atom_coords[None, :]) ** 2).sum(-1) ** (1 / 2), axis=1
        )[:, 0]
    if le:
        return min_dists <= cutoff
    else:
        return min_dists > cutoff

def molecular_bond_mid(atom_coords, atom_types, ob_bond=False):
    pdist = np.sqrt(
        ((atom_coords[None, :] - atom_coords[:, None]) ** 2).sum(axis=-1)
    )
    bonds = compute_bonds(pdist, atom_types).T
    if bonds.size == 0:
        return np.empty((0, 3), dtype=atom_coords.dtype), bonds
    bond_mid = atom_coords[bonds[0]] / 2 + atom_coords[bonds[1]] / 2
    bond_mid = np.unique(bond_mid, axis=0)
    return bond_mid, bonds

def pbc_bond_mid(struct):
    if isinstance(struct, Atoms):
        struct = AseAtomsAdaptor.get_structure(struct)
    crystal_graph = StructureGraph.with_local_env_strategy(struct, CrystalNN)

    edge_index, to_jimages = [], []
    for i, j, to_jimage in crystal_graph.graph.edges(data="to_jimage"):
        edge_index.append([j, i])
        to_jimages.append(to_jimage)
        edge_index.append([i, j])
        to_jimages.append(tuple(tj for tj in to_jimage))
    edge_index = np.array(edge_index).T
    to_jimages = np.array(to_jimages)

    coords = struct.cart_coords
    cell = struct.lattice.matrix
    shifts = to_jimages @ cell
    src = coords[edge_index[0]]
    dst = coords[edge_index[1]] + shifts
    bond_mid = ((src + (dst - src) / 2) @ np.linalg.inv(cell) % 1) @ cell
    bond_mid = np.unique(bond_mid, axis=0)
    return bond_mid, edge_index, to_jimages