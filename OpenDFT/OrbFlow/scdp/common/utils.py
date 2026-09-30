from typing import Tuple
import itertools
from copy import deepcopy as copy
import torch
import numpy as np

from ase.data import chemical_symbols
from pymatgen.core.structure import Structure, Molecule
from pymatgen.analysis.graphs import StructureGraph, MoleculeGraph

from scdp.common.constants import COVALENT_RADII, metals, alkali, lanthanides

# NOTE: future versions could consider more supercells.
SUPERCELLS = torch.FloatTensor(list(itertools.product((-1, 0, 1), repeat=3)))

def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # special case of torch_scatter.scatter with dim=0
    out = src.new_zeros(dim_size, *src.shape[1:])
    index = index.reshape(-1, *([1]*(len(src.shape)-1))).expand_as(src)
    return out.scatter_add_(0, index, src)

def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths

def frac2cart(fcoord, cell):
    return fcoord @ cell

def cart2frac(coord, cell):
    invcell = torch.linalg.inv(cell)
    return coord @ invcell


# NOTE: ase code computes a neighbor list without scaling the covalent radii.
def compute_bonds(distance_mat, atomic_numbers):
    """
    adapted from:
    https://github.com/peteboyd/lammps_interface/blob/master/lammps_interface/structure_data.py#L315
    """
    allatomtypes = [chemical_symbols[i] for i in atomic_numbers]
    all_bonds = []
    for i, e1 in enumerate(allatomtypes[:-1]):
        for j, e2 in enumerate(allatomtypes[i + 1 :]):
            elements = set([e1, e2])
            if elements < metals:  # FIXME no metal-metal bond allowed
                continue
            rad = COVALENT_RADII[e1] + COVALENT_RADII[e2]
            dist = distance_mat[i, i + j + 1]
            # check for atomic overlap:
            if dist < min(COVALENT_RADII[e1], COVALENT_RADII[e2]):
                print(dist)
                print(e1, e2)
                print("atomic overlap!")
            tempsf = 0.9
            # probably a better way to fix these kinds of issues..
            if (set("F") < elements) and (elements & metals):
                tempsf = 0.8
            if (set("C") < elements) and (elements & metals):
                tempsf = 0.95
            if (
                (set("H") < elements)
                and (elements & metals)
                and (not elements & alkali)
            ):
                tempsf = 0.75

            if (set("O") < elements) and (elements & metals):
                tempsf = 0.85
            if (set("N") < elements) and (elements & metals):
                tempsf = 0.82
            # fix for water particle recognition.
            if set(["O", "H"]) <= elements:
                tempsf = 0.8
            # very specific fix for Michelle's amine appended MOF
            if set(["N", "H"]) <= elements:
                tempsf = 0.67
            if set(["Mg", "N"]) <= elements:
                tempsf = 0.80
            if set(["C", "H"]) <= elements:
                tempsf = 0.80
            if set(["K"]) <= elements:
                tempsf = 0.95
            if lanthanides & elements:
                tempsf = 0.95
            if elements == set(["C"]):
                tempsf = 0.85
            if dist * tempsf < rad:  # and not (alkali & elements):
                all_bonds.append([i, i + j + 1])
                all_bonds.append([i + j + 1, i])
    return torch.LongTensor(all_bonds)


def graph_probe_to_atom(cell, atom_coords, probe_coords, cutoff=4.0, num_cells=1):
    """
    compute edges between atoms and probes under PBC.
    """
    pos = torch.arange(-num_cells, num_cells + 1, 1).to(cell.device)
    combos = (
        torch.stack(torch.meshgrid(pos, pos, pos, indexing="xy"))
        .permute(3, 2, 1, 0)
        .reshape(-1, 3)
        .to(cell.device)
    )
    shifts = torch.sum(cell.unsqueeze(0) * combos.unsqueeze(-1), dim=1)
    # n x supercell x 3
    shifted = atom_coords.unsqueeze(1) + shifts.unsqueeze(0)
    # m x 1 x 1 x 3 - 1 x n x supercell x 3
    diff_vec = probe_coords.unsqueeze(1).unsqueeze(1) - shifted.unsqueeze(0)
    # +eps to avoid nan in differentiation -> m x n x supercell
    dist_sqr = diff_vec.pow(2).sum(dim=-1)

    mask_within_cutoff = torch.le(dist_sqr, cutoff * cutoff)
    idx_probe, idx_atom, image = torch.nonzero(mask_within_cutoff, as_tuple=True)
    return idx_probe, idx_atom, image

def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1.0, 1.0)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack(
        [
            lengths[:, 0] * sins[:, 1],
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 0] * coses[:, 1],
        ],
        dim=1,
    )
    vector_b = torch.stack(
        [
            -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
            lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
            lengths[:, 1] * coses[:, 0],
        ],
        dim=1,
    )
    vector_c = torch.stack(
        [
            torch.zeros(lengths.size(0), device=lengths.device),
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 2],
        ],
        dim=1,
    )

    return torch.stack([vector_a, vector_b, vector_c], dim=1)

def compute_volume(batch_lattice):
    """
    Compute volume from batched lattice matrix.
    batch_lattice: (N, 3, 3)
    """
    vector_a, vector_b, vector_c = torch.unbind(batch_lattice, dim=1)
    return torch.abs(
        torch.einsum("bi,bi->b", vector_a, torch.cross(vector_b, vector_c, dim=1))
    )

def cell_heights(cell):
    volume = torch.linalg.det(cell).abs()
    crossproducts = torch.cross(cell[[1, 2, 0]], cell[[2, 0, 1]], dim=-1)
    heights = 1 / torch.norm(crossproducts / volume, p=2, dim=-1)
    return heights

def get_pmg_mol(cart_coords, atom_types, edge_index=None):
    atom_types[atom_types == 0] = 86
    sites = []
    for i in range(cart_coords.shape[0]):
        symbol = chemical_symbols[atom_types[i]]
        sites.append(
            {
                "species": [{"element": symbol, "occu": 1.0}],
                "xyz": cart_coords[i].tolist(),
                "properties": {},
            }
        )
    pymatgen_dict = {
        "@module": "pymatgen.core.structure",
        "@class": "Molecule",
        "charge": 0.0,
        "spin_multiplicity": 0,
        "sites": sites,
    }
    mol = Molecule.from_dict(pymatgen_dict)
    if edge_index is not None:
        remove_dir_mask = edge_index[0] <= edge_index[1]
        edge_index = edge_index[:, remove_dir_mask]
        edges = {
            (int(edge_index[0, i]), int(edge_index[1, i])): {}
            for i in range(edge_index.shape[1])
        }
    else:
        edges = {}
    graph = MoleculeGraph.with_edges(mol, edges)
    return graph


def get_pmg_struct(
    cart_coords, atom_types, lattice_matrix, edge_index=None, to_jimages=None
):
    """
    from pyg graph to pymatgen StructureGraph.
    """
    lattice = {
        "matrix": lattice_matrix.squeeze().tolist(),
    }

    frac_coords = cart_coords @ np.linalg.inv(lattice_matrix)
    sites = []
    for i in range(cart_coords.shape[0]):
        symbol = chemical_symbols[atom_types[i]]
        sites.append(
            {
                "species": [{"element": symbol, "occu": 1.0}],
                "abc": frac_coords[i].tolist(),
                "xyz": cart_coords[i].tolist(),
                "label": symbol,
                "properties": {},
            }
        )
    pymatgen_dict = {
        "@module": "pymatgen.core.structure",
        "@class": "Structure",
        "charge": None,
        "lattice": lattice,
        "sites": sites,
    }

    structure = Structure.from_dict(pymatgen_dict)

    if edge_index is not None:
        remove_dir_mask = edge_index[0] <= edge_index[1]
        edge_index = edge_index[:, remove_dir_mask]
        to_jimages = to_jimages[remove_dir_mask]
        edges = {
            (
                int(edge_index[0, i]),
                int(edge_index[1, i]),
                (0, 0, 0),
                tuple(to_jimages[i].tolist()),
            ): {}
            for i in range(edge_index.shape[1])
        }
    else:
        edges = {}
    graph = StructureGraph.with_edges(structure, edges)
    return graph

def radius_graph_pbc(
    cart_coords,
    lengths,
    angles,
    num_atoms,
    radius,
    max_num_neighbors_threshold,
    device,
    topk_per_pair=None,
    remove_self_edges=True,
):
    """
    Computes pbc graph edges under pbc.
    topk_per_pair: (num_atom_pairs,), select topk edges per atom pair
    Note: topk should take into account self-self edge for (i, i)
    """
    batch_size = len(num_atoms)
    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = num_atoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="trunc")
    ).long() + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand).long() + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(cart_coords, 0, index1)
    pos2 = torch.index_select(cart_coords, 0, index2)

    unit_cell = torch.tensor(SUPERCELLS, device=device).float()
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # lattice matrix
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 27 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)

    if topk_per_pair is not None:
        assert topk_per_pair.size(0) == num_atom_pairs
        atom_distance_sqr_sort_index = torch.argsort(atom_distance_sqr, dim=1)
        assert atom_distance_sqr_sort_index.size() == (num_atom_pairs, num_cells)
        atom_distance_sqr_sort_index = (
            atom_distance_sqr_sort_index
            + torch.arange(num_atom_pairs, device=device)[:, None] * num_cells
        ).view(-1)
        topk_mask = (
            torch.arange(num_cells, device=device)[None, :] < topk_per_pair[:, None]
        )
        topk_mask = topk_mask.view(-1)
        topk_indices = atom_distance_sqr_sort_index.masked_select(topk_mask)

        topk_mask = torch.zeros(num_atom_pairs * num_cells, device=device)
        topk_mask.scatter_(0, topk_indices, 1.0)
        topk_mask = topk_mask.bool()

    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    if remove_self_edges:
        mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
        mask = torch.logical_and(mask_within_radius, mask_not_same)
    else:
        mask = mask_within_radius
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    if topk_per_pair is not None:
        topk_mask = torch.masked_select(topk_mask, mask)

    num_neighbors = torch.zeros(len(cart_coords), device=device)
    num_neighbors.index_add_(0, index1, torch.ones(len(index1), device=device))
    num_neighbors = num_neighbors.long()
    max_num_neighbors = torch.max(num_neighbors).long()

    # Compute neighbors per image
    _max_neighbors = copy(num_neighbors)
    _max_neighbors[
        _max_neighbors > max_num_neighbors_threshold
    ] = max_num_neighbors_threshold
    _num_neighbors = torch.zeros(len(cart_coords) + 1, device=device).long()
    _natoms = torch.zeros(num_atoms.shape[0] + 1, device=device).long()
    _num_neighbors[1:] = torch.cumsum(_max_neighbors, dim=0)
    _natoms[1:] = torch.cumsum(num_atoms, dim=0)
    num_neighbors_image = _num_neighbors[_natoms[1:]] - _num_neighbors[_natoms[:-1]]

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        if topk_per_pair is None:
            return torch.stack((index2, index1)), -unit_cell, num_neighbors_image
        else:
            return (
                torch.stack((index2, index1)),
                -unit_cell,
                num_neighbors_image,
                topk_mask,
            )

    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with values greater than radius*radius so we can easily remove unused distances later.
    distance_sort = torch.zeros(
        len(cart_coords) * max_num_neighbors, device=device
    ).fill_(radius * radius + 1.0)

    # Create an index map to map distances from atom_distance_sqr to distance_sort
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index1 * max_num_neighbors
        + torch.arange(len(index1), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance_sqr)
    distance_sort = distance_sort.view(len(cart_coords), max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index1
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors_threshold
    )
    # Remove "unused pairs" with distances greater than the radius
    mask_within_radius = torch.le(distance_sort, radius * radius)
    index_sort = torch.masked_select(index_sort, mask_within_radius)

    # At this point index_sort contains the index into index1 of the closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index1), device=device).bool()
    mask_num_neighbors.index_fill_(0, index_sort, True)

    # Finally mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
    index1 = torch.masked_select(index1, mask_num_neighbors)
    index2 = torch.masked_select(index2, mask_num_neighbors)
    unit_cell = torch.masked_select(
        unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)

    if topk_per_pair is not None:
        topk_mask = torch.masked_select(topk_mask, mask_num_neighbors)

    edge_index = torch.stack((index2, index1))

    # fix to to_jimages: negate unit_cell.
    if topk_per_pair is None:
        return edge_index, -unit_cell, num_neighbors_image
    else:
        return edge_index, -unit_cell, num_neighbors_image, topk_mask