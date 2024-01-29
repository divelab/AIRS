import numpy as np
import torch
from torch_scatter import scatter


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
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack([
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)
    return torch.stack([vector_a, vector_b, vector_c], dim=1)


def frac_to_cart_coords(
    frac_coords,
    lengths,
    angles,
    num_atoms,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
    cart_coords = torch.einsum('bi,bij->bj', frac_coords.float(), lattice_nodes.float())  # cart coords
    return cart_coords


def cart_to_frac_coords(
    cart_coords,
    lengths,
    angles,
    num_atoms,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    # use pinv in case the predicted lattice is not rank 3
    inv_lattice = torch.pinverse(lattice)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)
    frac_coords = torch.einsum('bi,bij->bj', cart_coords.float(), inv_lattice_nodes.float())
    return frac_coords


def correct_cart_coords(cart_coords, lengths, angles, num_atoms, batch):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
    
    inv_lattice = torch.inverse(lattice)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)

    frac_coords = torch.einsum('bi,bij->bj', cart_coords, inv_lattice_nodes)
    frac_coords = correct_frac_coords(frac_coords, batch)
    
    cart_coords = torch.einsum('bi,bij->bj', frac_coords, lattice_nodes)  # cart coords
    return cart_coords


def correct_frac_coords(frac_coords, batch):
    new_frac_coords = (frac_coords + 0.5) % 1. - 0.5
    min_frac_coords = scatter(new_frac_coords, batch, dim=0, reduce='min')
    max_frac_coords = scatter(new_frac_coords, batch, dim=0, reduce='max')
    offset_frac_coords = (min_frac_coords + max_frac_coords) / 2.0
    new_frac_coords = new_frac_coords - offset_frac_coords[batch]
    
    return new_frac_coords


def get_pbc_distances(
    coords,
    edge_index,
    lengths,
    angles,
    to_jimages,
    num_atoms,
    num_bonds,
    coord_is_cart=False,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    if coord_is_cart:
        pos = coords
    else:
        lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
        pos = torch.einsum('bi,bij->bj', coords, lattice_nodes)  # cart coords

    j_index, i_index = edge_index

    distance_vectors = pos[i_index] - pos[j_index]

    # correct for pbc
    lattice_edges = torch.repeat_interleave(lattice, num_bonds, dim=0)
    offsets = torch.einsum('bi,bij->bj', to_jimages.float(), lattice_edges)
    distance_vectors -= offsets

    return pos, distance_vectors, offsets


OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]


def get_pbc_cutoff_graphs(cart_coords, lengths, angles, num_atoms, cutoff=7.0, max_num_neighbors_threshold=20):
    batch_size = len(num_atoms)

    # position of the atoms
    atom_pos = cart_coords

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = num_atoms
    num_atoms_per_image_sqr = (num_atoms_per_image ** 2).long()

    # index offset between images
    index_offset = (
        torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    )

    index_offset_expand = torch.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
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
    atom_count_sqr = (
        torch.arange(num_atom_pairs, device=num_atoms.device) - index_sqr_offset
    )

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        (atom_count_sqr // num_atoms_per_image_expand)
    ).long() + index_offset_expand
    index2 = (
        atom_count_sqr % num_atoms_per_image_expand
    ).long() + index_offset_expand
    # Get the positions for each atom
    # print(index1, index2)
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)
    
    # print(atom_pos.shape, pos1.shape, pos2.shape)
    unit_cell = torch.tensor(OFFSET_LIST, device=num_atoms.device).float()
    num_cells = len(unit_cell)
    # unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
    #     len(index2), 1, 1
    # )
    # unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, num_cells, 3).expand(
        batch_size, -1, -1
    )

    # lattice matrix
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    # Compute the x, y, z positional offsets for each cell in each image
    # data_cell = torch.transpose(lattice, 1, 2)
    # pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets = torch.bmm(unit_cell_batch, lattice)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    ).view(-1, 3)

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 1, 3).expand(-1, num_cells, -1).contiguous().view(-1, 3)
    pos2 = pos2.view(-1, 1, 3).expand(-1, num_cells, -1).contiguous().view(-1, 3)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    # pos2 = pos2 + pbc_offsets_per_atom
    distance_vectors = pos1 - pos2 - pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum(distance_vectors ** 2, dim=1)

    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, cutoff * cutoff)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    distance_vectors = distance_vectors[mask]
    pbc_offsets_per_atom = pbc_offsets_per_atom[mask]
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    
    num_neighbors = torch.zeros(len(cart_coords), device=cart_coords.device)
    num_neighbors.index_add_(0, index1, torch.ones(len(index1), device=index1.device))
    num_neighbors = num_neighbors.long()
    max_num_neighbors = torch.max(num_neighbors).long()

    if max_num_neighbors_threshold <= 0 or max_num_neighbors <= max_num_neighbors_threshold:
        edge_index = torch.stack((index2, index1))
        return edge_index, distance_vectors, pbc_offsets_per_atom

    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with values greater than radius*radius so we can easily remove unused distances later.
    distance_sort = torch.zeros(
        len(cart_coords) * max_num_neighbors, device=cart_coords.device
    ).fill_(cutoff * cutoff + 1.0)

    # Create an index map to map distances from atom_distance_sqr to distance_sort
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index1 * max_num_neighbors
        + torch.arange(len(index1), device=index1.device)
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
    mask_within_radius = torch.le(distance_sort, cutoff * cutoff)
    index_sort = torch.masked_select(index_sort, mask_within_radius)

    # At this point index_sort contains the index into index1 of the closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index1), device=index1.device).bool()
    mask_num_neighbors.index_fill_(0, index_sort, True)

    # Finally mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
    index1 = torch.masked_select(index1, mask_num_neighbors)
    index2 = torch.masked_select(index2, mask_num_neighbors)
    distance_vectors = distance_vectors[mask_num_neighbors]
    pbc_offsets_per_atom = pbc_offsets_per_atom[mask_num_neighbors]
    edge_index = torch.stack((index2, index1))

    return edge_index, distance_vectors, pbc_offsets_per_atom


def distance_matrix_pbc(cart_coords, lengths, angles):
    """Compute the pbc distance between atoms in cart_coords1 and cart_coords2.
    This function assumes that cart_coords1 and cart_coords2 have the same number of atoms
    in each data point.
    returns:
        basic return:
            min_atom_distance_sqr: (N_atoms, )
        return_vector == True:
            min_atom_distance_vector: vector pointing from cart_coords1 to cart_coords2, (N_atoms, 3)
        return_to_jimages == True:
            to_jimages: (N_atoms, 3), position of cart_coord2 relative to cart_coord1 in pbc
    """
    num_atoms = cart_coords.shape[0]

    unit_cell = torch.tensor(OFFSET_LIST, device=cart_coords.device).float()
    num_cells = len(unit_cell)
    unit_cell = torch.transpose(unit_cell, 0, 1).view(1, 3, num_cells)

    # lattice matrix
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = cart_coords.view(-1, 1, 3, 1).expand(-1, -1, -1, num_cells)
    pos2 = cart_coords.view(1, -1, 3, 1).expand(-1, -1, -1, num_cells)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the vector between atoms
    # shape (num_atom_squared_sum, 3, 27)
    atom_distances_pbc = (pos1 - pos2).norm(dim=-2)
    atom_distances, _ = atom_distances_pbc.min(dim=-1)
    
    return atom_distances


def align_gt_cart_coords(gt_cart_coords, cart_coords_perturbed, lengths, angles, num_atoms):
    num_graphs = len(num_atoms)

    unit_cell = torch.tensor(OFFSET_LIST, device=gt_cart_coords.device).float()
    num_cells = len(unit_cell)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(num_graphs, -1, -1)

    lattice = lattice_params_to_matrix_torch(lengths, angles)
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(pbc_offsets, num_atoms, dim=0)

    gt_cart_coords = gt_cart_coords.view(-1, 3, 1).expand(-1, -1, num_cells)
    cart_coords_perturbed = cart_coords_perturbed.view(-1, 3, 1).expand(-1, -1, num_cells)
    gt_cart_coords = gt_cart_coords + pbc_offsets_per_atom

    atom_distance_sqr = torch.sum((gt_cart_coords - cart_coords_perturbed) ** 2, dim=1)
    _, min_indices = atom_distance_sqr.min(dim=-1)
    min_indices = min_indices[:, None, None].repeat([1, 3, 1])
    aligned_gt_cart_coords = torch.gather(gt_cart_coords, 2, min_indices).squeeze(-1)

    return aligned_gt_cart_coords