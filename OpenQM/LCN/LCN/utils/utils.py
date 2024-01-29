import numpy as np
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix, csr_matrix, identity
from scipy import sparse
from numpy.random import randn
import os
import random
import torch
from torch_scatter import scatter_sum, scatter
from torch_geometric.utils import from_networkx, degree, sort_edge_index


def get_undirected_idx_list(data, periodic=True, square=False):
    edge_index = sort_edge_index(data.edge_index)[0]
    idx_list = []
    for i in range(edge_index.shape[1]):
        if edge_index[0, i] < edge_index[1, i]:
            idx_list.append(edge_index[:, i].numpy().tolist())

    # only works for square lattice
    if periodic and square:
        spin_num = data.num_nodes
        # assert spin_num == 36
        size = int(np.sqrt(spin_num))
        # assert size == 6
        for i in range(0, spin_num, size):
            idx_list.append([i, i + size - 1])
        for i in range(0, size):
            idx_list.append([i, spin_num - size + i])

    return idx_list


def flip(s, idx):
    sflip = deepcopy(s)
    sflip[idx] = -sflip[idx]
    return sflip


def heisenberg_loc(config, psi_func, psi_loc, idx_list, h=-0.5, J=-1):
    '''
    calculate local energy
    Args:
        config: one spin configuration
        psi_func: feedfoward function in model taking one input
        psi_loc: wavefunction value for one input spin configuration
        idx_list: undirected edge index list for graph
    '''
    # sigma_z * sigma_z
    e_part1 = 0
    for idx in idx_list:
        if config[idx[0]] == config[idx[1]]:
            e_part1 += 1
        else:
            e_part1 -= 1

    # sigma_x * sigma_x
    e_part2 = 0
    for idx in idx_list:
        config_i = flip(config, idx[0])
        config_i = flip(config_i, idx[1])
        e_part2 += (psi_func(config_i) / psi_loc)

    # sigma_y * sigma_y
    e_part3 = 0
    for idx in idx_list:
        config_i = flip(config, idx[0])
        config_i = flip(config_i, idx[1])
        if config[idx[0]] == config[idx[1]]:
            e_part3 -= (psi_func(config_i) / psi_loc)
        else:
            e_part3 += (psi_func(config_i) / psi_loc)

    return e_part1 + e_part2 + e_part3


def heisenberg_loc_fast(config, psi_func, psi_loc, idx_list, h=-0.5, J=-1):
    '''
    calculate local energy, fast version, using batch operation instead of for loop
    Args:
        config: one spin configuration
        psi_func: feedfoward function in model taking batch input
        psi_loc: wavefunction value for one input spin configuration
        idx_list: undirected edge index list for graph
    '''
    # sigma_z * sigma_z
    idx_array = np.array(idx_list)
    mask = config[idx_array[:, 0]] != config[idx_array[:, 1]]
    mask = mask.astype(int) * 2 - 1
    e_part1 = -np.sum(mask)

    # flip old config according to idx array to get new config list
    new_config_list = np.tile(np.array(config), (idx_array.shape[0], 1))
    row = np.arange(idx_array.shape[0])[:, None]
    new_config_list[row, idx_array] *= -1

    # calculate wave function value in new config list
    psi_list = psi_func(new_config_list)
    psi_ratio_list = psi_list / psi_loc  # torch.div(psi_list, psi_loc)

    # sigma_x * sigma_x
    e_part2 = torch.sum(psi_ratio_list)

    # sigma_y * sigma_y
    mask = torch.from_numpy(mask)
    e_part3 = torch.sum(psi_ratio_list.squeeze() * mask)

    return e_part1 + e_part2 + e_part3


## torch tensor version
def heisenberg_loc_batch_fast(config, psi_func, psi_loc, idx_list, h=-0.5, J=-1):
    '''
    calculate local energy, fast & batch version, using batch operation instead of for loop
    Args:
        config: batch spin configuration
        psi_func: feedfoward function in model taking batch input
        psi_loc: wavefunction value for batch input spin configuration
        idx_list: undirected edge index list for graph
    '''
    idx_array = torch.tensor(idx_list)

    # sigma_z * sigma_z
    mask = config[:, idx_array[:, 0]] != config[:, idx_array[:, 1]]
    mask = mask.int() * 2 - 1
    e_part1 = -torch.sum(mask, dim=1, keepdim=True)
    assert e_part1.shape[0] == config.shape[0]

    # flip old config according to idx array to get new config list
    config_batch_size = config.shape[0]
    new_config_list = torch.repeat_interleave(config, idx_array.shape[0], dim=0)
    row_batch = torch.arange(idx_array.shape[0] * config_batch_size)[:, None]
    idx_array_batch = torch.tile(idx_array, (config_batch_size, 1))
    new_config_list[row_batch, idx_array_batch] *= -1

    # calculate wave function value in new config list
    psi_loc = psi_loc.view(-1,1)
    psi_loc_batch = torch.repeat_interleave(psi_loc, idx_array.shape[0], dim=0)
    psi_list = psi_func(new_config_list).view(-1,1)
    assert psi_list.shape[0] == psi_loc_batch.shape[0]
    psi_ratio_list = psi_list / psi_loc_batch

    # sigma_x * sigma_x
    index = torch.arange(config_batch_size)
    index_batch = torch.repeat_interleave(index, idx_array.shape[0], dim=0).to(config.device)
    e_part2 = scatter_sum(src=psi_ratio_list, index=index_batch, dim=0)
    assert e_part2.shape[0] == config.shape[0]

    # sigma_y * sigma_y
    mask = mask.view(-1,1)
    e_part3 = scatter_sum(src=psi_ratio_list * mask, index=index_batch, dim=0)
    assert e_part3.shape[0] == config.shape[0]

    return (e_part1 + e_part2 + e_part3) / 4.0


def heisenberg_loc_batch_fast_J1_J2(config, psi_func, psi_loc, idx_list, J2=None):
    '''
    calculate local energy, fast & batch version, using batch operation instead of for loop
    Args:
        config: batch spin configuration
        psi_func: feedfoward function in model taking batch input
        psi_loc: wavefunction value for batch input spin configuration
        idx_list: undirected edge index list for graph
    '''
    energy = heisenberg_loc_batch_fast(config, psi_func, psi_loc, idx_list)
    if J2 is not None:
        idx_list_dist2 = get_dist2_idx(idx_list)
        energy += J2 * heisenberg_loc_batch_fast(config, psi_func, psi_loc, idx_list_dist2)

    return energy


def heisenberg_loc_it_swo(config, phi_func, psi_loc, idx_list, beta):
    '''
    calculate local energy, fast & batch version, using batch operation instead of for loop
    Args:
        config: batch spin configuration
        phi_func: feedfoward function in model taking batch input
        psi_loc: wavefunction value for batch input spin configuration
        idx_list: undirected edge index list for graph
    '''
    idx_array = torch.tensor(idx_list)

    # sigma_z * sigma_z
    mask = config[:, idx_array[:, 0]] != config[:, idx_array[:, 1]]
    mask = mask.int() * 2 - 1
    e_part1 = -torch.sum(mask, dim=1, keepdim=True)
    assert e_part1.shape[0] == config.shape[0]

    #print(config.shape, idx_array.shape)

    # flip old config according to idx array to get new config list
    config_batch_size = config.shape[0]
    new_config_list = torch.repeat_interleave(config, idx_array.shape[0], dim=0)
    row_batch = torch.arange(idx_array.shape[0] * config_batch_size)[:, None]
    idx_array_batch = torch.tile(idx_array, (config_batch_size, 1))
    new_config_list[row_batch, idx_array_batch] *= -1

    # calculate wave function value in new config list
    psi_loc = psi_loc.view(-1,1)
    psi_loc_batch = torch.repeat_interleave(psi_loc, idx_array.shape[0], dim=0)
    phi_list = phi_func(new_config_list).view(-1,1)
    assert phi_list.shape[0] == psi_loc_batch.shape[0]
    phi_ratio_list = phi_list / psi_loc_batch

    # sigma_x * sigma_x
    index = torch.arange(config_batch_size)
    index_batch = torch.repeat_interleave(index, idx_array.shape[0], dim=0).to(config.device)
    e_part2 = scatter_sum(src=phi_ratio_list, index=index_batch, dim=0)
    assert e_part2.shape[0] == config.shape[0]

    # sigma_y * sigma_y
    mask = mask.view(-1,1)
    e_part3 = scatter_sum(src=phi_ratio_list * mask, index=index_batch, dim=0)
    assert e_part3.shape[0] == config.shape[0]

    config_ratio = phi_func(config) / psi_loc

    return config_ratio - (e_part1 * config_ratio + e_part2 + e_part3) / 4.0 * beta


def heisenberg_loc_it_swo_J1_J2(config, phi_func, psi_loc, idx_list, J2, beta):
    '''
    calculate local energy, fast & batch version, using batch operation instead of for loop
    Args:
        config: batch spin configuration
        psi_func: feedfoward function in model taking batch input
        psi_loc: wavefunction value for batch input spin configuration
        idx_list: undirected edge index list for graph
    '''
    energy = heisenberg_loc_it_swo(config, phi_func, psi_loc, idx_list, beta)
    if J2 is not None:
        idx_list_dist2 = get_dist2_idx(idx_list)
        energy += J2 * heisenberg_loc_it_swo(config, phi_func, psi_loc, idx_list_dist2, beta)

    return energy


def get_dist2_idx(idx_list):
    idx_list_dist2 = []
    for idx in idx_list:
        for idx2 in idx_list:
            if idx[0] == idx2[0]:
                a, b = idx[1], idx2[1]
            elif idx[1] == idx2[0]:
                a, b = idx[0], idx2[1]
            elif idx[0] == idx2[1]:
                a, b = idx[1], idx2[0]
            elif idx[1] == idx2[1]:
                a, b = idx[0], idx2[0]
            else:
                a, b = None, None
            if a is not None and a != b:
                if b < a:
                    a, b = b, a
                if [a, b] not in idx_list and [a, b] not in idx_list_dist2:
                    idx_list_dist2.append([a, b])
    return idx_list_dist2


# def heisenberg_loc_batch_fast(config, psi_func, psi_loc, idx_list, h=-0.5, J=-1):
#     '''
#     calculate local energy, fast & batch version, using batch operation instead of for loop
#     Args:
#         config: batch spin configuration
#         psi_func: feedfoward function in model taking batch input
#         psi_loc: wavefunction value for batch input spin configuration
#         idx_list: undirected edge index list for graph
#     '''
#     # sigma_z * sigma_z
#     idx_array = np.array(idx_list)
#     mask = config[:, idx_array[:, 0]] != config[:, idx_array[:, 1]]
#     mask = mask.astype(int) * 2 - 1
#     e_part1 = torch.from_numpy(-np.sum(mask, axis=1, keepdims=True))
#     assert e_part1.shape[0] == config.shape[0]
#
#     # flip old config according to idx array to get new config list
#     config_batch_size = config.shape[0]
#     new_config_list = np.repeat(config, [idx_array.shape[0]], axis=0)
#     row_batch = np.arange(idx_array.shape[0] * config_batch_size)[:, None]
#     idx_array_batch = np.tile(idx_array, (config_batch_size, 1))
#     new_config_list[row_batch, idx_array_batch] *= -1
#
#     # calculate wave function value in new config list
#     psi_loc = psi_loc.reshape(-1,1)
#     psi_loc_batch = np.repeat(psi_loc, [idx_array.shape[0]], axis=0)
#     psi_list = psi_func(new_config_list).view(-1,1)
#     assert psi_list.shape[0] == psi_loc_batch.shape[0]
#     psi_ratio_list = psi_list / psi_loc_batch
#
#     # sigma_x * sigma_x
#     index = np.arange(config_batch_size)
#     index_batch = np.repeat(index, [idx_array.shape[0]], axis=0)
#     e_part2 = scatter_sum(src=psi_ratio_list, index=torch.from_numpy(index_batch), dim=0)
#     assert e_part2.shape[0] == config.shape[0]
#
#     # sigma_y * sigma_y
#     mask = mask.reshape(-1,1)
#     e_part3 = scatter_sum(src=psi_ratio_list * mask, index=torch.from_numpy(index_batch), dim=0)
#     assert e_part3.shape[0] == config.shape[0]
#
#     return e_part1 + e_part2 + e_part3

def tensor_prod_graph_Heisenberg(data, J=-1, h=-0.5, n=10):
    def tensor_prod(idx, s, size=10, J=-1, h=-0.5):
        "Tensor product of `s` acting on indexes `idx`. Fills rest with Id."
        Id = np.array([[1, 0], [0, 1]])
        idx, s = np.array(idx), np.array(s)
        matrices = [Id if k not in idx else s for k in range(size)]
        prod = matrices[0]
        for k in range(1, size):
            prod = np.kron(prod, matrices[k])
        return prod

    sx = np.array([[0, 1], [1, 0]])
    sz = np.array([[1, 0], [0, -1]])
    sy = np.array([[0, -1j], [1j, 0]])

    edge_index = sort_edge_index(data.edge_index)[0]
    idx_list = []
    for i in range(edge_index.shape[1]):
        if edge_index[0, i] < edge_index[1, i]:
            idx_list.append(edge_index[:, i].numpy().tolist())

    H_1 = sum([tensor_prod(idx, sz, size=n) for idx in idx_list])
    H_2 = sum([tensor_prod(idx, sx, size=n) for idx in idx_list])
    H_3 = sum([tensor_prod(idx, sy, size=n) for idx in idx_list])
    H = (H_1 + H_2 + H_3)

    return H


def tensor_prod_graph_Heisenberg_sparse(data, J=-1, h=-0.5, n=10):
    def tensor_prod(idx, s, size=10, J=-1, h=-0.5):
        "Tensor product of `s` acting on indexes `idx`. Fills rest with Id."
        Id = identity(2)
        idx = np.array(idx)
        matrices = [Id if k not in idx else s for k in range(size)]
        prod = matrices[0]
        for k in range(1, size):
            prod = sparse.kron(prod, matrices[k])
        return prod

    sx = csr_matrix(np.array([[0, 1], [1, 0]]))
    sz = csr_matrix(np.array([[1, 0], [0, -1]]))
    sy = csr_matrix(np.array([[0, -1j], [1j, 0]]))

    edge_index = sort_edge_index(data.edge_index)[0]
    idx_list = []
    for i in range(edge_index.shape[1]):
        if edge_index[0, i] < edge_index[1, i]:
            idx_list.append(edge_index[:, i].numpy().tolist())

    H_1 = csr_matrix(np.zeros((2 ** n, 2 ** n)))
    for idx in idx_list:
        H_1 = H_1 + tensor_prod(idx, sz, size=n)

    H_2 = csr_matrix(np.zeros((2 ** n, 2 ** n)))
    for idx in idx_list:
        H_2 = H_2 + tensor_prod(idx, sx, size=n)

    H_3 = csr_matrix(np.zeros((2 ** n, 2 ** n)))
    for idx in idx_list:
        H_3 = H_3 + tensor_prod(idx, sy, size=n)

    H = (H_1 + H_2 + H_3)

    return H