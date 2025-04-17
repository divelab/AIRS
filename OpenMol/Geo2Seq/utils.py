import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from rdkit import Chem
import threading

def generate_random_graph(n, p):
    """Generate a random undirected graph with n vertices and edge probability p."""
    adj_matrix = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
    return adj_matrix


def adjacency_matrix_to_adjacency_list(adj_matrix):
    """Convert an adjacency matrix to an adjacency list."""
    adjacency_dict = {}
    for i in range(adj_matrix.shape[0]):
        adjacency_dict[i] = [j for j in range(adj_matrix.shape[1]) if adj_matrix[i, j]]
    return adjacency_dict


def relabel_undirected_graph(node_attr, edge_index, edge_attr):
    n = node_attr.shape[0]

    unique_node_attrs, node_colors = np.unique(node_attr, axis=0, return_inverse=True)
    unique_edge_attrs, edge_colors = np.unique(edge_attr, axis=0, return_inverse=True)
    edge_colors += n

    adj_matrix_size = n + len(edge_index)
    adj_matrix = np.zeros((adj_matrix_size, adj_matrix_size), dtype=np.bool_)

    for idx, (src, tgt) in enumerate(edge_index):
        edge_vertex = n + idx
        adj_matrix[src][edge_vertex] = True
        adj_matrix[edge_vertex][src] = True
        adj_matrix[tgt][edge_vertex] = True
        adj_matrix[edge_vertex][tgt] = True

    # Combine the node and edge colors
    weights = np.concatenate((node_colors, edge_colors))

    return adj_matrix, weights


def coordinates_to_distances(coords):
    n = coords.shape[0]
    sum_sq = np.sum(coords ** 2, axis=1)
    dist_sq = np.add.outer(sum_sq, sum_sq) - 2 * np.dot(coords, coords.T)
    dist_sq[dist_sq < 0] = 0
    distances = np.sqrt(dist_sq)

    iu = np.triu_indices(n, 1)
    edge_index = np.vstack(iu).T
    edge_distances = distances[iu]
    return edge_index, edge_distances


def expand_edges_and_attributes(n, edge_indices, edge_attr, fill_value=0):
    edge_index = np.vstack(np.triu_indices(n, 1)).T

    k = edge_attr.shape[1]
    new_edge_attr = np.full((len(edge_index), k), fill_value=fill_value)

    edge_to_index = {tuple(edge): idx for idx, edge in enumerate(edge_indices.T)}
    for idx, edge in enumerate(edge_index):
        edge_tuple = tuple(edge)
        reverse_edge_tuple = tuple(edge[::-1])

        if edge_tuple in edge_to_index:
            new_edge_attr[idx] = edge_attr[edge_to_index[edge_tuple]]
        elif reverse_edge_tuple in edge_to_index:
            new_edge_attr[idx] = edge_attr[edge_to_index[reverse_edge_tuple]]

    return edge_index, new_edge_attr


def create_lab_ptn_from_weights(weights):
    inds = np.arange(len(weights))

    indices = np.lexsort((inds, weights))
    sorted_weights = np.array(weights)[indices]

    ptn = np.ones_like(weights, dtype=np.int32)
    ptn[-1] = 0

    for i in range(len(sorted_weights) - 1):
        if sorted_weights[i] != sorted_weights[i + 1]:
            ptn[i] = 0

    return indices, ptn


def permutation_array_to_matrix(permutation_array):
    n = len(permutation_array)
    # Initialize an n x n matrix with zeros
    permutation_matrix = np.zeros((n, n), dtype=int)

    # Set the appropriate entries to 1 based on the permutation array
    for i, p in enumerate(permutation_array):
        permutation_matrix[i, p] = 1

    return permutation_matrix


def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None, prop = None, scaffold = None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence
    """
    block_size = model.get_block_size()   
    model.eval()

    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        logits, _, _ = model(x_cond, prop = prop, scaffold = scaffold)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        x = torch.cat((x, ix), dim=1)

    return x

