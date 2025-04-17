# More ordering studies
import heapq
import pickle
import sys
from collections import deque
from multiprocessing import Pool
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, depth_first_order, reverse_cuthill_mckee
from sklearn.utils import shuffle
import torch
import os, json
import os.path as osp
from itertools import repeat
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from torch_cluster import radius_graph
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from tqdm import tqdm
import periodictable
from space_filling_curve_sort import *


def rbf(D, max, num_rbf):
    D_min, D_max, D_count = 0., max, num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count)
    D_mu = D_mu.view([1, 1, 1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


class QM9GenDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 name,
                 split,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 processed_filename='data.pt',
                 sample=False,
                 remove_h=False,
                 no_feature=True,
                 no_property=True
                 ):

        self.processed_filename = processed_filename
        self.root = root
        self.name = f"{name}{'_no_feature' if no_feature else '_with_feature'}{'_no_property' if no_property else '_with_property'}{'_no_h' if remove_h else '_with_h'}{'_sample' if sample else '_all'}"
        self.split = split
        self.sample = sample
        self.remove_h = remove_h
        self.no_feature = no_feature
        self.no_property = no_property

        super(QM9GenDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if osp.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process()

    @property
    def raw_dir(self):
        return osp.join(self.root)

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, self.name, name)

    @property
    def processed_file_names(self):
        return self.processed_filename

    def process(self):
        r"""Processes the dataset from raw data file to the :obj:`self.processed_dir` folder.
        """
        self.data, self.slices = self.pre_process()

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        print('making processed files:', self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get(self, idx):
        r"""Gets the data object at index :idx:.

        Args:
            idx: The index of the data that you want to reach.
        :rtype: A data object corresponding to the input index :obj:`idx` .
        """
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        return data

    def pre_process(self):
        data_list = []

        raw_file = osp.join(self.root, 'processed_edm', self.split + '.npz')
        all_data = np.load(raw_file)
        keys = all_data.files

        ############### all files ################
        # 'num_atoms', 'charges', 'positions', 
        # 'index', 
        # properties: 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 
        # properties: 'U0', 'U', 'H', 'G', 'Cv', 'omega1', 'zpve_thermo', 'U0_thermo', 'U_thermo', 'H_thermo', 'G_thermo', 'Cv_thermo'
        # need to change units: 
        # qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 
        # 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}

        num_mol = len(all_data[keys[0]])

        all_num_atoms = all_data['num_atoms']
        all_charges = all_data['charges']
        all_positions = all_data['positions']
        for i in tqdm(range(num_mol)):
            data = Data()
            num_atom = all_num_atoms[i]
            z = torch.tensor(all_charges[i][:num_atom], dtype=torch.int64)
            xyz = torch.tensor(all_positions[i][:num_atom], dtype=torch.float32)
            if self.remove_h:
                mask = z != 1
                z = z[mask]
                xyz = xyz[mask]
                num_atom = len(z)
            if not self.no_feature:
                print('!!! include node and edge features, not supported !!!')
                exit()
            if not self.no_property:
                print('!!! include molecular properties, not supported !!!')
                exit()
            data.z = z
            data.xyz = xyz
            data.no = num_atom
            data_list.append(data)
            if self.sample:
                if len(data_list) > 100:
                    break

        data, slices = self.collate(data_list)
        return data, slices


from utils import relabel_undirected_graph, create_lab_ptn_from_weights, \
    coordinates_to_distances


def bfs_seq(edge_index, start_node):
    num_nodes = edge_index.max().item() + 1  # Assuming nodes are 0-indexed and contiguous
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    order = []

    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        if not visited[current_node]:
            visited[current_node] = True
            order.append(current_node)

            # Find neighbors of the current node
            neighbors = edge_index[1][edge_index[0] == current_node]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    queue.append(neighbor)

    return torch.tensor(order, dtype=torch.long).numpy()


def bfs_seq_with_weights(edge_index, edge_weight, start_node):
    num_nodes = edge_index.max().item() + 1  # Assuming nodes are 0-indexed and contiguous
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    order = []

    # Use a priority queue instead of a simple deque
    # The priority queue will store tuples of (weight, node)
    priority_queue = []

    # Start with the start node, with a priority (weight) of 0
    heapq.heappush(priority_queue, (0, start_node))
    while priority_queue:
        # Pop the node with the smallest weight
        current_weight, current_node = heapq.heappop(priority_queue)

        if not visited[current_node]:
            visited[current_node] = True
            order.append(current_node)

            # Find neighbors and corresponding weights of the current node
            neighbors = edge_index[1][edge_index[0] == current_node]
            weights = edge_weight[edge_index[0] == current_node]

            for neighbor, weight in zip(neighbors, weights):
                if not visited[neighbor]:
                    # Add the neighbor with its weight to the priority queue
                    heapq.heappush(priority_queue, (weight.item(), neighbor.item()))

    return torch.tensor(order, dtype=torch.long).numpy()


def dijkstra_ordering(edge_index, edge_weight, start_node):
    num_nodes = edge_index.max().item() + 1
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    distances = torch.full((num_nodes,), float('inf'), dtype=torch.float)
    distances[start_node] = 0
    order = []

    # Min-heap priority queue. Each entry is (distance, node).
    priority_queue = [(0, start_node)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if visited[current_node]:
            continue

        visited[current_node] = True
        order.append(current_node)

        # Find neighbors of the current node
        mask = edge_index[0] == current_node
        neighbors = edge_index[1][mask]
        weights = edge_weight[mask]

        for neighbor, weight in zip(neighbors, weights):
            if not visited[neighbor]:
                new_distance = current_distance + weight.item()
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    heapq.heappush(priority_queue, (new_distance, neighbor))

    return torch.tensor(order, dtype=torch.long).numpy()


def generate_dijkstra(graph_data):
    node_attr, edge_index, xyz = graph_data
    n = node_attr.shape[0]
    i, j = edge_index
    coords = torch.FloatTensor(xyz)
    edge_weight = torch.norm(coords[i] - coords[j], dim=-1)
    indices = dijkstra_ordering(edge_index, edge_weight, 0)
    syms = node_attr[indices]
    symbols = [periodictable.elements[an].symbol for an in syms]
    xyz = xyz[indices].tolist()
    labels = [[symbols[i]] + xyz[i] for i in range(n)]
    return [item for sublist in labels for item in sublist]


def generate_bfs_weights(graph_data):
    node_attr, edge_index, xyz = graph_data
    n = node_attr.shape[0]
    i, j = edge_index
    coords = torch.FloatTensor(xyz)
    edge_weight = torch.norm(coords[i] - coords[j], dim=-1)
    indices = bfs_seq_with_weights(edge_index, edge_weight, 0)
    syms = node_attr[indices]
    symbols = [periodictable.elements[an].symbol for an in syms]
    xyz = xyz[indices].tolist()
    labels = [[symbols[i]] + xyz[i] for i in range(n)]
    return [item for sublist in labels for item in sublist]


def generate_bfs(graph_data):
    node_attr, cs_graph, xyz = graph_data
    n = node_attr.shape[0]
    indices = breadth_first_order(cs_graph, n - 1)[0]
    syms = node_attr[indices]
    symbols = [periodictable.elements[an].symbol for an in syms]
    xyz = xyz[indices].tolist()
    labels = [[symbols[i]] + xyz[i] for i in range(n)]
    return [item for sublist in labels for item in sublist]


def generate_dfs(graph_data):
    node_attr, cs_graph, xyz = graph_data
    n = node_attr.shape[0]
    indices = depth_first_order(cs_graph, n - 1)[0]
    syms = node_attr[indices]
    symbols = [periodictable.elements[an].symbol for an in syms]
    xyz = xyz[indices].tolist()
    labels = [[symbols[i]] + xyz[i] for i in range(n)]
    return [item for sublist in labels for item in sublist]


def generate_cuthill_mckee(graph_data):
    node_attr, cs_graph, xyz = graph_data
    n = node_attr.shape[0]
    cs_graph_csr = csr_matrix(cs_graph)
    indices = reverse_cuthill_mckee(cs_graph_csr)
    syms = node_attr[indices]
    symbols = [periodictable.elements[an].symbol for an in syms]
    xyz = xyz[indices].tolist()
    labels = [[symbols[i]] + xyz[i] for i in range(n)]
    return [item for sublist in labels for item in sublist]


def generate_so(graph_data):
    node_attr, edge_index, xyz = graph_data
    n = node_attr.shape[0]
    laplacian = to_scipy_sparse_matrix(edge_index, num_nodes=n)
    laplacian = scipy.sparse.csgraph.laplacian(laplacian, normed=True)
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(laplacian, k=2, which='SM', v0=np.ones(n))
    fiedler_vector = eigenvectors[:, 1]
    indices = torch.argsort(torch.from_numpy(fiedler_vector))
    syms = node_attr[indices]
    symbols = [periodictable.elements[an].symbol for an in syms]
    xyz = xyz[indices].tolist()
    labels = [[symbols[i]] + xyz[i] for i in range(n)]
    return [item for sublist in labels for item in sublist]


def generate_morton(graph_data):
    node_attr, xyz = graph_data
    xyz_normal = normalize_points(xyz.tolist(), 10)
    _, indices = sort_points_z_order(xyz_normal, dtype='morton')
    n = node_attr.shape[0]
    syms = node_attr[indices]
    symbols = [periodictable.elements[an].symbol for an in syms]
    xyz = xyz[indices].tolist()
    # xyz = [[round(c, 2) for c in coords] for coords in xyz]
    labels = [[symbols[i]] + xyz[i] for i in range(n)]
    return [item for sublist in labels for item in sublist]


def generate_hilbert(graph_data):
    node_attr, xyz = graph_data
    xyz_normal = normalize_points(xyz.tolist(), 4)
    _, indices = sort_points_z_order(xyz_normal, dtype='hilbert')
    n = node_attr.shape[0]
    syms = node_attr[indices]
    symbols = [periodictable.elements[an].symbol for an in syms]
    xyz = xyz[indices].tolist()
    # xyz = [[round(c, 2) for c in coords] for coords in xyz]
    labels = [[symbols[i]] + xyz[i] for i in range(n)]
    return [item for sublist in labels for item in sublist]


def generate_nonlocal_canonical_labels(graph_data):
    node_attr, xyz = graph_data
    edge_index, edge_distances = coordinates_to_distances(xyz)

    edge_attr = edge_distances[:, np.newaxis]

    adj_matrix, weights = relabel_undirected_graph(node_attr[:, np.newaxis], edge_index, edge_attr)
    lab, ptn = create_lab_ptn_from_weights(weights)
    N_py = Nauty(adj_matrix.shape[0], adj_matrix, lab, ptn, defaultptn=False)
    perms = np.array(N_py.generate_full_group())
    label = np.array(N_py.canonlab)
    del N_py
    return node_attr[:, np.newaxis], edge_index, edge_attr, xyz, perms, label


def generate_tokens(graph_data):
    node_attr, edge_index, edge_attr, xyz, perms, label = graph_data
    n = node_attr.shape[0]
    canons = []
    labels = [perm[label] for perm in perms]
    for label in labels:
        canon = []
        node_perm = label[:n]
        edge_perm = label[n:] - n
        value_to_idx = {value: idx for idx, value in enumerate(edge_perm)}

        syms = node_attr[:, 0][node_perm]
        symbols = [periodictable.elements[an].symbol for an in syms]
        dist_token = edge_attr[:, 0][edge_perm].tolist()
        canon.append(symbols)
        canon.append(symbols + dist_token)
        third_labeling = []
        last_added_indices = set()
        for i in range(n):
            third_labeling.append(symbols[i])
            indices = np.where((edge_index[:, 0] == node_perm[i]) | (edge_index[:, 1] == node_perm[i]))[0]
            indices = set(indices)
            indices -= last_added_indices

            sorted_indices = sorted(indices, key=lambda x: value_to_idx[x])
            for idx in sorted_indices:
                third_labeling.append(dist_token[idx])
            last_added_indices = indices

        canon.append(third_labeling)
        canon.append([[symbols[i]] + xyz[node_perm][i].tolist() for i in range(n)])

        canons.append(canon)

    return canons


def nan_to_num(vec, num=0.0):
    idx = np.isnan(vec)
    vec[idx] = num
    return vec


def _normalize(vec, axis=-1):
    return nan_to_num(
        np.divide(vec, np.linalg.norm(vec, axis=axis, keepdims=True)))


def write_xyz_file(atom_types, atom_coordinates, file_path):
    with open(file_path, 'w') as file:
        num_atoms = len(atom_types)
        file.write(f"{num_atoms}\n")
        file.write('\n')

        for atom_type, coords in zip(atom_types, atom_coordinates):
            x, y, z = coords
            file.write(
                f"{atom_type} {np.format_float_positional(x)} {np.format_float_positional(y)} {np.format_float_positional(z)}\n")


def order_nodes_by_degree(edge_index):
    # Calculate degree of each node by counting occurrences in the edge index
    all_nodes = torch.cat((edge_index[0], edge_index[1]))  # Combine both rows to count all occurrences
    degrees = torch.bincount(all_nodes)  # Counts the frequency of each element in the tensor

    # Sort nodes by degree
    # torch.sort returns two tensors, one of the sorted values and one of the indices that would sort the tensor
    _, nodes_ordered_by_degree = degrees.sort(descending=True)

    return nodes_ordered_by_degree


dict_qm9 = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}

if __name__ == '__main__':
    # for split in ['train', 'valid', 'test']:
    split = sys.argv[1]
    method = sys.argv[2]
    dataset = QM9GenDataset(root='data/QM9Gen/',
                            name='data',
                            processed_filename=split + '.pt',
                            split=split,
                            sample=False,
                            remove_h=False,
                            no_feature=True,
                            no_property=True)

    file_path = 'Molecule3D/example.xyz'

    def process_bfs(index):
        data = dataset[index]
        edge_index = radius_graph(data.xyz, 5.0)
        cs_graph = to_scipy_sparse_matrix(edge_index)
        graph_data = (data.z.numpy(), cs_graph, data.xyz.numpy())
        labels = generate_bfs(graph_data)
        return labels


    def process_dfs(index):
        data = dataset[index]
        edge_index = radius_graph(data.xyz, 5.0)
        cs_graph = to_scipy_sparse_matrix(edge_index)
        graph_data = (data.z.numpy(), cs_graph, data.xyz.numpy())
        labels = generate_dfs(graph_data)
        return labels


    def process_weighted_bfs(index):
        data = dataset[index]
        edge_index = radius_graph(data.xyz, 5.0)
        graph_data = (data.z.numpy(), edge_index, data.xyz.numpy())
        labels = generate_bfs_weights(graph_data)
        return labels

    def process_nonlocal(index):
        data = dataset[index]
        graph_data = (data.z.numpy(), data.xyz.numpy())
        labels = generate_nonlocal_canonical_labels(graph_data)
        return labels


    def process_morton(index):
        data = dataset[index]
        graph_data = (data.z.numpy(), data.xyz.numpy())
        labels = generate_morton(graph_data)
        return labels


    def process_hilbert(index):
        data = dataset[index]
        graph_data = (data.z.numpy(), data.xyz.numpy())
        labels = generate_hilbert(graph_data)
        return labels


    def process_dijkstra(index):
        data = dataset[index]
        edge_index = radius_graph(data.xyz, 5.0)
        graph_data = (data.z.numpy(), edge_index, data.xyz.numpy())
        labels = generate_dijkstra(graph_data)
        return labels


    def process_cuthill_mckee(index):
        data = dataset[index]
        edge_index = radius_graph(data.xyz, 5.0)
        cs_graph = to_scipy_sparse_matrix(edge_index)
        graph_data = (data.z.numpy(), cs_graph, data.xyz.numpy())
        labels = generate_cuthill_mckee(graph_data)
        return labels


    def process_so(index):
        data = dataset[index]
        edge_index = radius_graph(data.xyz, 5.0)
        graph_data = (data.z.numpy(), edge_index, data.xyz.numpy())
        labels = generate_so(graph_data)
        return labels


    num_items = len(dataset)  # Replace with the actual number of items in the dataset
    if method == 'CLnl':
        with Pool(processes=64) as pool:
            labels = list(tqdm(pool.imap(process_nonlocal, range(num_items)), total=num_items))
        labelings = labels


        def process_graph_label(index):
            label = labelings[index]
            canons = generate_tokens(label)
            return canons


        num_items = len(labelings)  # Replace with the actual number of items in the dataset
        with Pool(processes=64) as pool:
            labels = list(tqdm(pool.imap(process_graph_label, range(num_items)), total=num_items))
    elif method == 'morton':
        with Pool(processes=64) as pool:
            labels = list(tqdm(pool.imap(process_morton, range(num_items)), total=num_items))
    elif method == 'hilbert':
        with Pool(processes=64) as pool:
            labels = list(tqdm(pool.imap(process_hilbert, range(num_items)), total=num_items))
    elif method == 'bfs':
        with Pool(processes=64) as pool:
            labels = list(tqdm(pool.imap(process_bfs, range(num_items)), total=num_items))
    elif method == 'weighted_bfs':
        with Pool(processes=64) as pool:
            labels = list(tqdm(pool.imap(process_weighted_bfs, range(num_items)), total=num_items))
    elif method == 'dfs':
        with Pool(processes=64) as pool:
            labels = list(tqdm(pool.imap(process_dfs, range(num_items)), total=num_items))
    elif method == 'so':
        with Pool(processes=64) as pool:
            labels = list(tqdm(pool.imap(process_so, range(num_items)), total=num_items))
    elif method == 'dijkstra':
        with Pool(processes=64) as pool:
            labels = list(tqdm(pool.imap(process_dijkstra, range(num_items)), total=num_items))
    elif method == 'cuthill_mckee':
        with Pool(processes=64) as pool:
            labels = list(tqdm(pool.imap(process_cuthill_mckee, range(num_items)), total=num_items))
    else:
        raise NotImplementedError("Unknown ordering method")

    with open(f'data/Molecule3D/QM9_{split}_{method}_labels.pkl', 'wb') as file:
        pickle.dump(labels, file)

    with open(f'Molecule3D/QM9_{split}_{method}_labels.txt', 'w') as file:
        # Iterate through each sublist in the list
        for sublist in labels:
            # Join each element in the sublist into a string separated by whitespace
            line = ' '.join(str(item) for item in sublist)
            # Write the line to the file, followed by a newline character
            file.write(line + '\n')
