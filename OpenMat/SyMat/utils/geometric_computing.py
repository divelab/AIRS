import torch
from torch_scatter import scatter
from torch_sparse import SparseTensor
from math import pi as PI


def xyz_to_d(edge_index, num_nodes, distance_vectors):
    """
    Compute the diatance, angle, and torsion from geometric information.
    Args:
        pos: Geometric information for every node in the graph.
        edgee_index: Edge index of the graph.
        number_nodes: Number of nodes in the graph.
        use_torsion: If set to :obj:`True`, will return distance, angle and torsion, otherwise only return distance and angle (also retrun some useful index). (default: :obj:`False`)
    """
    j, i = edge_index  # j->i

    # Calculate distances. # number of edges
    dist = distance_vectors.pow(2).sum(dim=-1).sqrt()

    value = torch.arange(j.size(0), device=j.device)
    adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[j]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = i.repeat_interleave(num_triplets)
    idx_j = j.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    
    # Edge indices (k->j, j->i) for triplets.
    idx_ji = adj_t_row.storage.row()
    idx_kj = adj_t_row.storage.value()
    
    pos_ji = distance_vectors[idx_ji]
    pos_jk = -distance_vectors[idx_kj]
    mask = ~((pos_ji == pos_jk).sum(dim=-1) == 3)
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
    idx_ji, idx_kj = idx_ji[mask], idx_kj[mask]

    return dist, i, j, idx_kj, idx_ji


def xyz_to_dat(edge_index, num_nodes, num_edges, distance_vectors, use_torsion = False):
    """
    Compute the diatance, angle, and torsion from geometric information.
    Args:
        pos: Geometric information for every node in the graph.
        edgee_index: Edge index of the graph.
        number_nodes: Number of nodes in the graph.
        use_torsion: If set to :obj:`True`, will return distance, angle and torsion, otherwise only return distance and angle (also retrun some useful index). (default: :obj:`False`)
    """
    j, i = edge_index  # j->i

    # Calculate distances. # number of edges
    dist = distance_vectors.pow(2).sum(dim=-1).sqrt()

    value = torch.arange(j.size(0), device=j.device)
    adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[j]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    edge2graph = torch.arange(len(num_edges), device=num_edges.device)
    edge2graph = edge2graph.repeat_interleave(num_edges, dim=0)
    num_triplets_per_graph = scatter(num_triplets, edge2graph, dim=0, reduce='sum')

    # Node indices (k->j->i) for triplets.
    idx_i = i.repeat_interleave(num_triplets)
    idx_j = j.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    
    # Edge indices (k->j, j->i) for triplets.
    idx_ji = adj_t_row.storage.row()
    idx_kj = adj_t_row.storage.value()
    same_edge_diff = (num_edges // 2).repeat_interleave(num_triplets_per_graph, dim=0)
    mask = (idx_ji - idx_kj).abs() != same_edge_diff
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
    idx_ji, idx_kj = idx_ji[mask], idx_kj[mask]

    # Calculate angles. 0 to pi
    pos_ji = distance_vectors[idx_ji]
    pos_jk = -distance_vectors[idx_kj]
    a = (pos_ji * pos_jk).sum(dim=-1) # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(pos_ji, pos_jk).norm(dim=-1) # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)

    # Calculate torsions.
    if use_torsion:
        # k_n->j, k->j, j->i        
        idx_batch = torch.arange(len(idx_i),device=j.device)
        adj_t_row_t = adj_t[idx_j]
        idx_k_n = adj_t_row_t.storage.col()
        idx_k_n_j_t = adj_t_row_t.storage.value()
        repeat = num_triplets
        num_triplets_t = num_triplets.repeat_interleave(repeat)[mask]
        idx_i_t = idx_i.repeat_interleave(num_triplets_t)
        idx_j_t = idx_j.repeat_interleave(num_triplets_t)
        idx_k_t = idx_k.repeat_interleave(num_triplets_t)
        idx_ji_t = idx_ji.repeat_interleave(num_triplets_t)
        idx_kj_t = idx_kj.repeat_interleave(num_triplets_t)
        idx_batch_t = idx_batch.repeat_interleave(num_triplets_t)
        same_edge_diff_t = same_edge_diff[mask]
        same_edge_diff_t = same_edge_diff_t.repeat_interleave(num_triplets_t)
        mask = (idx_ji_t - idx_k_n_j_t).abs() != same_edge_diff_t       
        idx_i_t, idx_j_t, idx_k_t, idx_k_n, idx_batch_t = idx_i_t[mask], idx_j_t[mask], idx_k_t[mask], idx_k_n[mask], idx_batch_t[mask]
        idx_ji_t, idx_kj_t, idx_k_n_j_t = idx_ji_t[mask], idx_kj_t[mask], idx_k_n_j_t[mask]

        pos_jk = -distance_vectors[idx_kj_t]
        pos_ji = distance_vectors[idx_ji_t]
        pos_j_k_n = -distance_vectors[idx_k_n_j_t]
        # dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(pos_ji, pos_jk)
        plane2 = torch.cross(pos_ji, pos_j_k_n)
        a = (plane1 * plane2).sum(dim=-1) # cos_angle * |plane1| * |plane2|
        b = torch.cross(plane1, plane2).norm(dim=-1) # sin_angle * |plane1| * |plane2|
        torsion1 = torch.atan2(b, a) # -pi to pi
        torsion1[torsion1<=0] += 2*PI # 0 to 2pi
        torsion = scatter(torsion1, idx_batch_t, reduce='min')

        return dist, angle, torsion, i, j, idx_kj, idx_ji
    
    else:
        return dist, angle, i, j, idx_kj, idx_ji