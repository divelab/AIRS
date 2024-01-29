import torch
import torch.nn.functional as F
from torch_geometric.nn.acts import swish
from torch_scatter import scatter
from .modules import emb, init, update_v, update_e, update_u
import sys
sys.path.append("..")
from utils import xyz_to_dat
from utils import get_pbc_distances


class SphereNetEncoder(torch.nn.Module):
    r"""
         The spherical message passing neural network SphereNet from the `"Spherical Message Passing for 3D Graph Networks" <https://arxiv.org/abs/2102.05013>`_ paper.
        
        Args:
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`5.0`)
            num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            out_channels (int, optional): Size of each output sample. (default: :obj:`1`)
            int_emb_size (int, optional): Embedding size used for interaction triplets. (default: :obj:`64`)
            basis_emb_size_dist (int, optional): Embedding size used in the basis transformation of distance. (default: :obj:`8`)
            basis_emb_size_angle (int, optional): Embedding size used in the basis transformation of angle. (default: :obj:`8`)
            basis_emb_size_torsion (int, optional): Embedding size used in the basis transformation of torsion. (default: :obj:`8`)
            out_emb_channels (int, optional): Embedding size used for atoms in the output block. (default: :obj:`256`)
            num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`7`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`6`)
            envelop_exponent (int, optional): Shape of the smooth cutoff. (default: :obj:`5`)
            num_before_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`1`)
            num_after_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`2`)
            num_output_layers (int, optional): Number of linear layers for the output blocks. (default: :obj:`3`)
            act: (function, optional): The activation funtion. (default: :obj:`swish`)
            output_init: (str, optional): The initialization fot the output. It could be :obj:`GlorotOrthogonal` and :obj:`zeros`. (default: :obj:`GlorotOrthogonal`)
            
    """
    def __init__(
        self, cutoff=5.0, num_layers=4,
        hidden_channels=128, out_channels=1, int_emb_size=64,
        basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
        num_spherical=7, num_radial=6, envelope_exponent=5,
        num_before_skip=1, num_after_skip=2, num_output_layers=3,
        act=swish, output_init='GlorotOrthogonal', use_node_features=True):
        super(SphereNetEncoder, self).__init__()

        self.cutoff = cutoff

        self.init_e = init(num_radial, hidden_channels, act, use_node_features=use_node_features)
        self.init_v = update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init)
        self.init_u = update_u()
        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)

        self.update_vs = torch.nn.ModuleList([
            update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init) for _ in range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e(hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion, num_spherical, num_radial, num_before_skip, num_after_skip,act) for _ in range(num_layers)])

        self.update_us = torch.nn.ModuleList([update_u() for _ in range(num_layers)])

        self.reset_parameters()


    def reset_parameters(self):
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.emb.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()


    def forward(self, batch_data):
        z, edge_index, frac_coords, batch = batch_data.atom_types - 1, batch_data.edge_index, batch_data.frac_coords, batch_data.batch
        lattice_lengths, lattice_angles = batch_data.lengths, batch_data.angles
        to_jimages, num_atoms, num_bonds = batch_data.to_jimages, batch_data.num_atoms, batch_data.num_bonds
        
        _, distance_vectors, _ = get_pbc_distances(frac_coords, edge_index, lattice_lengths, lattice_angles, 
            to_jimages, num_atoms, num_bonds)
        num_nodes = z.shape[0]
        dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(edge_index, num_nodes, num_bonds, distance_vectors, use_torsion=True)

        emb = self.emb(dist, angle, torsion, idx_kj)

        #Initialize edge, node, graph features
        e = self.init_e(z, emb, i, j)
        v = self.init_v(e, i)
        u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch) #scatter(v, batch, dim=0)

        for update_e, update_v, update_u in zip(self.update_es, self.update_vs, self.update_us):
            e = update_e(e, emb, idx_kj, idx_ji)
            v = update_v(e, i)
            u = update_u(u, v, batch) #u += scatter(v, batch, dim=0)

        return u