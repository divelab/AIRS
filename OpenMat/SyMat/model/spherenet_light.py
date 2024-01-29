import torch
from math import pi as PI
from torch import nn
from torch.nn import Linear, Embedding
from torch_geometric.nn.acts import swish
from torch_geometric.nn.inits import glorot_orthogonal
from torch_scatter import scatter
from math import sqrt
import sys
sys.path.append("..")
from utils import xyz_to_d


class dist_emb(torch.nn.Module):
    def __init__(self, num_radial, cutoff=5.0):
        super(dist_emb, self).__init__()
        self.cutoff = cutoff

        self.freq = torch.nn.Parameter(torch.Tensor(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return (self.freq * dist).sin()


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class init(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish, use_node_features=True):
        super(init, self).__init__()
        self.act = act
        self.use_node_features = use_node_features
        if self.use_node_features:
            self.emb = Embedding(100, hidden_channels)
        else: # option to use no node features and a learned embedding vector for each node instead
            self.node_embedding = nn.Parameter(torch.empty((hidden_channels,)))
            nn.init.normal_(self.node_embedding)
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_node_features:
            self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, x, emb, i, j):
        rbf = emb
        if self.use_node_features:
            x = self.emb(x)
        else:
            x = self.node_embedding[None, :].expand(x.shape[0], -1)
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(torch.cat([x[i], x[j], rbf0], dim=-1)))
        e2 = self.lin_rbf_1(rbf) * e1

        return e1, e2


class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size_dist, num_radial,
        num_before_skip, num_after_skip, act=swish):
        super(update_e, self).__init__()
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(self, x, emb, idx_kj, idx_ji):
        rbf0 = emb
        x1,_ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))
        x_kj = x_kj[idx_kj]

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1

        return e1, e2


class SphereNetLightDecoder(torch.nn.Module):
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
        num_before_skip=1, num_after_skip=2,
        act=swish, use_node_features=True):
        super(SphereNetLightDecoder, self).__init__()

        self.cutoff = cutoff

        self.init_e = init(num_radial, hidden_channels, act, use_node_features=use_node_features)
        self.emb = dist_emb(num_radial, self.cutoff)

        self.update_es = torch.nn.ModuleList([
            update_e(hidden_channels, int_emb_size, basis_emb_size_dist, num_radial, num_before_skip, num_after_skip,act) for _ in range(num_layers)])

        self.reset_parameters()


    def reset_parameters(self):
        self.init_e.reset_parameters()
        self.emb.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()


    def forward(self, z, edge_index, distance_vectors):
        num_nodes = z.shape[0]
        dist, i, j, idx_kj, idx_ji = xyz_to_d(edge_index, num_nodes, distance_vectors)

        emb = self.emb(dist)

        #Initialize edge, node, graph features
        e = self.init_e(z, emb, i, j)

        for update_e in self.update_es:
            e = update_e(e, emb, idx_kj, idx_ji)

        return e[1]