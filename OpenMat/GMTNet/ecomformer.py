import torch
from torch import nn
from utils import RBFExpansion
from transformer import ComformerConv
from torch_scatter import scatter

import math
from e3nn import o3
from typing import Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor


class TensorProductConvLayer(torch.nn.Module):
    # from Torsional diffusion
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, n_edge_features),
            nn.Softplus(),
            nn.Linear(n_edge_features, tp.weight_numel)
        )

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):

        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)
        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        return out


class MatformerConvEqui(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        edge_dim: Optional[int] = None,
        use_second_order_repr: bool = True,
        ns: int = 64,
        nv: int = 8,
        residual: bool = True,
    ):
        super().__init__()

        irrep_seq = [
            f'{ns}x0e',
            f'{ns}x0e + {nv}x1o + {nv}x2e',
            f'{ns}x0e'
        ]
        self.ns, self.nv = ns, nv
        self.node_linear = nn.Linear(in_channels, ns)
        self.skip_linear = nn.Linear(in_channels, out_channels)
        self.sh = '1x0e + 1x1o + 1x2e'
        # self.sh = '1x0e + 1x1o' # ablation
        self.nlayer_1 = TensorProductConvLayer(
            in_irreps=irrep_seq[0],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[1],
            n_edge_features=edge_dim,
            residual=residual
        )
        self.nlayer_2 = TensorProductConvLayer(
            in_irreps=irrep_seq[1],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[2],
            n_edge_features=edge_dim,
            residual=False
        )
        self.softplus = nn.Softplus()
        self.bn = nn.BatchNorm1d(ns)
        self.node_linear_2 = nn.Linear(ns, out_channels)

    def forward(self, data, node_feature: Union[Tensor, PairTensor], edge_index: Adj, edge_feature: Union[Tensor, PairTensor], 
                    edge_nei_len: OptTensor = None):
        edge_vec = data.edge_attr
        edge_irr = o3.spherical_harmonics(self.sh, edge_vec, normalize=True, normalization='component')
        n_ = node_feature.shape[0]
        skip_connect = node_feature
        node_feature = self.node_linear(node_feature)
        node_feature = self.nlayer_1(node_feature, edge_index, edge_feature, edge_irr)
        node_feature = self.nlayer_2(node_feature, edge_index, edge_feature, edge_irr)
        node_feature = self.softplus(self.node_linear_2(self.softplus(self.bn(node_feature))))
        node_feature += self.skip_linear(skip_connect)

        return node_feature


def bond_cosine(r1, r2):
    bond_cosine = torch.sum(r1 * r2, dim=-1) / (
        torch.norm(r1, dim=-1) * torch.norm(r2, dim=-1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine

def equality_adjustment(equality, batch):
    """
    Adjust the second batch of matrices based on the equality of entries in the first batch.
    """
    b, l1, l2 = batch.size()
    batch = batch.reshape(b, l1 * l2)
    for i in range(b):
        mask = equality[i]
        for j in range(l1 * l2):
            for k in range(j + 1, l1 * l2):
                if mask[j, k]:
                    # Average the entries in the second batch
                    batch[i, j] = batch[i, k] = (batch[i, j] + batch[i, k]) / 2
    return batch.reshape(b, l1, l2)

class EComformerEquivariant(nn.Module):
    def __init__(self, args):
        super().__init__()
        embsize = 128
        self.atom_embedding = nn.Linear(
            92, embsize
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=-4.0,
                vmax=0.0,
                bins=512,
            ),
            nn.Linear(512, embsize),
            nn.Softplus(),
        )

        self.att_layers = nn.ModuleList(
            [
                ComformerConv(in_channels=embsize, out_channels=embsize, heads=1, edge_dim=embsize)
                for _ in range(2)
            ]
        )

        self.equi_update = MatformerConvEqui(in_channels=embsize, out_channels=embsize, edge_dim=embsize)
        
        self.output_ln = nn.Linear(embsize, 9)
        

    def forward(self, data) -> torch.Tensor:
        node_features = self.atom_embedding(data.x)
        edge_feat = -0.75 / torch.norm(data.edge_attr, dim=1)
        # edge_feat = torch.norm(data.edge_attr, dim=1)
        edge_features = self.rbf(edge_feat)

        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        node_features = self.equi_update(data, node_features, data.edge_index, edge_features)
        node_features = self.output_ln(node_features)
        crystal_features = scatter(node_features, data.batch, dim=0, reduce="mean")
        outputs = crystal_features.view(-1, 3, 3) # ablation for no equivariance.

        return outputs