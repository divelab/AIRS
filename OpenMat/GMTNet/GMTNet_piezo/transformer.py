import math
from e3nn import o3
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_scatter import scatter
from torch.autograd import grad
from e3nn.io import CartesianTensor

class ComformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(ComformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels)
        self.lin_concate = nn.Linear(heads * out_channels, out_channels)
        
        self.lin_msg_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                        nn.SiLU(),
                                        nn.Linear(out_channels, out_channels))
        self.softplus = nn.Softplus()
        self.silu = nn.SiLU()
        self.key_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                        nn.SiLU(),
                                        nn.Linear(out_channels, out_channels))
        self.bn = nn.BatchNorm1d(out_channels)
        self.bn_att = nn.BatchNorm1d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        
        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)
        
        out = out.view(-1, self.heads * self.out_channels)
        out = self.lin_concate(out)
        
        return self.softplus(x[1] + out)

    def message(self, query_i: Tensor, key_i: Tensor, key_j: Tensor, value_j: Tensor, value_i: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        key_j = self.key_update(torch.cat((key_i, key_j, edge_attr), dim=-1))
        alpha = (query_i * key_j) / math.sqrt(self.out_channels)
        out = self.lin_msg_update(torch.cat((value_i, value_j, edge_attr), dim=-1))
        out = out * self.sigmoid(self.bn_att(alpha.view(-1, self.out_channels)).view(-1, self.heads, self.out_channels))
        return out



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


class ComformerConvEqui(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        edge_dim: Optional[int] = None,
        ns: int = 16,
        nv: int = 2,
        residual: bool = True,
    ):
        super().__init__()

        irrep_seq = [
            f'{ns}x0e',
            f'{ns}x0e + {nv}x1o + {nv}x2e',
            f'{ns}x0e + {nv}x1o + {nv}x1e + {nv}x2e + {nv}x2o',
            # f'{ns}x0e + {nv}x1o + {nv}x1e + {nv}x2e + {nv}x2o + {nv}x3e + {nv}x3o',
            # f'{ns}x0e + {nv}x1o + {nv}x1e + {nv}x2e + {nv}x2o + {nv}x3e + {nv}x3o + {nv}x4e + {nv}x4o',
            '2x0e + 2x0o + 2x1e + 2x1o + 2x2e + 2x2o + 2x3e + 2x3o',
            # '9x0e', # ablation for no equivariance
        ]
        self.ns, self.nv = ns, nv
        self.node_linear = nn.Linear(in_channels, ns)
        self.sh = '1x0e + 1x1o + 1x2e'
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
        self.nlayer_3 = TensorProductConvLayer(
            in_irreps=irrep_seq[2],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[3],
            n_edge_features=edge_dim,
            residual=False
        )

    def forward(self, data, node_feature: Union[Tensor, PairTensor], edge_index: Adj, edge_feature: Union[Tensor, PairTensor]):
        edge_vec = data.edge_attr
        edge_irr = o3.spherical_harmonics(self.sh, edge_vec, normalize=True, normalization='component')
        n_ = node_feature.shape[0]
        skip_connect = node_feature
        node_feature = self.node_linear(node_feature)
        node_feature = self.nlayer_1(node_feature, edge_index, edge_feature, edge_irr)
        node_feature = self.nlayer_2(node_feature, edge_index, edge_feature, edge_irr)
        node_feature = self.nlayer_3(node_feature, edge_index, edge_feature, edge_irr)
        
        return node_feature
    

class Gradient_block(nn.Module):
    def __init__(
        self,
        nv: int = 2,
    ):
        super().__init__()

        irrep_seq = [
            '2x0e + 2x0o + 2x1e + 2x1o + 2x2e + 2x2o + 2x3e + 2x3o',
            f'1x1o',
        ]
        self.nv = nv
        self.sh = '1x1o'
        self.tp = tp = o3.FullyConnectedTensorProduct(irrep_seq[0], self.sh, irrep_seq[1], internal_weights=False)
        self.constant_w = nn.Parameter(torch.ones(tp.weight_numel), requires_grad=False)

    def forward(self, node_feature):
        bs = node_feature.shape[0]
        outer_E = torch.ones(bs, 3).to(node_feature.device)
        outer_E.requires_grad_(True)
        E_ = o3.spherical_harmonics(self.sh, outer_E, normalize=False)
        D_ = self.tp(node_feature, E_, self.constant_w.to(node_feature.device))
        dielectric = []
        for i in range(3):
            grad_outputs = torch.zeros(bs, 3).to(node_feature.device)
            grad_outputs[:, i] = 1.0
            dielectric.append(grad(D_, outer_E, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0])
        return torch.stack(dielectric).transpose(0, 1)


class Piezo_block(nn.Module):
    def __init__(
        self,
        nv: int = 2,
    ):
        super().__init__()

        irrep_seq = [
            '2x0e + 2x0o + 2x1e + 2x1o + 2x2e + 2x2o + 2x3e + 2x3o',
            f'1x1o',
        ]
        self.nv = nv
        self.stress = '1x0e + 1x2e'
        self.converter = CartesianTensor("ij=ji")
        self.tp = tp = o3.FullyConnectedTensorProduct(irrep_seq[0], self.stress, irrep_seq[1], internal_weights=False)
        self.constant_w = nn.Parameter(torch.ones(tp.weight_numel))
        self.idx = [0, 4, 8, 1, 5, 6]

    def forward(self, node_feature):
        bs = node_feature.shape[0]
        outer_S = torch.ones(bs, 3, 3).to(node_feature.device)
        outer_S.requires_grad_(True)
        stress = self.converter.from_cartesian(outer_S)
        D_ = self.tp(node_feature, stress, self.constant_w.to(node_feature.device))
        piezo = []
        for i in range(3):
            grad_outputs = torch.zeros(bs, 3).to(node_feature.device)
            grad_outputs[:, i] = 1.0
            piezo.append(grad(D_, outer_S, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0].reshape(bs, 9)[:, [0, 4, 8, 1, 5, 6]])
        return torch.stack(piezo).transpose(0, 1)


class Elastic_block(nn.Module):
    def __init__(
        self,
        nv: int = 2,
    ):
        super().__init__()

        irrep_seq = [
            '1x0e + 1x0o + 1x1e + 1x1o + 1x2e + 1x2o + 1x3e + 1x3o',
            f'1x0e + 1x1e + 1x2e',
        ]
        self.nv = nv
        self.strain = '1x0e + 1x1e + 1x2e'
        self.converter = CartesianTensor("ij")
        self.tp = tp = o3.FullyConnectedTensorProduct(irrep_seq[0], self.strain, irrep_seq[1], internal_weights=False)
        self.constant_w = nn.Parameter(torch.ones(tp.weight_numel))
        self.idx = [0, 4, 8, 1, 5, 6]

    def forward(self, node_feature):
        bs = node_feature.shape[0]
        outer_Strain = torch.ones(bs, 3, 3).to(node_feature.device)
        outer_Strain.requires_grad_(True)
        strain = self.converter.from_cartesian(outer_Strain)
        stress = self.tp(node_feature, strain, self.constant_w.to(node_feature.device))
        final_stress = self.converter.to_cartesian(stress).view(bs, -1)
        elastic = torch.zeros((bs, 6, 6)).to(node_feature[0].device)
        elastic.requires_grad_(True)
        for i in range(6):
            grad_outputs = torch.zeros(bs, 9).to(node_feature.device)
            grad_outputs[:, self.idx[i]] = 1.0
            elastic[:, i, :] += grad(final_stress, outer_Strain, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0].view(bs, -1)[self.idx]
        return elastic