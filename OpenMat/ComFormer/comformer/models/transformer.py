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
from comforemr.models.utils import softmax
from torch_scatter import scatter


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
        super(MatformerConv, self).__init__(node_dim=0, **kwargs)

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
        print('I am using the correct version of matformer')

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
        
        return self.softplus(x[1] + self.bn(out))

    def message(self, query_i: Tensor, key_i: Tensor, key_j: Tensor, value_j: Tensor, value_i: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        key_j = self.key_update(torch.cat((key_i, key_j, edge_attr), dim=-1))
        alpha = (query_i * key_j) / math.sqrt(self.out_channels)
        out = self.lin_msg_update(torch.cat((value_i, value_j, edge_attr), dim=-1))
        out = out * self.sigmoid(self.bn_att(alpha.view(-1, self.out_channels)).view(-1, self.heads, self.out_channels))
        return out


class ComformerConv_edge(nn.Module):
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
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.lemb = nn.Embedding(num_embeddings=3, embedding_dim=32)
        self.embedding_dim = 32
        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)
        # for test
        self.lin_key_e1 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_value_e1 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_key_e2 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_value_e2 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_key_e3 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_value_e3 = nn.Linear(in_channels[0], heads * out_channels)
        # for test ends
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_edge_len = nn.Linear(in_channels[0] + self.embedding_dim, in_channels[0])
        self.lin_concate = nn.Linear(heads * out_channels, out_channels)
        self.lin_msg_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                        nn.SiLU(),
                                        nn.Linear(out_channels, out_channels))
        self.silu = nn.SiLU()
        self.softplus = nn.Softplus()
        self.key_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                        nn.SiLU(),
                                        nn.Linear(out_channels, out_channels))
        self.bn_att = nn.BatchNorm1d(out_channels)
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.sigmoid = nn.Sigmoid()
        print('I am using the invariant version of EPCNet')

    def forward(self, edge: Union[Tensor, PairTensor], edge_nei_len: OptTensor = None, edge_nei_angle: OptTensor = None):
        # preprocess for edge of shape [num_edges, hidden_dim]

        H, C = self.heads, self.out_channels
        if isinstance(edge, Tensor):
            edge: PairTensor = (edge, edge)
        device = edge[1].device
        query_x = self.lin_query(edge[1]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        key_x = self.lin_key(edge[0]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        value_x = self.lin_value(edge[0]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        num_edge = query_x.shape[0]
        
        key_y = torch.cat((self.lin_key_e1(edge_nei_len[:,0,:]).view(-1, 1, H, C),
                            self.lin_key_e2(edge_nei_len[:,1,:]).view(-1, 1, H, C),
                            self.lin_key_e3(edge_nei_len[:,2,:]).view(-1, 1, H, C)), dim=1)
        value_y = torch.cat((self.lin_value_e1(edge_nei_len[:,0,:]).view(-1, 1, H, C),
                            self.lin_value_e2(edge_nei_len[:,1,:]).view(-1, 1, H, C),
                            self.lin_value_e3(edge_nei_len[:,2,:]).view(-1, 1, H, C)), dim=1)

        # preprocess for interaction of shape [num_edges, 3, hidden_dim]
        edge_xy = self.lin_edge(edge_nei_angle).view(-1, 3, H, C)

        key = self.key_update(torch.cat((key_x, key_y, edge_xy), dim=-1))
        alpha = (query_x * key) / math.sqrt(self.out_channels)
        out = self.lin_msg_update(torch.cat((value_x, value_y, edge_xy), dim=-1))
        out = out * self.sigmoid(self.bn_att(alpha.view(-1, self.out_channels)).view(-1, 3, self.heads, self.out_channels))

        out = out.view(-1, 3, self.heads * self.out_channels)
        out = self.lin_concate(out)
        # aggregate the msg
        out = out.sum(dim=1)

        return self.softplus(edge[1] + self.bn(out))



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