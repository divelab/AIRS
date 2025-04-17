import math
from typing import Optional, Tuple, Union
from collections import OrderedDict

from hienet.nn.linear import AtomReduce, FCN_e3nn, IrrepsLinear, eComfIrrepsLinear, DynamicIrrepsLinear

import torch
import hienet.util as util
import hienet._keys as KEY
from hienet._const import AtomGraphDataType
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from e3nn import o3
from e3nn.o3 import Irreps, TensorProduct, Linear

from hienet.nn.convolution import IrrepsConvolution
from e3nn.util.jit import compile_mode
import numpy as np
from e3nn.nn import FullyConnectedNet, BatchNorm
from hienet.nn.equivariant_gate import EquivariantGate, eComfEquivariantGate
import random


from hienet.nn.self_connection import (
    SelfConnectionIntro,
    SelfConnectionLinearIntro,
    SelfConnectionOutro,
)

import numpy as np
import torch

@compile_mode('script')
class ComformerNodeConvLayer(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        denominator: float = 1.0,
        concat: bool = True,
        beta: bool = False,
        dropout_mlp: float = 0.0,
        dropout_attn: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):

        kwargs.setdefault('aggr', 'add')
        super(ComformerNodeConvLayer, self).__init__(node_dim=0, **kwargs)
        self.denominator = nn.Parameter(
            torch.FloatTensor([denominator]), requires_grad=False
        )
    
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.edge_dim = edge_dim
        self._alpha = None


        self.cnt = 0

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.linear_key = nn.Linear(in_channels[0], heads * out_channels)
        self.linear_query = nn.Linear(in_channels[1], heads * out_channels)
        self.linear_value = nn.Linear(in_channels[0], heads * out_channels)
        self.linear_edge = nn.Linear(edge_dim, heads * out_channels)
        self.linear_concate = nn.Linear(heads * out_channels, out_channels)
        
        self.msg_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                        nn.SiLU(),
                                        nn.Linear(out_channels, out_channels))
        self.softplus = nn.Softplus()
        self.silu = nn.SiLU()
        self.key_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                        nn.SiLU(),
                                        nn.Linear(out_channels, out_channels))
        self.bn = nn.LayerNorm(out_channels)
        self.bn_att = nn.LayerNorm(out_channels)
        self.sigmoid = nn.Sigmoid()

        self.linear_reset = nn.Linear(in_channels[1], out_channels)
        self.linear_update = nn.Linear(in_channels[1], out_channels)

        self.dropout_mlp = nn.Dropout(p=dropout_mlp)
        self.dropout_attn = nn.Dropout(p=dropout_attn)
        
    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[KEY.NODE_FEATURE]

        edge_index = data[KEY.EDGE_IDX]
        edge_attr = data[KEY.EDGE_EMBEDDING]

        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        
        query = self.linear_query(x[1]).view(-1, H, C)
        key = self.linear_key(x[0]).view(-1, H, C)
        value = self.linear_value(x[0]).view(-1, H, C)

        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)
        
        out = out.view(-1, self.heads * self.out_channels)
        out = self.linear_concate(out)

        # previous implementation:
        #x = self.softplus(x[1] + self.bn(out))


        # gated implementation:
        reset_gate = self.sigmoid(self.linear_reset(x[1]))
        update_gate = self.sigmoid(self.linear_update(x[1]))


        x_reset = reset_gate * x[1]
        
        x = (1 - update_gate) * x[1] + update_gate * self.softplus(self.bn(out))

        x = (1 - update_gate) * x_reset + update_gate * x        

        data[KEY.NODE_FEATURE] = x
        
        return data

    def message(self, query_i: Tensor, key_i: Tensor, key_j: Tensor, value_j: Tensor, value_i: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        edge_attr = self.linear_edge(edge_attr).view(-1, self.heads, self.out_channels)
        edge_attr =  self.dropout_mlp(edge_attr)
        key_j = self.key_update(torch.cat((key_i, key_j, edge_attr), dim=-1))
        alpha = (query_i * key_j) / math.sqrt(self.out_channels)
        alpha = self.dropout_attn(alpha)
        out = self.msg_update(torch.cat((value_i, value_j, edge_attr), dim=-1))
        out = out * self.sigmoid(self.bn_att(alpha.view(-1, self.out_channels)).view(-1, self.heads, self.out_channels))      
        return out



@compile_mode('script')
class ComformerConvEdgeLayer(nn.Module):
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
        self.embedding_dim = 32
        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_key_e1 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_value_e1 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_key_e2 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_value_e2 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_key_e3 = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_value_e3 = nn.Linear(in_channels[0], heads * out_channels)

        self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_concate = nn.Linear(heads * out_channels, out_channels)
        self.lin_msg_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                            nn.SiLU(),
                                            nn.Linear(out_channels, out_channels))

        # Gating mechanism
        # self.gate_x = nn.Linear(out_channels, out_channels)
        # self.gate_y = nn.Linear(out_channels, out_channels)
        # self.gate_xy = nn.Linear(out_channels, out_channels)

        self.silu = nn.SiLU()
        self.softplus = nn.Softplus()
        self.key_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                        nn.SiLU(),
                                        nn.Linear(out_channels, out_channels))
        self.bn_att = nn.LayerNorm(out_channels)
        
        self.bn = nn.LayerNorm(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:   
        
        edge = data[KEY.EDGE_EMBEDDING]
        edge_nei_len = data[KEY.LATTICE_EMBEDDING]
        edge_nei_angle = data[KEY.ANGLE_EMBEDDING]

        H, C = self.heads, self.out_channels
        if isinstance(edge, Tensor):
            edge: PairTensor = (edge, edge)
        query_x = self.lin_query(edge[1]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        key_x = self.lin_key(edge[0]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        value_x = self.lin_value(edge[0]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
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

        # Gating mechanism
        # gate_x = self.sigmoid(self.gate_x(self.bn(value_x)))
        # gate_y = self.sigmoid(self.gate_y(self.bn(value_y)))
        # gate_xy = self.sigmoid(self.gate_xy(self.bn(edge_xy)))

        # gated_value_x = gate_x * value_x
        # gated_value_y = gate_y * value_y
        # gated_edge_xy = gate_xy * edge_xy

        # out = torch.cat((gated_value_x, gated_value_y, gated_edge_xy), dim=-1)
        # out = self.lin_msg_update(out)

        out = self.lin_msg_update(torch.cat((value_x, value_y, edge_xy), dim=-1))

        out = out * self.sigmoid(self.bn_att(alpha.view(-1, self.out_channels)).view(-1, 3, self.heads, self.out_channels))
        out = out.view(-1, 3, self.heads * self.out_channels)
        out = self.lin_concate(out)
        # aggregate the msg
        out = out.sum(dim=1)
        
        data[KEY.EDGE_EMBEDDING] = self.softplus(edge[1] + self.bn(out))

        return data


@compile_mode('script')
class eComfEquivariantConvLayer(nn.Module):

    def __init__(
        self,
        node_features_in: Union[int, Tuple[int, int]],
        node_features_out: Union[int, Tuple[int, int]],
        edge_dim: Optional[int] = None,
        lmax: int = 2,
        parity_mode: str = 'full',
        sh = '1x0e + 1x1e + 1x2e',
        act_gate = None,
        act_scalar = None,
        dropout: float = 0.0,
        denominator: float = 1.0,
        weight_layer_input_to_hidden = [12],
        weight_layer_act = 'relu',
        use_bias_in_linear: bool = False,
    ):

        super().__init__()
        self.node_features_in = node_features_in
        self.node_features_out = node_features_out
        self.lmax = lmax
        self.parity_mode = parity_mode
        self.sh = sh

        node_features_in = Irreps(node_features_in) 
        node_features_out = Irreps(node_features_out)  

        tp_irreps_out = util.infer_irreps_out(
            node_features_in,  # node feature irreps
            self.sh,  # filter irreps
            drop_l=self.lmax,
            parity_mode=self.parity_mode,
        )   
        
        self.gate = eComfEquivariantGate(node_features_out, act_scalar, act_gate)
        irreps_for_gate_in = self.gate.get_gate_irreps_in()

        self.skip_linear = eComfIrrepsLinear(node_features_in, irreps_for_gate_in) 
        self.node_linear =  eComfIrrepsLinear(node_features_in, node_features_in, biases = use_bias_in_linear) 
        
       
        self.convolution = TensorProductConvLayer(
            in_irreps=node_features_in,
            sh_irreps=self.sh,
            out_irreps=tp_irreps_out,
            n_edge_features=edge_dim,
            residual=False,
            denominator=denominator,
            weight_layer_input_to_hidden=weight_layer_input_to_hidden,
            weight_layer_act=weight_layer_act,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.l0_indices = [i for i, l in enumerate(tp_irreps_out.ls) if l == 0]
        self.node_linear_2 = eComfIrrepsLinear(tp_irreps_out, irreps_for_gate_in, biases = use_bias_in_linear)

    def forward(self, data: AtomGraphDataType,
                edge_nei_len: OptTensor = None) -> AtomGraphDataType:

        edge_index = data[KEY.EDGE_IDX]
        edge_attr = data[KEY.EDGE_VEC]
        edge_feature = data[KEY.EDGE_EMBEDDING]
        node_feature = data[KEY.NODE_FEATURE]

        skip_connect = self.skip_linear(node_feature)  
        
        node_feature = self.node_linear(node_feature)

        edge_irr = o3.spherical_harmonics(self.sh, edge_attr, normalize=True, normalization='component')

        tp = self.convolution(node_feature, edge_index, edge_feature, edge_irr)

        tp[:, self.l0_indices] = self.dropout(tp[:, self.l0_indices])
        
        node_feature = self.node_linear_2(tp)
        
        node_feature = node_feature + skip_connect

        node_feature = self.gate(node_feature)

        data[KEY.NODE_FEATURE] = node_feature

        return data

class TensorProductConvLayer(torch.nn.Module):
    # from Torsional diffusion
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True,  denominator=1.0, weight_layer_input_to_hidden=[12],weight_layer_act="relu"):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual

        # eComformer's tp:
        #self.tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        irreps_x = Irreps(in_irreps)
        irreps_filter = Irreps(sh_irreps)
        irreps_out = Irreps(out_irreps)

        #### Tensor product following sevenNet ####
        instructions = []
        irreps_mid = []
        for i, (mul_x, ir_x) in enumerate(irreps_x):
            for j, (_, ir_filter) in enumerate(irreps_filter):
                for ir_out in ir_x * ir_filter:
                    if ir_out in irreps_out:  # here we drop l > lmax
                        k = len(irreps_mid)
                        irreps_mid.append((mul_x, ir_out))
                        instructions.append((i, j, k, 'uvu', True))

        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]
        self.convolution = TensorProduct(
            irreps_x,
            irreps_filter,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # eComformer equivariant fc:
        # self.fc = nn.Sequential(
        #     nn.Linear(n_edge_features, n_edge_features),
        #     nn.Softplus(),
        #     nn.Linear(n_edge_features, self.convolution.weight_numel)
        # )

        self.fc = FullyConnectedNet(
            weight_layer_input_to_hidden + [self.convolution.weight_numel],
            weight_layer_act,
        )

        self.denominator = denominator

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):

        #edge_src = edge_index[1]
        #edge_dst = edge_index[0]
        edge_src, edge_dst = edge_index
        #tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))
        tp = self.convolution(node_attr[edge_dst], edge_sh, self.fc(edge_attr))
        
        out_nodes = out_nodes or node_attr.shape[0]

        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes)# reduce=reduce)
        out = out.div(self.denominator)


        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded
        return out
    
    # def forward(self, data : AtomGraphDataType) -> AtomGraphDataType:
    #     edge_index = data[KEY.EDGE_IDX]
    #     edge_attr = data[KEY.EDGE_VEC]
    #     edge_feature = data[KEY.EDGE_EMBEDDING]
    #     node_feature = data[KEY.NODE_FEATURE]


    #     edge_irr = o3.spherical_harmonics("1x0e + 1x1e + 1x2e", edge_attr, normalize=True, normalization='component')

    #     data[KEY.EDGE_ATTR] = edge_irr

    #     out = self._forward(node_feature, edge_index, edge_feature, edge_irr)
    #     data[KEY.NODE_FEATURE] = out
    #     return data
