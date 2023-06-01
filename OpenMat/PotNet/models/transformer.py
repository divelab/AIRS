from typing import Optional, Tuple, Union, Dict
import math
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.models.schnet import ShiftedSoftplus
from torch_sparse import SparseTensor
import torch.nn as nn
from torch_geometric.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.typing import Adj, OptTensor, PairTensor


class TransformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
            self,
            hidden_dim: int,
            edge_dim: int,
            bias: bool = True,
            root_weight: bool = True,
            aggr: str = 'mean',
            **kwargs,
    ):
        kwargs.setdefault('aggr', aggr)
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.hidden_dim = hidden_dim
        self.root_weight = root_weight
        self.edge_dim = edge_dim
        self._alpha = None

        self.lin_edge = Linear(edge_dim, hidden_dim)

        self.lin_concate = Linear(hidden_dim, hidden_dim)

        self.lin_skip = Linear(hidden_dim, hidden_dim, bias=bias)

        self.lin_msg_update = Linear(hidden_dim, hidden_dim)
        self.msg_layer = Linear(hidden_dim, hidden_dim)
        self.msg_ln = nn.LayerNorm(hidden_dim)

        self.alpha_ln = nn.LayerNorm(hidden_dim)

        self.bn = BatchNorm(hidden_dim)
        self.skip_bn = BatchNorm(hidden_dim)
        self.act = ShiftedSoftplus()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: Tensor, return_attention_weights=None):
        out = self.propagate(edge_index, query=x, key=x, value=x, edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        out = self.act(self.bn(out))

        if self.root_weight:
            out = out + x

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_i: Tensor, key_j: Tensor, value_j: Tensor, value_i: Tensor,
                edge_attr: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        edge_attr = self.lin_edge(edge_attr)
        alpha = (query_i * key_j) / math.sqrt(self.hidden_dim) + edge_attr
        alpha = F.sigmoid(self.act(self.alpha_ln(alpha)))
        self._alpha = alpha

        out = self.lin_msg_update(value_j) * alpha
        out = F.relu(self.msg_ln(self.msg_layer(out)))
        return out


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(, '
                f'{self.hidden_dim})')
