import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from utils import RBFExpansion
from torch_scatter import scatter
from torch.nn import Embedding
from typing import Union

from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn import Set2Set as set2set


class MEGConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        v_input_dim=16,
        e_input_dim=100,
        u_input_dim=2,
        n1 = 64,
        n2 = 32,
        concat: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'mean')
        super(MEGConv, self).__init__(node_dim=0, **kwargs)

        self.n1 = n1
        self.n2 = n2
        self.v_input_dim = v_input_dim
        self.e_input_dim = e_input_dim
        self.u_input_dim = u_input_dim
        self.concat = concat
        self.dropout = dropout

        # Linear for v
        self.linear_v1 = nn.Linear(v_input_dim + u_input_dim + n2, n1)
        self.linear_v2 = nn.Linear(n1, n1)
        self.linear_v3 = nn.Linear(n1, n2)

        # Linear for e
        self.linear_e1 = nn.Linear(2 * v_input_dim + e_input_dim + u_input_dim, n1)
        self.linear_e2 = nn.Linear(n1, n1)
        self.linear_e3 = nn.Linear(n1, n2)

        # Linear for u
        self.linear_u1 = nn.Linear(n2 + u_input_dim + n2, n1)
        self.linear_u2 = nn.Linear(n1, n1)
        self.linear_u3 = nn.Linear(n1, n2)
        
        self.softplus = nn.Softplus()
        self.reset_parameters()

    def reset_parameters(self):
        self.linear_v1.reset_parameters()
        self.linear_v2.reset_parameters()
        self.linear_v3.reset_parameters()

        self.linear_e1.reset_parameters()
        self.linear_e2.reset_parameters()
        self.linear_e3.reset_parameters()

        self.linear_u1.reset_parameters()
        self.linear_u2.reset_parameters()
        self.linear_u3.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, batch=None, edge_attr: OptTensor = None, u_rep=None ,return_attention_weights=None):
       
        v = x 
        e = edge_attr
        u = u_rep # batch * k
        # phi_e calculation
        e_p = torch.cat((v[edge_index[0]], v[edge_index[1]], u[batch][edge_index[0]], e), dim=-1)
        e_p = self.softplus(self.linear_e1(e_p))
        e_p = self.softplus(self.linear_e2(e_p))
        e_p = self.softplus(self.linear_e3(e_p))

        # v_p calculation
        edge_to_v = scatter(e_p, edge_index[1], dim=0, reduce="mean")
        v_p = torch.cat((v, edge_to_v, u[batch]), dim=-1)
        v_p = self.softplus(self.linear_v1(v_p))
        v_p = self.softplus(self.linear_v2(v_p))
        v_p = self.softplus(self.linear_v3(v_p))

        # u_p calculation
        ue = scatter(edge_to_v, batch, dim=0, reduce="mean")
        uv = scatter(v_p, batch, dim=0, reduce="mean")
        u_p = torch.cat((ue, uv, u), dim=-1)
        u_p = self.softplus(self.linear_u1(u_p))
        u_p = self.softplus(self.linear_u2(u_p))
        u_p = self.softplus(self.linear_u3(u_p))

        return v_p, e_p, u_p



class MEGNET(nn.Module):
    """megnet pyg implementation."""

    def __init__(self,larger=1):
        """Set up megnet modules."""
        super().__init__()
        # self.embedding = self.embedding = Embedding(100, 16*larger)
        self.embedding = nn.Linear(92, 16*larger)
        self.batchsize = 64
        self.rbf = RBFExpansion(
                        vmin=0,
                        vmax=5.0,
                        bins=100,
                        lengthscale=0.5,
                    )
        
        self.softplus = nn.Softplus()

        self.meglayer1 = MEGConv(v_input_dim=32*larger, e_input_dim=32*larger, u_input_dim=32*larger)
        self.meglayer2 = MEGConv(v_input_dim=32*larger, e_input_dim=32*larger, u_input_dim=32*larger)
        self.meglayer3 = MEGConv(v_input_dim=32*larger, e_input_dim=32*larger, u_input_dim=32*larger)
        self.ffv0 = nn.Sequential(nn.Linear(16*larger, 64*larger), nn.Softplus(), nn.Linear(64*larger, 32*larger), nn.Softplus())
        self.ffv1 = nn.Sequential(nn.Linear(32*larger, 64*larger), nn.Softplus(), nn.Linear(64*larger, 32*larger), nn.Softplus())
        self.ffv2 = nn.Sequential(nn.Linear(32*larger, 64*larger), nn.Softplus(), nn.Linear(64*larger, 32*larger), nn.Softplus())

        self.ffe0 = nn.Sequential(nn.Linear(100, 64*larger), nn.Softplus(), nn.Linear(64*larger, 32*larger), nn.Softplus())
        self.ffe1 = nn.Sequential(nn.Linear(32*larger, 64*larger), nn.Softplus(), nn.Linear(64*larger, 32*larger), nn.Softplus())
        self.ffe2 = nn.Sequential(nn.Linear(32*larger, 64*larger), nn.Softplus(), nn.Linear(64*larger, 32*larger), nn.Softplus())

        self.ffu0 = nn.Sequential(nn.Linear(2, 64*larger), nn.Softplus(), nn.Linear(64*larger, 32*larger), nn.Softplus())
        self.ffu1 = nn.Sequential(nn.Linear(32*larger, 64*larger), nn.Softplus(), nn.Linear(64*larger, 32*larger), nn.Softplus())
        self.ffu2 = nn.Sequential(nn.Linear(32*larger, 64*larger), nn.Softplus(), nn.Linear(64*larger, 32*larger), nn.Softplus())
        self.node_linear = nn.Linear(32*larger, 16*larger)
        self.node_s2s = set2set(in_channels=16*larger, processing_steps=3)
        self.edge_linear = nn.Linear(32*larger, 16*larger)
        self.edge_s2s = set2set(in_channels=16*larger, processing_steps=3)
        
        self.fc_out = nn.Sequential(nn.Linear(96*larger, 32*larger), nn.Softplus(), nn.Linear(32*larger, 16*larger), nn.Softplus(), nn.Linear(16*larger, 9))

    def forward(self, data, test=False) -> torch.Tensor:
        z = data.x
        # z = z.squeeze(-1).long()
        # calculate v 16
        v = self.embedding(z)
        # calculate e 100
        e = torch.norm(data.edge_attr, dim=1)
        e = self.rbf(e)
        # calculate u batch*2
        u = torch.zeros(self.batchsize, 2).float().to(z.device)
        if test:
            u = torch.zeros(1, 2).float().to(z.device)
        v = self.ffv0(v)
        e = self.ffe0(e)
        u = self.ffu0(u)
        # meg layer one
        v1, e1, u1 = self.meglayer1(x=v, edge_index=data.edge_index, batch=data.batch, edge_attr=e, u_rep=u)
        v1 = v1 + v
        e1 = e1 + e
        u1 = u1 + u
        v_t = v1
        e_t = e1
        u_t = u1
        # fc
        v1 = self.ffv1(v1)
        e1 = self.ffe1(e1)
        u1 = self.ffu1(u1)
        # meg layer two
        v1, e1, u1 = self.meglayer2(x=v1, edge_index=data.edge_index, batch=data.batch, edge_attr=e1, u_rep=u1)
        v1 = v1 + v_t
        e1 = e1 + e_t
        u1 = u1 + u_t
        v_t = v1
        e_t = e1
        u_t = u1
        # fc
        v1 = self.ffv2(v1)
        e1 = self.ffe2(e1)
        u1 = self.ffu2(u1)
        # meg layer three
        v1, e1, u1 = self.meglayer3(x=v1, edge_index=data.edge_index, batch=data.batch, edge_attr=e1, u_rep=u1)
        v1 = v1 + v_t
        e1 = e1 + e_t
        u1 = u1 + u_t
        # set2set
        v1 = self.node_linear(v1)
        e1 = self.edge_linear(e1)
        node_vec = self.node_s2s(v1, data.batch)
        edge_vec = self.edge_s2s(e1, data.batch[data.edge_index[1]])
        final_vec = torch.cat((node_vec, edge_vec, u1), dim=-1)
        
        out = self.fc_out(final_vec)

        return out
