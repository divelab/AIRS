"""
===============================================================================
File:           painn
Date:           6/16/2024
Description:    Code is adapted from PaiNN OC20 implementation https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/core/models/painn.

All rights reserved to original authors.

===============================================================================
"""

from __future__ import annotations

import math
import typing

import torch
from torch import nn
from torch.nn import SiLU
from torch_cluster import radius_graph

if typing.TYPE_CHECKING:
    from torch_geometric.data.batch import Batch
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter, segment_coo

from torch_geometric.utils import remove_isolated_nodes

_MAX_ATOM_TYPE = 80

class GaussianSmearing(torch.nn.Module):
    def __init__(
            self,
            start: float = 0.0,
            stop: float = 5.0,
            num_gaussians: int = 50,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class RadiusInteractionGraph(torch.nn.Module):
    r"""Creates edges based on atom positions :obj:`pos` to all points within
    the cutoff distance.

    Args:
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance with the
            default interaction graph method.
            (default: :obj:`32`)
    """

    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos, batch):
        r"""Forward pass.

        Args:
            pos (Tensor): Coordinates of each atom.
            batch (LongTensor, optional): Batch indices assigning each atom to
                a separate molecule.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight


class PaiNN(nn.Module):
    r"""PaiNN model based on the description in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties
    and molecular spectra, https://arxiv.org/abs/2102.03150.
    """

    def __init__(
        self,
        hidden_channels: int = 512,
        num_interactions: int = 6,
        num_gaussians: int = 128,
        cutoff: float = 12.0,
        max_neighbors: int = 100,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.gradient_force = gradient_force

        #### Learnable parameters #############################################

        self.atom_emb = nn.Embedding(_MAX_ATOM_TYPE, hidden_channels)
        self.interaction_graph = RadiusInteractionGraph(cutoff, max_neighbors)

        self.radial_basis = GaussianSmearing(
            stop=cutoff,
            num_gaussians=num_gaussians,
        )

        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()

        for i in range(num_interactions):
            self.message_layers.append(
                PaiNNMessage(hidden_channels, num_gaussians).jittable()
            )
            self.update_layers.append(PaiNNUpdate(hidden_channels))

        self.out_energy = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            SiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.out_energy[0].weight)
        self.out_energy[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_energy[2].weight)
        self.out_energy[2].bias.data.fill_(0)


    def forward(self, data):
        batch_size = data.batch.max().item() + 1
        
        edge_index = radius_graph(data.pos, self.cutoff, data.batch, max_num_neighbors=self.max_neighbors)
        edge_index, _, mask = remove_isolated_nodes(edge_index, num_nodes=data.num_nodes)
        
        pos = data.pos[mask]
        batch = data.batch[mask]
        z = data.x[mask].long().squeeze()
        if self.gradient_force:
            pos = pos.requires_grad_(True)

        edge_index, edge_dist = self.interaction_graph(pos, batch)
        edge_vector = pos[edge_index[1]] - pos[edge_index[0]]
        assert z.dim() == 1
        assert z.dtype == torch.long

        edge_rbf = self.radial_basis(edge_dist)  # rbf * envelope

        x = self.atom_emb(z)
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)

        #### Interaction blocks ###############################################

        for i in range(self.num_interactions):
            dx, dvec = self.message_layers[i](x, vec, edge_index, edge_rbf, edge_vector)

            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            dx, dvec = self.update_layers[i](x, vec)

            x = x + dx
            vec = vec + dvec

        #### Output block #####################################################

        per_atom_energy = self.out_energy(x).squeeze(1)
        energy = scatter(per_atom_energy, batch, dim=0, reduce='mean', dim_size=batch_size)
            
        forces = -1 * (
            torch.autograd.grad(
                energy,
                pos,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
            )[0]
        )
        return energy, forces, mask



class PaiNNMessage(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        num_gaussians,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)

        self.hidden_channels = hidden_channels

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.rbf_proj = nn.Linear(num_gaussians, hidden_channels * 3)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.x_layernorm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)
        self.x_layernorm.reset_parameters()

    def forward(self, x, vec, edge_index, edge_rbf, edge_vector):
        xh = self.x_proj(self.x_layernorm(x))

        # TODO(@abhshkdz): Nans out with AMP here during backprop. Debug / fix.
        rbfh = self.rbf_proj(edge_rbf)

        # propagate_type: (xh: Tensor, vec: Tensor, rbfh_ij: Tensor, r_ij: Tensor)
        dx, dvec = self.propagate(
            edge_index,
            xh=xh,
            vec=vec,
            rbfh_ij=rbfh,
            r_ij=edge_vector,
            size=None,
        )

        return dx, dvec

    def message(self, xh_j, vec_j, rbfh_ij, r_ij):
        x, xh2, xh3 = torch.split(xh_j * rbfh_ij, self.hidden_channels, dim=-1)
        xh2 = xh2 * self.inv_sqrt_3

        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        vec = vec * self.inv_sqrt_h

        return x, vec

    def aggregate(
        self,
        features: tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        dim_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return inputs


class PaiNNUpdate(nn.Module):
    def __init__(self, hidden_channels) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.xvec_proj[0].weight)
        self.xvec_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xvec_proj[2].weight)
        self.xvec_proj[2].bias.data.fill_(0)

    def forward(self, x, vec):
        vec1, vec2 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1) * self.inv_sqrt_h

        # NOTE: Can't use torch.norm because the gradient is NaN for input = 0.
        # Add an epsilon offset to make sure sqrt is always positive.
        x_vec_h = self.xvec_proj(
            torch.cat([x, torch.sqrt(torch.sum(vec2**2, dim=-2) + 1e-8)], dim=-1)
        )
        xvec1, xvec2, xvec3 = torch.split(x_vec_h, self.hidden_channels, dim=-1)

        dx = xvec1 + xvec2 * vec_dot
        dx = dx * self.inv_sqrt_2

        dvec = xvec3.unsqueeze(1) * vec1

        return dx, dvec