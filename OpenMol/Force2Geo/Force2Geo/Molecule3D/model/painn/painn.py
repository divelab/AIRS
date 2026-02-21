"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

---

MIT License

Copyright (c) 2021 www.compscience.org

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
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
        num_layers: int = 6,
        num_rbf: int = 128,
        cutoff: float = 12.0,
        max_neighbors: int = 100,
        gradient_force: bool = True,
        num_elements: int = 80,
        return_embedding: bool = False,
        conformer: str = 'GT',
        return_vec: bool = False,
        noise_std: float = 0.02,
        add_noise_to_relax: bool = False,
        frac_dn: bool = False,
        DeNS: bool = False,
    ) -> None:
        super().__init__()
        
        self.conformer = conformer
        self.return_vec = return_vec
        self.noise_std = noise_std
        self.add_noise_to_relax = add_noise_to_relax
        self.frac_dn = frac_dn
        self.DeNS = DeNS
        

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.regress_forces = gradient_force
        self.return_embedding = return_embedding

        #### Learnable parameters #############################################

        self.atom_emb = nn.Embedding(num_elements, hidden_channels)
        self.interaction_graph = RadiusInteractionGraph(cutoff, max_neighbors)

        self.radial_basis = GaussianSmearing(
            stop=cutoff,
            num_gaussians=num_rbf,
        )

        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()


        for i in range(num_layers):
            self.message_layers.append(
                PaiNNMessage(hidden_channels, num_rbf).jittable()
            )
            self.update_layers.append(PaiNNUpdate(hidden_channels))

        self.out_energy = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            SiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

        # if self.regress_forces is True and self.direct_forces is True:
        #     self.out_forces = PaiNNOutput(hidden_channels)
        if self.return_vec:
            self.out_position = PaiNNOutput(hidden_channels)

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.out_energy[0].weight)
        self.out_energy[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_energy[2].weight)
        self.out_energy[2].bias.data.fill_(0)


    def forward(self, data):
        # pos = data.pos
        # batch = data.batch
        # z = data.x.long().squeeze()
        if self.conformer == 'GT':
            pos = data.xyz
            z = data.z.long().squeeze()
        elif self.conformer == 'rdkit':
            pos = data.rdkit_xyz
            z = data.z.long().squeeze()
        elif self.conformer == 'relaxation':
            pos = data.relaxed_xyz
            z = data.DFT1st_z.long().squeeze()
        elif self.conformer == 'DFT_1st':
            pos = data.DFT1st_init_xyz
            z = data.DFT1st_z.long().squeeze()
        elif self.conformer == 'mix_GT_relax':
            # import pdb; pdb.set_trace()
            noise_target = None
            if self.training:
                if torch.rand(1).item() > 0.5:
                    if self.frac_dn:
                        noise_target = data.frac_dn_target
                        pos = data.frac_perturbed_GT_pos
                        z = data.z.long().squeeze()
                    else:
                        noise = torch.randn_like(data.xyz) * self.noise_std
                        pos = data.xyz + noise
                        z = data.z.long().squeeze()
                        noise_target = noise
                else:
                    if self.add_noise_to_relax:
                        noise = torch.randn_like(data.relaxed_xyz) * self.noise_std
                        pos = data.relaxed_xyz + noise
                        noise_target = data.xyz - data.relaxed_xyz - noise
                    else:
                        pos = data.relaxed_xyz
                    z = data.DFT1st_z.long().squeeze()
                # import pdb; pdb.set_trace()
            else:
                pos = data.relaxed_xyz
                z = data.DFT1st_z.long().squeeze()
        else:
            raise ValueError(f"Conformer type {self.conformer} not supported.")
        
        # pos = data.xyz
        batch = data.batch
        # z = data.z.long().squeeze()

        pos = pos.requires_grad_(True)

        edge_index, edge_dist = self.interaction_graph(pos, batch)
        edge_vector = pos[edge_index[1]] - pos[edge_index[0]]
        
        assert z.dim() == 1
        assert z.dtype == torch.long

        edge_rbf = self.radial_basis(edge_dist)  # rbf * envelope

        x = self.atom_emb(z)
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)
        # import pdb; pdb.set_trace()
        #### Interaction blocks ###############################################

        for i in range(self.num_layers):
            dx, dvec = self.message_layers[i](x, vec, edge_index, edge_rbf, edge_vector)

            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            dx, dvec = self.update_layers[i](x, vec)

            x = x + dx
            vec = vec + dvec
            
        if self.return_embedding:
            return x

        #### Output block #####################################################

        per_atom_energy = self.out_energy(x).squeeze(1)
        energy = scatter(per_atom_energy, batch, dim=0, reduce='mean')
        
        if self.return_vec:
            vec = self.out_position(x, vec)
            # import pdb; pdb.set_trace()
            assert vec.shape[0] == x.shape[0]
            assert vec.shape[1] == 3
            if self.conformer == 'mix_GT_relax' and noise_target is not None:
                return energy.squeeze(), vec, noise_target
            return energy.squeeze(), vec
        else:
            return energy.squeeze()



class PaiNNMessage(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        num_rbf,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)

        self.hidden_channels = hidden_channels

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.rbf_proj = nn.Linear(num_rbf, hidden_channels * 3)

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
        # import pdb; pdb.set_trace()
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


class PaiNNOutput(nn.Module):
    def __init__(self, hidden_channels) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, vec):
        for layer in self.output_network:
            x, vec = layer(x, vec)
        return vec.squeeze()


# Borrowed from TorchMD-Net
class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            SiLU(),
            nn.Linear(hidden_channels, out_channels * 2),
        )

        self.act = SiLU()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        x = self.act(x)
        return x, v
    
if __name__ == "__main__":
    # Test the model
    from torch_geometric.data import Batch
    model = PaiNN()
    data = Batch()
    data.xyz = torch.randn(10, 3)
    data.z = torch.randint(0, 80, (10,))
    data.batch = torch.zeros(10, dtype=torch.long)
    output = model(data)
    print(output)