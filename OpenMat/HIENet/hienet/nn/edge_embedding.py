import math
from typing import Optional, Tuple, Union

import torch
import hienet._keys as KEY
from hienet._const import AtomGraphDataType
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from e3nn import o3
from e3nn.o3 import Irreps, SphericalHarmonics
from e3nn.util.jit import compile_mode
import numpy as np


import random
import numpy as np
import torch



@compile_mode('script')
class EdgePreprocess(nn.Module):
    """
    preprocessing pos to edge vectors and edge lengths
    """

    def __init__(self, is_stress):
        super().__init__()
        # controlled by 'AtomGraphSequential'
        self.is_stress = is_stress
        self._is_batch_data = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if self._is_batch_data:
            cell = data[KEY.CELL].view(-1, 3, 3)
        else:
            cell = data[KEY.CELL].view(3, 3)
        cell_shift = data[KEY.CELL_SHIFT]
        pos = data[KEY.POS]

        batch = data[KEY.BATCH]  # for deploy, must be defined first
        if self.is_stress:
            if self._is_batch_data:
                num_batch = int(batch.max().cpu().item()) + 1
                strain = torch.zeros(
                    (num_batch, 3, 3),
                    dtype=pos.dtype,
                    device=pos.device,
                )
                strain.requires_grad_(True)
                data['_strain'] = strain

                sym_strain = 0.5 * (strain + strain.transpose(-1, -2))
                pos = pos + torch.bmm(
                    pos.unsqueeze(-2), sym_strain[batch]
                ).squeeze(-2)
                cell = cell + torch.bmm(cell, sym_strain)
            else:
                strain = torch.zeros(
                    (3, 3),
                    dtype=pos.dtype,
                    device=pos.device,
                )
                strain.requires_grad_(True)
                data['_strain'] = strain

                sym_strain = 0.5 * (strain + strain.transpose(-1, -2))
                pos = pos + torch.mm(pos, sym_strain)
                cell = cell + torch.mm(cell, sym_strain)

        idx_src = data[KEY.EDGE_IDX][0]
        idx_dst = data[KEY.EDGE_IDX][1]

        edge_vec = pos[idx_dst] - pos[idx_src]

        if self._is_batch_data:
            edge_vec = edge_vec + torch.einsum(
                'ni,nij->nj', cell_shift, cell[batch[idx_src]]
            )
        else:
            edge_vec = edge_vec + torch.einsum(
                'ni,ij->nj', cell_shift, cell.squeeze(0)
            )
        data[KEY.EDGE_VEC] = edge_vec
        data[KEY.EDGE_LENGTH] = torch.linalg.norm(edge_vec, dim=-1)
        return data


class BesselBasis(nn.Module):
    """
    f : (*, 1) -> (*, bessel_basis_num)
    """

    def __init__(
        self,
        cutoff_length: float,
        bessel_basis_num: int = 8,
        trainable_coeff: bool = True,
    ):
        super().__init__()
        self.num_basis = bessel_basis_num
        self.prefactor = 2.0 / cutoff_length
        self.coeffs = torch.FloatTensor(
            [n * math.pi / cutoff_length for n in range(1, bessel_basis_num + 1)]
        )
        if trainable_coeff:
            self.coeffs = nn.Parameter(self.coeffs)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        ur = r.unsqueeze(-1)  # to fit dimension
        return self.prefactor * torch.sin(self.coeffs * ur) / ur


class PolynomialCutoff(nn.Module):
    """
    f : (*, 1) -> (*, 1)
    https://arxiv.org/pdf/2003.03123.pdf
    """

    def __init__(
        self,
        cutoff_length: float,
        poly_cut_p_value: int = 6,
    ):
        super().__init__()
        p = poly_cut_p_value
        self.cutoff_length = cutoff_length
        self.p = p
        self.coeff_p0 = (p + 1.0) * (p + 2.0) / 2.0
        self.coeff_p1 = p * (p + 2.0)
        self.coeff_p2 = p * (p + 1.0) / 2.0

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r = r / self.cutoff_length
        return (
            1
            - self.coeff_p0 * torch.pow(r, self.p)
            + self.coeff_p1 * torch.pow(r, self.p + 1.0)
            - self.coeff_p2 * torch.pow(r, self.p + 2.0)
        )


class XPLORCutoff(nn.Module):
    """
    https://hoomd-blue.readthedocs.io/en/latest/module-md-pair.html
    """

    def __init__(
        self,
        cutoff_length: float,
        cutoff_on: float,
    ):
        super().__init__()
        self.r_on = cutoff_on
        self.r_cut = cutoff_length
        assert self.r_on < self.r_cut

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # r > r_cut switch is not necessary since edges are already based on cutoff
        r_sq = r * r
        r_on_sq = self.r_on * self.r_on
        r_cut_sq = self.r_cut * self.r_cut
        return torch.where(
            r < self.r_on,
            1.0,
            (r_cut_sq - r_sq) ** 2
            * (r_cut_sq + 2 * r_sq - 3 * r_on_sq)
            / (r_cut_sq - r_on_sq) ** 3,
        )


class CosineCutoff(torch.nn.Module):
    """Cosine cutoff function."""

    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        """Compute the cutoff function."""
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


class ExpNormalSmearing(torch.nn.Module):
    """Exponential normal smearing function."""

    def __init__(self, cutoff_lower=0.0, cutoff_upper=10.0, num_rbf=50, trainable=True):
        """Exponential normal smearing function.

        Distances are expanded into exponential radial basis functions.
        Basis function parameters are initialised as proposed by Unke & Mewly 2019 Physnet,
        https://arxiv.org/pdf/1902.08408.pdf.
        A cosine cutoff function is used to ensure smooth transition to 0.

        Args:
            cutoff_lower (float): Lower cutoff radius.
            cutoff_upper (float): Upper cutoff radius.
            num_rbf (int): Number of radial basis functions.
            trainable (bool): Whether the parameters are trainable.
        """
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        #self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = cutoff_upper / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", torch.nn.Parameter(means))
            self.register_parameter("betas", torch.nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(float(start_value), 1.0, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        """Reset the parameters to their default values.""" ""
        means, betas = self._initial_params()
        self.means.data.copy_(means)  # type: ignore
        self.betas.data.copy_(betas)  # type: ignore

    def forward(self, dist):
        """Expand incoming distances into basis functions."""
        dist = dist.unsqueeze(-1)
        assert isinstance(self.betas, torch.Tensor)
        return torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )







@compile_mode('script')
class SphericalEncoding(nn.Module):
    """
    Calculate spherical harmonics from 0 to lmax
    taking displacement vector (EDGE_VEC) as input.

    lmax: maximum angular momentum quantum number used in model
    normalization : {'integral', 'component', 'norm'}
        normalization of the output tensors
        Valid options:
        * *component*: :math:`\|Y^l(x)\|^2 = 2l+1, x \in S^2`
        * *norm*: :math:`\|Y^l(x)\| = 1, x \in S^2`, ``component / sqrt(2l+1)``
        * *integral*: :math:`\int_{S^2} Y^l_m(x)^2 dx = 1`, ``component / sqrt(4pi)``

    Returns
    -------
    `torch.Tensor`
        a tensor of shape ``(..., (lmax+1)^2)``
    """

    def __init__(
        self, lmax: int, parity: int = -1, normalization: str = 'component', normalize = True,
    ):
        super().__init__()
        self.lmax = lmax
        self.normalization = normalization
        self.irreps_in = Irreps('1x1o') if parity == -1 else Irreps('1x1e')
        self.irreps_out = Irreps.spherical_harmonics(lmax, parity) # Pay attention to this line
        self.sph = SphericalHarmonics(
            self.irreps_out,
            normalize=normalize,
            normalization=normalization,
            irreps_in=self.irreps_in,
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.sph(r)


def bond_cosine(r1, r2):
    bond_cosine = torch.sum(r1 * r2, dim=-1) / (
        torch.norm(r1, dim=-1) * torch.norm(r2, dim=-1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine

def bond_sine(r1, r2):
    bond_sine = torch.norm(torch.cross(r1, r2, dim=-1), dim=-1) / (
        torch.norm(r1, dim=-1) * torch.norm(r2, dim=-1)
    )
    bond_sine = torch.clamp(bond_sine, -1, 1)
    return bond_sine



@compile_mode('script')
class EdgeEmbedding(nn.Module):
    """
    embedding layer of |r| by
    RadialBasis(|r|)*CutOff(|r|)
    f : (N_edge) -> (N_edge, basis_num)
    """

    def __init__(
        self,
        basis_module: nn.Module,
        cutoff_module: nn.Module,
        spherical_module: nn.Module,

        use_edge_conv: bool = False,
        angle_module: nn.Module = None,
        use_sine: bool = True,
    ):
        super().__init__()
        self.basis_module = basis_module
        self.cutoff_function = cutoff_module
        self.spherical = spherical_module

        self.use_edge_conv = use_edge_conv 
        # edge convolution:
        if self.use_edge_conv:
            self.angle_module = angle_module

            if use_sine:
                self.angle_embedding = bond_sine
            else:
                self.angle_embedding = bond_cosine


    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        rvec = data[KEY.EDGE_VEC]
        r = torch.linalg.norm(data[KEY.EDGE_VEC], dim=-1)
        data[KEY.EDGE_LENGTH] = r

        data[KEY.EDGE_EMBEDDING] = self.basis_module(
            r
        ) * self.cutoff_function(r).unsqueeze(-1)
        data[KEY.EDGE_ATTR] = self.spherical(rvec)


        if self.use_edge_conv:
            rvec = data[KEY.EDGE_VEC]
            edge_features = -0.75 / r        

            r2 = rvec.unsqueeze(1).repeat(1,3,1)

            cells = data[KEY.CELL].view(-1,3,3)
            edge_neighbors = cells[data[KEY.BATCH][data[KEY.EDGE_IDX][0]]]  # [n_edges, 3]

            edge_angles = self.angle_embedding(edge_neighbors, r2)  
            edge_angles = edge_angles.reshape(-1)                               # [n_edges * 3]

            edge_angle_embeddings = self.angle_module(edge_angles)                 # [3*n_edges, node_features]
            edge_angle_embeddings = edge_angle_embeddings.reshape(edge_features.shape[0], 3, -1)

            edge_lattice_lengths = -0.75 / torch.norm(edge_neighbors, dim=-1)   # [n_edges, 3]
            edge_lattice_lengths = edge_lattice_lengths.reshape(-1)             # [n_edges * 3]
            edge_lattice_embeddings = self.basis_module(edge_lattice_lengths)            # [3*n_edges, node_features]
            edge_lattice_embeddings = edge_lattice_embeddings.reshape(edge_features.shape[0], 3, -1)

            data[KEY.ANGLE_EMBEDDING] = edge_angle_embeddings
            data[KEY.LATTICE_EMBEDDING] = edge_lattice_embeddings
        return data



from typing import List, Optional, Tuple



@compile_mode('script')
class ComformerEdgeEmbedding(nn.Module):
    def __init__(
        self,
        basis_module: nn.Module,
        radial_basis_num: int,
        out_dim: int,
        # triplet_features: int,
        spherical_module: nn.Module,
        use_sine: bool = True,
    ):
        super().__init__()
        self.radial_module = nn.Sequential(
            basis_module,
            nn.Linear(radial_basis_num, out_dim),
            nn.Softplus(),
        )
        self.spherical = spherical_module
        
        use_edge_conv = False
        if use_edge_conv:
            
            self.rbf_angle = nn.Sequential(
                RBFExpansion(
                    r_min=-1.0,
                    r_max=1.0,
                    n_bins=triplet_features,
                ),
                nn.Linear(triplet_features, out_dim),
                nn.Softplus(),
            )

            if use_sine:
                self.angle_embedding = bond_sine
            else:
                self.angle_embedding = bond_cosine

       
    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:

        rvec = data[KEY.EDGE_VEC]

        r = torch.linalg.norm(data[KEY.EDGE_VEC], dim=-1)
        data[KEY.EDGE_LENGTH] = r
        edge_features = -0.75 / r
        data[KEY.EDGE_EMBEDDING] = self.radial_module(edge_features)
        use_edge_conv = False

        # used to be required for equivariant convolution to work
        data[KEY.EDGE_ATTR] = self.spherical(rvec)

        if use_edge_conv:
            r2 = rvec.unsqueeze(1).repeat(1,3,1)

            cells = data[KEY.CELL].view(-1,3,3)

            edge_neighbors = cells[data[KEY.BATCH][data[KEY.EDGE_IDX][0]]]  # [n_edges, 3]


            edge_angles = self.angle_embedding(edge_neighbors, r2)  
            edge_angles = edge_angles.reshape(-1)                               # [n_edges * 3]

            edge_angle_embeddings = self.rbf_angle(edge_angles)                 # [3*n_edges, node_features]
            edge_angle_embeddings = edge_angle_embeddings.reshape(edge_features.shape[0], 3, -1)

            edge_lattice_lengths = -0.75 / torch.norm(edge_neighbors, dim=-1)   # [n_edges, 3]
            edge_lattice_lengths = edge_lattice_lengths.reshape(-1)             # [n_edges * 3]
            edge_lattice_embeddings = self.rbf(edge_lattice_lengths)            # [3*n_edges, node_features]
            edge_lattice_embeddings = edge_lattice_embeddings.reshape(edge_features.shape[0], 3, -1)

            data[KEY.ANGLE_EMBEDDING] = edge_angle_embeddings
            data[KEY.LATTICE_EMBEDDING] = edge_lattice_embeddings

        return data

@compile_mode('script')
class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        r_min: float = 0,
        r_max: float = 8,
        n_bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()

        self.r_min = r_min
        self.r_max = r_max
        self.n_bins = n_bins
        self.register_buffer(
            "centers", torch.linspace(self.r_min, self.r_max, self.n_bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            # lengthscale = (r_max - r_min) / n_bins
            self.lengthscale = np.diff(self.centers).mean() # average distance between centers
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )
