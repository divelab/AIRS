###########################################################################################
# from MACE and PhiSNet
###########################################################################################

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from e3nn import o3


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.lin2 = torch.nn.Linear(hidden_dim, output_dim, bias=False)
        self.dropout = dropout
        self.activation = torch.nn.SiLU()
        self.normalization = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.activation(self.normalization(self.lin1(x)))
        x = self.lin2(x)
        return x

#############################################
# Atom embedding
#############################################
class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, node_attrs):
        return self.linear(node_attrs)


#############################################
# Edge embedding
#############################################

"""
IMPORTANT NOTE: The cutoff and the switch function are numerically a bit tricky:
Right at the "seems" of these functions, i.e. where the piecewise definition changes,
there is formally a division by 0 (i.e. 0/0). This is of no issue for the function
itself, but when automatic differentiation is used, this division will lead to NaN 
gradients. In order to circumvent this, the input needs to be masked as well.
"""

"""
shifted softplus activation function
"""
_log2 = math.log(2)
def shifted_softplus(x):
    return F.softplus(x) - _log2

"""
cutoff function that smoothly goes from y = 1..0 in the interval x = 0..cutoff
(this cutoff function has infinitely many smooth derivatives)
"""
def cutoff_function(x, cutoff):
    zeros = torch.zeros_like(x)
    x_ = torch.where(x < cutoff, x, zeros)
    return torch.where(x < cutoff, torch.exp(-x_**2/((cutoff-x_)*(cutoff+x_))), zeros)

"""
switch function that smoothly and symmetrically goes from y = 1..0 in the interval x = cuton..cutoff and
is 1 for x <= cuton and 0 for x >= cutoff (this switch function has infinitely many smooth derivatives)
(when cuton < cutoff, it goes from 1 to 0, if cutoff < cuton, it goes from 0 to 1)
NOTE: the implementation with the "_switch_component" function is numerically more stable than
a simplified version, DO NOT CHANGE THIS!
"""
def _switch_component(x, ones, zeros):
    x_ = torch.where(x <= 0, ones, x)
    return torch.where(x <= 0, zeros, torch.exp(-ones/x_))

def switch_function(x, cuton, cutoff):
    x = (x-cuton)/(cutoff-cuton)
    ones  = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    fp = _switch_component(x, ones, zeros)
    fm = _switch_component(1-x, ones, zeros)
    return torch.where(x <= 0, ones, torch.where(x >= 1, zeros, fm/(fp+fm)))

"""
inverse softplus transformation, this is useful for initialization of parameters that are constrained to be positive
"""
def softplus_inverse(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x + torch.log(-torch.expm1(-x))


class ExponentialBernsteinRadialBasisFunctions(nn.Module):
    def __init__(self, num_basis_functions, cutoff, ini_alpha=0.5):
        super(ExponentialBernsteinRadialBasisFunctions, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.ini_alpha = ini_alpha
        # compute values to initialize buffers
        logfactorial = np.zeros((num_basis_functions))
        for i in range(2,num_basis_functions):
            logfactorial[i] = logfactorial[i-1] + np.log(i)
        v = np.arange(0,num_basis_functions)
        n = (num_basis_functions-1)-v
        logbinomial = logfactorial[-1]-logfactorial[v]-logfactorial[n]
        #register buffers and parameters
        self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float32))
        self.register_buffer('logc', torch.tensor(logbinomial, dtype=torch.float32))
        self.register_buffer('n', torch.tensor(n, dtype=torch.float32))
        self.register_buffer('v', torch.tensor(v, dtype=torch.float32))
        self.register_parameter('_alpha', nn.Parameter(torch.tensor(1.0, dtype=torch.float32)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self._alpha,  softplus_inverse(self.ini_alpha))

    def forward(self, r):
        alpha = F.softplus(self._alpha)
        x = - alpha * r
        x = self.logc + self.n * x + self.v * torch.log(- torch.expm1(x) )
        rbf = cutoff_function(r, self.cutoff) * torch.exp(x)
        return rbf


class BesselBasis(torch.nn.Module):
    """
    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (7)
    """

    def __init__(self, r_max, num_basis=8, trainable=False):
        super().__init__()

        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(self, x):
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )

class PolynomialCutoff(torch.nn.Module):
    """
    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (8)
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max, p=6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, x):
        # yapf: disable
        envelope = (
                1.0
                - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x / self.r_max, self.p)
                + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1)
                - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2)
        )
        # yapf: enable

        # noinspection PyUnresolvedReferences
        return envelope * (x < self.r_max)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max})"
    

class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(self, r_max, num_bessel, num_polynomial_cutoff):
        super().__init__()
        self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(self, edge_lengths):
        bessel = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        return bessel * cutoff  # [n_edges, n_basis]

