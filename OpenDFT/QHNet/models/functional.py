"""
from PhiSNet SE(3)-equivariant prediction of molecular wavefunctions and electronic densities 
<https://arxiv.org/abs/2106.02347>
"""

import math
import torch
import torch.nn.functional as F

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
