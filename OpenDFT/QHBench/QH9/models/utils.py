import math
import torch
import torch.nn.functional as F


def ShiftedSoftPlus(x):
    return torch.nn.functional.softplus(x) - math.log(2.0)


def get_nonlinear(nonlinear: str):
    if nonlinear.lower() == 'ssp':
        return ShiftedSoftPlus
    elif nonlinear.lower() == 'silu':
        return F.silu
    elif nonlinear.lower() == 'tanh':
        return F.tanh
    elif nonlinear.lower() == 'abs':
        return torch.abs
    else:
        raise NotImplementedError


def prod(x):
    """Compute the product of a sequence."""
    out = 1
    for a in x:
        out *= a
    return out
