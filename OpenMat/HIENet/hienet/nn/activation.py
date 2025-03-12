import math

import torch


@torch.jit.script
def ShiftedSoftPlus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x) - math.log(2.0)
