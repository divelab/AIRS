# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn.functional as F


def scaledlp_loss(input: torch.Tensor, target: torch.Tensor, p: int = 2, reduction: str = "mean"):
    assert input.dim() == 5 and target.dim() == 5 # B, T, C, S, S
    input = input.flatten(3)
    target = target.flatten(3)
    diff_norms = torch.norm(input - target, p, dim=-1)
    target_norms = torch.norm(target, p, dim=-1)
    val = diff_norms / target_norms
    if reduction == "mean":
        return torch.mean(val)
    elif reduction == "sum":
        return torch.sum(val)
    elif reduction == "none":
        return val
    else:
        raise NotImplementedError(reduction)


def custommse_loss(input: torch.Tensor, target: torch.Tensor, reduction: str = "mean"):
    loss = F.mse_loss(input, target, reduction="none")
    # avg across space
    reduced_loss = torch.mean(loss, dim=(3, 4))
    # sum across time + fields
    reduced_loss = reduced_loss.sum(dim=(1, 2))
    # reduce along batch
    if reduction == "mean":
        return torch.mean(reduced_loss)
    elif reduction == "sum":
        return torch.sum(reduced_loss)
    elif reduction == "none":
        return reduced_loss
    else:
        raise NotImplementedError(reduction)


class ScaledLpLoss(torch.nn.Module):
    """Scaled Lp loss for PDEs.

    Args:
        p (int, optional): p in Lp norm. Defaults to 2.
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, p: int = 2, reduction: str = "mean") -> None:
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return scaledlp_loss(input, target, p=self.p, reduction=self.reduction)


class CustomMSELoss(torch.nn.Module):
    """Custom MSE loss for PDEs.

    MSE but summed over time and fields, then averaged over space and batch.

    Args:
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return custommse_loss(input, target, reduction=self.reduction)
