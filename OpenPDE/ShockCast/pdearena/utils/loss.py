# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter


def lp_norm_weighted(input: torch.Tensor, weights: torch.Tensor, batch_idx: torch.Tensor, p: int):
    if p == 2:
        return scatter(src=input.square() * weights, index=batch_idx, dim=0, reduce="sum").sqrt()
    else:
        return scatter(src=input.pow(p) * weights, index=batch_idx, dim=0, reduce="sum").pow(1/p)

def lp_norm(input: torch.Tensor, batch_idx: torch.Tensor, p: int):
    if p == 2:
        return scatter(src=input.square(), index=batch_idx, dim=0, reduce="sum").sqrt()
    else:
        return scatter(src=input.pow(p), index=batch_idx, dim=0, reduce="sum").pow(1/p)

def mse(input: torch.Tensor, batch_idx: torch.Tensor):
    return scatter(src=input.square(), index=batch_idx, dim=0, reduce="mean")

def mse_loss(input: torch.Tensor, target: torch.Tensor, batch_idx: torch.Tensor, reduction: str = "mean"):
    loss = mse(input=input - target, batch_idx=batch_idx)
    if reduction == "mean":
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
    elif reduction == "none":
        return loss
    else:
        raise NotImplementedError(reduction)


def scaledlp_loss(input: torch.Tensor, target: torch.Tensor, batch_idx: torch.Tensor, p: int = 2, clamp_denom: float = 1, reduction: str = "mean"):
    diff_norms = lp_norm(input=input - target, batch_idx=batch_idx, p=p)
    target_norms = lp_norm(input=target, batch_idx=batch_idx, p=p).clamp_min(clamp_denom)
    val = diff_norms / target_norms
    if reduction == "mean":
        return torch.mean(val)
    elif reduction == "sum":
        return torch.sum(val)
    elif reduction == "none":
        return val
    else:
        raise NotImplementedError(reduction)


def scaledlp_loss_weighted(
        input: torch.Tensor, 
        target: torch.Tensor,
        weights: torch.Tensor, 
        batch_idx: torch.Tensor, 
        p: int = 2, 
        reduction: str = "mean"
    ):
    diff_norms = lp_norm_weighted(
        input=input - target, 
        weights=weights, 
        batch_idx=batch_idx, 
        p=p
    )
    target_norms = lp_norm_weighted(
        input=target, 
        weights=weights,
        batch_idx=batch_idx, 
        p=p
    )
    val = diff_norms / target_norms
    if reduction == "mean":
        return torch.mean(val)
    elif reduction == "sum":
        return torch.sum(val)
    elif reduction == "none":
        return val
    else:
        raise NotImplementedError(reduction)


def scaledlp_loss_grid(input: torch.Tensor, target: torch.Tensor, p: int = 2, reduction: str = "mean"):
    diff_norms = (input - target).flatten(-2).norm(p, -1)
    target_norms = target.flatten(-2).norm(p, -1)
    val = diff_norms / target_norms
    if reduction == "mean":
        return torch.mean(val)
    elif reduction == "sum":
        return torch.sum(val)
    elif reduction == "none":
        return val
    else:
        raise NotImplementedError(reduction)

class ScaledLpLoss(torch.nn.Module):
    """Scaled Lp loss for PDEs.

    Args:
        p (int, optional): p in Lp norm. Defaults to 2.
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, p: int = 2, clamp_denom: float = 0, reduction: str = "mean") -> None:
        super().__init__()
        self.clamp_denom = clamp_denom
        self.p = p
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        return scaledlp_loss(input=input, target=target, batch_idx=batch_idx, p=self.p, clamp_denom=self.clamp_denom, reduction=self.reduction)

class ScaledLpLossWeighted(ScaledLpLoss):
    """Scaled Lp loss for PDEs.

    Args:
        p (int, optional): p in Lp norm. Defaults to 2.
        reduction (str, optional): Reduction method. Defaults to "mean".
    """
    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        dim_diff = input.ndim - weights.ndim
        if dim_diff > 0:
            weights = weights.view(weights.shape + (1,) * dim_diff)
        return scaledlp_loss_weighted(input=input, target=target, weights=weights, batch_idx=batch_idx, p=self.p, reduction=self.reduction)

class MSELoss(torch.nn.Module):
    """Mean squared error loss for PDEs.
    
    Args: 
        reduction (str, optional): Reduction method. Defaults to "mean".
    """
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor, batch_idx: torch.Tensor):
        return mse_loss(input=input, target=target, batch_idx=batch_idx, reduction=self.reduction)

def correlation(
        input: torch.Tensor, 
        target: torch.Tensor, 
        batch_idx: torch.Tensor
    ) -> torch.Tensor:
    
    # scatter + mean reduce doesn't work with DDP devices > 1
    p = torch.ones(len(input)).type_as(input)
    p = 1 / scatter(p, batch_idx)
    p = p[batch_idx].view([len(input)] + [-1] * (input.ndim - 1))
    mu_x = scatter(p * input, batch_idx)
    mu_y = scatter(p * target, batch_idx)
    x0 = input - mu_x[batch_idx]
    y0 = target - mu_y[batch_idx]

    sigma_x = scatter(p * x0.square(), batch_idx).sqrt()
    sigma_y = scatter(p * y0.square(), batch_idx).sqrt()

    cov_xy = scatter(p * x0 * y0, batch_idx)

    # return torch.ones([len(batch_idx.unique()), input.shape[1]], device=input.device)
    return cov_xy / (sigma_x * sigma_y)

class Correlation(torch.nn.Module):
    def forward(
            self, 
            input: torch.Tensor, 
            target: torch.Tensor, 
            batch_idx: torch.Tensor
        ) -> torch.Tensor:
        assert len(input) == len(target) == len(batch_idx), [len(input), len(target), len(batch_idx)]                
        input = input.view(len(input), -1)
        target = target.view(len(target), -1)
        batch_idx = batch_idx.flatten()
        return correlation(
            input=input, 
            target=target,
            batch_idx=batch_idx
        )

if __name__ == "__main__":
    torch.manual_seed(1)
    batch_size = 5
    nt = 4
    nfields = 3
    nx = ny = 128

    scale = lambda N: torch.linspace(0, 2, N)

    y = torch.randn(batch_size, nt, nfields, nx, ny)
    noise_scale = scale(batch_size * nt * nfields).view(batch_size, nt, nfields, 1, 1) 
    x = y + noise_scale * torch.rand_like(y)

    x_node = x.permute(0, 3, 4, 1, 2).flatten(0, 2)
    y_node = y.permute(0, 3, 4, 1, 2).flatten(0, 2)
    batch_idx = torch.arange(batch_size).repeat_interleave(nx * ny)

    assert torch.allclose(scaledlp_loss(input=x_node, target=y_node, batch_idx=batch_idx, reduction="none"), 
                          scaledlp_loss_grid(input=x, target=y, reduction="none"))
    
    min_num_nodes = 10
    max_num_nodes = 100
    num_nodes = torch.randint(low=min_num_nodes, high=max_num_nodes, size=(batch_size,))
    batch_idx = torch.arange(batch_size).repeat_interleave(num_nodes)
    y_node_data = [torch.randn(num, nt, nfields) for num in num_nodes]
    x_node_data = [ydata + scale(nt * nfields).view(1, nt, nfields) * torch.rand_like(ydata) for ydata in y_node_data]
    
    x_nodes = torch.cat(x_node_data)
    y_nodes = torch.cat(y_node_data)

    diff_norms = torch.stack([(xdata - ydata).norm(dim=0) for xdata, ydata in zip(x_node_data, y_node_data)])
    target_norms = torch.stack([ydata.norm(dim=0) for ydata in y_node_data])
    rel_errs = diff_norms / target_norms

    assert torch.allclose(scaledlp_loss(input=x_nodes, target=y_nodes, batch_idx=batch_idx, reduction="none"), 
                          rel_errs)