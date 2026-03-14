import torch
import torch_scatter


def mae(pred, target, mask):
    """
    pred: T, N, 9, 2 already masked
    target: T, N, 9, 2
    """
    err = pred - target
    return torch.mean(err.abs()) * mask.numel() / mask.sum()


# atom-wise
def l2mae(pred, target, mask=None):
    """
    pred: T, N, 9, 2 already masked
    target: T, N, 9, 2
    """
    err = pred - target
    err_norm = err.flatten(start_dim=2).norm(dim=-1)  # T, N
    return torch.mean(err_norm)


def scaled_l2(pred, target, graph_batch, state_ind_batch=None, eband_batch=None):
    # Backward compatibility: older callers pass `eband_batch`.
    if state_ind_batch is None:
        state_ind_batch = eband_batch
    if state_ind_batch is None:
        raise ValueError("scaled_l2 requires `state_ind_batch` (or legacy `eband_batch`).")

    err = pred - target
    err_norm = err.flatten(start_dim=2).norm(dim=-1)  # T, N
    err_norm = torch_scatter.scatter_sum(err_norm ** 2, graph_batch, dim=1).sqrt()  # T, num_ebands
    err_norm = torch.sum(err_norm ** 2, dim=0).sqrt()  # num_ebands
    err_norm = torch_scatter.scatter_mean(err_norm, state_ind_batch, dim=0)  # batch_size
    target_norm = target.flatten(start_dim=2).norm(dim=-1)  # T, N
    target_norm = torch_scatter.scatter_sum(target_norm ** 2, graph_batch, dim=1).sqrt()  # T, num_ebands
    target_norm = torch.sum(target_norm ** 2, dim=0).sqrt()  # num_ebands
    target_norm = torch_scatter.scatter_mean(target_norm, state_ind_batch, dim=0)  # batch_size
    return torch.mean(err_norm / target_norm)


def mae_loss(pred, target, reduction='mean'):
    pred = pred.flatten(-2)
    target = target.flatten(-2)  # * 10

    loss = (pred - target).abs().mean(dim=-1)

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'none':
        pass  # do nothing
    else:
        raise NotImplementedError('unknown reduction')

    return loss


def scaled_l2_loss(pred, target, reduction='mean'):
    """
    Currently support same size
    Values can be complex (wfc) or real (H)
    pred: B, T, W, H
    target: B, T, W, H
    """
    pred = pred.flatten(-2)
    target = target.flatten(-2)

    target_norm = torch.norm(target, dim=-1, p=2)  # B, T
    diff_norm = torch.norm(pred - target, dim=-1, p=2)
    loss = diff_norm / (target_norm + 1e-12)

    # loss = torch.mean((pred - target).abs())

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'none':
        pass  # do nothing
    else:
        raise NotImplementedError('unknown reduction')

    return loss
