import torch
import torch.nn as nn
import torch.nn.functional as F


def l2loss_atomwise(pred, target, reduction="mean"):
    dist = torch.linalg.vector_norm((pred - target), dim=-1)
    if reduction == "mean":
        return torch.mean(dist)
    elif reduction == "sum":
        return torch.sum(dist)
    else:
        return dist


class L2Loss(nn.Module):
    def __init__(self, reduction="mean"):
        super(L2Loss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        loss = l2loss_atomwise(pred, target, self.reduction)
        return loss


# def cosine_loss_atomwise(pred, target, reduction="mean"):
#     # Normalize vectors
#     pred_norm = F.normalize(pred, dim=-1)
#     target_norm = F.normalize(target, dim=-1)

#     # Compute cosine similarity
#     cos_sim = (pred_norm * target_norm).sum(dim=-1)  # [N]
#     loss = 1.0 - cos_sim  # Cosine similarity ranges [-1, 1], we want to minimize (1 - sim)

#     if reduction == "mean":
#         return loss.mean()
#     elif reduction == "sum":
#         return loss.sum()
#     else:
#         return loss  # shape: [N]

# class CosineLoss(nn.Module):
#     def __init__(self, reduction="mean"):
#         super(CosineLoss, self).__init__()
#         self.reduction = reduction

#     def forward(self, pred, target):
#         return cosine_loss_atomwise(pred, target, self.reduction)

class CosineLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(CosineLoss, self).__init__()
        self.reduction = reduction
        self.cosine = nn.CosineSimilarity(dim=-1)

    def forward(self, pred, target):
        sim = self.cosine(pred, target)
        loss = 1.0 - sim
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss