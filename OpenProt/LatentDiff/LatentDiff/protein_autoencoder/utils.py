import torch
import numpy as np
from typing import List
from torch import Tensor

def build_edge_idx(num_nodes):
    # Initialize edge index matrix
    E = torch.zeros((2, num_nodes * (num_nodes - 1)), dtype=torch.long)

    # Populate 1st row
    for node in range(num_nodes):
        for neighbor in range(num_nodes - 1):
            E[0, node * (num_nodes - 1) + neighbor] = node

    # Populate 2nd row
    neighbors = []
    for node in range(num_nodes):
        neighbors.append(list(np.arange(node)) + list(np.arange(node + 1, num_nodes)))
    E[1, :] = torch.Tensor([item for sublist in neighbors for item in sublist])

    return E


class KabschRMSD(torch.nn.Module):
    def __init__(self) -> None:
        super(KabschRMSD, self).__init__()

    def forward(self, protein_coords_pred: List[Tensor], protein_coords: List[Tensor]) -> Tensor:
        rmsds = []
        for protein_coords_pred, protein_coords in zip(protein_coords_pred, protein_coords):
            protein_coords_pred_mean = protein_coords_pred.mean(dim=0, keepdim=True)  # (1,3)
            protein_coords_mean = protein_coords.mean(dim=0, keepdim=True)  # (1,3)

            A = (protein_coords_pred - protein_coords_pred_mean).transpose(0, 1) @ (
                        protein_coords - protein_coords_mean)

            U, S, Vt = torch.linalg.svd(A)

            # corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=protein_coords_pred.device))
            corr_mat = torch.diag(
                torch.tensor([1, 1, torch.sign(torch.det(Vt.t() @ U.t()))], device=protein_coords_pred.device))
            rotation = (U @ corr_mat) @ Vt
            # corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(Vt.t() @ U.t()))], device=protein_coords_pred.device))
            # rotation = (Vt.t() @ corr_mat) @ U.t()
            translation = protein_coords_pred_mean - torch.t(rotation @ protein_coords_mean.t())  # (1,3)

            protein_coords = (rotation @ protein_coords.t()).t() + translation

            rmsds.append(torch.sqrt(torch.mean(torch.sum(((protein_coords_pred - protein_coords) ** 2), dim=1))))
        return torch.tensor(rmsds).mean()

class RMSD(torch.nn.Module):
    def __init__(self) -> None:
        super(RMSD, self).__init__()

    def forward(self, protein_coords_pred: List[Tensor], protein_coords: List[Tensor]) -> Tensor:
        rmsds = []
        for protein_coords_pred, protein_coords in zip(protein_coords_pred, protein_coords):
            rmsds.append(torch.sqrt(torch.mean(torch.sum(((protein_coords_pred - protein_coords) ** 2), dim=1))))
        return torch.tensor(rmsds).mean()

