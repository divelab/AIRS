import torch
from torch_geometric.loader import DataLoader
from dataset.PygrMD17 import rMD17


class WeightedEnergyForcesLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, gt_batch, pred):
        num_atoms = gt_batch.ptr[1:] - gt_batch.ptr[:-1]

        loss_e = torch.mean(torch.square((gt_batch.y - pred["energy"]) / num_atoms))
        loss_f = torch.mean(torch.square((gt_batch.force - pred["force"])))
        return self.energy_weight * loss_e + self.forces_weight * loss_f


class L1Loss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.L1Loss()

    def forward(self, gt_batch, pred):
        return self.loss(pred["energy"], gt_batch.y)


if __name__ == "__main__":
    mode = "train_01"
    dataset = rMD17(root="./rmd17/", name="benzene", mode=mode)

    split_idx = dataset.get_idx_split(len(dataset.data.y), valid_fraction=0.05, seed=3)
    print(dataset[split_idx["train"]])
    train_dataset, valid_dataset = (
        dataset[split_idx["train"]],
        dataset[split_idx["valid"]],
    )
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    data = next(iter(train_loader))
    print(data)
    print(data.ptr)
    num_atoms = data.ptr[1:] - data.ptr[:-1]
    print(num_atoms)
