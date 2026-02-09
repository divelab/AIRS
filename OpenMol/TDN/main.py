import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, ReduceLROnPlateau
from torch_scatter import scatter
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
from tap import tapify

from data import LMDBDataLoader, _STD_ENERGY, _STD_FORCE_SCALE
from models.painn import PaiNN
from models.schnet import SchNet
from models.tdn import TensorDecompositionNetwork


# Loss helpers
def compute_training_force_loss(forces, data, mask, loss_name: str):
    name = loss_name.lower()

    if name == "rmse":
        return (
            scatter(
                (forces - data.y_force[mask]).pow(2).sum(dim=-1),
                data.batch[mask],
                reduce="mean",
                dim=0,
                dim_size=data.batch.max().item() + 1,
            )
            .sqrt()
            .mean()
        )

    if name == "cosine":
        return torch.mean(
            (1.0 - torch.cosine_similarity(forces, data.y_force[mask], dim=-1))
            * (
                (
                    torch.abs(torch.norm(forces, dim=-1) - torch.norm(data.y_force[mask], dim=-1))
                    * 1.0
                )
                ** 2
            )
        )

    if name == "mse":
        return scatter(
            ((forces - data.y_force[mask]) ** 2).sum(dim=-1),
            data.batch[mask],
            reduce="mean",
            dim=0,
            dim_size=data.batch.max().item() + 1,
        ).mean()

    if name == "mae":
        return scatter(
            torch.abs(forces - data.y_force[mask]).sum(dim=-1),
            data.batch[mask],
            reduce="mean",
            dim=0,
            dim_size=data.batch.max().item() + 1,
        ).mean()

    if name == "p2":
        return torch.mean((forces - data.y_force[mask]).norm(p=2, dim=-1))

    if name == "p1":
        return torch.mean((forces - data.y_force[mask]).norm(p=1, dim=-1))

    raise Exception("Unknown training force")


class RMSE:
    def __call__(self, pred, target, batch):
        return (
            scatter(
                (pred - target).pow(2).sum(dim=-1),
                batch,
                reduce="mean",
                dim=0,
                dim_size=batch.max().item() + 1,
            )
            .sqrt()
            .mean()
        )


# Train / eval loops
def train(
    model,
    rank,
    device,
    train_loader,
    optimizer,
    criterion,
    training_force_loss,
    scheduler=None,
    energy_weight=1.0,
    force_weight=1.0,
    ema=None,
):
    model.train()
    total_energy_loss = 0.0
    total_force_loss = 0.0

    progress = (
        tqdm(train_loader, desc="Training", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
        if rank == 0
        else train_loader
    )

    for batch in progress:
        optimizer.zero_grad()

        data = batch.to(device, non_blocking=True)
        energies, forces, mask = model(data)

        energy_loss = criterion(energies, data.y)
        force_loss = compute_training_force_loss(forces, data, mask, training_force_loss)

        loss = energy_weight * energy_loss + force_weight * force_loss
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update()

        total_energy_loss += energy_loss.item()
        total_force_loss += force_loss.item()

        if rank == 0:
            progress.set_description(
                f"Training - Energy Loss: {energy_loss * _STD_ENERGY:.5f}, "
                f"Force Loss: {force_loss * _STD_FORCE_SCALE:.5f}"
            )

    n = len(train_loader)
    return total_energy_loss / n, total_force_loss / n


def evaluate(model, rank, device, loader, criterion, criterion_force, ema):
    model.eval()
    torch.cuda.empty_cache()

    total_energy_loss = 0.0
    total_force_loss = 0.0

    ctx = ema.average_parameters() if ema is not None else nullcontext()
    with ctx:
        progress = tqdm(loader, desc="Evaluating", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
        for batch in progress:
            data = batch.to(device, non_blocking=True)
            energies, forces, mask = model(data)

            energy_loss = criterion(energies, data.y)
            force_loss = criterion_force(forces, data.y_force[mask], data.batch[mask])

            total_energy_loss += energy_loss.item()
            total_force_loss += force_loss.item()

            if rank == 0:
                progress.set_description(
                    f"Evaluation - Energy Loss: {energy_loss * _STD_ENERGY:.5f}, "
                    f"Force Loss: {force_loss * _STD_FORCE_SCALE:.5f}"
                )

    n = len(loader)
    return total_energy_loss / n, total_force_loss / n



def main(
    root: str,
    model_name: str,
    stage: str,
    optimizer_name: str,
    batch_size: int,
    total_traj: bool = False,
    gradient_force: bool = False,
    data_parallel: bool = False,
    cutoff: float = 4.5,
    num_gaussians: int = 128,
    hidden_channels: int = 128,
    num_interactions: int = 4,
    lr: float = 1e-3,
    epochs: int = 100,
    weight_decay: float = 0.0,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    num_workers: int = 1,
    subset: bool = False,
    scheduler_name: str = "",
    training_force_loss: str = "rmse",
    resume_from_checkpoint: str = None,
    EMA: bool = False,
):
    # ---- device / DDP setup ----
    if data_parallel:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0

    # ---- logging ----
    if rank == 0:
        print(f"root: {root}")
        print(f"model_name: {model_name}")
        print(f"stage: {stage}")
        print(f"optimizer_name: {optimizer_name}")
        print(f"batch_size: {batch_size}")
        print(f"total_traj: {total_traj}")
        print(f"gradient_force: {gradient_force}")
        print(f"data_parallel: {data_parallel}")
        print(f"cutoff: {cutoff}")
        print(f"num_gaussians: {num_gaussians}")
        print(f"hidden_channels: {hidden_channels}")
        print(f"num_interactions: {num_interactions}")
        print(f"lr: {lr}")
        print(f"epochs: {epochs}")
        print(f"weight_decay: {weight_decay}")
        print(f"energy_weight: {energy_weight}")
        print(f"force_weight: {force_weight}")
        print(f"num_workers: {num_workers}")
        print(f"subset: {subset}")

        print(
            "#IN#",
            "#" * 80,
            f"Optimizer: {optimizer_name}, batch size: {batch_size}, lr: {lr}, epochs: {epochs}",
            f"Weight decay: {weight_decay}",
            "#" * 80,
            sep="\n",
        )

    # ---- data ----
    dataload = LMDBDataLoader(
        root,
        batch_size,
        num_workers=num_workers,
        train_val_ratio=(0.6, 0.2) if subset else (0.8, 0.1),
        stage=stage,
        total_traj=total_traj,
        subset=subset,
    )
    train_loader = dataload.train_loader(distributed=data_parallel)
    val_loader = dataload.val_loader(distributed=data_parallel)
    test_loader = dataload.test_loader(distributed=data_parallel)

    # ---- model ----
    name = model_name.lower()
    if name == "schnet":
        model = SchNet(
            num_gaussians=num_gaussians,
            hidden_channels=hidden_channels,
            num_interactions=num_interactions,
            cutoff=cutoff,
        )
    elif name == "painn":
        model = PaiNN(
            num_gaussians=num_gaussians,
            hidden_channels=hidden_channels,
            num_interactions=num_interactions,
            cutoff=cutoff,
        )
    elif name == "tdn":
        model = TensorDecompositionNetwork(
            num_gaussians=num_gaussians,
            hidden_channels=hidden_channels,
            num_interactions=num_interactions,
            cutoff=cutoff,
            gradient_force=gradient_force,
        )
    else:
        raise Exception("Invalid model name.")

    torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if rank == 0:
        print(f"#IN#Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    model = model.to(device)
    if data_parallel:
        model = DDP(model, device_ids=[rank])

    # ---- optimizer ----
    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = optimizer_name.lower()

    if opt_name == "adam":
        optimizer = Adam([{"params": params}], lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD([{"params": params}], lr=lr)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW([{"params": params}], lr=lr, weight_decay=weight_decay)
    else:
        raise Exception(f"Unknown optimizer name {optimizer_name}")

    # ---- scheduler ----
    sch_name = scheduler_name.lower()
    if sch_name == "reducelronplateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2)
    elif sch_name == "cyclic":
        scheduler = CyclicLR(optimizer=optimizer, base_lr=1e-6, max_lr=lr, step_size_up=7052 * 2)
    elif sch_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-4)
    else:
        scheduler = None

    # ---- EMA ----
    ema = ExponentialMovingAverage(params, decay=0.995) if EMA else None

    # ---- criteria / bookkeeping ----
    criterion = nn.L1Loss()
    criterion_force = RMSE()
    best_val_loss = 1e3
    start_epoch = 0

    # ---- resume ----
    if resume_from_checkpoint is not None:
        if rank == 0:
            print(f"Loading checkpoint from {resume_from_checkpoint}")

        map_location = {f"cuda:{0}": f"cuda:{rank}"}
        checkpoint = torch.load(resume_from_checkpoint, map_location=map_location)

        exclude = {"tp.cp_tp.cp_u", "tp.cp_tp.cp_v", "tp.cp_tp.cp_t"}
        filtered_state = {
            k: v for k, v in checkpoint["model_state_dict"].items() if not any(s in k for s in exclude)
        }

        if data_parallel:
            model.module.load_state_dict(filtered_state, strict=False)
        else:
            model.load_state_dict(filtered_state, strict=False)

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if ema is not None and "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        if rank == 0:
            print(f"Resuming from epoch {start_epoch}")

    # ---- train loop ----
    for epoch in range(start_epoch, epochs):
        per_epoch_scheduler = sch_name in {"reducelronplateau", "cosine", "cosine_decay"}

        if per_epoch_scheduler:
            train_energy_loss, train_force_loss_value = train(
                model=model,
                rank=rank,
                device=device,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                training_force_loss=training_force_loss,
                scheduler=None,
                energy_weight=energy_weight,
                force_weight=force_weight,
                ema=ema,
            )
        else:
            train_energy_loss, train_force_loss_value = train(
                model=model,
                rank=rank,
                device=device,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                training_force_loss=training_force_loss,
                scheduler=scheduler,
                energy_weight=energy_weight,
                force_weight=force_weight,
                ema=ema,
            )

        val_energy_loss, val_force_loss = evaluate(
            model=model,
            rank=rank,
            device=device,
            loader=val_loader,
            criterion=criterion,
            criterion_force=criterion_force,
            ema=ema,
        )

        if rank == 0:
            print(
                f"#IN#Epoch {epoch + 1}, "
                f"Train Energy Loss: {train_energy_loss * _STD_ENERGY:.5f}, "
                f"Val Energy Loss: {val_energy_loss * _STD_ENERGY:.5f}, "
                f"Train Force Loss: {train_force_loss_value * _STD_FORCE_SCALE:.5f}, "
                f"Val Force Loss: {val_force_loss * _STD_FORCE_SCALE:.5f}"
            )

        if sch_name == "reducelronplateau":
            if scheduler is not None:
                scheduler.step(val_energy_loss)
        elif sch_name in {"cosine", "cosine_decay"}:
            if scheduler is not None:
                scheduler.step()

        torch.cuda.empty_cache()

        test_energy_loss, test_force_loss = evaluate(
            model=model,
            rank=rank,
            device=device,
            loader=test_loader,
            criterion=criterion,
            criterion_force=criterion_force,
            ema=ema,
        )

        # ---- save best ----
        if val_energy_loss < best_val_loss and rank == 0:
            print(
                f"Saving model... "
                f"Test Energy loss: {test_energy_loss * _STD_ENERGY:.5f}, "
                f"Test Force loss: {test_force_loss * _STD_FORCE_SCALE:.5f}"
            )
            best_val_loss = val_energy_loss

            best_model = model.state_dict()
            if subset:
                torch.save(best_model, f"{model_name}_best_model_{total_traj}_{num_interactions}_{batch_size}_{lr}.pth")
            else:
                torch.save(
                    best_model,
                    f"{model_name}_best_model_full_{total_traj}_{num_interactions}_{batch_size}_{lr}.pth",
                )

        # ---- checkpoint each epoch (rank 0 only) ----
        if rank == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict() if data_parallel else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            if ema is not None:
                checkpoint["ema_state_dict"] = ema.state_dict()

            if subset:
                torch.save(checkpoint, f"{model_name}_checkpoint_{total_traj}_{num_interactions}_{batch_size}_{lr}.pth")
            else:
                torch.save(
                    checkpoint,
                    f"{model_name}_checkpoint_full_{total_traj}_{num_interactions}_{batch_size}_{lr}.pth",
                )

        if data_parallel:
            dist.barrier()

    # ---- final test ----
    final_test_energy_loss, final_test_force_loss = evaluate(
        model=model,
        rank=rank,
        device=device,
        loader=test_loader,
        criterion=criterion,
        criterion_force=criterion_force,
        ema=ema,
    )

    if rank == 0:
        print(
            f"#IN#Final Test Energy Loss: {final_test_energy_loss * _STD_ENERGY:.5f}, "
            f"Test Force Loss: {final_test_force_loss * _STD_FORCE_SCALE:.5f}"
        )

    if data_parallel:
        dist.destroy_process_group()


if __name__ == "__main__":
    tapify(main)
