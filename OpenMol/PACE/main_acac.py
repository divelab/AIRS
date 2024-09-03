import torch
from torch_geometric.loader import DataLoader
from torch_ema import ExponentialMovingAverage

import logging
import numpy as np
from prettytable import PrettyTable

from config import build_arg_parser
from dataset.PygAcAc import PygAcAc
from dataset.Preprocess_AcAc import PreprocessAcAc
from eval import evaluate
from loss import WeightedEnergyForcesLoss
from train import train
from utils import *
from model.get_model import get_model


def main():
    ### Setup
    args = build_arg_parser().parse_args()
    setup_seed(args.seed)
    setup_logger(args.output_dir)
    logging.info(f"Configuration: {args}")

    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    ### Load data
    dataset = PygAcAc(
        root="./dataset/acac/pyg_data/", name="train_300K"
    )
    split_idx = dataset.get_idx_split(
        len(dataset.data.y), valid_fraction=0.1, seed=args.seed
    )
    train_dataset = PreprocessAcAc(dataset[split_idx["train"]], args.cutoff)
    valid_dataset = PreprocessAcAc(dataset[split_idx["valid"]], args.cutoff)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
    )

    test_dataset_1 = PygAcAc(
        root="./dataset/acac/pyg_data/", name="test_MD_300K"
    )
    test_dataset_1 = PreprocessAcAc(test_dataset_1, args.cutoff)
    test_loader_1 = DataLoader(
        test_dataset_1, batch_size=args.val_batch_size, shuffle=False, drop_last=False
    )
    test_dataset_2 = PygAcAc(
        root="./dataset/acac/pyg_data/", name="test_MD_600K"
    )
    test_dataset_2 = PreprocessAcAc(test_dataset_2, args.cutoff)
    test_loader_2 = DataLoader(
        test_dataset_2, batch_size=args.val_batch_size, shuffle=False, drop_last=False
    )

    statistics = {
        "z_table": train_dataset.z_table,
        "average_energy": train_dataset.average_energy,
        "avg_num_neighbors": train_dataset.avg_num_neighbors,
        "std": train_dataset.std,
        "mean": train_dataset.mean,
    }
    logging.info(f"Statistics: {statistics}")

    # Build model & optimzer & scheduler
    model = get_model(args, statistics, device)
    logging.info(f"Number of parameters: {count_parameters(model)}")

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.lr, amsgrad=True
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=args.lr_factor,
        patience=args.scheduler_patience,
    )

    # Continue run
    if args.continue_run:
        start_epoch, _ = load_checkpoint(
            model, optimizer, scheduler, args.output_dir, device, "checkpoint.pt"
        )
        logging.info(f"Loaded model from epoch {start_epoch} from {args.output_dir}")
    else:
        start_epoch = -1

    # EMA & loss
    ema = ExponentialMovingAverage(model.parameters(), decay=0.99)
    loss_fn = WeightedEnergyForcesLoss(
        energy_weight=args.energy_weight, forces_weight=args.force_weight
    )

    ### Training & validation
    logging.info("Started training")
    lowest_loss, valid_loss = np.inf, np.inf
    patience_counter = 0
    for epoch in range(1 + start_epoch, args.epochs):
        train_metrics = train(model, train_loader, loss_fn, optimizer, ema, device)

        if epoch % args.eval_interval == 0:
            eval_metrics = evaluate(model, valid_loader, loss_fn, ema, device)
            valid_loss = eval_metrics["loss"]
            error_e = eval_metrics["rmse_e"] * 1e3
            error_f = eval_metrics["rmse_f"] * 1e3
            lr_cur = optimizer.param_groups[0]["lr"]
            logging.info(
                f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E={error_e:.2f} meV, RMSE_F={error_f:.2f} meV / A, lr_cur={lr_cur:.5f}"
            )

        if epoch % 500 == 0:
            save_checkpoint(
                epoch,
                model,
                optimizer,
                scheduler,
                lowest_loss,
                args.output_dir,
                device,
                ema,
                name="epoch_{}.pt".format(epoch),
            )
        
        save_checkpoint(
            epoch,
            model,
            optimizer,
            scheduler,
            lowest_loss,
            args.output_dir,
            device,
            ema,
        )

    ### Testing
    logging.info("Computing metrics for training, validation, and test sets")

    epoch, _ = load_checkpoint(model, optimizer, scheduler, args.output_dir, device)
    logging.info(f"Loaded model from epoch {epoch}")

    table = PrettyTable()
    table.field_names = [
        "config_type",
        "RMSE E / meV",
        "RMSE F / meV / A",
    ]

    test_metrics_1 = evaluate(model, test_loader_1, loss_fn, None, device)
    test_metrics_2 = evaluate(model, test_loader_2, loss_fn, None, device)

    add_row_3bpa(table, test_metrics_1, "300K")
    add_row_3bpa(table, test_metrics_2, "600K")
    logging.info("\n" + str(table))


if __name__ == "__main__":
    main()
