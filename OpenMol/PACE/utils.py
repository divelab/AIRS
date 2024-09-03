import os
import sys
import logging
import shutil
import random
import torch
import numpy as np
from datetime import datetime


def save_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    lowest_loss,
    output_dir,
    device,
    ema,
    name="checkpoint.pt",
):
    if ema is not None:
        with ema.average_parameters():
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.cpu().state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "lowest_val_loss": lowest_loss,
            }
            torch.save(checkpoint, os.path.join(output_dir, name))
            model.to(device)
    else:
        checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.cpu().state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "lowest_val_loss": lowest_loss,
            }
        torch.save(checkpoint, os.path.join(output_dir, name))
        model.to(device)


def load_checkpoint(
    model, optimizer, scheduler, output_dir, device, name="checkpoint.pt"
):
    checkpoint = torch.load(os.path.join(output_dir, name))
    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    lowest_loss = checkpoint["lowest_val_loss"]
    return epoch, lowest_loss


def setup_logger(directory):
    logger = logging.getLogger()
    logger.setLevel("INFO")

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if directory is not None:
        os.makedirs(name=directory, exist_ok=True)
        path = os.path.join(directory, "PCAE_run.log")
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)

        logger.addHandler(fh)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_row(table, metrics, name):
    table.add_row(
        [
            name,
            f"{metrics['mae_e'] * 1e3:.2f}",
            f"{metrics['mae_f'] * 1e3:.2f}",
        ]
    )

def add_row_3bpa(table, metrics, name):
    table.add_row(
        [
            name,
            f"{metrics['rmse_e'] * 1e3:.2f}",
            f"{metrics['rmse_f'] * 1e3:.2f}",
        ]
    )


def make_output_dirs(output_dir='outputs'):
    date = datetime.now().strftime('%Y-%m-%d')
    time = datetime.now().strftime('%H-%M-%S')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    current_dir = os.path.join(output_dir, date)
    if not os.path.isdir(current_dir):
        os.makedirs(current_dir)
    current_dir = os.path.join(current_dir, time)
    if not os.path.isdir(current_dir):
        os.makedirs(current_dir)
    return current_dir


def save_code():
    root_path = os.getcwd()
    output_dir = make_output_dirs(os.path.join(root_path, 'code_copy'))
    shutil.copytree(os.path.join(root_path, 'model/'), os.path.join(output_dir, 'model/'))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
