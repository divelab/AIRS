import glob
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import hydra.utils
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities import rank_zero_only

JOB_TYPES = ["train", "test", "predict"]
logger = logging.getLogger()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def close_loggers(
    logger: List,
) -> None:
    """Makes sure everything closed properly."""
    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


def load_envs():
    envs = glob.glob("*.env")
    for env in envs:
        load_dotenv(env)


@rank_zero_only
def init_wandb():
    wandb.login()


def set_additional_params(config: DictConfig) -> DictConfig:
    datamodule_cls = config.datamodule._target_

    # if config.name in ["SchNet", "PaiNN", "DimeNet++"]:
    #     with open_dict(config):
    #         config.trainer.inference_mode = False
    if len(config.devices) > 1:
        with open_dict(config):
            strategy_cfg = OmegaConf.create({"_target_": "pytorch_lightning.strategies.ddp.DDPStrategy"})
            config.trainer.strategy = strategy_cfg

    # config.trainer.find_unused_parameters = True    
        
    return config


def check_cfg_parameters(cfg: DictConfig):
    if cfg.job_type not in JOB_TYPES:
        raise ValueError(f"job_type must be one of {JOB_TYPES}, got {cfg.job_type}")
    # if cfg.pretrained and cfg.ckpt_path:
    #     raise ValueError(
    #         "Config parameters ckpt_path and pretrained are mutually exclusive. Consider set ckpt_path "
    #         "to null, if you plan to use pretrained checkpoints."
    #     )



def load_from_checkpoint(config: DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(config.model)
    ckpt = torch.load(config.ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    logger.info(f"Restore model weights from {config.ckpt_path}")
    model.eval()
    return model