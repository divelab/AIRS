import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig

import os
from pathlib import Path
from typing import List

from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)

from pytorch_lightning.loggers import Logger

from utils import (
    init_wandb,
    check_cfg_parameters,
    close_loggers,
    load_from_checkpoint,
    seed_everything,
    set_additional_params,
    TransformOptim3D,
    TransformRDKit3D
)

from collections import OrderedDict

    
def run(config: DictConfig):

    check_cfg_parameters(config)
    if config.get("seed"):
        seed_everything(config.seed)
    job_type = config.job_type
    # import pdb; pdb.set_trace()
    if config.get("ckpt_path"):
        config.ckpt_path = Path(hydra.utils.get_original_cwd()) / config.get("ckpt_path")
    else:
        config.ckpt_path = None
    config = set_additional_params(config)

    model: LightningModule = hydra.utils.instantiate(config.model)
    
    # import pdb; pdb.set_trace()
    
    if config.finetune_type is not None:
        if config.finetune_type == 'property':
            # load pretained foundation model checkpoint
            checkpoint = torch.load(config.pretrained_foundation, map_location='cpu')
            # model.model.load_state_dict(checkpoint)
            
            if config.model.model_name == 'PaiNN':
                new_state_dict = checkpoint['model_state_dict']
            else:
                raise ValueError('model_name not recognized')
                
            if config.finetune_layer == 'head':
                model.model.backbone.load_state_dict(new_state_dict)
            elif config.finetune_layer == 'all':
                model.model.load_state_dict(new_state_dict, strict=False)
            else:
                raise ValueError('finetune_layer not recognized')
    
        elif config.finetune_type == 'geometry':
            checkpoint = torch.load(config.pretrained_downstream, map_location='cpu')
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            assert config.conformer != "GT"
            print(f"Finetune downstream model on {config.conformer} conformer !!")
            # model.load_from_checkpoint(config.pretrained_downstream)
        else:
            raise ValueError('finetune_type not recognized')
    
    # import pdb; pdb.set_trace()
    
    # Callbacks
    callbacks: List[Callback] = []
    for _, callback_cfg in config.callbacks.items():
        callbacks.append(hydra.utils.instantiate(callback_cfg))
    # Loggers
    loggers: List[Logger] = []
    for _, logger_cfg in config.loggers.items():
        loggers.append(hydra.utils.instantiate(logger_cfg))
    # Trainer
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=loggers)
    
    transform = None
    pre_transform = None
    
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule,
        transform=transform,
        pre_transform=pre_transform
    )
    
    if job_type == "train":
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=config.ckpt_path)
    elif job_type == "test":
        trainer.test(model=model, datamodule=datamodule, ckpt_path=config.ckpt_path)

    # Finalize
    close_loggers(
        logger=loggers,
    )

@hydra.main(config_path="./config", config_name=None, version_base="1.2")
def main(config: DictConfig):
    init_wandb()
    run(config)
    

if __name__ == "__main__":
    main()
