import os
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import Logger

from nablaDFT import model_registry
from nablaDFT.dataset import ASENablaDFT
from nablaDFT.optimization import BatchwiseOptimizeTask
from nablaDFT.utils import (
    check_cfg_parameters,
    close_loggers,
    load_from_checkpoint,
    seed_everything,
    set_additional_params,
    write_predictions_to_db,
)
import torch
from collections import OrderedDict


def predict(
    trainer: Trainer,
    model: LightningModule,
    datamodule: LightningDataModule,
    ckpt_path: str,
    model_name: str,
    output_dir: str,
):
    """Function for prediction loop execution.

    NOTE: experimental function, may be changed in the future.

    Saves model prediction to "predictions" directory.
    """
    trainer.logger = False  # need this to disable save_hyperparameters during predict,
    # otherwise OmegaConf DictConf can't be dumped to YAML
    output_db_path = Path(output_dir) / f"{model_name}_{datamodule.dataset_name}.db"
    if isinstance(datamodule, ASENablaDFT):
        input_db_path = Path(datamodule.datapath).parent / f"raw/{datamodule.dataset_name}.db"
    else:
        input_db_path = Path(datamodule.root) / f"raw/{datamodule.dataset_name}.db"
    predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    write_predictions_to_db(input_db_path, output_db_path, predictions)


def optimize(config: DictConfig, model: LightningModule):
    """Function for batched molecules optimization.
    Uses model defined in config.

    Args:
        config (DictConfig): config for task. see r'config/' for examples.
        model (LightningModule): instantiated lightning model.
    """
    output_dir = config.get("output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_datapath = os.path.join(output_dir, f"{config.name}_{config.dataset_name}.db")
    # append to existing database not supported
    if os.path.exists(output_datapath):
        os.remove(output_datapath)
    if config.ckpt_path:
        model = load_from_checkpoint(config)
    calculator = hydra.utils.instantiate(config.calculator, model)
    optimizer = hydra.utils.instantiate(config.optimizer, calculator)
    task = BatchwiseOptimizeTask(
        config.input_db,
        output_datapath,
        optimizer=optimizer,
        batch_size=config.batch_size,
        fmax=config.fmax,
        steps=config.steps,
    )
    task.run()


def run(config: DictConfig):
    """Main function to perform task runs on nablaDFT datasets.
    Refer to r'nablaDFT/README.md' for detailed description of run configuration.

    Args:
        config (DictConfig): config for task. see r'config/' for examples.
    """
    check_cfg_parameters(config)
    if config.get("seed"):
        seed_everything(config.seed)
    job_type = config.job_type
    if config.get("ckpt_path"):
        config.ckpt_path = Path(hydra.utils.get_original_cwd()) / config.get("ckpt_path")
    else:
        config.ckpt_path = None
    config = set_additional_params(config)
    # download checkpoint if pretrained
    # if config.get("pretrained"):
    #     model: LightningModule = model_registry.get_pretrained_model("lightning", config.pretrained)
    # else:
    #     model: LightningModule = hydra.utils.instantiate(config.model)
    
    model: LightningModule = hydra.utils.instantiate(config.model)
    
    if config.pretrained_foundation:
        print("load pretrained foundation model")
        # load pretained foundation model checkpoint
        checkpoint = torch.load(config.pretrained_foundation)
        # model.model.load_state_dict(checkpoint)
        
        # Remove "module." prefix from keys
        new_state_dict = OrderedDict()
        for key, value in checkpoint.items():
            new_key = key.replace("module.", "")  # Remove "module." prefix
            new_state_dict[new_key] = value
        model.model.load_state_dict(new_state_dict)
    
    if job_type == "optimize":
        return optimize(config, model)
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
    # Datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    if job_type == "train":
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=config.ckpt_path)
    elif job_type == "test":
        trainer.test(model=model, datamodule=datamodule, ckpt_path=config.ckpt_path)
    elif job_type == "predict":
        predict(trainer, model, datamodule, config.ckpt_path, config.model.model_name, config.output_dir)
    # Finalize
    close_loggers(
        logger=loggers,
    )
