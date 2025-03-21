# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os

from pytorch_lightning.cli import LightningCLI

from pdearena import utils
from pdearena.data.cond_datamodule import CondPDEDataModule
from pdearena.lr_scheduler import LinearWarmupCosineAnnealingLR  # noqa: F401
from pdearena.models.cond_pdemodel import CondPDEModel

logger = utils.get_logger(__name__)


def setupdir(path):
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "tb"), exist_ok=True)
    os.makedirs(os.path.join(path, "ckpts"), exist_ok=True)


class CondCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("data.pde.n_scalar_components", "model.pdeconfig.n_scalar_components")
        parser.link_arguments("data.pde.n_vector_components", "model.pdeconfig.n_vector_components")
        parser.link_arguments("data.pde.trajlen", "model.pdeconfig.trajlen")
        parser.link_arguments("data.pde.n_spatial_dim", "model.pdeconfig.n_spatial_dim")


def main():
    cli = CondCLI(
        CondPDEModel,
        CondPDEDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        parser_kwargs={"parser_mode": "omegaconf"},
    )
    if cli.trainer.default_root_dir is None:
        logger.warning("No default root dir set, using: ")
        cli.trainer.default_root_dir = os.environ.get("AMLT_OUTPUT_DIR", "./outputs")
        logger.warning(f"\t {cli.trainer.default_root_dir}")

    setupdir(cli.trainer.default_root_dir)
    logger.info(f"Checkpoints and logs will be saved in {cli.trainer.default_root_dir}")
    
    train_args = {"model":cli.model, "datamodule":cli.datamodule}
    if cli.trainer.resume_from_checkpoint:
        logger.info(f"Found ckpt to resume: {cli.trainer.resume_from_checkpoint}")
        train_args['ckpt_path'] = cli.trainer.resume_from_checkpoint
    
    logger.info("Starting training...")
    cli.trainer.fit(**train_args)
    if not cli.trainer.fast_dev_run:
        logger.info("Starting testing...")
        cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
