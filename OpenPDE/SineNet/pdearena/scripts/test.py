# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os

from pdearena import utils
from pdearena.data.datamodule import PDEDataModule
from pdearena.lr_scheduler import LinearWarmupCosineAnnealingLR  # noqa: F401
from pdearena.models.pdemodel import PDEModel

logger = utils.get_logger(__name__)


def setupdir(path):
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "tb"), exist_ok=True)
    os.makedirs(os.path.join(path, "ckpts"), exist_ok=True)


def main():
    cli = utils.PDECLI(
        PDEModel,
        PDEDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        parser_kwargs={"parser_mode": "omegaconf"},
    )
    # logger.info("Starting testing...")
    # cli.trainer.test(ckpt_path=cli.trainer.resume_from_checkpoint, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
