# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# RuntimeError: Pin memory thread exited unexpectedly
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (10_000, hard))
# >>> print(resource.getrlimit(resource.RLIMIT_NOFILE)) 
# (10000, 1048576)
# >>> subprocess.run("ulimit -n", shell=True)
# 10000

import os
from argparse import ArgumentParser
import subprocess
from copy import deepcopy
import json
import torch

NTHREADS = 2
torch.set_num_threads(NTHREADS)
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(NTHREADS)

# silence torchdata warning
import torchdata
torchdata.deprecation_warning = lambda: None

from pdearena.utils import utils
from pdearena.data.datamodule import PDEDataModule
from pdearena.models.registry import MODEL_REGISTRY
from pdearena.pl_models.pdemodel import PDEModel
from pdearena.pl_models.registry import MODELS
from pdearena.configs.registry import (
    get_config, 
    config_str, 
    validate_config
)
from pdearena.configs.config import Config, PretrainedCFDConfig
from pdearena.utils.callbacks import (
    ParamCounter,
    StepCounter,
    StepTimer,
    GradNormCallback,
    GPUStatsCallback
)

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import (
    Timer,
    RichModelSummary,
    LearningRateMonitor,
    TQDMProgressBar,
    ModelCheckpoint
)

logger = utils.get_logger(__name__)


def main(args):

    if args.mode in ["predict", "test"] and args.config is None:
        assert args.ckpt_path is not None
        ckpt_path = args.ckpt_path
        directories = ckpt_path.split(os.path.sep)[::-1]
        while len(directories) > 0:
            config = directories.pop()
            try:
                get_config(config, verbose=False)
                break
            except:
                assert len(directories) > 0
        args.config = config
    else:
        assert args.config is not None

    config: Config = get_config(args.config)
    if isinstance(config, PretrainedCFDConfig):
        assert args.mode == "test"
        if config.solver_ckpt is None:
            assert args.ckpt_path is not None
            config.solver_ckpt = args.ckpt_path
            config.solver_cfg = config.cfg_from_ckpt(config.solver_ckpt)
            args.ckpt_path = None
            assert validate_config(config.cfl_cfg), f"Invalid cfl config {config.cfl_cfg}"
            assert validate_config(config.solver_cfg), f"Invalid solver config {config.solver_cfg}"
        else:
            assert args.ckpt_path is None
    logger.info(f"Using config: {args.config}\n{config_str(config)}")
    if args.seed is not None:
        logger.info(f"Overriding config seed {config.seed}--->{args.seed}")
        config.seed = args.seed
    seed_everything(config.seed)

    model: PDEModel = MODELS[config.plmodel](args=config)
    datamodule = PDEDataModule(args=config)
    should_log = True

    if args.debug:
        config.num_workers = 0
        config.epochs = 1

    if args.debug or args.nolog:    
        config.save_dir = os.path.join(config.save_dir, "DEBUG")

    if args.mode in ["predict", "test"]:
        # config.num_workers = 0
        should_log = False
        config.devices = 1
        config.save_dir = os.path.join(config.save_dir, args.mode)

    if args.mode == "test" and hasattr(config, "solver_ckpt"):
        ckpts_dir = os.path.join("", "ckpts", "")
        model_dir = config.solver_ckpt.split(ckpts_dir)
        assert len(model_dir) == 2, model_dir
        save_dir = model_dir[0]
    else:
        num_commits_master = int(subprocess.run(['git', 'rev-list', '--count', 'master'], capture_output=True, text=True).stdout.strip())
        num_commits_current = int(subprocess.run(['git', 'rev-list', '--count', 'HEAD'], capture_output=True, text=True).stdout.strip()) - num_commits_master
        curr_branch_name = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True).stdout.strip()
        save_dir = os.path.join(config.save_dir, f"master-{num_commits_master}", f"{curr_branch_name}-{num_commits_current}", args.config)
        if args.mode != "train":
            save_dir += f"_{args.mode}"
    if not os.path.exists(save_dir) and should_log:
        os.makedirs(save_dir)
    ckpt_path = os.path.join(save_dir, "ckpts")
    if not os.path.exists(ckpt_path) and should_log:
        os.makedirs(ckpt_path)

    if should_log:
        trainer_logger = TensorBoardLogger(save_dir=save_dir, name="tb")
        version = trainer_logger.version
        arg_dir = os.path.join(save_dir, "args", f"version_{version}")
        if not os.path.exists(arg_dir):
            os.makedirs(arg_dir)
        args_file = os.path.join(arg_dir, f"{args.config}.json")
        with open(args_file, 'w') as f:
            f.write(config.model_dump_json(indent=4))
        model_args_file = os.path.join(arg_dir, f"{config.model}.json")
        assert config.model in MODEL_REGISTRY, f"Model {config.model} not found in registry."
        with open(model_args_file, 'w') as f:
            model_cfg = deepcopy(MODEL_REGISTRY[config.model])
            model_cfg["init_args"]["args"] = model_cfg["init_args"]["args"].model_dump()
            json.dump(model_cfg, f, indent=4)
    else:
        # ckpt_callback = False
        trainer_logger = False

    ckpt_callback = ModelCheckpoint(
            monitor=config.checkpoint_monitor,
            save_last=True,
            dirpath=ckpt_path,
            filename=f'epoch={{epoch}}-val={{{config.checkpoint_monitor}:.4f}}',
            auto_insert_metric_name=False
    )
    callbacks = [
        Timer(interval="epoch"), 
        RichModelSummary(max_depth=-1), 
        LearningRateMonitor(logging_interval="step"), 
        TQDMProgressBar(), 
        ckpt_callback,
        ParamCounter(),
        StepCounter(),
        StepTimer(),
        GPUStatsCallback(),
        GradNormCallback(),
    ]
    
    trainer_args = dict(
        logger=trainer_logger,
        callbacks=callbacks,
        default_root_dir=save_dir,
        gradient_clip_val=config.gradient_clip_val,
        gradient_clip_algorithm=config.gradient_clip_algorithm,
        devices=config.devices,
        max_epochs=config.epochs,
        min_epochs=1,
        accelerator="gpu",
        strategy=config.strategy,
        deterministic=False,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        num_sanity_val_steps=1
    )
    if args.debug:
        limit_batches = 2
        trainer_args.update(
            dict(
                limit_train_batches=limit_batches,
                limit_val_batches=limit_batches,
                limit_test_batches=limit_batches,
                log_every_n_steps=limit_batches,
                devices=1,
                num_sanity_val_steps=1,
                )
            )

    trainer = Trainer(**trainer_args)

    match args.mode:
        case "train":
            trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
            if not trainer.fast_dev_run:
                logger.info("Starting testing...")
                trainer.test(model=model, datamodule=datamodule)
        case "predict":
            assert args.ckpt_path is not None
            model.predict_train = args.predict_train
            predictions = trainer.predict(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
            pred_path = args.ckpt_path.replace(".ckpt", ".preds")
            torch.save(predictions, pred_path)
        case "test":
            trainer.test(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
            results = dict(config=config.model_dump())
            results.update(results={k: v.item() for k, v in trainer.callback_metrics.items()})
            result_dir = os.path.join(save_dir, "results")
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            result_file = os.path.join(result_dir, f"results.json")
            logger.info(f"Saving results to {result_file}")
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=4)
            if hasattr(model, "predictions"):                    
                pred_path = os.path.join(result_dir, f"predictions.pt")
                logger.info(f"Saving predictions to {pred_path}")
                torch.save(model.predictions, pred_path)
        case _:
            raise ValueError(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--nolog", action="store_true")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--predict-train", action="store_true")
    parser.add_argument("--ckpt-path", type=str, default=None)
    args = parser.parse_args()
    main(args)
