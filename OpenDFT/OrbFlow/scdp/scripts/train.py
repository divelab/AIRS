import hashlib
import logging
import json
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

import hydra
import lightning.pytorch as pl
import omegaconf
import torch
import wandb
from lightning.pytorch import Callback
from omegaconf import DictConfig, ListConfig

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from scdp.common.system import log_hyperparameters, PROJECT_ROOT
from scdp.common.checkpoint_utils import (
    load_weights_only_checkpoint,
    resolve_fit_checkpoint,
    save_fit_end_checkpoint,
    save_weights_only_checkpoint,
)

# NOTE: disable slurm detection of lightning
from lightning.pytorch.plugins.environments import SLURMEnvironment
SLURMEnvironment.detect = lambda: False

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

def build_callbacks(cfg: ListConfig, *args: Callback) -> List[Callback]:
    """Instantiate the callbacks given their configuration.

    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated

    Returns:
        the complete list of callbacks to use
    """
    callbacks: List[Callback] = list(args)

    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


def stable_wandb_run_id(expname: str, length: int = 8) -> str:
    """Deterministic W&B run id from experiment name (stable across Slurm resubmits)."""
    return hashlib.sha256(expname.encode("utf-8")).hexdigest()[:length]


def is_global_zero() -> bool:
    """True on the process that should own W&B (DDP rank 0 or single-GPU)."""
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"]) == 0
    return True


def _shutdown_after_fit(trainer: pl.Trainer, logger) -> None:
    """
    Tear down DDP/W&B so multi-phase Slurm scripts can start the next train.py.

    Finish W&B only after destroy_process_group(). Calling experiment.finish()
    while other ranks block on a dist barrier causes multi-hour NCCL hangs.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception as exc:
            pylogger.warning("pre-teardown barrier failed: %s", exc)

    try:
        trainer.strategy.teardown()
    except Exception as exc:
        pylogger.warning("strategy teardown failed: %s", exc)

    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as exc:
            pylogger.warning("destroy_process_group failed: %s", exc)

    if logger is not None and trainer.is_global_zero:
        try:
            logger.experiment.finish()
        except Exception as exc:
            pylogger.warning("W&B finalize failed: %s", exc)


def _has_training_checkpoint(storage_dir: Path) -> bool:
    if (storage_dir / "last.ckpt").exists():
        return True
    return bool(list(storage_dir.glob("*epoch*.ckpt")))


def _read_persisted_wandb_id(storage_dir: Path, expname: str) -> Optional[str]:
    id_path = storage_dir / "wandb_id.txt"
    if not id_path.is_file():
        return None
    run_id = id_path.read_text().strip()
    if not run_id or run_id.lower() == "none":
        pylogger.warning(
            "Ignoring invalid wandb_id.txt (%r); using stable id for %s",
            run_id,
            expname,
        )
        return None
    return run_id


def _persist_wandb_id(storage_dir: Path, run_id: str) -> None:
    if not run_id or str(run_id).lower() == "none":
        return
    (storage_dir / "wandb_id.txt").write_text(str(run_id))


def resolve_wandb_logger_kwargs(
    wandb_cfg: DictConfig, storage_dir: str, expname: str
) -> Dict[str, Any]:
    """
    Build WandbLogger kwargs so Slurm restarts continue the same W&B run/curves.

    - Stable ``id`` from ``wandb_id.txt`` or hash of ``core.expname``.
    - ``resume=allow`` when ``wandb_id.txt`` exists (continue if the remote run
      exists; otherwise start fresh with the same id).
    - ``resume=never`` on a fresh run, including finetune warm-starts that only
      stage ``last.ckpt`` without an existing W&B run for this experiment.
    """
    storage = Path(storage_dir)
    cfg: Dict[str, Any] = dict(omegaconf.OmegaConf.to_container(wandb_cfg, resolve=True))

    run_id = _read_persisted_wandb_id(storage, expname)
    if run_id is None and cfg.get("id"):
        cand = str(cfg["id"]).strip()
        if cand and cand.lower() != "none":
            run_id = cand
    if run_id is None:
        run_id = stable_wandb_run_id(expname)

    cfg["id"] = run_id
    # Always "allow": stable ids can already exist on W&B from a prior partial
    # init (resume=never then fails with UsageError). "allow" continues if the
    # remote run exists and creates it if not — correct for Slurm restarts.
    cfg["resume"] = "allow"
    cfg["save_dir"] = str(storage / "wandb")
    cfg["settings"] = wandb.Settings(init_timeout=300)
    return cfg


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    if cfg.train.deterministic:
        seed_everything(cfg.train.seed)

    # Instantiate datamodule
    pylogger.info(f"Instantiating <{cfg.data['_target_']}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data, _recursive_=False
    )
    datamodule.setup(stage="fit")
        
    metadata = getattr(datamodule, "metadata", None)
    if metadata is None:
        pylogger.warning(
            f"No 'metadata' attribute found in datamodule <{datamodule.__class__.__name__}>"
        )

    # Instantiate model
    pylogger.info(f"Instantiating <{cfg.model['_target_']}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model, train=cfg.train, _recursive_=False, metadata=metadata,
    )

    storage_dir: str = cfg.core.storage_dir

    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks)
    # DDP workers use logger=False; LearningRateMonitor requires a logger.
    if not is_global_zero():
        callbacks = [
            cb for cb in callbacks if not isinstance(cb, LearningRateMonitor)
        ]
    from scdp.model.flow_loss_spike_callback import FlowLossSpikeCallback

    spike_cb = FlowLossSpikeCallback.from_env(storage_dir)
    if spike_cb is not None:
        pylogger.info(
            "Adding <FlowLossSpikeCallback> threshold_abs=%s spike_factor=%s",
            spike_cb.threshold_abs,
            spike_cb.spike_factor,
        )
        callbacks.append(spike_cb)

    logger = None
    if "wandb" in cfg.train.logging:
        if is_global_zero():
            Path(storage_dir).mkdir(parents=True, exist_ok=True)
            wandb_kwargs = resolve_wandb_logger_kwargs(
                cfg.train.logging.wandb, storage_dir, cfg.core.expname
            )
            pylogger.info(
                "Instantiating <WandbLogger> id=%s resume=%s name=%s save_dir=%s",
                wandb_kwargs.get("id"),
                wandb_kwargs.get("resume"),
                wandb_kwargs.get("name"),
                wandb_kwargs.get("save_dir"),
            )
            logger = WandbLogger(**wandb_kwargs)
            watch_log = cfg.train.logging.wandb_watch.log
            if watch_log is None or str(watch_log).lower() in ("null", "none", ""):
                pylogger.info(
                    "Skipping wandb.watch (train.logging.wandb_watch.log=null); "
                    "loss/nMAPE metrics still log to W&B."
                )
            else:
                pylogger.info(f"W&B is now watching <{watch_log}>!")
                logger.watch(
                    model,
                    log=watch_log,
                    log_freq=cfg.train.logging.wandb_watch.log_freq,
                )
            _persist_wandb_id(Path(storage_dir), str(wandb_kwargs["id"]))
        else:
            pylogger.info(
                "Skipping WandbLogger on rank %s (only global rank 0 logs to W&B)",
                os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?")),
            )
    else:
        if is_global_zero():
            logger = TensorBoardLogger(**cfg.train.logging.tensorboard)
            pylogger.info(
                f"TensorBoard Logger logs into <{cfg.train.logging.tensorboard.save_dir}>."
            )

    ckpt = None
    weights_only_init = False

    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        logger=logger if is_global_zero() else False,
        callbacks=callbacks,
        **cfg.train.trainer,
    )

    # save the config yaml file.
    yaml_conf: str = omegaconf.OmegaConf.to_yaml(cfg)
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    (Path(storage_dir) / "config.yaml").write_text(yaml_conf)
    log_hyperparameters(cfg, model, trainer)
    with open(Path(storage_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    weights_only_on_cold_start = bool(
        omegaconf.OmegaConf.select(cfg, "train.restore.weights_only", default=False)
    )
    ckpt, weights_only_init = resolve_fit_checkpoint(
        storage_dir, weights_only_on_cold_start=weights_only_on_cold_start
    )

    if cfg.model.expo_trainable:
        if ckpt is not None:
            model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)
        ckpt = None
        weights_only_init = False
    elif weights_only_init and ckpt is not None:
        load_weights_only_checkpoint(model, ckpt)
        ckpt = None

    pylogger.info("starting training.")
    trainer.fit(
        model, datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=ckpt
    )

    save_fit_end_checkpoint(trainer, Path(storage_dir))

    run_post_fit_test = bool(
        omegaconf.OmegaConf.select(cfg, "train.run_post_fit_test", default=True)
    )
    if not run_post_fit_test:
        pylogger.info(
            "Skipping post-fit test (train.run_post_fit_test=false); "
            "use scdp/scripts/test_flow.py for held-out evaluation."
        )
    elif (
        datamodule.test_dataset is not None
        and trainer.checkpoint_callback.best_model_path is not None
    ):
        pylogger.info("starting testing.")
        trainer.test(dataloaders=[datamodule.test_dataloader()])

    _shutdown_after_fit(trainer, logger)

@hydra.main(config_path=str(PROJECT_ROOT / "scdp" / "config"), config_name="default", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    # pylint: disable=E1120
    main()
