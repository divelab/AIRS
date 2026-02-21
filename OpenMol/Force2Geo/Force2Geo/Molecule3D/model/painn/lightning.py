import logging
from typing import Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter, segment_coo



class PaiNNLightning(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        losses: Dict,
        metric,
        loss_coefs,
        target_index,
        auxilary_task: bool = False,
        mix_GT_relax: bool = False,
    ) -> None:
        super(PaiNNLightning, self).__init__()
        self.model = model
        self.save_hyperparameters(logger=True, ignore=["net"])
        
        # import pdb; pdb.set_trace()

    def forward(self, data):
        out = self.model(data)
        return out

    def step(self, batch, calculate_metrics: bool = False):
        # y = batch.y[:, self.hparams.target_index]
        bsz = self._get_batch_size(batch)
        y = batch.props.view(bsz, -1)[:, self.hparams.target_index]
        # make dense batch from PyG batch
        out = self.model(batch)
        if self.hparams.auxilary_task:
            if self.hparams.mix_GT_relax:
                if len(out) == 3:
                    target_displacement = out[2] # noise
                else:
                    target_displacement = batch.xyz - batch.relaxed_xyz
                preds = {"out": out[0], "position": out[1]}
                target = {"out": y, "position": target_displacement}
            else:
                preds = {"out": out[0], "position": out[1]}
                target = {"out": y, "position": batch.xyz - batch.relaxed_xyz}
        else:
            preds = {"out": out}
            target = {"out": y}
        loss = self._calculate_loss(preds, target)
        if calculate_metrics:
            metrics = self._calculate_metrics(preds, target)
            return loss, metrics
        return loss

    def training_step(self, batch, batch_idx):
        bsz = self._get_batch_size(batch)
        loss = self.step(batch, calculate_metrics=False)
        self._log_current_lr()
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=bsz,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        bsz = self._get_batch_size(batch)
        loss, metrics = self.step(batch, calculate_metrics=True)
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=bsz,
        )
        # workaround for checkpoint callback
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=False,
            sync_dist=True,
            batch_size=bsz,
        )
        return loss

    def test_step(self, batch, batch_idx):
        bsz = self._get_batch_size(batch)
        loss, metrics = self.step(batch, calculate_metrics=True)
        self.log(
            "test/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=bsz,
        )
        return loss

    def predict_step(self, data, **kwargs):
        out = self(data)
        return out

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.lr_scheduler is not None:
            scheduler = self.hparams.lr_scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def on_fit_start(self) -> None:
        self._check_devices()

    def on_test_start(self) -> None:
        self._check_devices()

    def on_validation_epoch_end(self) -> None:
        self._reduce_metrics(step_type="val")

    def on_test_epoch_end(self) -> None:
        self._reduce_metrics(step_type="test")

    def _calculate_loss(self, y_pred, y_true) -> float:
        total_loss = 0.0
        for name, loss in self.hparams.losses.items():
            total_loss += self.hparams.loss_coefs[name] * loss(y_pred[name], y_true[name])
        return total_loss

    def _calculate_metrics(self, y_pred, y_true) -> Dict:
        """Function for metrics calculation during step."""
        metric = self.hparams.metric(y_pred, y_true)
        return metric

    def _log_current_lr(self) -> None:
        opt = self.optimizers()
        current_lr = opt.optimizer.param_groups[0]["lr"]
        self.log("LR", current_lr, logger=True)

    def _reduce_metrics(self, step_type: str = "train"):
        metric = self.hparams.metric.compute()
        for key in metric.keys():
            self.log(
                f"{step_type}/{key}",
                metric[key],
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            
            # workaround for checkpoint callback
            self.log(
                f"{step_type}_{key}",
                metric[key],
                logger=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            
        self.hparams.metric.reset()

    def _check_devices(self):
        self.hparams.metric = self.hparams.metric.to(self.device)

    def _get_batch_size(self, batch):
        """Function for batch size infer."""
        bsz = batch.batch.max().detach().item() + 1  # get batch size
        return bsz
