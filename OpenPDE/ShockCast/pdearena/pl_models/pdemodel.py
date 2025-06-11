# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from abc import ABC, abstractmethod
from typing import Dict, List

import torch
from torchmetrics import MetricCollection
from torch_geometric.data import Batch

from pytorch_lightning import LightningModule
from pytorch_lightning.cli import instantiate_class

from pdearena.utils import utils
from pdearena.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from pdearena.utils.loss import ScaledLpLoss, MSELoss, Correlation
from pdearena.configs.config import Config
from pdearena.utils.metrics import ErrorStats
from pdearena.utils.normalizer import Normalizer
from pdearena.models.registry import MODEL_REGISTRY

import numpy as np

logger = utils.get_logger(__name__)


def get_model(args: Config) -> torch.nn.Module:
    if args.model in MODEL_REGISTRY:
        _model = MODEL_REGISTRY[args.model].copy()
    else:
        raise ValueError(f"Model {args.model} not found in registry.")
    
    _model["init_args"].update(args=args)
    model = instantiate_class(tuple(), _model)

    if args.pretrained_path is not None:
        logger.info(f"Loading pretrained model from {args.pretrained_path}")
        state_dict = torch.load(args.pretrained_path, map_location="cpu", weights_only=True)
        state_dict = {k.replace("model.", ""): v for k,v in state_dict["state_dict"].items() if k.startswith("model.")}
        model.load_state_dict(state_dict)

    return model


class PDEModel(LightningModule, ABC):
    def __init__(self, args: Config) -> None:
        super().__init__()
        self.save_hyperparameters(args.model_dump())
        self.args = args

        self.time_dependent = args.time_dependent
        output_fields = args.fields
        self.output_fields = {i: field for i, field in enumerate(output_fields)}
        assert len(self.output_fields) == self.args.n_output_fields, [self.output_fields, self.args.n_output_fields]

        self.model = get_model(self.args)
        self.normalizer = Normalizer(args=self.args)
        match args.loss:
            case "rel":
                self.train_loss_fn = ScaledLpLoss(reduction="none", clamp_denom=0)
            case "mse":
                self.train_loss_fn = MSELoss(reduction="none")
            case "mae":
                self.train_loss_fn = None
            case _:
                raise NotImplementedError(args.loss)
        self.eval_loss_fn = ScaledLpLoss(reduction="none", clamp_denom=1)
        self.rollout_criterion = ScaledLpLoss(reduction="none", clamp_denom=1)
        self.correlation = Correlation()
        self.correlation_threshold = [0.8, 0.9, 0.95]
        time_resolution = args.trajlen
        # Max number of previous points solver can eat
        reduced_time_resolution = time_resolution - args.time_history
        # Number of future points to predict
        self.max_start_time = (
            reduced_time_resolution - args.time_future * args.max_num_steps - args.time_gap
        )

        if self.time_dependent:
            log_loss_t_interval = int(np.ceil(self.args.max_num_steps / self.args.log_losses_t))
            log_loss_t_steps = torch.arange(self.args.max_num_steps)[::log_loss_t_interval]
            self.register_buffer("log_loss_t_steps", log_loss_t_steps)


    @abstractmethod
    def forward(self, **kwargs):
        pass

    @abstractmethod
    def get_forward_inputs(self, batch: Batch):
        pass

    def setup(self, stage) -> None:
        if self._trainer is not None:
            self.effective_batch_size = self.args.onestep_batch_size * self.trainer.num_devices
            self.ntrain = self.trainer.datamodule.train_dp1.len()
            # TODO: still not getting the right number of steps per epoch
            # self.steps_per_epoch = self.ntrain // self.effective_batch_size
            num_workers = max(1, self.args.num_workers)
            self.steps_per_epoch = self.ntrain // (self.effective_batch_size * num_workers) * num_workers
            self.estimated_stepping_batches = self.trainer.max_epochs * self.steps_per_epoch
        self.init_metrics()
    
    def on_train_epoch_start(self):
        if self.trainer.current_epoch == 1:
            steps_match = self.global_step == self.steps_per_epoch
            steps_match_message = "" if steps_match else " doesn't"
            logger.info(f"epoch 0 steps ({self.global_step}){steps_match_message} match computed steps per epoch ({self.steps_per_epoch})")
            if not steps_match:
                diagnostics_message = f"effective batch size: {self.effective_batch_size}, num_workers: {self.args.num_workers}, ntrain: {self.ntrain}"
                logger.warning(diagnostics_message)
        return super().on_train_epoch_start()

    def init_onestep_metrics(self):
        return dict(
            loss=ErrorStats(n_fields=1, track_sd=True),
            **{f"{self.output_fields[i]}_loss": ErrorStats(n_fields=1, track_sd=True) for i in self.output_fields}
        )

    def init_time_dependent_metrics(self):
        corr_metrics = dict()
        for threshold in self.correlation_threshold:
            threshold = int(threshold * 100)
            for field_name in self.output_fields.values():
                corr_metrics[f"{field_name}_corr_steps_{threshold}"] = ErrorStats(n_fields=1, track_sd=True)
        return dict(
            unrolled_loss=ErrorStats(n_fields=1, track_sd=True),
            **{f"{self.output_fields[i]}_unrolled_loss": ErrorStats(n_fields=1, track_sd=True) for i in self.output_fields},
            **{f"{self.output_fields[i]}_loss_t": ErrorStats(n_fields=len(self.log_loss_t_steps), track_sd=False) for i in self.output_fields},
            loss_t=ErrorStats(n_fields=len(self.log_loss_t_steps), track_sd=False),
            **corr_metrics
        )

    def init_metrics(self):
        onestep_metrics = MetricCollection(self.init_onestep_metrics())
        # self.onestep_metrics = list(onestep_metrics.keys())
        eval_metrics = onestep_metrics.clone()
        if self.time_dependent:
            eval_metrics = dict(eval_metrics)
            eval_metrics.update(self.init_time_dependent_metrics())
            eval_metrics = MetricCollection(eval_metrics)
        self.train_metrics = onestep_metrics.clone()
        self.valid_metrics = eval_metrics.clone()
        self.test_metrics = eval_metrics.clone()
    
    def compute_metrics(self, mode: str):
        metrics: ErrorStats = getattr(self, f"{mode}_metrics")
        for metric_name, metric in metrics.items():
            if metric_name.endswith("_t") or "_t_" in metric_name:
                try:
                    loss_timesteps = metrics[metric_name].compute()["mean"]
                except Exception as e:
                    print(f"\n\n\n{metric_name}\n\n\n")
                    raise e
                if loss_timesteps.ndim == 0:
                    loss_timesteps = loss_timesteps.unsqueeze(0)
                for i, t in enumerate(self.log_loss_t_steps):
                    self.log(name=f"{mode}/{metric_name}_{t}_loss", value=loss_timesteps[i], sync_dist=True)
            else:
                if metric.update_called:
                    try:
                        stats = metric.compute()
                    except Exception as e:
                        print(f"\n\n\n{metric_name}\n\n\n")
                        raise e
                    self.log(f"{mode}/{metric_name}_mean", stats["mean"], sync_dist=True)
                    if metric.track_sd:
                        self.log(f"{mode}/{metric_name}_std", stats["sd"], sync_dist=True)
                else:
                    if metric.warn:
                        logger.warning(f"update was never called for {mode} metric {metric_name}.")
                        metric.warn = False

    def preprocess(self, batch: Batch, mode: str):
        proc_batch = batch.clone()
        proc_batch.x0 = proc_batch.x[:, -1:]  # TODO: hasn't been tested
        if mode == "train":
            if self.args.predict_diff:
                proc_batch.y = proc_batch.y - proc_batch.x0
            proc_batch.y = self.normalizer.normalize(
                x=proc_batch.y,
                mean=proc_batch.mean_y,
                sd=proc_batch.sd_y
            )
        proc_batch.x = self.normalizer.normalize(
            x=proc_batch.x,
            mean=proc_batch.mean_x,
            sd=proc_batch.sd_x
        )
        return proc_batch

    def postprocess(self, batch: Batch, pred: torch.Tensor, mode: str):
        if mode != "train":
            pred = self.normalizer.denormalize(
                x=pred,
                mean=batch.mean_y,
                sd=batch.sd_y
            )
            if self.args.predict_diff:
                pred = batch.x0 + pred 
        return pred

    def step(self, batch: Batch, mode: str):
        proc_batch = self.preprocess(batch=batch, mode=mode)
        inputs = self.get_forward_inputs(batch=proc_batch)
        pred = self(**inputs)
        pred = self.postprocess(batch=proc_batch, pred=pred, mode=mode)
        if mode == "rollout":
            return pred
        else:
            return self.compute_onesteploss(batch=proc_batch, pred=pred, mode=mode)

    def log_losses(self, batch: Batch, losses: Dict, mode: str):
        log_step = mode == "train"
        metrics = getattr(self, f"{mode}_metrics")

        for error_name, errors in losses.items():
            if errors.numel() > 0:
                if errors.ndim == 1:
                    errors = errors.unsqueeze(1)
                else:
                    assert errors.ndim == 2, [error_name, errors.ndim, errors.shape]
                metric = metrics[error_name]
                error_mean = metric(errors=errors)["mean"]
                if log_step:
                    prog_bar = error_name == "loss" and mode == "train"
                    self.log(
                        name=f"{mode}/{error_name}",
                        value=error_mean,
                        sync_dist=True,
                        on_step=log_step,
                        on_epoch=False,
                        prog_bar=prog_bar,
                        batch_size=len(batch)
                    )

    def compute_loss(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
            batch_idx: torch.Tensor,
            mode: str
    ):
        if mode == "train":
            loss = self.train_loss_fn
        else:
            loss = self.eval_loss_fn
        return loss(
            input=input,
            target=target,
            batch_idx=batch_idx
        )

    def compute_onesteploss_(
            self, 
            batch: Batch, 
            pred, 
            mode: str
        ):

        loss = self.compute_loss(
            input=pred, 
            target=batch.y, 
            batch_idx=batch.batch,
            mode=mode
        )
        with torch.no_grad():
            losses = dict(
                loss=loss.mean(dim=-1),
                **{f"{self.output_fields[i]}_loss": loss[:, :, i] for i in self.output_fields}
            )
        return loss, losses


    def compute_onesteploss(
            self, 
            batch: Batch, 
            pred, 
            mode: str
        ):                 
        loss, losses = self.compute_onesteploss_(
            batch=batch, 
            pred=pred, 
            mode=mode
        )
        self.log_losses(
            batch=batch,
            losses=losses,
            mode=mode
        )
        return loss

    def get_correlation_steps(
            self,
            corr: torch.Tensor,
            threshold: float
        ):
        corr_below = corr < threshold
        nsteps = []
        for traj_corr_bools in corr_below:
            if traj_corr_bools.any():
                nsteps_traj = traj_corr_bools.nonzero().min()
            else:
                nsteps_traj = torch.tensor(len(traj_corr_bools)).to(dtype=torch.long, device=corr.device)
            nsteps.append(nsteps_traj)
        nsteps = torch.stack(nsteps)
        return nsteps

    def compute_rolloutloss_(
            self, 
            batch: Batch, 
            pred: torch.Tensor, 
            targ: torch.Tensor, 
            mode: str
        ):
        batch = batch.to_data_list()
        for b in batch:
            b.batch = torch.zeros(len(b.u), dtype=torch.long, device=b.u.device)
        batch = Batch.from_data_list(batch)
        losses = self.rollout_criterion(input=pred, target=targ, batch_idx=batch.batch)
        losses_t = losses.mean(dim=-1)

        # TODO: irregular number of time steps
        
        # averaging across time axis
        loss = dict(
            unrolled_loss=losses_t.mean(dim=1),
            **{f"{self.output_fields[i]}_unrolled_loss": losses[:, :, i].mean(dim=1) for i in self.output_fields}
        )
        assert losses_t.shape[1] == self.args.max_num_steps, [losses_t.shape, self.args.max_num_steps]
        losses_t = losses_t[:, self.log_loss_t_steps]
        loss["loss_t"] = losses_t
        for i, field_name in self.output_fields.items():
            field_losses_t = losses[:, self.log_loss_t_steps, i]
            assert field_losses_t.shape[1] == len(self.log_loss_t_steps), [field_losses_t.shape, len(self.log_loss_t_steps)]
            loss[f"{field_name}_loss_t"] = field_losses_t

        corr = [self.correlation(
            input=pred[..., i:i+1],
            target=targ[..., i:i+1],
            batch_idx=batch.batch
        ) for i in self.output_fields]
        corr = torch.stack(corr)

        for i, field_name in self.output_fields.items():
            for threshold in self.correlation_threshold:
                field_corr = corr[i]
                nsteps = self.get_correlation_steps(
                    corr=field_corr,
                    threshold=threshold
                )
                threshold_ = int(threshold * 100)
                loss[f"{field_name}_corr_steps_{threshold_}"] = nsteps.float()
        return loss


    def compute_rolloutloss(
            self, 
            batch: Batch, 
            pred: torch.Tensor, 
            targ: torch.Tensor, 
            mode: str
        ):
        loss = self.compute_rolloutloss_(
            batch=batch, 
            pred=pred, 
            targ=targ, 
            mode=mode
        )
        self.log_losses(
            batch=batch,
            losses=loss,
            mode=mode
        )

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch=batch, mode="train")
        return loss.mean()

    def on_train_epoch_end(self) -> None:
        self.compute_metrics(mode="train")
        self.train_metrics.reset()

    def rollout(self, batch: Batch):
        rollout_batch = batch.clone()
        end_time = self.args.time_history
        target_start_time = end_time + self.args.time_gap
        rollout_batch.x = batch.u[:, :end_time]
        targ_u = batch.u[:, target_start_time:]
        num_steps = targ_u.shape[1]
        traj_ls = []
        for _ in range(num_steps):
            pred = self.step(batch=rollout_batch, mode="rollout")
            rollout_batch.x = torch.cat([rollout_batch.x, pred], dim=1)[:, -self.args.time_history:]
            traj_ls.append(pred)
        traj = torch.cat(traj_ls, dim=1)
        return traj, targ_u

    def evaluation_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0, mode: str = "valid"):
        match dataloader_idx:
            case 0:
                # one-step loss
                self.step(batch=batch, mode=mode)
            case 1:
                # rollout loss
                pred, targ = self.rollout(batch=batch)
                self.compute_rolloutloss(batch=batch, pred=pred, targ=targ, mode=mode)
            case _:
                raise NotImplementedError(f"Unknown dataloader index {dataloader_idx}")

    def evaluation_epoch_end(self, mode: str):
        self.compute_metrics(mode=mode)
        metrics = getattr(self, f"{mode}_metrics")
        metrics.reset()

    def validation_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0):
        self.evaluation_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx, mode="valid")

    def on_validation_epoch_end(self) -> None:
        self.evaluation_epoch_end(mode="valid")

    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0):
        self.evaluation_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx, mode="test")

    def on_test_epoch_end(self) -> None:
        self.evaluation_epoch_end(mode="test")

    def predict_step(self, batch: List, batch_idx: int, dataloader_idx: int=0):
        assert self.args.time_history == 1
        assert self.args.time_future == 1
        match dataloader_idx:
            case 0:
                if not self.predict_train:
                    return
                mode = "train"
            case 1:
                mode = "eval"
            case _:
                raise NotImplementedError
        onestep_data, rollout_data = self.get_preds(batch=batch)
        result = {
            mode: dict(
                onestep_data=onestep_data,
                rollout_data=rollout_data
            )
        }
        return result

    @abstractmethod
    def get_preds(self, batch: List):
        pass

    def configure_optimizers(self):
        return self.configure_optimizers_(model=self.model)

    def configure_optimizers_(self, model: torch.nn.Module):
        if self.estimated_stepping_batches < 1:
            raise ValueError(f"Couldn't estimate stepping batches; got {self.estimated_stepping_batches}")

        optimizer_args = self.args.optimizer_args
        optimizer_arg_keys = set(list(optimizer_args.keys()))
        logger.info(f"Optimizer args: {optimizer_args}")
        match self.args.optimizer:
            case "AdamW":
                assert len(optimizer_arg_keys.intersection({"betas"})) == len(optimizer_arg_keys), optimizer_arg_keys
                optimizer = torch.optim.AdamW(
                    model.parameters(), 
                    lr=self.args.lr, 
                    weight_decay=self.args.weight_decay,
                    **optimizer_args
                )
            case _:
                raise ValueError(f"Optimizer {self.args.optimizer} not supported")
        
        scheduler_args = self.args.scheduler_args
        scheduler_arg_keys = set(list(scheduler_args.keys()))
        match self.args.lr_scheduler:
            case "cosine_epoch":
                assert scheduler_arg_keys == {"warmup_epochs", "warmup_start_lr", "eta_min"}, scheduler_arg_keys
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer=optimizer,
                    warmup_epochs=scheduler_args["warmup_epochs"],
                    max_epochs=self.args.epochs,
                    warmup_start_lr=scheduler_args["warmup_start_lr"],
                    eta_min=scheduler_args["eta_min"],
                )
            case "cosine_step":
                assert len(scheduler_arg_keys) == 0, scheduler_arg_keys
                scheduler = dict(
                    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, 
                        T_max=self.estimated_stepping_batches
                    ),
                    interval="step"
                )
            case "onecycle":
                assert len(scheduler_arg_keys) == 0, scheduler_arg_keys
                scheduler = dict(
                    scheduler=torch.optim.lr_scheduler.OneCycleLR(
                        optimizer, 
                        max_lr=self.args.lr, 
                        total_steps=self.estimated_stepping_batches
                    ),
                    interval="step"
                )
            case _:
                raise ValueError(f"LR scheduler {self.args.lr_scheduler} not supported")
            
        return dict(optimizer=optimizer, lr_scheduler=scheduler)
