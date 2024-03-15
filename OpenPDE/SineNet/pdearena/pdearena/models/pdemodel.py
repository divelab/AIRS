# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import instantiate_class
import torch.distributed as dist
import os

from pdearena import utils
from pdearena.data.utils import PDEDataConfig
from pdearena.modules.loss import CustomMSELoss, ScaledLpLoss
from pdearena.rollout import rollout2d, rollout3d_maxwell

from pdearena.models.normalizer import normalizer

from .registry import MODEL_REGISTRY
from tqdm import tqdm

logger = utils.get_logger(__name__)


def get_model(args, pde):
    assert args.name in MODEL_REGISTRY, f"{args.name} not in model registry"
    _model = MODEL_REGISTRY[args.name].copy()
    if "Maxwell" in args.name:
        _model["init_args"].update(
            dict(
                time_history=args.time_history,
                time_future=args.time_future,
                activation=args.activation,
            )
        )
    else:
        _model["init_args"].update(
            dict(
                n_input_scalar_components=pde.n_scalar_components,
                n_output_scalar_components=pde.n_scalar_components,
                n_input_vector_components=pde.n_vector_components,
                n_output_vector_components=pde.n_vector_components,
                time_history=args.time_history,
                time_future=args.time_future,
                activation=args.activation,
            )
        )
    model = instantiate_class(tuple(), _model)

    return model


class PDEModel(LightningModule):
    def __init__(
        self,
        name: str,
        time_history: int,
        time_future: int,
        time_gap: int,
        max_num_steps: int,
        activation: str,
        criterion: str,
        lr: float,
        pdeconfig: PDEDataConfig,
        model: Optional[Dict] = None,
        normalize: bool = False,
        noise: float = 0
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="pdeconfig")
        self.pde = pdeconfig
        if (self.pde.n_spatial_dim) == 3:
            self._mode = "3DMaxwell"
            assert self.pde.n_scalar_components == 0
            assert self.pde.n_vector_components == 2
        elif (self.pde.n_spatial_dim) == 2:
            self._mode = "2D"
        else:
            raise NotImplementedError(f"{self.pde}")

        self.model = get_model(self.hparams, self.pde)
        if criterion == "mse":
            self.train_criterion = CustomMSELoss()
        elif criterion == "scaledl2":
            self.train_criterion = ScaledLpLoss()
        else:
            raise NotImplementedError(f"Criterion {criterion} not implemented yet")

        self.val_criterions = {"mse": CustomMSELoss(), "scaledl2": ScaledLpLoss()}
        self.rollout_criterion = ScaledLpLoss(reduction="none") # torch.nn.MSELoss(reduction="none")
        time_resolution = self.pde.trajlen
        # Max number of previous points solver can eat
        reduced_time_resolution = time_resolution - self.hparams.time_history
        # Number of future points to predict
        self.max_start_time = (
            reduced_time_resolution - self.hparams.time_future * self.hparams.max_num_steps - self.hparams.time_gap
        )
        self.normalizer = normalizer(hparams=self.hparams, pde=self.pde)

    def setup(self, stage):
        datamodule = self.trainer.datamodule
        if self.hparams.normalize:
            task = datamodule.hparams.task
            if task == 'CFD':
                mu = torch.tensor([[[[[5.041397571563721]], [[25.086105346679688]], [[-3.756210207939148e-05]], [[1.1378322597010992e-05]]]]])
                sigma = torch.tensor([[[[[2.9829258918762207]], [[21.71123504638672]], [[0.16352088749408722]], [[0.16349086165428162]]]]])
            elif task == 'NavierStokes2D':
                mu = torch.tensor([[[[[0.7171192169189453]], [[-5.4855921888252723e-11]], [[-3.2108783187823065e-08]]]]])
                sigma = torch.tensor([[[[[0.4428091049194336]], [[0.4393102824687958]], [[0.508793294429779]]]]])
            elif task == 'ShallowWater2DVel-2Day':
                mu = torch.tensor([[[[[0.0009767287410795689]], [[-0.4592400789260864]], [[-0.0141550088301301]]]]])
                sigma = torch.tensor([[[[[1.0120806694030762]], [[3.4489285945892334]], [[3.4050679206848145]]]]])
            else:
                raise ValueError(f'Normalization not implemented for {task}')
            
            self.normalizer.register_buffer("mean", mu)
            self.normalizer.register_buffer("sd", sigma)
            self.normalizer.normalize = True

        if self.global_rank == 0 and "ffno" in self.hparams.name.lower() and self.model.should_normalize:
            setup_ckpt = os.path.join(self.trainer.checkpoint_callback.dirpath, "setup.ckpt") 
            par_dict = {}
            for n, (x, y) in enumerate(tqdm(datamodule.train_dataloader())):
                self.model._build_features(self.normalizer(x))
                # break
            self.model.normalizer.max_accumulations = torch.tensor(n)
            par_dict['ffno_normalizer'] = self.model.normalizer
            os.makedirs(os.path.dirname(setup_ckpt), exist_ok=True)
            torch.save(par_dict, setup_ckpt)
        
        dist.barrier()
        
        if "ffno" in self.hparams.name.lower() and self.model.should_normalize:
            par_dict = torch.load(setup_ckpt)
            self.model.normalizer = par_dict['ffno_normalizer']

            print(f"Rank {self.global_rank} FFNO: sum - {self.model.normalizer.sum}, sum squared - {self.model.normalizer.sum_squared}, max accumulations - {self.model.normalizer.max_accumulations}, n accumulations - {self.model.normalizer.n_accumulations}")
            assert not (self.model.normalizer.sum == 0).all()
            assert not (self.model.normalizer.sum_squared == 1).all()
            
    def on_train_start(self):
        par_ct = sum(par.numel() for par in self.model.parameters())
        self.log("model/parameters", par_ct, sync_dist=True)

    def forward(self, *args):
        return self.model(*args)

    def train_step(self, batch):
        x, y = batch
        x = self.normalizer(x)
        x = self.normalizer.noise(x)
        y = self.normalizer(y)
        pred = self.model(x)
        loss = self.train_criterion(pred, y)
        return loss, pred, y

    def eval_step(self, batch):
        x, y = batch
        x = self.normalizer(x)
        pred = self.model(x)
        pred = self.normalizer.inverse(pred)
        loss = {k: vc(pred, y) for k, vc in self.val_criterions.items()}
        # if any([vc == float('inf') for vc in loss.values()]):
        #     print("here")
        # loss = {k: vc(pred, y) for k, vc in self.val_criterions.items()}
        return loss, pred, y

    def training_step(self, batch, batch_idx: int):
        loss, preds, targets = self.train_step(batch)

        if self._mode == "2D":
            scalar_loss = self.train_criterion(
                preds[:, :, 0 : self.pde.n_scalar_components, ...],
                targets[:, :, 0 : self.pde.n_scalar_components, ...],
            )

            if self.pde.n_vector_components > 0:
                vector_loss = self.train_criterion(
                    preds[:, :, self.pde.n_scalar_components :, ...],
                    targets[:, :, self.pde.n_scalar_components :, ...],
                )
            else:
                vector_loss = torch.tensor(0.0)
            self.log("train/loss", loss, sync_dist=True)
            self.log("train/scalar_loss", scalar_loss, sync_dist=True)
            self.log("train/vector_loss", vector_loss, sync_dist=True)
            return {
                "loss": loss,
                "scalar_loss": scalar_loss.detach(),
                "vector_loss": vector_loss.detach(),
            }
        else:
            raise NotImplementedError(f"{self._mode}")

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        for key in outputs[0].keys():
            if "loss" in key:
                loss_vec = torch.stack([outputs[i][key] for i in range(len(outputs))])
                mean, std = loss_vec.mean(), None # utils.bootstrap(loss_vec, 64, 1)
                self.log(f"train/{key}_mean", mean, sync_dist=True)
                # self.log(f"train/{key}_std", std)

    def compute_rolloutloss2D(self, batch: Any, return_rollout: bool = False):
        (u, v, cond, grid) = batch

        losses = []
        for start in range(
            0,
            self.max_start_time + 1,
            self.hparams.time_future + self.hparams.time_gap,
        ):
            end_time = start + self.hparams.time_history
            target_start_time = end_time + self.hparams.time_gap
            target_end_time = target_start_time + self.hparams.time_future * self.hparams.max_num_steps

            init_u = u[:, start:end_time, ...]
            if self.pde.n_vector_components > 0:
                init_v = v[:, start:end_time, ...]
            else:
                init_v = None

            pred_traj = rollout2d(
                self.model,
                self.normalizer,
                init_u,
                init_v,
                grid,
                self.pde,
                self.hparams.time_history,
                self.hparams.max_num_steps,
            )
            targ_u = u[:, target_start_time:target_end_time, ...]
            if self.pde.n_vector_components > 0:
                targ_v = v[:, target_start_time:target_end_time, ...]
                targ_traj = torch.cat((targ_u, targ_v), dim=2)
            else:
                targ_traj = targ_u
            loss = self.rollout_criterion(pred_traj, targ_traj).mean(dim=(0, 2))
            losses.append(loss)
        loss_vec = torch.stack(losses, dim=0).mean(dim=0)
        if return_rollout:
            loss_vec = {"pred": pred_traj, "targ_traj": targ_traj, "loss": loss_vec}
        return loss_vec

    def evaluation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0, test: bool = False):
        mode = "test" if test else "valid"
        if dataloader_idx == 0:
            # one-step loss
            loss, preds, targets = self.eval_step(batch)
            if self._mode == "2D":
                loss["scalar_mse"] = self.val_criterions["mse"](
                    preds[:, :, 0 : self.pde.n_scalar_components, ...],
                    targets[:, :, 0 : self.pde.n_scalar_components, ...],
                )
                loss["vector_mse"] = self.val_criterions["mse"](
                    preds[:, :, self.pde.n_scalar_components :, ...],
                    targets[:, :, self.pde.n_scalar_components :, ...],
                )
                for k in loss.keys():
                    self.log(f"{mode}/loss/{k}", loss[k], sync_dist=True)
                return {f"{k}_loss": v for k, v in loss.items()}
            else:
                raise NotImplementedError(f"{self._mode}")

        elif dataloader_idx == 1:
            # rollout loss
            if self._mode == "2D":
                loss_vec = self.compute_rolloutloss2D(batch)
            else:
                raise NotImplementedError(f"{self._mode}")
            # summing across "time axis"
            loss = loss_vec.mean()
            loss_t = loss_vec#.cumsum(0)
            # chan_avg_loss = loss / (self.pde.n_scalar_components + self.pde.n_vector_components)
            self.log(f"{mode}/unrolled_loss", loss, sync_dist=True)
            return {
                "unrolled_loss": loss,
                "loss_timesteps": loss_t,
                # "unrolled_chan_avg_loss": chan_avg_loss,
            }

    def evaluation_epoch_end(self, outputs: List[Any], test: bool = False):
        mode = "test" if test else "valid"
        assert len(outputs) > 1
        if len(outputs[0]) > 0:
            for key in outputs[0][0].keys():
                if "loss" in key:
                    loss_vec = torch.stack([outputs[0][i][key] for i in range(len(outputs[0]))])
                    mean, std = loss_vec.mean(), None # utils.bootstrap(loss_vec, 64, 1)
                    self.log(f"{mode}/{key}_mean", mean, sync_dist=True)
                    # self.log(f"valid/{key}_std", std)
        if len(outputs[1]) > 0:
            unrolled_loss = torch.stack([outputs[1][i]["unrolled_loss"] for i in range(len(outputs[1]))])
            loss_timesteps_B = torch.stack([outputs[1][i]["loss_timesteps"] for i in range(len(outputs[1]))])
            loss_timesteps = loss_timesteps_B.mean(0)
            for i in range(self.hparams.max_num_steps):
                self.log(f"{mode}/intime_{i}_loss", loss_timesteps[i], sync_dist=True)

            mean, std = unrolled_loss.mean(), None # utils.bootstrap(unrolled_loss, 64, 1)
            self.log(f"{mode}/unrolled_loss_mean", mean, sync_dist=True)
            # self.log("valid/unrolled_loss_std", std)

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        return self.evaluation_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)

    def validation_epoch_end(self, outputs: List[Any]):
        if len(outputs) > 1:
            self.evaluation_epoch_end(outputs=outputs)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        return self.evaluation_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx, test=True)

    def test_epoch_end(self, outputs: List[Any]):
        self.evaluation_epoch_end(outputs=outputs, test=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer
