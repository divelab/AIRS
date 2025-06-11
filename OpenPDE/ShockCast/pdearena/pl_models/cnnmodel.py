import torch
from torch_geometric.data import Batch, Data
from typing import Tuple, List

from pdearena.pl_models.pdemodel import PDEModel
from pdearena.configs.config import CFDConfig, PretrainedCFDConfig
from pdearena.configs.registry import get_config
from pdearena.utils.metrics import ErrorStats
from pdearena.utils.interpolate import linear_interpolate
from pdearena.utils.mean_flow import compute_mean_flow_tke
from pdearena.utils import utils

logger = utils.get_logger(__name__)

class CNNModel(PDEModel):

    def __init__(self, args: CFDConfig):
        super().__init__(args)
        self.xpadding = args.xpad
        self.ypadding = args.ypad

    def pad(
        self, 
        x: torch.Tensor,
        unpad: bool
    ) -> torch.Tensor:
        if self.xpadding > 0 or self.ypadding > 0:
            assert x.ndim == 5
            if unpad:
                x = x[..., self.ypadding:x.shape[-2] - self.ypadding, self.xpadding:x.shape[-1] - self.xpadding]
            else:
                shape = x.shape[:2]
                x = torch.nn.functional.pad(
                    input=x.flatten(0, 1),
                    pad=(self.xpadding, self.xpadding, self.ypadding, self.ypadding),
                    mode="replicate"
                ).unflatten(dim=0, sizes=shape)
        return x

    def forward(self, batch: Batch):
        return self.model(batch)
    
    def get_forward_inputs(self, batch: Batch):
        return dict(batch=batch)
    
    def preprocess(self, batch: Batch, mode: str):
        proc_batch = batch.clone()
        proc_batch.nx = proc_batch.nx[0]
        proc_batch.ny = proc_batch.ny[0]
        proc_batch.x = self.mesh_to_grid(
            x=proc_batch.x,
            nx=proc_batch.nx,
            ny=proc_batch.ny,
            batch_size=len(proc_batch)
        )
        proc_batch.x0 = proc_batch.x[:, -1:]  
        proc_batch.x = self.normalizer.normalize(
            x=proc_batch.x,
            mean=proc_batch.mean_x,
            sd=proc_batch.sd_x,
            dim=2
        )
        proc_batch.x = self.pad(proc_batch.x, unpad=False)
        if mode == "train":
            proc_batch.y = self.mesh_to_grid(
                x=proc_batch.y,
                nx=proc_batch.nx,
                ny=proc_batch.ny,
                batch_size=len(proc_batch)
            )
            if self.args.predict_diff:
                proc_batch.y = proc_batch.y - proc_batch.x0
            proc_batch.y = self.normalizer.normalize(
                x=proc_batch.y,
                mean=proc_batch.mean_y,
                sd=proc_batch.sd_y,
                dim=2
            )
            proc_batch.y = self.grid_to_mesh(proc_batch.y)

        return proc_batch
    
    def postprocess(self, batch: Batch, pred, mode: str):
        pred = self.pad(pred, unpad=True)
        if mode != "train":
            pred = self.normalizer.denormalize(
                x=pred,
                mean=batch.mean_y,
                sd=batch.sd_y,
                dim=2
            )
            if self.args.predict_diff:
                pred = batch.x0 + pred
        pred = self.grid_to_mesh(x=pred)
        return pred
    
    def mesh_to_grid(
            self, 
            x: torch.Tensor, 
            nx: int, 
            ny: int,
            batch_size: int
        ):
        return x.unflatten(dim=0, sizes=[batch_size, ny, nx]).permute(0, 3, 4, 1, 2)
    
    def grid_to_mesh(self, x: torch.Tensor):
        return x.permute(0, 3, 4, 1, 2).flatten(0, 2)
    
    def get_preds(self, batch):
        raise NotImplementedError

class NeuralSolver(CNNModel):

    def __init__(self, args: CFDConfig):
        super().__init__(args)
        self.dt_norm = args.dt_norm and args.predict_diff
        self.log_loss_t_steps: torch.Tensor 
        log_loss_t_steps = torch.arange(0, self.args.sim_time * (1 + 1e-12), self.args.log_losses_dt)
        self.register_buffer("log_loss_t_steps", log_loss_t_steps)        

    def init_time_dependent_metrics(self):
        corr_metrics = dict()
        for threshold in self.correlation_threshold:
            threshold = int(threshold * 100)
            for field_name in self.output_fields.values():
                corr_metrics[f"{field_name}_corr_time_{threshold}"] = ErrorStats(n_fields=1, track_sd=True)
                corr_metrics[f"{field_name}_corr_prop_{threshold}"] = ErrorStats(n_fields=1, track_sd=True)

        metrics = PDEModel.init_time_dependent_metrics(self)

        metrics.update(corr_metrics)
        return metrics

    def preprocess(self, batch: Batch, mode: str):
        proc_batch = batch.clone()
        if self.dt_norm:
            proc_batch.dt_unscaled, proc_batch.dt = proc_batch.dt.view(-1, 2).T
            dt_unscaled = proc_batch.dt_unscaled.unsqueeze(1)
            proc_batch.mean_y = proc_batch.mean_y * dt_unscaled
            proc_batch.sd_y = proc_batch.sd_y * dt_unscaled
        proc_batch.dt = (proc_batch.dt - proc_batch.delta_t_shift) / proc_batch.delta_t_scale
        return super().preprocess(batch=proc_batch, mode=mode)
    
    def rollout(self, batch: Batch):
        end_time = self.args.time_history
        target_start_time = end_time + self.args.time_gap
        assert end_time == 1  # TODO
        assert target_start_time == 1  # TODO
        dt = batch.dt  # TODO: time step is not known a priori
        targ_u = batch.u[:, self.args.time_history:]
        rollout_batch = batch.clone()
        del rollout_batch.u
        rollout_batch.x = batch.u[:, :end_time]
        num_steps = targ_u.shape[1]
        traj_ls = []
        for t in range(num_steps):
            rollout_batch.dt = dt[t].unsqueeze(0)
            pred = self.step(batch=rollout_batch, mode="rollout")
            rollout_batch.x = torch.cat([rollout_batch.x, pred], dim=1)[:, -self.args.time_history:]
            traj_ls.append(pred)
        traj = torch.cat(traj_ls, dim=1)
        return traj, targ_u

    def compute_rolloutloss_(
            self,
            batch: Batch,
            pred: torch.Tensor,
            targ: torch.Tensor,
            mode: str
        ):
        batch.batch = torch.zeros(len(pred), device=pred.device, dtype=torch.long)
        losses = self.rollout_criterion(input=pred, target=targ, batch_idx=batch.batch)
        losses_t = losses.mean(dim=-1)

        # TODO: irregular number of time steps
        
        # averaging across time axis
        loss = dict(
            unrolled_loss=losses_t.mean(dim=1),
            **{f"{self.output_fields[i]}_unrolled_loss": losses[:, :, i].mean(dim=1) for i in self.output_fields}
        )

        times = batch.times[self.args.time_history:]
        log_steps = torch.cdist(times.unsqueeze(1), self.log_loss_t_steps.unsqueeze(1)).argmin(dim=0)
        loss["loss_t"] = losses_t[:, log_steps]
 
        for i, field_name in self.output_fields.items():
            field_losses_t = losses[:, log_steps, i]
            loss[f"{field_name}_loss_t"] = field_losses_t

        corr = [self.correlation(
            input=pred[..., i:i+1],
            target=targ[..., i:i+1],
            batch_idx=batch.batch
        ) for i in self.output_fields]
        corr = torch.stack(corr)
        final_time = times[-1]
        for i, field_name in self.output_fields.items():
            for threshold in self.correlation_threshold:
                field_corr = corr[i]
                nsteps = self.get_correlation_steps(
                    corr=field_corr,
                    threshold=threshold
                )

                threshold_ = int(threshold * 100)
                loss[f"{field_name}_corr_steps_{threshold_}"] = nsteps.unsqueeze(0).float()
                corr_time = times[nsteps - 1] if nsteps > 0 else torch.zeros(1).type_as(corr)
                loss[f"{field_name}_corr_time_{threshold_}"] = corr_time.unsqueeze(0)
                loss[f"{field_name}_corr_prop_{threshold_}"] = (corr_time / final_time).unsqueeze(0)
        return loss

    def get_preds(self, batch: Batch):
        onestep_data = []
        for t in range(batch.u.shape[1] - 1):
            b = batch.clone()
            b.x = b.u[:, t:t+1]
            del b.u
            b.dt = b.dt[t:t+1]
            
            pred = self.step(batch=b, mode="rollout")
            onestep_data.append(pred.cpu())
        onestep_data = Data(x=torch.cat(onestep_data, dim=1))
        onestep_data.times = b.times
        onestep_data.z = b.z

        pred, targ = self.rollout(batch=batch)
        return onestep_data, pred

class NeuralCFL(NeuralSolver):
    def __init__(self, args: CFDConfig):
        self.noise_level = args.noise_level
        if args.include_derivatives:
            args.n_input_fields = 3 * args.n_input_fields
        if args.include_cfl_features:
            args.n_input_fields += 4
        super().__init__(args)
        xvel_ind = [i for i, field in enumerate(args.fields) if "xvel" in field.lower()]
        assert len(xvel_ind) == 1, f"Expected one xvel field, found {len(xvel_ind)}"
        yvel_ind = [i for i, field in enumerate(args.fields) if "yvel" in field.lower()]
        assert len(yvel_ind) == 1, f"Expected one yvel field, found {len(yvel_ind)}"
        self.xvel_ind = xvel_ind[0]
        self.yvel_ind = yvel_ind[0]
        temp_ind = [i for i, field in enumerate(args.fields) if "temp" in field.lower()]
        assert len(temp_ind) == 1, f"Expected one temp field, found {len(temp_ind)}"
        self.temp_ind = temp_ind[0]
        match args.loss:
            case "rel":
                raise NotImplementedError
            case "mse":
                self.train_loss_fn = torch.nn.MSELoss(reduction="none")
            case "mae":
                self.train_loss_fn = torch.nn.L1Loss(reduction="none")
            case _:
                raise NotImplementedError(args.loss)
        self.eval_loss_fn = torch.nn.L1Loss(reduction="none")

    def init_time_dependent_metrics(self):
        return dict()

    def preprocess(self, batch: Batch, mode: str):
        proc_batch = batch.clone()
        proc_batch = batch.clone()
        proc_batch.nx = proc_batch.nx[0]
        proc_batch.ny = proc_batch.ny[0]
        proc_batch.x = self.mesh_to_grid(
            x=proc_batch.x,
            nx=proc_batch.nx,
            ny=proc_batch.ny,
            batch_size=len(proc_batch)
        )
        proc_batch.x0 = proc_batch.x[:, -1:]  

        temp = proc_batch.x[:, :, self.temp_ind:self.temp_ind + 1]
        xvel = proc_batch.x[:, :, self.xvel_ind:self.xvel_ind + 1]
        yvel = proc_batch.x[:, :, self.yvel_ind:self.yvel_ind + 1]

        if self.args.include_cfl_features:
            gamma = 1.4
            R = 287
            sound_speed = (gamma * R * temp).sqrt()
            xvel_mag = xvel.abs()
            yvel_mag = yvel.abs()
            wave_speed = torch.max(xvel.abs() + sound_speed, yvel.abs() + sound_speed)
            cfl_features = torch.cat([sound_speed, xvel_mag, yvel_mag, wave_speed], dim=2)
            cfl_mean = torch.stack([proc_batch.sound_speed_mean, proc_batch.xvel_mag_mean, proc_batch.yvel_mag_mean, proc_batch.wave_speed_mean], dim=1)
            cfl_sd = torch.stack([proc_batch.sound_speed_sd, proc_batch.xvel_mag_sd, proc_batch.yvel_mag_sd, proc_batch.wave_speed_sd], dim=1)
            cfl_features = self.normalizer.normalize(
                x=cfl_features,
                mean=cfl_mean,
                sd=cfl_sd,
                dim=2
            )

        if self.args.include_derivatives:
            dy, dx = torch.gradient(proc_batch.x, dim=[-2, -1])
            dy = self.normalizer.normalize(
                x=dy,
                mean=proc_batch.mean_yderiv,
                sd=proc_batch.sd_yderiv,
                dim=2
            )
            dx = self.normalizer.normalize(
                x=dx,
                mean=proc_batch.mean_xderiv,
                sd=proc_batch.sd_xderiv,
                dim=2
            )

        proc_batch.x = self.normalizer.normalize(
            x=proc_batch.x,
            mean=proc_batch.mean_x,
            sd=proc_batch.sd_x,
            dim=2
        )

        if self.args.include_derivatives:
            proc_batch.x = torch.cat([proc_batch.x, dy, dx], dim=2)

        if self.args.include_cfl_features:
            proc_batch.x = torch.cat([proc_batch.x, cfl_features], dim=2)

        if mode == "train" and self.noise_level > 0:
            noise = torch.randn_like(proc_batch.x) * self.noise_level
            proc_batch.x = proc_batch.x + noise

        if mode != "rollout":
            shift = proc_batch.delta_t_shift
            scale = proc_batch.delta_t_scale
            proc_batch.dt = self.normalize_dt(
                dt=proc_batch.dt,
                mean=shift,
                sd=scale,
                unnormalize=False
            )
             
        return proc_batch
    
    def normalize_dt(
            self,
            dt: torch.Tensor,
            mean: torch.Tensor,
            sd: torch.Tensor,
            unnormalize: bool
        ) -> torch.Tensor:
        if unnormalize:
            return dt * sd + mean
        else:
            return (dt - mean) / sd

    def postprocess(self, batch: Batch, pred: torch.Tensor, mode: str):
        if mode == "rollout":
            shift = batch.delta_t_shift
            scale = batch.delta_t_scale
            pred = self.normalize_dt(
                dt=pred,
                mean=shift,
                sd=scale,
                unnormalize=True
            )

        return pred

    def compute_loss(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
            mode: str
    ):
        if mode == "train":
            loss = self.train_loss_fn
        else:
            loss = self.eval_loss_fn
        return loss(
            input=input,
            target=target,
        )

    def compute_onesteploss_(
            self, 
            batch: Batch, 
            pred, 
            mode: str
        ):

        loss = self.compute_loss(
            input=pred, 
            target=batch.dt, 
            mode=mode
        )
        losses = dict(
            loss=loss
        )
        return loss, losses

    def evaluation_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0, mode: str = "valid"):
        if dataloader_idx == 0:
            # one-step loss
            self.step(batch=batch, mode=mode)

    def predict_step(self, batch: List, batch_idx: int, dataloader_idx: int=0):
        raise NotImplementedError


class ShockCast(NeuralSolver):
    def __init__(self, args: PretrainedCFDConfig):
        super().__init__(args)
        self.vel_inds = args.vel_inds
        self.solver_args = get_config(args.solver_cfg).copy()

        self.model = NeuralSolver.load_from_checkpoint(
            args=self.solver_args,
            checkpoint_path=args.solver_ckpt,
            map_location="cpu"
        )

        info_keys = ["epoch", "global_step"]
        model_info = {k:v for k,v in torch.load(args.solver_ckpt, weights_only=True).items() if k in info_keys}
        model_epochs = model_info["epoch"] + 1
        model_global_step = model_info["global_step"]
        assert model_epochs == self.solver_args.epochs, f"Model epochs {model_epochs} do not match config epochs {self.solver_args.epochs}"
        logger.info(f"Loaded {args.solver_cfg} trained for {model_epochs} epochs, {model_global_step} steps")

        self.cfl_args = get_config(args.cfl_cfg).copy()
        self.cfl_model = NeuralCFL.load_from_checkpoint(
            args=self.cfl_args,
            checkpoint_path=args.cfl_ckpt,
            map_location="cpu"
        )
        cfl_info = {k:v for k,v in torch.load(args.cfl_ckpt, weights_only=True).items() if k in info_keys}
        cfl_epochs = cfl_info["epoch"] + 1
        cfl_global_step = cfl_info["global_step"]
        assert cfl_epochs == self.cfl_args.epochs, f"Model epochs {cfl_epochs} do not match config epochs {self.cfl_args.epochs}"
        logger.info(f"Loaded {args.cfl_cfg} trained for {cfl_epochs} epochs, {cfl_global_step} steps")

        self.dt_eval_loss_fn = torch.nn.L1Loss(reduction="none")
        self.register_buffer("log_loss_t_steps", torch.tensor([]))
        self.predictions = dict(onestep=[], rollout=[])

    def init_onestep_metrics(self):
        metrics = super().init_onestep_metrics()
        gt_dt_metrics = super().init_onestep_metrics()
        gt_dt_metrics = {f"{k}_gt_dt": v for k, v in gt_dt_metrics.items()}
        metrics.update(gt_dt_metrics)
        metrics["dt_loss"] = ErrorStats(n_fields=1, track_sd=True)
        return metrics

    def init_time_dependent_metrics(self):
        metrics = super().init_time_dependent_metrics()
        gt_dt_metrics = super().init_time_dependent_metrics()
        gt_dt_metrics = {f"{k}_gt_dt": v for k, v in gt_dt_metrics.items()}        
        metrics.update(gt_dt_metrics)
        mean_flow_tke_metrics = dict(
            mean_flow_loss=ErrorStats(n_fields=1, track_sd=True),
            mean_flow_loss_gt_dt=ErrorStats(n_fields=1, track_sd=True),
            tke_loss=ErrorStats(n_fields=1, track_sd=True),
            tke_loss_gt_dt=ErrorStats(n_fields=1, track_sd=True),
            **{f"mean_flow_{self.output_fields[i]}_loss": ErrorStats(n_fields=1, track_sd=True) for i in self.output_fields},
            **{f"mean_flow_{self.output_fields[i]}_loss_gt_dt": ErrorStats(n_fields=1, track_sd=True) for i in self.output_fields},
        )
        metrics.update(mean_flow_tke_metrics)
        return metrics

    def evaluation_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0, mode: str = "valid"):
        match dataloader_idx:
            case 0:
                # one-step loss
                return self.onestep(batch=batch, mode=mode)
            case 1:
                # rollout loss
                return super().evaluation_step(
                    batch=batch,
                    batch_idx=batch_idx,
                    dataloader_idx=dataloader_idx,
                    mode=mode
                )
            case _:
                raise NotImplementedError(f"Unknown dataloader index: {dataloader_idx}")

    def onestep(self, batch: Batch, mode: str):
        pred_gt_dt = self.model.step(batch=batch.clone(), mode="rollout")        
        pred_dt = self.cfl_model.step(batch=batch.clone(), mode="rollout")
        proc_batch = batch.clone()
        proc_batch.dt = pred_dt
        pred = self.model.step(batch=proc_batch, mode="rollout")
        return self.compute_onesteploss(batch=batch, pred=[pred, pred_dt, pred_gt_dt], mode=mode)

    def compute_onesteploss_(
            self, 
            batch: Batch, 
            pred, 
            mode: str
        ):
        pred, pred_dt, pred_gt_dt = pred
        loss_gt_dt, losses_gt_dt = NeuralSolver.compute_onesteploss_(
            self=self,
            batch=batch,
            pred=pred_gt_dt,
            mode=mode
        )
        pred_dt_norm = self.cfl_model.normalize_dt(
            dt=pred_dt,
            mean=batch.delta_t_shift,
            sd=batch.delta_t_scale,
            unnormalize=False
        )
        dt_norm = self.cfl_model.normalize_dt(
            dt=batch.dt,
            mean=batch.delta_t_shift,
            sd=batch.delta_t_scale,
            unnormalize=False
        )
        dt_loss = self.dt_eval_loss_fn(
            input=pred_dt_norm,
            target=dt_norm
        )
        losses_gt_dt = {f"{k}_gt_dt": v for k, v in losses_gt_dt.items()}
        loss_pred_dt, losses_pred_dt = NeuralSolver.compute_onesteploss_(
            self=self,
            batch=batch,
            pred=pred,
            mode=mode
        )
        losses = {**losses_gt_dt, **losses_pred_dt, "dt_loss": dt_loss}
        preds = dict(
            loss={k:v.cpu() for k, v in losses.items()},
            pred_gt_dt=pred_gt_dt.cpu(),
            pred_dt=pred_dt.cpu(),
            pred=pred.cpu(),
            z=batch.z.cpu()
        )
        self.predictions["onestep"].append(preds)
        return losses_pred_dt, losses

    def compute_rolloutloss_(
            self,
            batch: Batch,
            pred: torch.Tensor,
            targ: torch.Tensor,
            mode: str
        ):
        pred, pred_time, pred_gt_dt = pred
        loss_gt_dt = NeuralSolver.compute_rolloutloss_(
            self=self,
            batch=batch,
            pred=pred_gt_dt,
            targ=targ,
            mode=mode
        )
        loss_gt_dt = {f"{k}_gt_dt": v for k, v in loss_gt_dt.items()}
        pred_interp = linear_interpolate(
            values=pred,
            times=pred_time,
            target_times=batch.times[self.args.time_history:],
            dim=1
        )
        assert pred_interp.shape == targ.shape, [pred_interp.shape, targ.shape]
        loss_interp = NeuralSolver.compute_rolloutloss_(
            self=self,
            batch=batch,
            pred=pred_interp,
            targ=targ,
            mode=mode
        )
        loss = {**loss_gt_dt, **loss_interp}
        times = batch.times[self.args.time_history:]
        pred_mean_flow_gt_dt, pred_tke_gt_dt = compute_mean_flow_tke(
            times=times,
            u=pred_gt_dt,
            time_dim=1,
            field_dim=2,
            velo_inds=self.vel_inds
        )
        pred_mean_flow, pred_tke = compute_mean_flow_tke(
            times=pred_time,
            u=pred,
            time_dim=1,
            field_dim=2,
            velo_inds=self.vel_inds
        )
        mean_flow, tke = compute_mean_flow_tke(
            times=times,
            u=targ,
            time_dim=1,
            field_dim=2,
            velo_inds=self.vel_inds
        )
        one_batch_idx = torch.zeros(len(tke), device=tke.device, dtype=torch.long)
        tke_loss_gt_dt = self.compute_loss(
            input=pred_tke_gt_dt,
            target=tke,
            batch_idx=one_batch_idx,
            mode=mode
        )
        mean_flow_loss_gt_dt = self.compute_loss(
            input=pred_mean_flow_gt_dt,
            target=mean_flow,
            batch_idx=one_batch_idx,
            mode=mode
        )
        tke_loss = self.compute_loss(
            input=pred_tke, 
            target=tke, 
            batch_idx=one_batch_idx,
            mode=mode
        )
        mean_flow_loss = self.compute_loss(
            input=pred_mean_flow,
            target=mean_flow,
            batch_idx=one_batch_idx,
            mode=mode
        )
        loss.update(
            tke_loss_gt_dt=tke_loss_gt_dt,
            tke_loss=tke_loss,
            mean_flow_loss_gt_dt=mean_flow_loss_gt_dt.mean(dim=1),
            mean_flow_loss=mean_flow_loss.mean(dim=1),
            **{f"mean_flow_{self.output_fields[i]}_loss_gt_dt": mean_flow_loss_gt_dt[:, i] for i in self.output_fields},
            **{f"mean_flow_{self.output_fields[i]}_loss": mean_flow_loss[:, i] for i in self.output_fields}
        )
        preds = dict(
            loss={k:v.cpu() for k, v in loss.items()},
            pred=pred.cpu(),
            pred_time=pred_time.cpu(),
            pred_interp=pred_interp.cpu(),
            pred_gt_dt=pred_gt_dt.cpu(),
            pred_mean_flow=pred_mean_flow.cpu(),
            pred_tke=pred_tke.cpu(),
            pred_mean_flow_gt_dt=pred_mean_flow_gt_dt.cpu(),
            pred_tke_gt_dt=pred_tke_gt_dt.cpu(),
            gt_tke=tke.cpu(),
            gt_mean_flow=mean_flow.cpu(),
        )
        self.predictions["rollout"].append(preds)
        return loss
    # """
    def rollout(self, batch: Batch):

        pred_gt_dt, targ = self.model.rollout(batch=batch.clone())

        end_time = self.args.time_history
        target_start_time = end_time + self.args.time_gap
        assert end_time == 1  # TODO
        assert target_start_time == 1  # TODO
        targ_u = batch.u[:, self.args.time_history:]
        rollout_batch = batch.clone()
        del rollout_batch.u, rollout_batch.dt, rollout_batch.times
        rollout_batch.x = batch.u[:, :end_time]
        tmax = batch.times[-1]
        time = 0
        traj_ls = []
        time_ls = []
        while time < tmax:
            dt = self.cfl_model.step(batch=rollout_batch, mode="rollout")
            time += dt
            rollout_batch.dt = dt
            time_ls.append(time.clone())
            pred = self.model.step(batch=rollout_batch, mode="rollout")
            rollout_batch.x = torch.cat([rollout_batch.x, pred], dim=1)[:, -self.args.time_history:]
            traj_ls.append(pred)
        traj = torch.cat(traj_ls, dim=1)
        time = torch.cat(time_ls)
        return [traj, time, pred_gt_dt], targ_u