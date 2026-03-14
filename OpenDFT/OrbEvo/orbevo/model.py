import os
import torch
os.environ["MKL_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
torch.set_num_threads(1)

import math

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm

from omegaconf import OmegaConf
from orbevo.datamodule import DataModule
import importlib
from orbevo.loss import mae, l2mae, scaled_l2

import numpy as np
import torch_scatter
from torch_geometric.data import Data



# DDP helpers
def setup(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def get_model(cfg, **kwargs):
    model_cls = getattr(importlib.import_module('orbevo.models'), cfg.model.name)
    if cfg.model.args is not None:
        net = model_cls(**cfg.model.args, **kwargs)
    else:
        net = model_cls(**kwargs)
    return net


class Model:
    def __init__(self, cfg, rank, world_size, output_dir) -> None:
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.output_dir = output_dir  # output directory
        self.has_cuda = torch.cuda.is_available()
        self.device = torch.device(f"cuda:{self.rank}" if self.has_cuda else "cpu")
        self.use_amp = bool(self.cfg.use_amp and self.has_cuda)
        self.autocast_device_type = "cuda" if self.has_cuda else "cpu"
        if self.cfg.use_ddp:
            if not self.has_cuda:
                raise RuntimeError("DDP training requires CUDA in this setup.")
            setup(self.rank, self.world_size, str(self.cfg.ddp_port))  # setup ddp

        net = get_model(
            cfg, 
            time_cond=cfg.time_cond, 
            time_future=cfg.time_future, 
            avg_num_nodes=self.cfg.dataset.avg_num_nodes, 
            avg_degree=self.cfg.dataset.avg_degree
        )
        par= sum(par.numel() for par in net.parameters())
        print(f'# par: {par}')

        # resume
        if self.cfg.model_path != "":
            print(f'resuming from {self.cfg.model_path}')
            ckpt = torch.load(self.cfg.model_path, map_location='cpu')
            net.load_state_dict(ckpt['model_state_dict'])
            del ckpt
            if self.has_cuda:
                torch.cuda.empty_cache()

        if self.cfg.use_ddp:
            self.model = DDP(net.to(self.device), device_ids=[self.rank], find_unused_parameters=self.cfg.find_unused_parameters)

        else:
            self.model = net.to(self.device)

        self.best_ckpts = {}  # record for saving ckpts {epoch: valid_loss}

        self.train_data_module = DataModule(cfg=cfg, world_size=world_size,
                                            time_start=cfg.time_start, time_cond=cfg.time_cond, time_future=cfg.time_future, T=cfg.dataset.T)

        self.valid_onestep_data_module = DataModule(cfg=cfg, world_size=world_size,
                                                    time_start=cfg.time_start, time_cond=cfg.time_cond, time_future=cfg.time_future, T=cfg.dataset.T)
        self.valid_rollout_data_module = DataModule(cfg=cfg, world_size=world_size,
                                                    time_start=cfg.time_start, time_cond=cfg.time_cond, time_future=cfg.dataset.T, T=cfg.dataset.T)

        self.rng = np.random.default_rng(seed=rank)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        assert len(self.cfg.model) == 2, 'Please put model arguments under model.args'

    def cleanup(self):
        # clean up ddp
        dist.destroy_process_group()


    def __del__(self):
        if self.cfg.use_ddp:
            self.cleanup()


    def train_one_epoch(self, optimizer, loader, device, writer, global_iter, scheduler, push_forward):
        total_loss = torch.Tensor([0.])
        iter = 0

        if self.rank == 0:
            writer.add_scalar('lr', scheduler.get_last_lr()[0], global_iter) # TODO: move inside the loop

        for batch in tqdm(loader, disable=(self.rank != 0), ncols=80):
            for key in batch:
                batch[key] = batch[key].to(device)

            efield_clone = batch['molecule_data'].efield.clone()
            batch['molecule_data'].efield = batch['molecule_data'].efield[:, :(self.cfg.time_cond + self.cfg.time_future) * 10]
            device_batch_size = batch['state_data'].mol_batch.max().cpu().item() + 1

            # Push-forward needs at least 2 samples, otherwise the sampled PF subset is empty.
            do_pf = push_forward and device_batch_size >= 2

            optimizer.zero_grad()

            if do_pf:
                # Shuffling should have ensured the randomness. But do another sampling anyway to make sure.
                # Assume even batch size
                pf_batch_mask = np.zeros(device_batch_size, dtype=bool)
                pf_batch_mask[:device_batch_size // 2] = True
                self.rng.shuffle(pf_batch_mask)
                pf_batch_mask = torch.tensor(pf_batch_mask).to(device) # which samples in the batch should do pf
                pf_mol_mask = pf_batch_mask[batch['state_data'].mol_batch] # which nodes in graph_data should do pf
                pf_global_mask = pf_batch_mask[batch['molecule_data'].batch] # which nodes in global_data should do pf

                def sample_batch(batch, mask):
                    _, contiguous_batch = torch.unique(batch[mask], return_inverse=True)
                    return contiguous_batch

                # Sample pf batch
                pf_batch = {}
                # Filter global data, using pf_batch_mask and pf_global_mask
                pf_batch['molecule_data'] = Data(atom_type=batch['molecule_data'].atom_type[pf_global_mask],
                                            atom_pos=batch['molecule_data'].atom_pos[pf_global_mask],
                                            batch=sample_batch(batch['molecule_data'].batch, pf_global_mask),
                                            occ=batch['molecule_data'].occ[pf_batch_mask[batch['molecule_data'].occ_batch]],
                                            # occ_batch=, # don't really need this
                                            num_atoms=batch['molecule_data'].num_atoms[pf_batch_mask],
                                            efield=batch['molecule_data'].efield[pf_batch_mask],
                                            coef_mask=batch['molecule_data'].coef_mask[pf_global_mask],
                                            t=batch['molecule_data'].t[pf_batch_mask])
                
                # Filter graph data using pf_mol_mask, pf_global_mask and pf_batch_mask
                pf_batch['state_data'] = Data(atom_type=batch['state_data'].atom_type[pf_mol_mask],
                                atom_pos=batch['state_data'].atom_pos[pf_mol_mask],
                                coef_0=batch['state_data'].coef_0[:, pf_mol_mask, :, :],
                                delta_coef_cond=batch['state_data'].delta_coef_cond[:, pf_mol_mask, :, :],
                                delta_coef_target=batch['state_data'].delta_coef_target[:, pf_mol_mask, :, :],
                                state_phase_cond=batch['state_data'].state_phase_cond[:, pf_batch_mask[batch['state_data'].state_ind_batch]],
                                state_phase_target=batch['state_data'].state_phase_target[:, pf_batch_mask[batch['state_data'].state_ind_batch]],
                                batch=sample_batch(batch['state_data'].batch, pf_mol_mask),
                                state_atom_batch=sample_batch(batch['state_data'].state_atom_batch, pf_mol_mask),
                                state_ind_batch=sample_batch(batch['state_data'].state_ind_batch, pf_batch_mask[batch['state_data'].state_ind_batch]), # same as occ_batch in global_data
                                mol_batch=sample_batch(batch['state_data'].mol_batch, pf_mol_mask),
                                coef_mask=batch['state_data'].coef_mask[pf_mol_mask],
                                num_atoms=batch['state_data'].num_atoms[pf_batch_mask[batch['state_data'].state_ind_batch]],
                                num_states=batch['state_data'].num_states[pf_batch_mask])
                
                # Foward the sampled part
                self.model.eval()
                with torch.no_grad():
                    with torch.autocast(device_type=self.autocast_device_type, enabled=self.use_amp):
                        pred = self.model(pf_batch, max_state_samples=None)

                    pf_ratio = 1.0
                    new_cond = pf_batch['state_data'].delta_coef_target[:self.cfg.time_future] * (1 - pf_ratio) + pred['delta_coef_t_norm'] * pf_ratio

                    batch['state_data'].delta_coef_cond[:, pf_mol_mask, :, :] = new_cond
                    batch['molecule_data'].efield[pf_batch_mask] = efield_clone[pf_batch_mask, self.cfg.time_future * 10:]
                    batch['molecule_data'].t[pf_batch_mask] += self.cfg.time_future 
                self.model.train()
            
            with torch.autocast(device_type=self.autocast_device_type, enabled=self.use_amp):
                pred = self.model(batch, max_state_samples=self.cfg.max_state_samples)

            if do_pf:
                coef_target = batch['state_data'].delta_coef_target[:self.cfg.time_future]
                coef_target[:, pf_mol_mask, :, :] = batch['state_data'].delta_coef_target[self.cfg.time_future : self.cfg.time_future * 2, pf_mol_mask, :, :]
            else:
                coef_target = batch['state_data'].delta_coef_target[:self.cfg.time_future]

            coef_target_norm = coef_target[:, pred['sampled_mask'], :, :] # normalized in dataset
            # Construct mask for sampled states.
            
            with torch.autocast(device_type=self.autocast_device_type, enabled=self.use_amp):
                err = pred['delta_coef_t_norm'] - coef_target_norm
                atomwise_l2mae = err.flatten(start_dim=2).norm(dim=-1).mean(0)  # N
                if self.cfg.loss == 'atomwise':
                    loss = atomwise_l2mae
                else:
                    loss = torch_scatter.scatter_mean(atomwise_l2mae, index=batch['state_data'].mol_batch[pred['sampled_mask']], dim=0) # B

                # Assign weights to balance push forward for the first and last steps
                loss_weights = torch.ones_like(loss)
                if do_pf:
                    # Training input does not have previous or next bundle
                    no_previous_bundle_batch = batch['molecule_data'].t < self.cfg.time_start + self.cfg.time_future
                    no_next_bundle_batch = batch['molecule_data'].t >= self.cfg.dataset.T - self.cfg.time_future
                    if self.cfg.loss == 'atomwise':
                        no_previous_bundle = no_previous_bundle_batch[batch['state_data'].mol_batch[pred['sampled_mask']]]
                        no_next_bundle = no_next_bundle_batch[batch['state_data'].mol_batch[pred['sampled_mask']]]
                    loss_weights[no_previous_bundle] = 2.
                    loss_weights[no_next_bundle] = 0.
                loss = torch.sum(loss * loss_weights) / torch.sum(loss_weights)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
            else:
                loss.backward()

            # clip gradient if needed
            if self.cfg.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)

            if self.use_amp:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()

            scheduler.step()

            total_loss = total_loss + loss.detach().cpu()
            iter += 1

            # increment global iter
            global_iter += 1

            # average and log
            loss_list = [
                         ('loss_l2mae', loss),
                        ]

            for tag, value in loss_list:
                value = value.detach().cpu()
                if self.cfg.use_ddp:
                    dist.all_reduce(value)
                    value /= self.world_size
                if self.rank == 0:
                    writer.add_scalar(f'Train/{tag}', value.item(), global_iter)

        if self.cfg.use_ddp:
            dist.all_reduce(total_loss)
            total_loss = total_loss / self.world_size
        train_loss = total_loss.item() / iter
        if self.rank == 0:
            writer.add_scalar(f'Train/loss', train_loss, global_iter)
            if do_pf:
                writer.add_scalar(f'Train/pf_ratio', pf_ratio, global_iter)

        return train_loss, global_iter


    def valid_one_epoch_onestep(self, loader, device, writer, global_iter):
        loss_l2mae = torch.Tensor([0.])

        iter = 0
        with torch.no_grad():
            for batch in tqdm(loader, disable=(self.rank != 0), ncols=80):
                for key in batch:
                    batch[key] = batch[key].to(device)

                batch['molecule_data'].efield = batch['molecule_data'].efield[:, :(self.cfg.time_cond + self.cfg.time_future) * 10]

                with torch.autocast(device_type=self.autocast_device_type, enabled=self.use_amp):
                    pred = self.model(batch)

                coef_target = batch['state_data'].delta_coef_target[:self.cfg.time_future]
                coef_target_norm = coef_target
                loss_l2mae += l2mae(pred=pred['delta_coef_t_norm'], target=coef_target_norm, mask=batch['state_data'].coef_mask).cpu()
              
                iter += 1

        all_losses = {}
        # average and log
        loss_list = [
                     ('loss_l2mae', loss_l2mae),
                    ]

        for tag, value in loss_list:
            value = value.detach().cpu()
            if self.cfg.use_ddp:
                dist.all_reduce(value)
                value = value / self.world_size
            value = value / iter
            all_losses[tag] = value.item()
            if self.rank == 0:
                writer.add_scalar(f'Valid_onestep/{tag}', value.item(), global_iter)

        return all_losses['loss_l2mae']


    def valid_one_epoch_rollout(self, loader, device, writer, global_iter):
        Ts = [8, 16, 32, 64, 100]

        loss_l2mae = {}
        loss_scaled_l2 = {}
        for T in Ts:
            loss_l2mae[T] = torch.Tensor([0.])
            loss_scaled_l2[T] = torch.Tensor([0.])

        iter = 0
        with torch.no_grad():
            for batch in tqdm(loader, disable=(self.rank != 0), ncols=80):
                for key in batch:
                    batch[key] = batch[key].to(device).clone()

                efield_clone = batch['molecule_data'].efield.clone()

                preds_norm = []
                preds_phase = []
                rollout_steps = math.ceil(self.cfg.dataset.T / self.cfg.time_future)
                for i in range(rollout_steps):
                    device_batch_size = batch['state_data'].mol_batch.max().cpu().item() + 1
                    if i == 0:
                        batch['pf_cond'] = torch.zeros(device_batch_size, dtype=torch.long).to(device)
                    else:
                        batch['pf_cond'] = torch.ones(device_batch_size, dtype=torch.long).to(device)

                    batch['molecule_data'].efield = efield_clone[:, self.cfg.time_future * 10 * i : self.cfg.time_future * 10 * i + (self.cfg.time_cond + self.cfg.time_future) * 10]
                    if batch['molecule_data'].efield.shape[1] < (self.cfg.time_cond + self.cfg.time_future) * 10:
                        # pad
                        pad_len = (self.cfg.time_cond + self.cfg.time_future) * 10 - batch['molecule_data'].efield.shape[1]
                        batch['molecule_data'].efield = torch.cat([batch['molecule_data'].efield, torch.zeros(efield_clone.shape[0], pad_len).to(efield_clone.device)], dim=1)

                    with torch.autocast(device_type=self.autocast_device_type, enabled=self.use_amp):
                        pred = self.model(batch)
                    preds_norm.append(pred['delta_coef_t_norm'])
                    batch['state_data'].delta_coef_cond = pred['delta_coef_t_norm']
                    if self.cfg.pred_phase:
                        preds_phase.append(pred['state_phase_t'])
                        batch['state_data'].state_phase_cond = pred['state_phase_t']

                preds_norm = torch.cat(preds_norm, dim=0)[:self.cfg.dataset.T]
                coef_target = batch['state_data'].delta_coef_target
           
                coef_target_norm = coef_target
                for T in Ts:
                    loss_l2mae[T] += l2mae(pred=preds_norm[:T], target=coef_target_norm[:T], mask=batch['state_data'].coef_mask).cpu()
                    loss_scaled_l2[T] += scaled_l2(pred=preds_norm[:T], target=coef_target_norm[:T], 
                                                   graph_batch=batch['state_data'].batch, state_ind_batch=batch['state_data'].state_ind_batch).cpu()

                iter += 1

        all_losses = {}
        # average and log
        loss_list = []
        for T in Ts:
            loss_list.append((f'Valid_rollout_l2mae/Avg@{T}', loss_l2mae[T]))
            loss_list.append((f'Valid_rollout_scaled_l2/Avg@{T}', loss_scaled_l2[T]))

        for tag, value in loss_list:
            value = value.detach().cpu()
            if self.cfg.use_ddp:
                dist.all_reduce(value)
                value = value / self.world_size
            value = value / iter
            all_losses[tag] = value.item()
            if self.rank == 0:
                writer.add_scalar(f'{tag}', value.item(), global_iter)

        return all_losses['Valid_rollout_scaled_l2/Avg@100']



    def train(self):
        print(f"Running DDP training on rank {self.rank}.")
        print(self.output_dir)

        if self.rank == 0:
            writer = SummaryWriter(os.path.join(self.output_dir, 'tb'))
            # log config
            writer.add_text('config', OmegaConf.to_yaml(self.cfg), 0)
        else:
            writer = None

        global_iter = 0
        start_epoch = 0
        self.valid_losses = []  # history of valid losses for early stop and increase prediction steps

        assert self.cfg.num_sessions == len(self.cfg.num_epochs)
        global_epoch = start_epoch

        iters_per_epoch = math.ceil(len(self.train_data_module.train_ds) / self.cfg.batch_size)
        total_num_iters = iters_per_epoch * (self.cfg.num_epochs[0] + self.cfg.num_epochs[1])

        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=1e-5)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=1e-3)
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=0.0)

        scheduler = CosineAnnealingLR(optimizer, T_max=total_num_iters)

        train_loader = self.train_data_module.train_dataloader(rank=self.rank)
        valid_onestep_loader = self.valid_onestep_data_module.valid_dataloader(rank=self.rank)
        valid_rollout_loader = self.valid_rollout_data_module.valid_dataloader(rank=self.rank)

        for session in range(self.cfg.num_sessions): # potentially do training with multiple sessions, not used

            if self.cfg.use_ddp:
                self.model.module.add_noise = self.cfg.add_noise
            else:
                self.model.add_noise = self.cfg.add_noise
        
            do_pf = session > 0 and self.cfg.push_forward
          
            self.best_ckpts = {} # restart saving
                
            for epoch in range(self.cfg.num_epochs[session]):
                # set epoch for ddp sampler
                if self.cfg.use_ddp:
                    self.train_data_module.train_sampler.set_epoch(global_epoch)

                # log lr
                if self.rank == 0:
                    writer.add_scalar('epoch', global_epoch, global_iter)

                # training
                self.model.train(True)
                loss_train, global_iter = self.train_one_epoch(optimizer=optimizer, loader=train_loader,
                                                            device=self.device, writer=writer, 
                                                            global_iter=global_iter, scheduler=scheduler, push_forward=do_pf)

                # validation
                if (epoch + 1) % self.cfg.valid_every == 0:
                    self.model.eval()
                    loss_valid_onestep = self.valid_one_epoch_onestep(loader=valid_onestep_loader, device=self.device, writer=writer, global_iter=global_iter)
                    loss_valid_rollout = self.valid_one_epoch_rollout(loader=valid_rollout_loader, device=self.device, writer=writer, global_iter=global_iter)

                    if self.rank == 0:
                        print('epoch {:3d}, train loss {:10.5f}, valid loss onetep {:10.5f}, valid loss rollout {:10.5f}'.format(global_epoch, loss_train, loss_valid_onestep, loss_valid_rollout))

                if self.rank == 0 and (epoch + 1) % self.cfg.valid_every == 0:
                    print('saving checkpoint...')
                    self.save(epoch=global_epoch, global_iter=global_iter, optimizer=optimizer, scheduler=scheduler, valid_loss=loss_valid_onestep, session=session)

                global_epoch += 1
                if self.rank == 0:
                    writer.flush()


    def save(self, epoch, global_iter, optimizer, scheduler, valid_loss, session):
        save_dir = os.path.join(self.output_dir, 'ckpt')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        def save_one(save_path):
            torch.save({
            'epoch': epoch,
            'global_iter': global_iter,
            'model_state_dict': self.model.module.state_dict() if self.cfg.use_ddp else self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)

        # save checkpoint if valid loss is smaller than one of the saved ckpts losses
        if len(self.best_ckpts) == 0 or valid_loss <= max(self.best_ckpts.values()):
            # save_one(os.path.join(save_dir, f'epoch={epoch:03d}.pt'))
                
            # update best if needed
            if len(self.best_ckpts) == 0 or valid_loss <= min(self.best_ckpts.values()):
                save_one(os.path.join(save_dir, f'best_s{session}.pt'))

            # update record
            self.best_ckpts[epoch] = valid_loss
