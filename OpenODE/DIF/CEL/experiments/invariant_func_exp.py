import copy

import torch
import os
from tqdm import tqdm
import numpy as np
from CEL.utils.register import register
import torch.nn as nn
import optax
import equinox as eqx
import jax
from CEL.data.data_manager import MyDataLoader
from CEL.networks.models.meta_model import ModelClass
import pathlib
from munch import Munch
import wandb
import torch.nn.functional as F
from torchdiffeq import odeint as nn_odeint
from torch_scatter import scatter, scatter_std

from .meta_exp import ExpClass

import time


class InvariantFuncExperiment(ExpClass):

    def __init__(self, dataloader: MyDataLoader, model: ModelClass,
                 weight_decay: float = 0.0, lr: float = 1e-3, 
                 max_epoch: int = 100, ctn_epoch: int = 0,
                 device: int = 0, log_interval: int = 20, wandb_logger: bool = True,
                 DEBUG: bool = False, num_sampling_per_batch: int = 1, partial_eval: bool = False,
                 lambda_inv: float = 1.0, inference_out: str = 'inv', inv_type: str = 'VREx', lambda_side: float = 1.0,
                 adapt_steps: int = 10, adapt_lr: float = 1e-2, **kwargs):
        r'''
        Invariant Physical Experiment
        Args:
            dataloader:
            model:
            weight_decay:
            lr:
            max_epoch:
            ctn_epoch:
            device:
            log_interval: Interval for wandb logging
            wandb_logger:
            DEBUG:
            num_sampling_per_batch: Each batch trajectory will be sampled this many times
            partial_eval: Whether to partially evaluate the model during testing phase because of the long computation time
            lambda_inv: Parameter for VREx or other invariant learning loss
            inference_out: During inference phases, whether to output the invariant or the combined derivative
        '''
        super().__init__()
        self.model = model
        self.loader = {set_name: dataloader.__getattribute__(set_name) for set_name in
                       ['train', 'test', 'id_test', 'ood_test'] if hasattr(dataloader, set_name)}
        self.pbar_setting = {'colour': '#a48fff', 'bar_format': '{l_bar}{bar:20}{r_bar}',
                             'dynamic_ncols': True, 'ascii': '░▒█'}
        
        # exp configs
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.ctn_epoch = ctn_epoch
        self.device = device
        self.log_interval = log_interval
        self.wandb_logger = wandb_logger
        self.DEBUG = DEBUG

        self.num_sampling_per_batch = num_sampling_per_batch
        self.partial_eval = partial_eval
        self.inference_out = inference_out

        # --- random seed generator ---
        self.sampling_generator = np.random.default_rng(0)

        # --- optimizer ---
        self.inv_type = inv_type
        self.lambda_inv = lambda_inv
        self.lambda_side = lambda_side

        # --- adaptation ---
        self.adapt_steps = adapt_steps
        self.adapt_lr = adapt_lr

    def inv_traj_inference(self, data, input_length):

        states = data['states'].to(self.device)
        W = data['W'].to(self.device)
        t = data['t'].to(self.device)[0]
        dydt = data['dy'].to(self.device)
        adapt_optim = torch.optim.Adam([self.model.hyper_network.adaptation_embedding], lr=self.adapt_lr)
        self.model.hyper_network.init_adaptation()

        sample_ids = self.sampling_generator.integers(0, input_length,
                                                      size=(self.adapt_steps, states.shape[0]))
        sample_ids = torch.from_numpy(sample_ids).to(self.device)
        arange_idx = torch.arange(states.shape[0]).to(self.device)
        loss = {}
        pbar = tqdm(sample_ids, **self.pbar_setting)
        for idx in pbar:
            pred_dydt = self.model(states[arange_idx, idx], None, W)  # Select (#0, idx[0]), (#1, idx[1]), ...
            pred_loss = F.mse_loss(pred_dydt, dydt[arange_idx, idx])
            adapt_loss = {'pred': {'weight': 1.0, 'value': pred_loss}}
            self.log_loss(loss, {key: loss_term['value'].item() for key, loss_term in adapt_loss.items()})
            adapt_optim.zero_grad()
            self.loss_wsum(adapt_loss).backward()
            # If RuntimeError: leaf variable has been moved into the graph interior, deepcopy the input model before deserializing.
            adapt_optim.step()
            pbar.set_description('|'.join([f'{loss_key}: {loss_value:.2e}' for loss_key, loss_value in self.reduce_loss(loss).items()]))
        # print(self.reduce_loss(loss))
        self.count = 0

        with torch.no_grad():
            def inv_forward_ode_func(t, y):
                print(f"\rCount: {self.count}", end="")
                self.count += 1
                pred_y = self.model(y, None, W, forecast=True)
                return pred_y

            def combine_forward_ode_func(t, y):
                pred_y = self.model(y, None, W, forecast=True)
                return pred_y

            forward_ode_func = {'inv': inv_forward_ode_func, 'combine': combine_forward_ode_func}

            try:
                tik = time.time()
                pred_y = nn_odeint(forward_ode_func[self.inference_out], states[:, 0], t)
                print(f"\nODE integral time taken: {time.time() - tik}") # about 5 mins
            except Exception as e:
                print(f"Exception {e}")
                pred_y = nn_odeint(forward_ode_func[self.inference_out], states[:, 0], t, method='rk4')

            nrmse = torch.sqrt(torch.mean((pred_y[input_length:] - states.transpose(0, 1)[input_length:]) ** 2)) / torch.std(states.transpose(0, 1)[input_length:])
        return nrmse.item()

    def train_batch(self, data) -> dict:
        r"""
        Train a batch. (Project use only)

        Returns:
            Calculated loss.
        """
        # data = data.to(self.config.device)
        states = data['states'].to(self.device)
        dydt = data['dy'].to(self.device)
        W = data['W'].to(self.device)
        t = data['t'].to(self.device)[0]
        env = data['env'].to(self.device)

        # --- Sample pairs ---
        loss = {}
        sample_ids = self.sampling_generator.integers(0, states.shape[1], size=(self.num_sampling_per_batch, states.shape[0]))
        sample_ids = torch.from_numpy(sample_ids).to(self.device)
        arange_idx = torch.arange(states.shape[0]).to(self.device)
        for idx in tqdm(sample_ids, disable=True):
            pred_dydt = self.model(states[arange_idx, idx], env, W)  # Select (#0, idx[0]), (#1, idx[1]), ...
            # print(pred_dydt.shape)
            # print(dydt[arange_idx, idx].shape)
            pred_loss = F.mse_loss(pred_dydt, dydt[arange_idx, idx])
            if self.inv_type == 'VREx':
                inv_loss = F.mse_loss(inv_dydt, dydt[arange_idx, idx], reduction='none')
                inv_loss = scatter(inv_loss, env, dim=0, reduce='mean')
                inv_loss = inv_loss.var(dim=0) # VREx loss across each state
                inv_loss = inv_loss.mean()
                training_loss = {'pred': {'weight': 1.0, 'value': pred_loss},
                                 'inv': {'weight': self.lambda_inv, 'value': inv_loss}}
            elif self.inv_type == 'SVREx':
                inv_loss = F.mse_loss(inv_dydt, dydt[arange_idx, idx], reduction='none')
                inv_loss = inv_loss.var(dim=0)
                inv_loss = inv_loss.mean()
                training_loss = {'pred': {'weight': 1.0, 'value': pred_loss},
                                 'inv': {'weight': self.lambda_inv, 'value': inv_loss}}
            elif self.inv_type == 'IVREx':
                inv_loss = F.mse_loss(inv_dydt, dydt[arange_idx, idx], reduction='none')
                inv_loss = scatter_std(inv_loss, env, dim=0, unbiased=False).square()
                inv_loss = inv_loss.mean()
                training_loss = {'pred': {'weight': 1.0, 'value': pred_loss},
                                 'inv': {'weight': self.lambda_inv, 'value': inv_loss}}
            elif self.inv_type == 'sideIVREx':
                inv_loss = F.mse_loss(inv_dydt, dydt[arange_idx, idx], reduction='none')
                inv_loss = scatter_std(inv_loss, env, dim=0, unbiased=False).square()
                inv_loss = inv_loss.mean()
                side_loss = F.mse_loss(inv_dydt, dydt[arange_idx, idx])
                training_loss = {'pred': {'weight': 0.0, 'value': pred_loss},
                                 'inv': {'weight': self.lambda_inv, 'value': inv_loss},
                                 'side': {'weight': self.lambda_side, 'value': side_loss}}
            elif self.inv_type == 'DANN':
                inv_dydt, inv_eoutput = inv_dydt
                env_loss = F.cross_entropy(inv_eoutput, env)
                side_loss = F.mse_loss(inv_dydt, dydt[arange_idx, idx])
                training_loss = {'pred': {'weight': 1.0, 'value': pred_loss},
                                 'env': {'weight': 1.0, 'value': env_loss},
                                 'side': {'weight': self.lambda_side, 'value': side_loss}}
            else:
                training_loss = {'pred': {'weight': 1.0, 'value': pred_loss}}
            self.log_loss(loss, {key: loss_term['value'].item() for key, loss_term in training_loss.items()})
            self.optimizer.zero_grad()
            self.loss_wsum(training_loss).backward()
            # If RuntimeError: leaf variable has been moved into the graph interior, deepcopy the input model before deserializing.
            self.optimizer.step()

        return self.reduce_loss(loss)

    def reduce_loss(self, loss):
        return {key: np.mean(value) for key, value in loss.items()}

    def loss_wsum(self, loss):
        return sum(loss_term['weight'] * loss_term['value'] for loss_term in loss.values())

    def log_loss(self, loss, new_loss):
        for key, value in new_loss.items():
            if key not in loss:
                loss[key] = []
            loss[key].append(value)
        return loss

    def __call__(self, *args, **kwargs):
        # config model
        print('#D#Config model')
        self.config_model('train')

        # config optimizer
        print('#D#Config optimizer')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epoch)
        best_test_error = 1e10
        best_ood_error = 1e10
        # train the model
        for epoch in range(self.ctn_epoch, self.max_epoch):
            self.epoch = epoch
            print(f'#IN#Epoch {epoch}:')

            loss_log = {}
            mean_loss = 0
            spec_loss = 0

            # self.ood_algorithm.stage_control(self.config)

            pbar = tqdm(enumerate(self.loader['train']), total=len(self.loader['train']), **self.pbar_setting)
            for index, traj_batch in pbar:

                # Parameter for DANN
                p = (index / len(self.loader['train']) + epoch) / self.max_epoch
                self.alpha = 2. / (1. + np.exp(-10 * p)) - 1

                # train a batch
                train_loss = self.train_batch(traj_batch)
                loss_log = self.log_loss(loss_log, train_loss)
                mean_loss = self.reduce_loss(loss_log)
                # print(f'iteration: {it}, mean loss: {mean_loss:.4f}')
                pbar.set_description('|'.join([f'{loss_key}: {loss_value:.2e}' for loss_key, loss_value in mean_loss.items()]))

            # mean_test_error = self.evaluate('id_test')
            # if mean_test_error < best_test_error:
            #     best_test_error = mean_test_error
            #     print(f'Best test error: {best_test_error:.4f}')
            # print(f'Test error: {mean_test_error:.4f}, Best test error: {best_test_error:.4f}')
            # self.evaluate('train')
            if epoch % 10 == 0:
                mean_ood_error = self.evaluate('ood_test')
                if mean_ood_error < best_ood_error:
                    best_ood_error = mean_ood_error
                    self.save_model(path=pathlib.Path(os.environ['STORAGE_DIR']) / 'exp' / kwargs['exp_hash'],
                                    dumped_config=kwargs['dumped_config'],
                                    epoch=epoch)
                    print(f'Best OOD test error: {best_ood_error:.4f}')
                print(f'Test OOD error: {mean_ood_error:.4f}, Best OOD test error: {best_ood_error:.4f}')
                print(f'lr: {self.optimizer.param_groups[0]["lr"]}')
                # --- Logging ---
                if epoch % self.log_interval == 0 and self.wandb_logger:
                    wandb.log({
                        'train_loss': mean_loss,
                        'test_NRMSE': {
                            # 'in_domain': mean_test_error,
                            'ood': mean_ood_error,
                        },
                        'Best_NRMSE': {
                            # 'in_domain': best_test_error,
                            'ood': best_ood_error,
                        },
                        'lr': self.optimizer.param_groups[0]['lr'],
                    }, step=epoch)
            # self.scheduler.step(epoch)

        print('#IN#Training end.')

    def evaluate(self, split: str):
        self.model.eval()
        test_errors = []
        input_length = self.loader[split].dataset.input_length
        for index, data in tqdm(enumerate(self.loader[split])):
            # errors = self.inference_JAX(self.model, data, input_length)
            errors = self.inv_traj_inference(data, input_length)
            test_errors.append(errors)
            print(np.mean(test_errors))
            if self.partial_eval and index >= 0:
                break
        mean_test_error = np.mean(test_errors)
        self.model.train()
        return mean_test_error

    def save_model(self, path: pathlib.Path, dumped_config: str, epoch: int):
        os.makedirs(path, exist_ok=True)
        with open(path / f'config.yml', 'w') as f:
            f.write(dumped_config)
        if issubclass(self.model.__class__, eqx.Module):
            with open(path / f'{epoch}.eqx', 'wb') as f:
                eqx.tree_serialise_leaves(f, self.model)
        else:
            torch.save(self.model.state_dict(), path / f'{epoch}.pt')

    def load_model(self, path: pathlib.Path, epoch: int):
        if issubclass(self.model.__class__, eqx.Module):
            with open(path / f'{epoch}.eqx', 'rb') as f:
                return eqx.tree_deserialise_leaves(f, self.model)
        else:
            self.model.load_state_dict(torch.load(path / f'{epoch}.pt'))

    def config_model(self, mode: str, load_param=False):
        r"""
        A model configuration utility. Responsible for transiting model from CPU -> GPU and loading checkpoints.
        Args:
            mode (str): 'train' or 'test'.
            load_param: When True, loading test checkpoint will load parameters to the GNN model.

        Returns:
            Test score and loss if mode=='test'.
        """
        self.model.to(self.device)
        self.model.train()

        # load checkpoint
        # if mode == 'train' and self.tr_ctn:
        #     ckpt = torch.load(os.path.join(self.config.ckpt_dir, f'last.ckpt'))
        #     self.model.load_state_dict(ckpt['state_dict'])
        #     best_ckpt = torch.load(os.path.join(self.config.ckpt_dir, f'best.ckpt'))
        #     self.config.metric.best_stat['score'] = best_ckpt['val_score']
        #     self.config.metric.best_stat['loss'] = best_ckpt['val_loss']
        #     self.ctn_epoch = ckpt['epoch'] + 1
        #     print(f'#IN#Continue training from Epoch {ckpt["epoch"]}...')
        #
        # if mode == 'test':
        #     try:
        #         ckpt = torch.load(self.config.test_ckpt, map_location=self.config.device)
        #     except FileNotFoundError:
        #         print(f'#E#Checkpoint not found at {os.path.abspath(self.config.test_ckpt)}')
        #         exit(1)
        #     if os.path.exists(self.config.id_test_ckpt):
        #         id_ckpt = torch.load(self.config.id_test_ckpt, map_location=self.config.device)
        #         # model.load_state_dict(id_ckpt['state_dict'])
        #         print(f'#IN#Loading best In-Domain Checkpoint {id_ckpt["epoch"]}...')
        #         print(f'#IN#Checkpoint {id_ckpt["epoch"]}: \n-----------------------------------\n'
        #               f'Train {self.config.metric.score_name}: {id_ckpt["train_score"]:.4f}\n'
        #               f'Train Loss: {id_ckpt["train_loss"].item():.4f}\n'
        #               f'ID Validation {self.config.metric.score_name}: {id_ckpt["id_val_score"]:.4f}\n'
        #               f'ID Validation Loss: {id_ckpt["id_val_loss"].item():.4f}\n'
        #               f'ID Test {self.config.metric.score_name}: {id_ckpt["id_test_score"]:.4f}\n'
        #               f'ID Test Loss: {id_ckpt["id_test_loss"].item():.4f}\n'
        #               f'OOD Validation {self.config.metric.score_name}: {id_ckpt["val_score"]:.4f}\n'
        #               f'OOD Validation Loss: {id_ckpt["val_loss"].item():.4f}\n'
        #               f'OOD Test {self.config.metric.score_name}: {id_ckpt["test_score"]:.4f}\n'
        #               f'OOD Test Loss: {id_ckpt["test_loss"].item():.4f}\n')
        #         print(f'#IN#Loading best Out-of-Domain Checkpoint {ckpt["epoch"]}...')
        #         print(f'#IN#Checkpoint {ckpt["epoch"]}: \n-----------------------------------\n'
        #               f'Train {self.config.metric.score_name}: {ckpt["train_score"]:.4f}\n'
        #               f'Train Loss: {ckpt["train_loss"].item():.4f}\n'
        #               f'ID Validation {self.config.metric.score_name}: {ckpt["id_val_score"]:.4f}\n'
        #               f'ID Validation Loss: {ckpt["id_val_loss"].item():.4f}\n'
        #               f'ID Test {self.config.metric.score_name}: {ckpt["id_test_score"]:.4f}\n'
        #               f'ID Test Loss: {ckpt["id_test_loss"].item():.4f}\n'
        #               f'OOD Validation {self.config.metric.score_name}: {ckpt["val_score"]:.4f}\n'
        #               f'OOD Validation Loss: {ckpt["val_loss"].item():.4f}\n'
        #               f'OOD Test {self.config.metric.score_name}: {ckpt["test_score"]:.4f}\n'
        #               f'OOD Test Loss: {ckpt["test_loss"].item():.4f}\n')
        #
        #         print(f'#IN#ChartInfo {id_ckpt["id_test_score"]:.4f} {id_ckpt["test_score"]:.4f} '
        #               f'{ckpt["id_test_score"]:.4f} {ckpt["test_score"]:.4f} {ckpt["val_score"]:.4f}', end='')
        #
        #     else:
        #         print(f'#IN#No In-Domain checkpoint.')
        #         # model.load_state_dict(ckpt['state_dict'])
        #         print(f'#IN#Loading best Checkpoint {ckpt["epoch"]}...')
        #         print(f'#IN#Checkpoint {ckpt["epoch"]}: \n-----------------------------------\n'
        #               f'Train {self.config.metric.score_name}: {ckpt["train_score"]:.4f}\n'
        #               f'Train Loss: {ckpt["train_loss"].item():.4f}\n'
        #               f'Validation {self.config.metric.score_name}: {ckpt["val_score"]:.4f}\n'
        #               f'Validation Loss: {ckpt["val_loss"].item():.4f}\n'
        #               f'Test {self.config.metric.score_name}: {ckpt["test_score"]:.4f}\n'
        #               f'Test Loss: {ckpt["test_loss"].item():.4f}\n')
        #
        #         print(
        #             f'#IN#ChartInfo {ckpt["test_score"]:.4f} {ckpt["val_score"]:.4f}', end='')
        #     if load_param:
        #         if self.config.ood.ood_alg != 'EERM':
        #             self.model.load_state_dict(ckpt['state_dict'])
        #         else:
        #             self.model.gnn.load_state_dict(ckpt['state_dict'])
        #     return ckpt["test_score"], ckpt["test_loss"]

    def run(self):
        self.intervention.run(self.intervention_type, **self.intervention_kwargs)
        self.model.fit(self.data)
        self.results = self.model.evaluate(self.data)
