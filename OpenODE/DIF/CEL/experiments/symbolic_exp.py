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

from .meta_exp import ExpClass


@register.experiment_register
class SymbolicExp(ExpClass):
    def __init__(self, dataloader: MyDataLoader, model: ModelClass,
                 weight_decay: float = 0.0, lr: float = 1e-3, xi_lr: float = 1e-3,
                 max_epoch: int = 100, ctn_epoch: int = 0,
                 device: int = 0, log_interval: int = 20, wandb_logger: bool = True,
                 DEBUG: bool = False):
        super().__init__()
        self.model = model
        self.loader = {set_name: dataloader.__getattribute__(set_name) for set_name in
                       ['train', 'test', 'id_test', 'ood_test'] if hasattr(dataloader, set_name)}
        self.pbar_setting = {'colour': '#a48fff', 'bar_format': '{l_bar}{bar:20}{r_bar}',
                             'dynamic_ncols': True, 'ascii': '░▒█'}
        
        # exp configs
        self.lr = lr
        self.xi_lr = xi_lr
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.ctn_epoch = ctn_epoch
        self.device = device
        self.log_interval = log_interval
        self.wandb_logger = wandb_logger
        self.DEBUG = DEBUG
        

    # @eqx.filter_jit()
    def reduced_loss(self, diff_model, static_model, states, t, X, input_length, prediction_length):
        model = eqx.combine(diff_model, static_model)

        errors = self.loss(model, states, t, X, input_length, prediction_length)
        mean_error = errors['prediction'].mean() + 0 * errors['alignment'].mean() + 0 * errors['l2'].mean() + 1e-2 * errors['xi'].mean()
        return mean_error

    @eqx.filter_jit
    def model_forward(self, model, states, t, X, input_length, prediction_length):
        # states are cropped here to avoid future data leaking.
        y_past = states[:, :input_length, :]
        pred = jax.vmap(model, (0, 0, 0, None, None))(y_past, t, X, input_length, prediction_length)
        return pred

    # @eqx.filter_jit
    def loss(self, model, states, t, X, input_length, prediction_length):
        pred, Ws, xi = self.model_forward(model, states, t, X, input_length, prediction_length)
        # MSE loss
        prediction_errors = optax.l2_loss(pred[:, :prediction_length], states[:, input_length: input_length + prediction_length])
        alignment_errors = optax.l2_loss(Ws[:, -1], Ws[:, -2])
        l2_errors = Ws[:, -1] ** 2
        xi_errors = xi

        if self.equation_print:
            sigmoid_xi = 1 / (1 + np.exp( - np.array(model.symcde_func.xi.val).reshape(2, -1)))
            equation = sigmoid_xi * np.array(Ws[0, -1].val).reshape(2, -1) * np.tile(np.array(model.symcde_func.sym_equation)[None, :], (2, 1))
            # equation_str = [' + '.join([str(term) for term in eq if str(term) > 0]) for eq in equation]
            print(f'\nXi: {np.array(model.symcde_func.xi.val).reshape(2, -1)}\n'
                  f'Equation: {equation}')
            self.equation_print = False
        return {'prediction': prediction_errors, 'alignment': alignment_errors, 'l2': l2_errors, 'xi': xi_errors}

    # @eqx.filter_jit
    def train_batch(self, model, data, opt_state, input_length, prediction_length):
        # standard shape: (batch, times, channals)
        X, states, t = self.unpack_data(data)

        # --- freeze the SINDy model ---
        model_tree = jax.tree_util.tree_map(lambda x: True, model)
        freeze_filter = jax.tree_util.tree_map(lambda x: False, model.symcde_func.symbolic_model)
        dyn_fre_filter = eqx.tree_at(lambda tree: tree.symcde_func.symbolic_model, model_tree, freeze_filter)
        diff_model, static_model = eqx.partition(model, dyn_fre_filter)


        loss, grads = eqx.filter_value_and_grad(self.reduced_loss)(diff_model, static_model, states, t, X, input_length, prediction_length)
        # assert False, 'Check the gradients! Whether it is properly freezed.' CHECK COMPLETED
        updates, opt_state = self.optimizer.update([grads], opt_state)   # wrap grads with a list to avoid optax "callable" bug
        model = eqx.apply_updates(model, updates[0])  # remove the list wrapping
        return model, loss, opt_state

    def unpack_data(self, data):
        states = data['states'].numpy()
        if data.get('X') is not None:
            X = data['X'].numpy()
            # check X dimension
            if X.ndim == 2:
                X = X[:, :, None]
        else:
            X = None
        t = data['t'].numpy()
        return X, states, t

    def inference(self, model, data, input_length, prediction_length):
        X, states, t = self.unpack_data(data)
        # error = self.loss(model, states, t, X)
        pred, Ws, xi = self.model_forward(model, states, t, X, input_length, prediction_length)
        nrmse = np.sqrt(np.mean((pred - states[:, input_length:]) ** 2)) / np.std(states[:, input_length:])
        return nrmse

    def __call__(self, *args, **kwargs):
        print('#D#Config optimizer')
        default_optimizer = optax.adam(self.lr)
        xi_optimizer = optax.adam(self.xi_lr)

        optim_selection = [jax.tree_util.tree_map(lambda x: 'xi' if x is self.model.symcde_func.xi else 'default', self.model)] # wrap with a list which is not callable to avoid bugs
        self.optimizer = optax.multi_transform(
            {'default': default_optimizer, 'xi': xi_optimizer},
            optim_selection)
        # self.optimizer = optax.adam(self.lr)
        optim_state = self.optimizer.init([eqx.filter(self.model, eqx.is_inexact_array)])

        best_test_error = 1e10
        best_ood_error = 1e10
        # train the model
        prediction_length = 1
        for epoch in range(self.ctn_epoch, self.max_epoch):
            self.epoch = epoch
            self.equation_print = True
            max_prediciton_length = self.loader['train'].dataset[0]['t'].shape[0] - self.loader['train'].dataset.input_length
            print(f'#IN#Epoch {epoch}:')

            loss_log = []
            mean_loss = 0
            spec_loss = 0

            # self.ood_algorithm.stage_control(self.config)

            pbar = tqdm(enumerate(self.loader['train']), total=len(self.loader['train']), **self.pbar_setting)
            for index, data in pbar:

                # visualize a data
                # if index == 0:
                #     print(f'States: {data["states"]}\n'
                #           f'T: {data["t"]}')
                #     # visualize it using matplotlib
                #     import matplotlib.pyplot as plt
                #     plt.plot(data['t'][0].numpy(), data['states'][0].numpy())
                #     plt.show()
                #     exit(0)

                # Parameter for DANN
                # p = (index / len(self.loader['train']) + epoch) / self.max_epoch
                # self.alpha = 2. / (1. + np.exp(-10 * p)) - 1

                # train a batch
                for it in range(1):
                    if issubclass(self.model.__class__, eqx.Module):
                        # with jax.disable_jit(self.DEBUG):
                        self.model, loss, optim_state = self.train_batch(self.model, data, optim_state,
                                                                         input_length=self.loader[
                                                                                 'train'].dataset.input_length,
                                                                         prediction_length=prediction_length)
                        loss_log.append(loss)
                    mean_loss = np.mean(loss_log)
                    # print(f'iteration: {it}, mean loss: {mean_loss:.4f}')
                pbar.set_description(f'Loss: {mean_loss:.4f}')
            if mean_loss < 1e-1:
                prediction_length = min(prediction_length + 10, max_prediciton_length)

            mean_test_error = self.evaluate('id_test', prediction_length)
            if mean_test_error < best_test_error:
                best_test_error = mean_test_error
                print(f'Best test error: {best_test_error:.4f}')
            print(f'Test error: {mean_test_error:.4f}, Best test error: {best_test_error:.4f}')

            mean_ood_error = self.evaluate('ood_test', prediction_length)
            if mean_ood_error < best_ood_error:
                best_ood_error = mean_ood_error
                self.save_model(path=pathlib.Path(os.environ['STORAGE_DIR']) / 'exp' / kwargs['exp_hash'],
                                dumped_config=kwargs['dumped_config'],
                                epoch=epoch)
                print(f'Best OOD test error: {best_ood_error:.4f}')
            print(f'Test OOD error: {mean_ood_error:.4f}, Best OOD test error: {best_ood_error:.4f}')

            # --- Logging ---
            if epoch % self.log_interval == 0 and self.wandb_logger:
                wandb.log({
                    'train_loss': mean_loss,
                    'test_NRMSE': {
                        'in_domain': mean_test_error,
                        'ood': mean_ood_error,
                    },
                    'Best_NRMSE': {
                        'in_domain': best_test_error,
                        'ood': best_ood_error,
                    }
                }, step=epoch)

        print('#IN#Training end.')

    def evaluate(self, split: str, prediction_length):
        test_errors = []
        input_length = self.loader[split].dataset.input_length
        for index, data in enumerate(self.loader[split]):
            errors = self.inference(self.model, data, input_length, prediction_length)
            test_errors.append(errors)
        mean_test_error = np.mean(test_errors)
        return mean_test_error

    def save_model(self, path: pathlib.Path, dumped_config: str, epoch: int):
        if issubclass(self.model.__class__, eqx.Module):
            os.makedirs(path, exist_ok=True)
            with open(path / f'config.yml', 'w') as f:
                f.write(dumped_config)
            with open(path / f'{epoch}.eqx', 'wb') as f:
                eqx.tree_serialise_leaves(f, self.model)
        else:
            torch.save(self.model.state_dict(), path)

    def load_model(self, path: pathlib.Path, epoch: int):
        if issubclass(self.model.__class__, eqx.Module):
            with open(path / f'{epoch}.eqx', 'rb') as f:
                return eqx.tree_deserialise_leaves(f, self.model)
        else:
            self.model.load_state_dict(torch.load(path))

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
