import munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics

from CEL.utils.utils import fix_seed, make_basedir, convert_tensor
from CEL.utils.utils import CalculateNorm, Logger

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
# matplotlib.rcParams["figure.dpi"] = 100

_EPSILON = 1e-5

import os, time, io, logging, sys
from functools import partial

from torchvision.utils import make_grid

import collections

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from CEL.definitions import STORAGE_DIR
from CEL.data.data_manager import MyDataLoader
from CEL.networks.models.meta_model import ModelClass

from CEL.utils.register import register


class BaseExperiment(object):
    def __init__(self, exp_config):
        self.device = exp_config.device
        self.path = make_basedir(exp_config.path)
            
        if exp_config.random_seed is not None:
            fix_seed(exp_config.random_seed)

    def training(self, mode=True):
        for m in self.modules():
            m.train(mode)

    def evaluating(self):
        self.training(mode=False)

    def zero_grad(self):
        for optimizer in self.optimizers():
            optimizer.zero_grad()        

    def to(self, device):
        for m in self.modules():
            m.to(device)
        return self

    def modules(self):
        for name, module in self.named_modules():
            yield module

    def named_modules(self):
        for name, module in self._modules.items():
            yield name, module

    def datasets(self):
        for name, dataset in self.named_datasets():
            yield dataset

    def named_datasets(self):
        for name, dataset in self._datasets.items():
            yield name, dataset

    def optimizers(self):
        for name, optimizer in self.named_optimizers():
            yield optimizer

    def named_optimizers(self):
        for name, optimizer in self._optimizers.items():
            yield name, optimizer

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            if not hasattr(self, '_modules'):
                self._modules = collections.OrderedDict()
            self._modules[name] = value
        elif isinstance(value, DataLoader):
            if not hasattr(self, '_datasets'):
                self._datasets = collections.OrderedDict()
            self._datasets[name] = value
        elif isinstance(value, Optimizer):
            if not hasattr(self, '_optimizers'):
                self._optimizers = collections.OrderedDict()
            self._optimizers[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        if '_datasets' in self.__dict__:
            datasets = self.__dict__['_datasets']
            if name in datasets:
                return datasets[name]
        if '_optimizers' in self.__dict__:
            optimizers = self.__dict__['_optimizers']
            if name in optimizers:
                return optimizers[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __delattr__(self, name):
        if name in self._modules:
            del self._modules[name]
        elif name in self._datasets:
            del self._datasets[name]
        elif name in self._optimizers:
            del self._optimizers[name]
        else:
            object.__delattr__(self, name)

def show(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

class LoopExperiment(BaseExperiment):
    def __init__(
        self, dataloader, exp_config, root=None, **kwargs):
        super().__init__(exp_config, **kwargs)
        self.train = dataloader['train']
        self.test = dataloader['test']
        self.nepoch = exp_config.max_epoch
        self.logger = Logger(filename=os.path.join(self.path, 'log.txt'))
        print(' '.join(sys.argv))

    def train_step(self, batch, val=False):
        self.training()
        batch = convert_tensor(batch, self.device)
        loss, output = self.step(batch)
        with torch.no_grad():
            metric = self.metric(**output, **batch)

        return batch, output, loss, metric

    def val_step(self, batch, val=False):
        self.evaluating()
        with torch.no_grad():
            batch = convert_tensor(batch, self.device)
            loss, output = self.step(batch, backward=False)
            metric = self.metric(**output, **batch)

        return batch, output, loss, metric

    def log(self, epoch, iteration, metrics):
        message = '[{step}][{epoch}/{max_epoch}][{i}/{max_i}]'.format(
            step=epoch *len(self.train)+ iteration+1,
            epoch=epoch+1,
            max_epoch=self.nepoch,
            i=iteration+1,
            max_i=len(self.train)
        )
        for name, value in metrics.items():
            message += ' | {name}: {value:.2e}'.format(name=name, value=value.item() if isinstance(value, torch.Tensor) else value)
            
        print(message)

    def step(self, **kwargs):
        raise NotImplementedError

from .meta_exp import ExpClass
@register.experiment_register
class APHYNITYExp(LoopExperiment, ExpClass):
    def __init__(self, dataloader: MyDataLoader, net: ModelClass, exp_config: dict, **kwargs):
        exp_config = munch.Munch(exp_config)
        dataloader = {'train': dataloader.train, 'test': dataloader.test}
        super().__init__(dataloader, exp_config, **kwargs)

        self.traj_loss = nn.MSELoss()
        self.net = net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=exp_config.lr)

        self.min_op = exp_config.min_op
        self.tau_2 = exp_config.tau_2
        self.niter = exp_config.niter
        self._lambda = exp_config.lambda_0

        self.nlog = exp_config.log_interval
        self.nupdate = exp_config.save_gap
    
    def lambda_update(self, loss):
        self._lambda = self._lambda + self.tau_2 * loss
    
    def _forward(self, states, t, backward):
        target = states
        y0 = states[:, :, 0]
        pred = self.net(y0, t)
        loss = self.traj_loss(pred, target)
        aug_deriv = self.net.net_aug.get_derivatives(states)

        if self.min_op == 'l2_normalized':
            loss_op = ((aug_deriv.norm(p=2, dim=1) / (states.norm(p=2, dim=1) + _EPSILON)) ** 2).mean()
        elif self.min_op == 'l2':
            loss_op = (aug_deriv.norm(p=2, dim=1) ** 2).mean()

        if backward:
            loss_total = loss * self._lambda + loss_op

            loss_total.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        loss = {
            'loss': loss,
            'loss_op': loss_op,
        }

        output = {
            'states_pred'     : pred,
        }
        return loss, output

    def step(self, batch, backward=True):
        states = batch['states']
        t = batch['t'][0]
        loss, output = self._forward(states, t, backward)
        return loss, output

    def metric(self, states, states_pred, **kwargs):
        metrics = {}
        metrics['param_error'] = statistics.mean(abs(v1-float(v2))/v1 for v1, v2 in zip(self.train.dataset.params.values(), self.net.get_pde_params().values()))
        metrics.update(self.net.get_pde_params())
        metrics.update({f'{k}_real': v for k, v in self.train.dataset.params.items() if k in metrics})
        return metrics

    def __call__(self):
        loss_test_min = None
        for epoch in range(self.nepoch): 
            for iteration, data in enumerate(self.train, 0):
                for _ in range(self.niter):
                    _, _, loss, metric = self.train_step(data)

                total_iteration = epoch * (len(self.train)) + (iteration + 1)
                loss_train = loss['loss'].item()
                self.lambda_update(loss_train)

                if total_iteration % self.nlog == 0:
                    self.log(epoch, iteration, loss | metric)

                if total_iteration % self.nupdate == 0:
                    with torch.no_grad():
                        loss_test = 0.
                        for j, data_test in enumerate(self.test, 0):
                            _, _, loss, metric = self.val_step(data_test)
                            loss_test += loss['loss'].item()
                            
                        loss_test /= j + 1

                        if loss_test_min == None or loss_test_min > loss_test:
                            loss_test_min = loss_test
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.net.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': loss_test_min, 
                            }, self.path + f'/model_{loss_test_min:.3e}.pt')
                        loss_test = {
                            'loss_test': loss_test,
                        }
                        print('#' * 80)
                        self.log(epoch, iteration, loss_test | metric)
                        print(f'lambda: {self._lambda}')
                        print('#' * 80)
                        