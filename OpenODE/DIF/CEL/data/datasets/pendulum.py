import pathlib

import numpy as np
import math, shelve
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp
import torch
import torchdiffeq
from collections import OrderedDict
from torch.distributions.normal import Normal
from typing import Literal

from CEL.utils.register import register
from .meta_dataset import DatasetRegistry

MAX = np.iinfo(np.int32).max


@register.dataset_register
class DampledPendulum(Dataset, metaclass=DatasetRegistry):
    __default_params = OrderedDict(omega0_square=(2 * math.pi / 6) ** 2, alpha=0.2)

    def __init__(self, path, num_seq, time_horizon, dt, params=None, group='train', **kwargs):
        super().__init__()
        self.len = num_seq[group]
        self.time_horizon = float(time_horizon)  # total time
        self.dt = float(dt)  # time step
        self.params = OrderedDict()
        if params is None:
            self.params.update(self.__default_params)
        else:
            self.params.update(params)
        self.group = group
        self.data = shelve.open(path)

    def _f(self, t, x):  # coords = [q,p]
        omega0_square, alpha = list(self.params.values())

        q, p = np.split(x, 2)
        dqdt = p
        dpdt = -omega0_square * np.sin(q) - alpha * p
        return np.concatenate([dqdt, dpdt], axis=-1)

    def _get_initial_condition(self, seed):
        np.random.seed(seed if self.group == 'train' else MAX - seed)
        y0 = np.random.rand(2) * 2.0 - 1
        radius = np.random.rand() + 1.3
        y0 = y0 / np.sqrt((y0 ** 2).sum()) * radius
        return y0

    def __getitem__(self, index):
        t_eval = torch.from_numpy(np.arange(0, self.time_horizon, self.dt))
        if self.data.get(str(index)) is None:
            print('create new data')
            y0 = self._get_initial_condition(index)
            states = solve_ivp(fun=self._f, t_span=(0, self.time_horizon), y0=y0, method='DOP853', t_eval=t_eval,
                               rtol=1e-10).y
            self.data[str(index)] = states
            states = torch.from_numpy(states).float()
        else:
            states = torch.from_numpy(self.data[str(index)]).float()
        return {'states': states, 't': t_eval.float()}

    def __len__(self):
        return self.len


@register.dataset_register
class IntervenableDampledPendulum(Dataset, metaclass=DatasetRegistry):
    __default_params = OrderedDict(omega0_square=(2 * math.pi / 6) ** 2, alpha=0.2)

    def __init__(self, path: pathlib.Path, num_seq: dict, time_horizon: int, dt: float, params=None, group='train',
                 intervention: Literal['omega0_square', 'alpha', 'delta'] = None, **kwargs):
        super().__init__()
        self.len = num_seq[group]
        self.intervention = intervention
        self.time_horizon = float(time_horizon)  # total time
        self.dt = float(dt)  # time step
        self.params = OrderedDict()
        if params is None:
            self.params.update(self.__default_params)
        else:
            self.params.update(params)
        self.group = group
        self.data = shelve.open(path)

    def _f(self, t, x):  # coords = [q,p]
        omega0_square, alpha = list(self.params.values())

        q, p = np.split(x, 2)
        dqdt = p
        dpdt = -omega0_square * np.sin(q) - alpha * p
        if self.intervention == 'delta':
            # print('intervene')
            dpdt = dpdt + torch.stack([amp * normal.log_prob(torch.tensor(t)).exp() for normal, amp in
                                       self.normals]).sum().numpy()  # This term can be considered as the integration of the change of acceleration
        return np.concatenate([dqdt, dpdt], axis=-1)

    def _get_initial_condition(self, seed):
        np.random.seed(seed if self.group == 'train' else MAX - seed)
        y0 = np.random.rand(2) * 2.0 - 1
        radius = np.random.rand() + 1.3
        y0 = y0 / np.sqrt((y0 ** 2).sum()) * radius
        return y0

    def _get_intervention(self, index):
        if self.intervention == 'delta':
            with torch.random.fork_rng():
                # Set the temporary seed
                torch.manual_seed(index)

                # Generate your random tensor
                self.normals = []
                X = torch.zeros((int(self.time_horizon / self.dt),), dtype=torch.float32)
                for mean, var, amp in torch.randn(3, 3):
                    action_time = (mean * 5 + 10).clamp(1, 19)
                    action_time_frame = (action_time / self.dt).round()
                    action_time_round = action_time_frame * self.dt
                    action_intensity = amp * 2
                    self.normals.append(
                        (Normal(action_time_frame, (var * 1e-1).abs().clamp(1e-2, 1)), action_intensity))
                    X[action_time_frame.long()] = action_intensity
                # self.normals = [(Normal((mean * 5 + 10).clamp(1, 19), (var * 1e-1).abs().clamp(1e-2, 1)), amp * 2) for mean, var, amp in torch.randn(3, 3)]
        else:
            raise NotImplementedError
        return X

    def __getitem__(self, index):
        t_eval = torch.from_numpy(np.arange(0, self.time_horizon, self.dt))
        if self.data.get(str(index)) is None:
            print('create new data')
            y0 = self._get_initial_condition(index)
            X = self._get_intervention(index)
            states = solve_ivp(fun=self._f, t_span=(0, self.time_horizon), y0=y0, method='DOP853', t_eval=t_eval,
                               rtol=1e-10).y
            # dim x time -> time x dim
            states = states.T
            self.data[str(index)] = {'states': states, 'X': X}
            states = torch.from_numpy(states).float()
        else:
            states = torch.from_numpy(self.data[str(index)]['states']).float()
            X = self.data[str(index)]['X']
        return {'states': states, 'X': X, 't': t_eval.float()}

    def __len__(self):
        return self.len
