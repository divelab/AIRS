import torch
from torch.utils.data import Dataset
from collections import OrderedDict
import numpy as np
from torchdiffeq import odeint
import shelve

from CEL.utils.register import register
from .meta_dataset import DatasetRegistry

MAX = np.iinfo(np.int32).max

@register.dataset_register
class DampedWaveEquation(Dataset, metaclass=DatasetRegistry):

    __default_params = OrderedDict(c=330, k=50)

    def __init__(self, path, size, seq_len, num_seq, dt, group='train', params=None):

        self.size = size
        self.dx = 1
        self.seq_len = seq_len
        self.params = OrderedDict()
        if params is None:
            self.params.update(self.__default_params)
        else:
            self.params.update(params)

        self.mesh = np.meshgrid(range(size), range(size))

        self.t = torch.arange(0, (num_seq+seq_len)*dt, dt)
        self.dt = dt
        self.data = shelve.open(path)
        self.group = group
        with torch.no_grad():
            self._simulate_wave(self.t)

    def _f(self, t, y, order=5):
        c, k = list(self.params.values())

        state = y[0]
        state_diff = y[1]
        
        state_zz = torch.zeros(state_diff.shape)
        state_xx = torch.zeros(state_diff.shape)

        if order == 3:
            state_zz[:,1:-1] = state[:,2:] - 2 * state[:, 1:-1] + state[:,:-2]
            state_xx[1:-1,:] = state[2:,:] - 2 * state[1:-1, :] + state[:-2,:]

        elif order == 5:
            state_zz[:,2:-2] = -1./12*state[:,4:]+4./3*state[:,3:-1]-5./2*state[:,2:-2]+4./3*state[:,1:-3]-1./12*state[:,:-4]
            state_xx[2:-2,:] = -1./12*state[4:,:]+4./3*state[3:-1,:]-5./2*state[2:-2,:]+4./3*state[1:-3,:]-1./12*state[:-4,:]

        lap = (c ** 2) * (state_zz + state_xx) / self.dx ** 2
        damping = state_diff * k
        derivative = torch.cat([state_diff.unsqueeze(0), (lap - damping).unsqueeze(0)], dim=0)

        return derivative

    def _get_initial_condition(self, seed=0):
        np.random.seed(seed if not self.group == 'train' else MAX-seed)
        rand_std = np.random.uniform(10, 100)
        condition = torch.tensor(self._gaussian_cond(self.mesh[0], self.mesh[1], mean=np.random.randint(low=20, high=40, size=(2,)), value=1., std=rand_std)).float().view(1, self.size, self.size)
        y0 = torch.cat([condition, torch.zeros(1, self.size, self.size)], dim=0)
        return y0

    def _simulate_wave(self, t):
        y0 = self._get_initial_condition()
        data = odeint(self._f, y0, t, method="rk4")
        # data = self._normalize(data)
        self.data['0'] = data.permute(1, 0, 2, 3)

    def _gaussian_cond(self, x, y, std, mean=(32, 32), value=1.):
        return value * np.exp(-((x - mean[0]) ** 2 + (y - mean[1]) ** 2) / std)

    # def _normalize(self, data):
    #     max_, min_ = data[:,0].max(), data[:,0].min()
    #     data[:,0] = (data[:,0] - min_) / (max_ - min_)
    #     data[:,1] =  data[:,1] / (max_ - min_)
    #     return data

    def __getitem__(self, idx):
        states = self.data['0'][:,idx:idx+self.seq_len]
        return {'states': states, 't': torch.arange(0, self.seq_len * self.dt, self.dt).float()}

    def __len__(self):
        return len(self.t) - self.seq_len

