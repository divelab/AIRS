import torch
from torch.utils.data import Dataset
from collections import OrderedDict
import numpy as np
from scipy.integrate import solve_ivp
import os, shelve

MAX = np.iinfo(np.int32).max
from CEL.utils.register import register
from .meta_dataset import DatasetRegistry

@register.dataset_register
class ReactionDiffusion(Dataset, metaclass=DatasetRegistry):

    __default_params = OrderedDict(a=1e-3, b=5e-3, k=5e-3)

    def __init__(self, path, num_seq=1600, size=32, time_horizon=3, dt=0.1, params=None, group='train'):
        super().__init__()
        self.len = num_seq
        self.size = size  # size of the 2D grid
        self.dx = 2. / size # space step
        self.time_horizon = float(time_horizon)  # total time
        self.dt_int = 1e-3  # time step
        self.dt = dt
        self.n = int(time_horizon / self.dt_int)  # number of iterations
        self.params = OrderedDict()
        if params is None:
            self.params.update(self.__default_params)
        else:
            self.params.update(params)
        self.group = group
        self.every = int(dt / self.dt_int)
        self.data = shelve.open(path)

    def _laplacian2D(self, a):
        return (
            - 4 * a
            + np.roll(a,+1,axis=0) 
            + np.roll(a,-1,axis=0)
            + np.roll(a,+1,axis=1)
            + np.roll(a,-1,axis=1)
        ) / (self.dx ** 2)
    
    def _vec_to_mat(self, vec_uv):
        UV = np.split(vec_uv, 2)
        U = np.reshape(UV[0], (self.size, self.size))
        V = np.reshape(UV[1], (self.size, self.size))
        return U, V

    def _mat_to_vec(self, mat_U, mat_V):
        dudt = np.reshape(mat_U, self.size * self.size)
        dvdt = np.reshape(mat_V, self.size * self.size)
        return np.concatenate((dudt, dvdt))

    def _f(self, t, uv):
        a, b, k = list(self.params.values())
        U, V = self._vec_to_mat(uv)

        deltaU = self._laplacian2D(U)
        deltaV = self._laplacian2D(V)

        dUdt = (a * deltaU + U - U**3 - V - k)
        dVdt = (b * deltaV + U - V)
        
        duvdt = self._mat_to_vec(dUdt, dVdt)
        return duvdt

    def _get_initial_condition(self, seed):
        np.random.seed(seed if self.group == 'train' else MAX-seed)
        U = np.random.rand(self.size, self.size)
        V = np.random.rand(self.size, self.size)
        y0 = self._mat_to_vec(U, V)
        return y0

    def __getitem__(self, index):
        if self.data.get(str(index)) is None:
            y0 = self._get_initial_condition(index)
            states = solve_ivp(self._f, (0., self.time_horizon), y0=y0, method='RK45', t_eval=np.arange(0., self.time_horizon, self.dt_int)).y
            u, v = list(), list()
            for i in range(0, self.n, self.every):
                res_U, res_V = self._vec_to_mat(states[:, i])
                u.append(torch.from_numpy(res_U))
                v.append(torch.from_numpy(res_V))
            u, v = torch.stack(u, dim=0), torch.stack(v, dim=0)
            states = torch.stack([u, v],dim=0).float()

            self.data[str(index)] = states.numpy()
        else:
            states = torch.from_numpy(self.data[str(index)]).float()

        return {'states': states, 't': torch.arange(0, self.time_horizon, self.dt).float()}

    def __len__(self):
        return self.len
