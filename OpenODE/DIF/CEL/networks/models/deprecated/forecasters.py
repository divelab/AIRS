import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint

# from CEL.networks.networks import *

class DerivativeEstimator(nn.Module):
    def __init__(self, model_phy, model_aug, is_augmented):
        super().__init__()
        self.model_phy = model_phy
        self.model_aug = model_aug
        self.is_augmented = is_augmented

    def forward(self, t, state):
        res_phy = self.model_phy(state)
        if self.is_augmented:
            res_aug = self.model_aug(state)
            return res_phy + res_aug
        else:
            return res_phy

class Forecaster(nn.Module):
    def __init__(self, model_phy, model_aug, is_augmented, method='rk4', options=None):
        super().__init__()

        self.model_phy = model_phy
        self.model_aug = model_aug

        self.derivative_estimator = DerivativeEstimator(self.model_phy, self.model_aug, is_augmented=is_augmented)
        self.method = method
        self.options = options
        self.int_ = odeint 
        
    def forward(self, y0, t):
        # y0 = y[:,:,0]
        res = self.int_(self.derivative_estimator, y0=y0, t=t, method=self.method, options=self.options)
        # res: T x batch_size x n_c (x h x w)
        dim_seq = y0.dim() + 1
        dims = [1, 2, 0] + list(range(dim_seq))[3:]
        return res.permute(*dims)   # batch_size x n_c x T (x h x w)
    
    def get_pde_params(self):
        return self.model_phy.params
