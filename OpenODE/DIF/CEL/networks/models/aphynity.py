import torch.nn as nn
from torchdiffeq import odeint
from CEL.networks.models.aphynity_networks import DampedPendulumParamPDE, ReactionDiffusionParamPDE, DampedWaveParamPDE, MLP, ConvNetEstimator
from functools import partial
from typing import Literal
from CEL.utils.register import register

# from CEL.networks.networks import *
from .meta_model import ModelClass
from CEL.data.data_manager import MyDataset
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

@register.model_register
class APHYNITY(nn.Module, ModelClass):

    dataset2model_phy = {'ReactionDiffusion': ReactionDiffusionParamPDE, 'DampedWaveEquation': DampedWaveParamPDE, 'DampledPendulum': DampedPendulumParamPDE,
                         'IntervenableDampledPendulum': DampedPendulumParamPDE}
    dataset2model_aug = {'ReactionDiffusion': ConvNetEstimator(state_c=2, hidden=16), 'DampedWaveEquation': ConvNetEstimator(state_c=2, hidden=16), 'DampledPendulum': MLP(state_c=2, hidden=200),
                         'IntervenableDampledPendulum': MLP(state_c=2, hidden=200)}

    def __init__(self, dataset_name: str, is_augmented: bool, model_phy_option: Literal['incomplete', 'complete', 'true'], dataset: MyDataset, method: str='rk4', options: dict=None, **kwargs):
        super().__init__()
        if dataset_name == 'ReactionDiffusion':
            self.net_phy = partial(self.dataset2model_phy[dataset_name], dx=dataset.train.dx)
        else:
            self.net_phy = self.dataset2model_phy[dataset_name]
        self.net_phy = self.net_phy(is_complete=False if model_phy_option == 'incomplete' else True,
                                    real_params=dataset.train.params if model_phy_option == 'true' else None)

        # self.net_phy = self.net_phy(is_complete=False, real_params=None)
        self.net_aug = self.dataset2model_aug[dataset_name]

        self.derivative_estimator = DerivativeEstimator(self.net_phy, self.net_aug, is_augmented=is_augmented)
        self.method = method
        self.options = options
        self.intergration = odeint

    def forward(self, y0, t):
        # y0 = y[:,:,0]
        res = self.intergration(self.derivative_estimator, y0=y0, t=t, method=self.method, options=self.options)
        # res: T x batch_size x n_c (x h x w)
        dim_seq = y0.dim() + 1
        dims = [1, 2, 0] + list(range(dim_seq))[3:]
        return res.permute(*dims)  # batch_size x n_c x T (x h x w)

    def get_pde_params(self):
        return self.net_phy.params
