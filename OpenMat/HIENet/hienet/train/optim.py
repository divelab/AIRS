import torch.nn as nn
import torch.optim as optim
import torch
import torch.optim.lr_scheduler as scheduler

from typing import List, Optional
import math
import warnings
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class CosineWarmupLR(_LRScheduler):
    """Cosine learning rate scheduler with warmup.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_factor (float): Initial learning rate factor during warmup. Default: 0.2
        warmup_epochs (float): Fraction of total epochs for warmup. Default: 0.1
        lr_min_factor (float): Minimum learning rate factor. Default: 0.01
        total_epochs (int): Total number of epochs. Default: 100
        last_epoch (int): The index of last epoch. Default: -1
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_factor: float = 0.2,
        warmup_epochs: float = 0.1,
        lr_min_factor: float = 0.01,
        total_epochs: int = 100,
        last_epoch: int = -1,
        verbose: bool = False
    ) -> None:
        # Validate parameters
        if not 0.0 <= warmup_factor < 1.0:
            raise ValueError(f'warmup_factor must be between 0 and 1, got {warmup_factor}')
        if not 0.0 <= warmup_epochs < 1.0:
            raise ValueError(f'warmup_epochs must be between 0 and 1, got {warmup_epochs}')
        if not 0.0 <= lr_min_factor < 1.0:
            raise ValueError(f'lr_min_factor must be between 0 and 1, got {lr_min_factor}')
        if total_epochs <= 0:
            raise ValueError(f'total_epochs must be positive, got {total_epochs}')

        self.warmup_epochs = int(warmup_epochs * total_epochs)
        self.warmup_factor = warmup_factor
        self.lr_min_factor = lr_min_factor
        self.total_epochs = total_epochs
        
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            if self.warmup_epochs == 0:
                factor = 1.0
            else:
                alpha = float(self.last_epoch) / float(self.warmup_epochs)
                factor = self.warmup_factor * (1 - alpha) + alpha
        else:
            # Cosine phase
            progress = float(self.last_epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
            factor = self.lr_min_factor + (1 - self.lr_min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
            
        return [base_lr * factor for base_lr in self.base_lrs]

    def state_dict(self) -> dict:
        """Returns the state of the scheduler as a :class:`dict`.
        
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Loads the schedulers state.
        
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

optim_dict = {
    'sgd': optim.SGD,
    'adagrad': optim.Adagrad,
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'radam': optim.RAdam,
}

optim_param_name_type_dict = {
    'universial': {'lr': float, 'weight_decay': float},
    'sgd': {
        'momentum': float,
        'dampening': float,
        'nesterov': bool,
        'maximize': bool,
        'foreach': bool,
    },
    'adagrad': {
        'lr_decay': float,
        'eps': float,
        'foreach': bool,
        'maximize': bool,
    },
    'adam': {
        'betas': tuple,  # How to make it tuple[float, float]?
        'eps': float,
        'amsgrad': bool,
        'foreach': bool,
        'maximize': bool,
        'capturable': bool,
        'fused': bool,
    },
    'adamw': {
        'betas': tuple,  # How to make it tuple[float, float]?
        'eps': float,
        'amsgrad': bool,
        'maximize': bool,
        'foreach': bool,
        'capturable': bool,
    },
    'radam': {
        'betas': tuple,  # How to make it tuple[float, float]?
        'eps': float,
        'foreach': bool,
    },
}

# Some scheduler use lambda function (e.g. LambdaLR) as input.
# This is not possible using simple yaml configuration.
# TODO: How to implement this?

scheduler_dict = {
    'steplr': scheduler.StepLR,
    'multisteplr': scheduler.MultiStepLR,
    'exponentiallr': scheduler.ExponentialLR,
    'cosineannealinglr': scheduler.CosineAnnealingLR,
    'reducelronplateau': scheduler.ReduceLROnPlateau,
    'linearlr': scheduler.LinearLR,
    'cosinewarmuplr': CosineWarmupLR,
}

scheduler_param_name_type_dict = {
    'universial': {'last_epoch': int, 'verbose': bool},
    'steplr': {'step_size': int, 'gamma': float},
    'multisteplr': {
        'milestones': list,  # How to make it list[int]?
        'gamma': float,
    },
    'exponentiallr': {'gamma': float},
    'cosineannealinglr': {'T_max': int, 'eta_min': float},
    'reducelronplateau': {
        'mode': str,
        'factor': float,
        'patience': int,
        'threshold': float,
        'threshold_mode': str,
        'cooldown': int,
        'min_lr': float,
        'eps': float,
    },
    'linearlr': {
        'start_factor': float,
        'end_factor': float,
        'total_iters': int,
        'last_epoch': int,
    },
    'cosinewarmuplr': {
        'warmup_factor': float,
        'warmup_epochs': float,
        'lr_min_factor': float,
        'total_epochs': int,
        'last_epoch': int,
        'verbose': bool,
    },
}







class L2NormLoss(nn.Module):
    """
    Currently this loss is intened to used with vectors.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        #assert target.dim() == 2
        #assert target.shape[1] != 1
        return torch.linalg.vector_norm(pred - target, ord=2, dim=-1)


loss_dict = {'mse': nn.MSELoss, 'huber': nn.HuberLoss, 'mae': nn.L1Loss, 'l2': L2NormLoss}
loss_param_name_type_dict = {
    'universial': {},
    'mse': {},
    'huber': {'delta': float},  # default = 1.0
    'mae': {}
}


