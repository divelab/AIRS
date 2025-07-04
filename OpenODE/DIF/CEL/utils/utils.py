import collections.abc as collections
import functools
import logging
import random
import warnings
import os
import sys
from typing import Any, Callable, Dict, Optional, TextIO, Tuple, Type, TypeVar, Union, cast

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torch import Tensor
from torch.nn import Parameter
from datetime import datetime

class Logger(object):
    "Lumberjack class - duplicates sys.stdout to a log file and it's okay"
    def __init__(self, filename, mode="a"):
        self.stdout = sys.stdout
        self.file = open(filename, mode)
        sys.stdout = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        if self.stdout != None:
            sys.stdout = self.stdout
            self.stdout = None

        if self.file != None:
            self.file.close()
            self.file = None

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def fix_seed(seed):
    import numpy as np
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def pretty_wrap(text, title=None, width=80):
    table = pt.PrettyTable(
        header=title is not None,
    )
    table.field_names = [title]
    for t in text.split('\n'):
        for i in range(0, len(t), width):
            table.add_row([t[i: i + width]])

    return table

def make_basedir(root, timestamp=None, attempts=5):
    """Takes 5 shots at creating a folder from root,
    adding timestamp if desired.
    """
    for i in range(attempts):
        basedir = root
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            basedir = os.path.join(basedir, timestamp)
        try:
            os.makedirs(basedir)
            return basedir
        except:
            sleep(0.01)
    raise FileExistsError(root)

################################################################################
# Adapted from http://
################################################################################

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class CalculateNorm:
    def __init__(self, module, power_iterations=5):
        self.module = module
        assert isinstance(module, nn.ModuleList)
        self.power_iterations = power_iterations
        self._make_params()

    def calculate_spectral_norm(self):
        # Adapted to complex weights
        sigmas = [0. for i in range(len(self.module))]
        for i, module in enumerate(self.module):
            for name, w in module.named_parameters():
                if name.find('bias') == -1 and name.find('beta') == -1:
                    u = self.u[f'{i},{name}']
                    v = self.v[f'{i},{name}']

                    height = w.data.shape[0]
                    for _ in range(self.power_iterations):
                        v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
                        u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

                    sigma = torch.conj(u).dot(w.view(height, -1).mv(v))
                    if torch.is_complex(sigma):
                        sigmas[i] = sigmas[i] + sigma.real ** 2
                    else:
                        sigmas[i] = sigmas[i] + sigma ** 2
        return torch.stack(sigmas)

    def calculate_frobenius_norm(self):
        # Only used for linear case
        sigmas = [0. for i in range(len(self.module))]
        for i, module in enumerate(self.module):
            for name, w in module.named_parameters():
                if name.find('bias') == -1 and name.find('beta') == -1:
                    sigmas[i] = sigmas[i] + torch.norm(w)
        return torch.stack(sigmas)

    def _make_params(self):
        self.u, self.v = dict(), dict()
        for i, module in enumerate(self.module):
            for name, w in module.named_parameters():
                if name.find('bias') == -1 and name.find('beta') == -1:
                    height = w.data.shape[0]
                    width = w.view(height, -1).data.shape[1]

                    u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
                    v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
                    u.data = l2normalize(u.data)
                    v.data = l2normalize(v.data)

                    self.u[f'{i},{name}'] = u
                    self.v[f'{i},{name}'] = v

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'default':
                pass
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.weight is not None:
                init.normal_(m.weight.data, 1.0, init_gain)
            if m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

################################################################################
# convert_tensor, apply_to_tensor
# 
# built-in functions of pytorch_ignite
# Source: https://github.com/pytorch/ignite/blob/master/ignite/utils.py  
################################################################################

def convert_tensor(
    x: Union[torch.Tensor, collections.Sequence, collections.Mapping, str, bytes],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    ) -> Union[torch.Tensor, collections.Sequence, collections.Mapping, str, bytes]:
    """Move tensors to relevant device.
    Args:
        x: input tensor or mapping, or sequence of tensors.
        device: device type to move ``x``.
        non_blocking: convert a CPU Tensor with pinned memory to a CUDA Tensor
            asynchronously with respect to the host if possible
    """

    def _func(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=device, non_blocking=non_blocking) if device is not None else tensor

    return apply_to_tensor(x, _func)


def apply_to_tensor(
    x: Union[torch.Tensor, collections.Sequence, collections.Mapping, str, bytes], func: Callable
    ) -> Union[torch.Tensor, collections.Sequence, collections.Mapping, str, bytes]:
    """Apply a function on a tensor or mapping, or sequence of tensors.
    Args:
        x: input tensor or mapping, or sequence of tensors.
        func: the function to apply on ``x``.
    """
    return apply_to_type(x, torch.Tensor, func)


def apply_to_type(
    x: Union[Any, collections.Sequence, collections.Mapping, str, bytes],
    input_type: Union[Type, Tuple[Type[Any], Any]],
    func: Callable,
    ) -> Union[Any, collections.Sequence, collections.Mapping, str, bytes]:
    """Apply a function on an object of `input_type` or mapping, or sequence of objects of `input_type`.
    Args:
        x: object or mapping or sequence.
        input_type: data type of ``x``.
        func: the function to apply on ``x``.
    """
    if isinstance(x, input_type):
        return func(x)
    if isinstance(x, (str, bytes)):
        return x
    if isinstance(x, collections.Mapping):
        return cast(Callable, type(x))({k: apply_to_type(sample, input_type, func) for k, sample in x.items()})
    if isinstance(x, tuple) and hasattr(x, "_fields"):  # namedtuple
        return cast(Callable, type(x))(*(apply_to_type(sample, input_type, func) for sample in x))
    if isinstance(x, collections.Sequence):
        return cast(Callable, type(x))([apply_to_type(sample, input_type, func) for sample in x])
    raise TypeError((f"x must contain {input_type}, dicts or lists; found {type(x)}"))

