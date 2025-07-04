# MIT License
#
# Copyright (c) 2022 Matthieu Kirchmeyer & Yuan Yin

from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torch.nn import init
import copy
import random
from torch import nn
import torch
import torch.nn.functional as F
from logging.handlers import RotatingFileHandler
import logging
import os
from PIL import Image
import numpy as np


def batch_transform(batch, minibatch_size):
    # batch: b x c x t
    t = batch.shape[2:]
    new_batch = []
    for i in range(minibatch_size):
        sample = batch[i::minibatch_size]  # n_env x c x t
        sample = sample.reshape(-1, *t)
        new_batch.append(sample)
    return torch.stack(new_batch)  # minibatch_size x n_env * c x t

def batch_transform_loss(batch, minibatch_size):
    # batch: b x c x t
    t = batch.shape[2:]
    new_batch = []
    for i in range(minibatch_size):
        sample = batch[i::minibatch_size]  # n_env x c x t
        new_batch.append(sample)
    return torch.stack(new_batch)

def batch_transform_inverse(new_batch, n_env):
    # new_batch: minibatch_size x n_env * c x t
    c = new_batch.size(1) // n_env
    t = new_batch.shape[2:]
    new_batch = new_batch.reshape(-1, n_env, c, *t)
    batch = []
    for i in range(n_env):
        sample = new_batch[:, i]  # minibatch_size x c x t
        batch.append(sample)
    return torch.cat(batch)  # b x c x t


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1 or classname.find('Bilinear') != -1):
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


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


class Sinus(nn.Module):
    def forward(self, input):
        return torch.sinus(input)


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices, minibatch_size=2):
        self.minibatch_size = minibatch_size
        if not any(isinstance(el, list) for el in indices):
            self.indices = [indices]
        else:
            self.indices = indices
        self.env_len = len(self.indices[0])

    def __iter__(self):
        if len(self.indices) > 1:
            l_indices = copy.deepcopy(self.indices)

            l_iter = list()
            for _ in range(0, self.env_len, self.minibatch_size):
                for i in range(len(l_indices)):
                    l_iter.extend(l_indices[i][:self.minibatch_size])
                    del l_indices[i][:self.minibatch_size]
        else:
            l_iter = copy.deepcopy(self.indices[0])
        return iter(l_iter)

    def __len__(self):
        return sum([len(el) for el in self.indices])


class SubsetRamdomSampler(Sampler):
    def __init__(self, indices, minibatch_size=2, same_order_in_groups=True):
        self.minibatch_size = minibatch_size
        self.same_order_in_groups = same_order_in_groups
        if not any(isinstance(el, list) for el in indices):
            self.indices = [indices]
        else:
            self.indices = indices
        self.env_len = len(self.indices[0])

    def __iter__(self):
        if len(self.indices) > 1:
            if self.same_order_in_groups:
                l_shuffled = copy.deepcopy(self.indices)
                random.shuffle(l_shuffled[0])
                for i in range(1, len(self.indices)):
                    l_shuffled[i] = [el + i * self.env_len for el in l_shuffled[0]]
            else:
                l_shuffled = copy.deepcopy(self.indices)
                for l in l_shuffled:
                    random.shuffle(l)

            l_iter = list()
            for _ in range(0, self.env_len, self.minibatch_size):
                for i in range(len(l_shuffled)):
                    l_iter.extend(l_shuffled[i][:self.minibatch_size])
                    del l_shuffled[i][:self.minibatch_size]
        else:
            l_shuffled = copy.deepcopy(self.indices[0])
            random.shuffle(l_shuffled)
            l_iter = l_shuffled
        return iter(l_iter)

    def __len__(self):
        return sum([len(el) for el in self.indices])


def create_logger(folder, outfile):
    try:
        os.makedirs(folder)
        print(f"Directory {folder} created")
    except FileExistsError:
        print(f"Directory {folder} already exists replacing files in this notebook")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_handler = RotatingFileHandler(outfile, "w")
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(logging.DEBUG)
    logger.addHandler(steam_handler)

    return logger


def DataLoaderODE(dataset, minibatch_size, n_env, is_train=True):
    if is_train:
        sampler = SubsetRamdomSampler(indices=dataset.indices, minibatch_size=minibatch_size)
    else:
        sampler = SubsetSequentialSampler(indices=dataset.indices, minibatch_size=minibatch_size)
    dataloader_params = {
        'dataset': dataset,
        'batch_size': minibatch_size * n_env,
        'num_workers': 0,
        'sampler': sampler,
        'pin_memory': True,
        'drop_last': False
    }
    return DataLoader(**dataloader_params)


def get_tensor(batch_gt, batch_pred, T_pred=0):
    _, _, seq_len, height, width = batch_pred.shape  # [n_env * minibatch_size, state_c, t_horizon / dt, h, w]
    state = batch_gt[:, 0:1, T_pred:].cpu().permute(0, 2, 1, 3, 4).contiguous().view(-1, 1, height, width)
    state_pred = batch_pred[:, 0:1].cpu().permute(0, 2, 1, 3, 4).contiguous().view(-1, 1, height, width)
    list_tensor = [state, state_pred]
    nrow = seq_len
    return torch.cat(list_tensor), nrow


def write_image(batch_gt, batch_pred, path, use_value_range=True):
    value_range = (-1, 1) if use_value_range else (0, 1)
    img_tensor, nrow = get_tensor(batch_gt, batch_pred)
    image = make_grid(img_tensor, nrow=nrow, normalize=True, value_range=value_range)
    if isinstance(image, torch.Tensor):
        ndarr = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    else:
        ndarr = image
    im = Image.fromarray(ndarr)
    filename = os.path.join(path)
    im.save(filename)


def save_numpy(batch_gt, batch_pred, path):
    filename = os.path.join(path)
    np.save(filename, torch.stack([batch_gt, batch_pred]).cpu().numpy())


def count_parameters(model, mode='ind'):
    if mode == 'ind':
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif mode == 'layer':
        return sum(1 for p in model.parameters() if p.requires_grad)
    elif mode == 'row':
        n_mask = 0
        for p in model.parameters():
            if p.dim() == 1:
                n_mask += 1
            else:
                n_mask += p.size(0) 
        return n_mask


def set_requires_grad(module, tf=False):
    module.requires_grad = tf
    for param in module.parameters():
        param.requires_grad = tf


def set_rdm_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_n_param_layer(net, layers):
    n_param = 0
    for name, p in net.named_parameters():
        if any(f"net.{layer}" in name for layer in layers):
            n_param += p.numel()
    return n_param
