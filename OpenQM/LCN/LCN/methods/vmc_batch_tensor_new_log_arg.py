import numpy as np
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from numpy.random import randn
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU
from torch_scatter import scatter_sum

from utils.utils import get_undirected_idx_list
import math
import copy
import time


### Adaptively adjust from https://github.com/GiggleLiu/marburg

def vmc_sample_batch(kernel, initial_config, num_bath, num_sample, prob_flip=0.0, chain_interval=False, generator=None):
    '''
    obtain a set of samples.
    Args:
        kernel (object): defines how to sample, requiring the following methods:
            * propose_config, propose a new configuration.
            * prob, get the probability of specific distribution.
        initial_config (1darray): initial configuration.
        num_bath (int): number of updates to thermalize.
        num_sample (int): number of samples.
    Return:
        list: a list of spin configurations.
    '''
    print_step = np.Inf  # steps between two print of accept rate, Inf to disable showing this information.

    batch_size = initial_config.shape[0]

    config = initial_config.clone()
    log_prob = kernel.log_prob(config)

    n_accepted = 0
    sample_list = []

    with torch.no_grad():
        for i in range(num_bath + 1):
            # generate new config and calculate probability ratio
            config_proposed = kernel.propose_config(config, prob_flip=prob_flip, chain_interval=chain_interval, generator=generator)
            log_prob_proposed = kernel.log_prob(config_proposed)

            # accept/reject a move by metropolis algorithm
            # accept_mask = torch.rand(size=(batch_size, 1)).to(initial_config.device) <= torch.exp(log_prob_proposed - log_prob)
            accept_mask = generator.get_random_accept_idx().to(initial_config.device) <= torch.exp(log_prob_proposed - log_prob)
            # accept_mask = torch.rand(size=(batch_size, 1)) <= prob_proposed / prob
            accept_mask = accept_mask.squeeze()

            config[accept_mask] = config_proposed[accept_mask]
            log_prob[accept_mask] = log_prob_proposed[accept_mask]
            n_accepted += torch.sum(accept_mask)

            # print statistics
            if i % print_step == print_step - 1:
                print('%-10s Accept rate: %.3f' %
                      (i + 1, n_accepted * 1. / (print_step * batch_size)))
                n_accepted = 0

            # add last sample
            if i >= num_bath:
                sample_list.append(config.clone())

    return torch.cat(sample_list, dim=0)[:num_sample]


class VMCKernel(object):
    '''
    variational monte carlo kernel.
    Attributes:
        energy_loc (func): local energy <x|H|\psi>/<x|\psi>.
        ansatz (Module): torch neural network.
    '''

    def __init__(self, data, energy_loc, ansatz, energy_phi=None):
        self.ansatz = ansatz
        self.energy_loc = energy_loc
        self.data = data
        self.energy_phi = energy_phi

    def log_prob(self, config):
        '''
        probability of configuration.
        Args:
            config (1darray): the bit string as a configuration.
        Returns:
            number: probability |<config|psi>|^2.
        '''
        # prob = (abs(self.ansatz.psi_batch(self.data, config)) ** 2)
        prob = self.ansatz.psi_batch(self.data, config)[0] * 2
        return prob.detach()

    def local_measure(self, config, idx_list):
        '''
        NOTE: not adapted to batch operations
        get local quantities energy_loc, grad_loc.
        Args:
            config (1darray): the bit string as a configuration.
        Returns:
            number, list: local energy and local gradients for variables.
        '''
        psi_loc = self.ansatz.psi(self.data, torch.from_numpy(config))

        # get gradients {d/dW}_{loc}
        self.ansatz.zero_grad()
        psi_loc.backward()
        grad_loc = [p.grad.data / psi_loc.item() for p in self.ansatz.parameters()]
        #         grad_loc = [p.grad.data for p in self.ansatz.parameters()]

        # E_{loc}
        eloc = self.energy_loc(config, lambda x: self.ansatz.psi(self.data, torch.from_numpy(x)).data, psi_loc.data,
                               idx_list)
        print('psi_loc, eloc: ', psi_loc, eloc)
        return eloc.item(), grad_loc

    def local_measure_two_path(self, config, idx_list, J2, return_eloc=False):
        '''
        get local quantities energy_loc, grad_loc.
        Args:
            config (1darray): the bit string as a configuration.
        Returns:
            number, list: local energy and local gradients for variables.
        '''
        with torch.no_grad():
            self.ansatz.eval()
            log_psi_loc, arg_psi_loc = self.ansatz.psi_batch(self.data, config)

            # E_{loc}
            eloc = self.energy_loc(config=config, psi_func=lambda x: self.ansatz.psi_batch(self.data, x), log_psi_loc=log_psi_loc, arg_psi_loc=arg_psi_loc,
                                   idx_list=idx_list, J2=J2)
            self.ansatz.train()

        # compute psi_loc again to get gradients
        log_psi, arg_psi = self.ansatz.psi_batch(self.data, config)

        # get gradients {d/dW}_{loc}
        self.ansatz.zero_grad()
        log_psi.mean().backward(retain_graph=True)
        # for name, p in self.ansatz.named_parameters():
        #     if p.grad is None:
        #         print(name)
        grad_loc_log = [p.grad.data.clone() for p in self.ansatz.parameters()]
        # get gradients {d/dW}_{loc} * eloc
        self.ansatz.zero_grad()
        (log_psi * eloc.real).mean().backward(retain_graph=True)
        grad_loc_log_eloc = [p.grad.data.clone() for p in self.ansatz.parameters()]

        self.ansatz.zero_grad()
        (arg_psi * eloc.imag).mean().backward()
        grad_loc_arg_eloc = [p.grad.data.clone() for p in self.ansatz.parameters()]

        energy = eloc.mean()
        energy_grad = [gl + ga for gl, ga in zip(grad_loc_log_eloc, grad_loc_arg_eloc)]
        grad = [2 * eg - 2 * energy.real * g for eg, g in zip(energy_grad, grad_loc_log)]

        if return_eloc:
            return eloc / config.shape[1]
        else:
            return energy / config.shape[1], grad

    def local_measure_two_path_weighted(self, config, idx_list, J2, return_eloc=False):
        '''
        get local quantities energy_loc, grad_loc.
        Args:
            config (1darray): the bit string as a configuration.
        Returns:
            number, list: local energy and local gradients for variables.
        '''
        with torch.no_grad():
            # unique
            config = torch.unique(config, dim=0)
            log_psi_loc, arg_psi_loc = self.ansatz.psi_batch(self.data, config)
            prob = (log_psi_loc - torch.mean(log_psi_loc, dim=0, keepdim=True) * 2).exp()
            prob = prob / prob.sum()

            # E_{loc}
            eloc = self.energy_loc(config=config, psi_func=lambda x: self.ansatz.psi_batch(self.data, x),
                                   log_psi_loc=log_psi_loc, arg_psi_loc=arg_psi_loc,
                                   idx_list=idx_list, J2=J2)

        # compute psi_loc again to get gradients
        log_psi, arg_psi = self.ansatz.psi_batch(self.data, config)

        # get gradients {d/dW}_{loc}
        self.ansatz.zero_grad()
        # log_psi.mean().backward(retain_graph=True)
        (log_psi * prob).sum().backward(retain_graph=True)
        # for name, p in self.ansatz.named_parameters():
        #     if p.grad is None:
        #         print(name)
        grad_loc_log = [p.grad.data.clone() for p in self.ansatz.parameters()]
        # get gradients {d/dW}_{loc} * eloc
        self.ansatz.zero_grad()
        # (log_psi * eloc.real).mean().backward(retain_graph=True)
        (log_psi * eloc.real * prob).sum().backward(retain_graph=True)

        grad_loc_log_eloc = [p.grad.data.clone() for p in self.ansatz.parameters()]

        self.ansatz.zero_grad()
        # (arg_psi * eloc.imag).mean().backward()
        (arg_psi * eloc.imag * prob).sum().backward()
        grad_loc_arg_eloc = [p.grad.data.clone() for p in self.ansatz.parameters()]

        # energy = eloc.mean()
        energy = (eloc * prob).sum()
        energy_grad = [gl + ga for gl, ga in zip(grad_loc_log_eloc, grad_loc_arg_eloc)]
        grad = [2 * eg - 2 * energy.real * g for eg, g in zip(energy_grad, grad_loc_log)]

        if return_eloc:
            return eloc / config.shape[1]
        else:
            return energy / config.shape[1], grad

    def local_measure_it_swo(self, config, idx_list, J2, beta, ansatz_phi):
        with torch.no_grad():
            log_psi_loc, arg_psi_loc = self.ansatz.psi_batch(self.data, config)
            eloc = self.energy_loc(config=config, psi_func=lambda x: self.ansatz.psi_batch(self.data, x), log_psi_loc=log_psi_loc, arg_psi_loc=arg_psi_loc,
                                   idx_list=idx_list, J2=J2)
            phi_loc = self.energy_phi(config=config, phi_func=lambda x: ansatz_phi.psi_batch(self.data, x), log_psi_loc=log_psi_loc, arg_psi_loc=arg_psi_loc,
                                      idx_list=idx_list, J2=J2, beta=beta) 

        # compute psi_loc again to get gradients
        log_psi, arg_psi = self.ansatz.psi_batch(self.data, config)

        self.ansatz.zero_grad()
        log_psi.mean().backward(retain_graph=True)
        grad_loc_log = [p.grad.data.clone() for p in self.ansatz.parameters()]

        self.ansatz.zero_grad()
        (log_psi * phi_loc.real + arg_psi * phi_loc.imag).mean().backward(retain_graph=True)
        gamma = [p.grad.data.clone() for p in self.ansatz.parameters()]

        self.ansatz.zero_grad()
        (log_psi * phi_loc.imag - arg_psi * phi_loc.real).mean().backward()
        eta = [p.grad.data.clone() for p in self.ansatz.parameters()]

        energy = eloc.mean()

        delta = phi_loc.mean().angle()
        grad = [0.5 * (grad - (gamma * torch.cos(delta) + eta * torch.sin(delta)) / phi_loc.mean().abs())
                for grad, gamma, eta in zip(grad_loc_log, gamma, eta)]

        return energy / config.shape[1], grad

    @staticmethod
    def propose_config(old_config, prob_flip=0.0, chain_interval=False, generator=None):
        '''
        flip two positions as suggested spin flips.
        Args:
            old_config (1darray): spin configuration, which is a [-1,1] string.
        Returns:
            1darray: new spin configuration.
        '''

        if np.random.random() < prob_flip:
            return old_config.clone() * -1

        batch_size, num_spin = old_config.shape
        upmask = old_config == 1

        if chain_interval:
            flips = torch.randint(0, num_spin // 2, (batch_size * num_spin, 2))
            idx = torch.arange(batch_size) * num_spin
            flips = flips[idx]
        else:
            # flips = torch.randint(0, num_spin // 2, (batch_size, 2))
            flips = generator.get_random_flip_idx()

        flips = flips + torch.div(torch.arange(batch_size)[:, None] * num_spin, 2, rounding_mode='floor')

        iflip0 = torch.nonzero(upmask)[:, 1][flips[:, 0]]
        iflip1 = torch.nonzero(~upmask)[:, 1][flips[:, 1]]

        config = old_config.clone()
        config[torch.arange(batch_size), iflip0] = -1
        config[torch.arange(batch_size), iflip1] = 1

        return config
