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

from torch_geometric.utils import from_networkx, degree, sort_edge_index
from torch_geometric.nn import GATConv, GraphConv, GCNConv, GINConv, GINEConv, Set2Set, GENConv, DeepGCNLayer
from torch_geometric.nn import global_mean_pool,global_max_pool,global_add_pool, LayerNorm, BatchNorm, GlobalAttention

from utils.utils import get_undirected_idx_list

### Adaptively adjust from https://github.com/GiggleLiu/marburg

def vmc_sample(kernel, initial_config, num_bath, num_sample):
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

    config = initial_config
    prob = kernel.prob(config)

    n_accepted = 0
    sample_list = []
    for i in range(num_bath + num_sample):
#         print('sample ', i)
        # generate new config and calculate probability ratio
        config_proposed = kernel.propose_config(config)
        prob_proposed = kernel.prob(config_proposed)

        # accept/reject a move by metropolis algorithm
        if np.random.random() <= prob_proposed / prob:
            config = config_proposed
            prob = prob_proposed
            n_accepted += 1

        # print statistics
        if i % print_step == print_step - 1:
            print('%-10s Accept rate: %.3f' %
                  (i + 1, n_accepted * 1. / print_step))
            n_accepted = 0

        # add a sample
        if i >= num_bath:
            sample_list.append(config)
    return sample_list


def vmc_measure(data, local_measure, sample_list, measure_step, num_bin=50):
    '''
    perform measurements on samples
    Args:
        local_measure (func): local measurements function, input configuration, return local energy and local gradient.
        sample_list (list): a list of spin configurations.
        num_bin (int): number of bins in binning statistics.
        meaure_step: number of samples skiped between two measurements + 1.
    Returns:
        tuple: expectation valued of energy, gradient, energy*gradient and error of energy.
    '''
    # measurements
    idx_list = get_undirected_idx_list(data)
    energy_loc_list, grad_loc_list = [], []
    for i, config in enumerate(sample_list):
        if i % measure_step == 0:
            # back-propagation is used to get gradients.
            energy_loc, grad_loc = local_measure(config, idx_list)
            energy_loc_list.append(energy_loc)
            grad_loc_list.append(grad_loc)

    # binning statistics for energy
    energy_loc_list = np.array(energy_loc_list)
    energy, energy_precision = binning_statistics(energy_loc_list, num_bin=num_bin)

    # get expectation values
    energy_loc_list = torch.from_numpy(energy_loc_list)
    if grad_loc_list[0][0].is_cuda: energy_loc_list = energy_loc_list.cuda()
    grad_mean = []
    energy_grad = []
    for grad_loc in zip(*grad_loc_list):
        grad_loc = torch.stack(grad_loc, 0)
        grad_mean.append(grad_loc.mean(0))
        energy_grad.append(
            (energy_loc_list[(slice(None),) + (None,) * (grad_loc.dim() - 1)] * grad_loc).mean(0))
    return energy.item(), grad_mean, energy_grad, energy_precision

def vmc_measure_two_path(data, local_measure, sample_list, measure_step, num_bin=50):
    '''
    perform measurements on samples
    Args:
        local_measure (func): local measurements function, input configuration, return local energy and local gradient.
        sample_list (list): a list of spin configurations.
        num_bin (int): number of bins in binning statistics.
        meaure_step: number of samples skiped between two measurements + 1.
    Returns:
        tuple: expectation valued of energy, gradient, energy*gradient and error of energy.
    '''
    # measurements
    idx_list = get_undirected_idx_list(data)
    energy_loc_list = []
    grad_loc_log_list = []
    grad_loc_arg_list = []
    for i, config in enumerate(sample_list):
        if i % measure_step == 0:
            # back-propagation is used to get gradients.
            energy_loc, grad_loc_log, grad_loc_arg = local_measure(config, idx_list)
            energy_loc_list.append(energy_loc)
            grad_loc_log_list.append(grad_loc_log)
            grad_loc_arg_list.append(grad_loc_arg)

    # binning statistics for energy
    energy_loc_list = np.array(energy_loc_list)
    energy, energy_precision = binning_statistics(energy_loc_list, num_bin=num_bin)
    
    # get expectation values
    energy_loc_list = torch.from_numpy(energy_loc_list)
    grad_log_mean = []
    energy_grad = []
    grad_loc_log_zip = zip(*grad_loc_log_list)
    grad_loc_arg_zip = zip(*grad_loc_arg_list)

    for grad_loc_log, grad_loc_arg in zip(grad_loc_log_zip, grad_loc_arg_zip):
        grad_loc_log = torch.stack(grad_loc_log, 0)
        grad_loc_arg = torch.stack(grad_loc_arg, 0)
        grad_log_mean.append(grad_loc_log.mean(0))
        energy_grad.append(
            (energy_loc_list[(slice(None),) + (None,) * (grad_loc_log.dim() - 1)].real * grad_loc_log +
             energy_loc_list[(slice(None),) + (None,) * (grad_loc_log.dim() - 1)].imag * grad_loc_arg).mean(0))
    return energy.real.item(), grad_log_mean, energy_grad, energy_precision

def vmc_measure_symmetry_gt(model, data, local_measure, sample_list, measure_step, num_bin=50):
    '''
    perform measurements on samples
    Args:
        local_measure (func): local measurements function, input configuration, return local energy and local gradient.
        sample_list (list): a list of spin configurations.
        num_bin (int): number of bins in binning statistics.
        meaure_step: number of samples skiped between two measurements + 1.
    Returns:
        tuple: expectation valued of energy, gradient, energy*gradient and error of energy.
    '''
    # measurements
    idx_list = get_undirected_idx_list(data)
    energy_loc_list, grad_loc_list = [], []
    symmetry_ground_truth = []
    symmetry_ground_truth_config = []
    for i, config in enumerate(sample_list):
        if i % measure_step == 0:
            # back-propagation is used to get gradients.
            energy_loc, grad_loc = local_measure(config, idx_list)
            energy_loc_list.append(energy_loc)
            grad_loc_list.append(grad_loc)
            
            # symmtry ground truth
            symmetry_ground_truth.append(model.ansatz.psi(data, torch.from_numpy(config)))
            symmetry_ground_truth_config.append(config)

    # binning statistics for energy
    energy_loc_list = np.array(energy_loc_list)
    energy, energy_precision = binning_statistics(energy_loc_list, num_bin=num_bin)

    # get expectation values
    energy_loc_list = torch.from_numpy(energy_loc_list)
    if grad_loc_list[0][0].is_cuda: energy_loc_list = energy_loc_list.cuda()
    grad_mean = []
    energy_grad = []
    for grad_loc in zip(*grad_loc_list):
        grad_loc = torch.stack(grad_loc, 0)
        grad_mean.append(grad_loc.mean(0))
        energy_grad.append(
            (energy_loc_list[(slice(None),) + (None,) * (grad_loc.dim() - 1)] * grad_loc).mean(0))
    return energy.item(), grad_mean, energy_grad, energy_precision, symmetry_ground_truth, symmetry_ground_truth_config

def binning_statistics(var_list, num_bin):
    '''
    binning statistics for variable list.
    '''
    # num_sample = len(var_list)
    # if num_sample % num_bin != 0:
    #     raise
    # size_bin = num_sample // num_bin
    #
    # # mean, variance
    mean = np.mean(var_list, axis=0)
    variance = np.var(var_list, axis=0)
    #
    # # binned variance and autocorrelation time.
    # variance_binned = np.var(
    #     [np.mean(var_list[size_bin * i:size_bin * (i + 1)]) for i in range(num_bin)])
    # t_auto = 0.5 * size_bin * \
    #     np.abs(np.mean(variance_binned) / np.mean(variance))
    # stderr = np.sqrt(variance_binned / num_bin)
    # print('Binning Statistics: Energy = %.4f +- %.4f, Auto correlation Time = %.4f' %
    #       (mean, stderr, t_auto))
    print('Energy = %.4f' % mean)
    stderr = np.var(var_list, axis=0)
    return mean, stderr

class VMCKernel(object):
    '''
    variational monte carlo kernel.
    Attributes:
        energy_loc (func): local energy <x|H|\psi>/<x|\psi>.
        ansatz (Module): torch neural network.
    '''
    def __init__(self, data, energy_loc, ansatz):
        self.ansatz = ansatz
        self.energy_loc = energy_loc
        self.data = data

    def prob(self, config):
        '''
        probability of configuration.
        Args:
            config (1darray): the bit string as a configuration.
        Returns:
            number: probability |<config|psi>|^2.
        '''
        return abs(self.ansatz.psi(self.data, torch.from_numpy(config)).item())**2

    def local_measure(self, config, idx_list):
        '''
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
        grad_loc = [p.grad.data/psi_loc.item() for p in self.ansatz.parameters()]
#         grad_loc = [p.grad.data for p in self.ansatz.parameters()]

        # E_{loc}
        eloc = self.energy_loc(config, lambda x: self.ansatz.psi(self.data, torch.from_numpy(x)).data, psi_loc.data, idx_list)
        return eloc.item(), grad_loc

    def local_measure_batch(self, config, idx_list):
        '''
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
        eloc = self.energy_loc(config, lambda x: self.ansatz.psi_batch(self.data, torch.from_numpy(x)).data, psi_loc.data,
                               idx_list)
        return eloc.item(), grad_loc

    def local_measure_two_path(self, config, idx_list):
        '''
        get local quantities energy_loc, grad_loc.
        Args:
            config (1darray): the bit string as a configuration.
        Returns:
            number, list: local energy and local gradients for variables.
        '''
        psi_loc = self.ansatz.psi(self.data, torch.from_numpy(config))

        # get gradients {d/dW}_{loc}
        self.ansatz.zero_grad()
        self.ansatz.log_psi.backward(retain_graph=True)
        grad_loc_log = [p.grad.data.clone() for p in self.ansatz.parameters()]

        self.ansatz.zero_grad()
        self.ansatz.arg_psi.backward()
        grad_loc_arg = [p.grad.data.clone() for p in self.ansatz.parameters()]

        # E_{loc}
        eloc = self.energy_loc(config, lambda x: self.ansatz.psi(self.data, torch.from_numpy(x)).data, psi_loc.data,
                               idx_list)
        return eloc.item(), grad_loc_log, grad_loc_arg

    @staticmethod
    def propose_config(old_config, prob_flip=0.5):
        '''
        flip two positions as suggested spin flips.
        Args:
            old_config (1darray): spin configuration, which is a [-1,1] string.
        Returns:
            1darray: new spin configuration.
        '''
        
#         if np.random.random() < prob_flip:
#             return old_config.copy() * -1

        num_spin = len(old_config)
        upmask = old_config == 1
        flips = np.random.randint(0, num_spin // 2, 2)
        iflip0 = np.where(upmask)[0][flips[0]]
        iflip1 = np.where(~upmask)[0][flips[1]]

        config = old_config.copy()
        config[iflip0] = -1
        config[iflip1] = 1
        
        
        # randomly flip one of the spins
#         def flip(config, idx):
#             new_config = config.copy()
#             new_config[idx] = -new_config[idx]
#             return new_config
        
#         idx = np.random.randint(0, len(old_config), 1)
#         config = flip(old_config, idx)
        
        
        return config