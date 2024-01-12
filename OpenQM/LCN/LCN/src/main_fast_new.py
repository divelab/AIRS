import sys

sys.path.append("..")

import numpy as np
import os
from copy import deepcopy
from scipy.sparse.linalg import eigsh
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import matplotlib.pyplot as plt
from collections import OrderedDict
from numpy.random import randn
import random
import time
import argparse
import torch

from torch_geometric.utils import degree

from methods.models import *
from methods.vmc_batch_tensor_new_log_arg import vmc_sample_batch, VMCKernel
from utils.utils_log_arg import tensor_prod_graph_Heisenberg, get_undirected_idx_list, heisenberg_loc_batch_fast_J1_J2, \
                        heisenberg_loc_it_swo_J1_J2, RandomGenerator

### some code are adjusted from https://github.com/GiggleLiu/marburg

num_spin = 10
num_hidden = 10

data = None

def test(model, num_spin, idx_list, J2, device='cpu', batch_size=1000, prob_flip=0.05,
         chain_interval=False, weighted=False, generator=None, save_dir=None, checkpoint_step=None):
    '''
    test model
    Args:
        model: trained model for testing
        num_spin: number of spin sites
        idx_list: unique edge index list
        device: gpu or cpu
    Returns: None
    '''
    model.ansatz.eval()

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    generator = RandomGenerator(batch_size, num_spin)

    total_sample = 200000

    assert total_sample % batch_size == 0
    assert batch_size % 100 == 0

    energy = 0
    bin_energy = 0
    total_step = int(total_sample / batch_size)
    bin_interval = total_sample / 100 // batch_size

    print(total_step)
    print(bin_interval)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    elif not os.path.exists(os.path.join(save_dir, f'{checkpoint_step}_test_energy.pt')):
        pass
    else:
        energy_list = torch.load(os.path.join(save_dir, f'{checkpoint_step}_test_energy.pt'))
        chain_avg_energies = torch.tensor([torch.mean(energy_list[i::batch_size]) for i in range(batch_size)])
        bins = torch.split(chain_avg_energies.real, split_size_or_sections=int(batch_size / 100))
        bins_avg = torch.stack(bins).mean(dim=1)
        print('Bin energies', bins_avg)
        print('Test Energy: ', torch.mean(energy_list), 'bin std: ', bins_avg.std())
        return

    initial_config = torch.Tensor([-1, 1] * (model.ansatz.num_visible // 2)).float()
    initial_config_batch = torch.tile(initial_config, (batch_size, 1)).to(device)

    indices = torch.argsort(torch.rand(*initial_config_batch.shape), dim=-1)
    initial_config_batch_rand = initial_config_batch[torch.arange(initial_config_batch.shape[0]).unsqueeze(-1), indices]
    # initial_config_batch_rand = initial_config_batch

    sample_batch = vmc_sample_batch(kernel=model, initial_config=initial_config_batch_rand, num_bath=500 * num_spin,
                                    num_sample=batch_size, prob_flip=prob_flip, chain_interval=chain_interval, generator=generator)

    bin = []
    energy_list = []
    for step in range(1, total_step+1):
        sample_batch = vmc_sample_batch(kernel=model, initial_config=sample_batch, num_bath=num_spin,
                                        num_sample=batch_size, prob_flip=prob_flip, chain_interval=chain_interval, generator=generator)
        with torch.no_grad():
            # psi_loc = model.ansatz.psi_batch(data, sample_batch)
            # eloc = model.energy_loc(sample_batch, lambda x: model.ansatz.psi_batch(data, x).data, psi_loc.data,
            #                        idx_list, J2)

            log_psi_loc, arg_psi_loc = model.ansatz.psi_batch(data, sample_batch)
            eloc = model.energy_loc(sample_batch, psi_func=lambda x: model.ansatz.psi_batch(data, x),
                                   log_psi_loc=log_psi_loc, arg_psi_loc=arg_psi_loc,
                                   idx_list=idx_list, J2=J2)

            energy_list.append(eloc / num_spin)

            # energy += (eloc.mean() / num_spin)
            # bin_energy += (eloc.mean() / num_spin)
            # energy_list.append(eloc / num_spin)
        # if step % bin_interval == 0:
        #     bin.append(bin_energy / bin_interval)
        #     print(bin[-1])
        #     bin_energy = 0

    energy_list = torch.cat(energy_list).cpu().squeeze()
    torch.save(energy_list, os.path.join(save_dir, f'{checkpoint_step}_test_energy.pt'))

    chain_avg_energies = torch.tensor([torch.mean(energy_list[i::batch_size]) for i in range(batch_size)])
    bins = torch.split(chain_avg_energies.real, split_size_or_sections=int(batch_size / 100))
    bins_avg = torch.stack(bins).mean(dim=1)
    print('Bin energies', bins_avg)
    print('Test Energy: ', torch.mean(energy_list), 'bin std: ', bins_avg.std())


    # energy = energy / total_step
    # std = torch.std(torch.stack(bin))
    # # print(bin)
    # print('Test Energy: ', energy, 'std: ', std)

def train_it_swo(model, num_spin, idx_list, optimizer, J2, first_batch=False, sample_batch=None, ansatz_phi=None,
                 device='cpu', batch_size=1000, generator=None, clip_grad=False):
    '''
    train a model.
    Args:
        model (obj): a model that meet VMC model definition.
        num_spin: number of spin sites
        optimizer: pytorch optimizer
    '''
    # Sample
    if first_batch:
        initial_config = torch.Tensor([-1, 1] * (model.ansatz.num_visible // 2)).float()
        initial_config_batch = torch.tile(initial_config, (batch_size, 1)).to(device)

        indices = torch.argsort(torch.rand(*initial_config_batch.shape), dim=-1)
        initial_config_batch_rand = initial_config_batch[torch.arange(initial_config_batch.shape[0]).unsqueeze(-1), indices]
        sample_batch = vmc_sample_batch(kernel=model, initial_config=initial_config_batch_rand, num_bath=500 * num_spin,
                                        num_sample=batch_size, generator=generator)
    else:
        sample_batch = vmc_sample_batch(kernel=model, initial_config=sample_batch, num_bath=num_spin,
                                        num_sample=batch_size, generator=generator)

    # Estimate gradient
    energy, grad = model.local_measure_it_swo(config=sample_batch, idx_list=idx_list, J2=J2, beta=0.04, ansatz_phi=ansatz_phi)

    optimizer.zero_grad()
    for param, g in zip(model.ansatz.parameters(), grad):
        param.grad.data = g

    if clip_grad:
        torch.nn.utils.clip_grad_norm_(model.ansatz.parameters(), max_norm=10)
    optimizer.step()

    precision = 0
    return energy.real.cpu(), precision, sample_batch


def train(step, model, num_spin, idx_list, optimizer, J2, first_batch=False, sample_batch=None,
          device='cpu', batch_size=1000, prob_flip=0.05, chain_interval=False, weighted=False, generator=None, clip_grad=False):
    '''
    train a model.
    Args:
        model (obj): a model that meet VMC model definition.
        num_spin: number of spin sites
        optimizer: pytorch optimizer
    '''
    # Sample
    if first_batch:
        initial_config = torch.Tensor([-1, 1] * (model.ansatz.num_visible // 2)).float()
        initial_config_batch = torch.tile(initial_config, (batch_size, 1)).to(device)

        indices = torch.argsort(torch.rand(*initial_config_batch.shape), dim=-1)
        initial_config_batch_rand = initial_config_batch[torch.arange(initial_config_batch.shape[0]).unsqueeze(-1), indices]
        sample_batch = vmc_sample_batch(kernel=model, initial_config=initial_config_batch_rand, num_bath=5 * num_spin,
                                        num_sample=batch_size, prob_flip=prob_flip, chain_interval=chain_interval, generator=generator)
    else:
        sample_batch = vmc_sample_batch(kernel=model, initial_config=sample_batch, num_bath=num_spin,
                                        num_sample=batch_size, prob_flip=prob_flip, chain_interval=chain_interval, generator=generator)

    # Estimate gradient
    if weighted:
        energy, grad = model.local_measure_two_path_weighted(config=sample_batch, idx_list=idx_list, J2=J2)
    else:
        energy, grad = model.local_measure_two_path(config=sample_batch, idx_list=idx_list, J2=J2)

    optimizer.zero_grad()
    for param, g in zip(model.ansatz.parameters(), grad):
        param.grad.data = g

    if clip_grad > 0 and step > 4000:
        torch.nn.utils.clip_grad_norm_(model.ansatz.parameters(), max_norm=clip_grad, norm_type=2.0, error_if_nonfinite=True)
    optimizer.step()

    precision = 0
    return energy.real.cpu(), precision, sample_batch



def main():
    parser = argparse.ArgumentParser(description='quantum many body problem with variational monte carlo method')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--J2', type=float, default=0.0,
                        help='J2 value in Heisenberg model')
    parser.add_argument('--model', type=str, default='cnn',
                        help='')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_spin', type=int, default=8,
                        help='spin number')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    #     parser.add_argument('--batch_size', type=int, default=256,
    #                         help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="../log/cnn",
                        help='tensorboard log directory')
    parser.add_argument('--data_dir', type=str, default='../dataset/10_node_chain.pt',
                        help='directory to load graph data')
    parser.add_argument('--dataname', type=str, default='10_node_chain', help='data name')
    parser.add_argument('--savefolder', type=str, default='', help='save folder')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='directory to save checkpoint')
    parser.add_argument('--save_dir', type=str, default='../result/Heisenberg/',
                        help='directory to save model parameter and energy list')
    parser.add_argument('--GPU', action='store_true', default=False, help="whether use GPU or not")
    parser.add_argument('--test', action='store_true', default=False, help="test mode")
    parser.add_argument('--optim', type=str, choices=['energy', 'it_swo', 'hybrid'], default='energy')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--first_kernel_size', type=int, default=3)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--non_local', action='store_true', default=False, help="add non local block")
    parser.add_argument('--remove_SE', action='store_true', default=False, help="remove SE")
    parser.add_argument('--only_nonlocal', action='store_true', default=False, help="use only nonlocal block")
    parser.add_argument('--mode', type=str, default='embedded')
    parser.add_argument('--filters_size', type=int, default=64)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--use_sublattice', action='store_true')
    parser.add_argument('--aggr', type=str, default='flatten')
    parser.add_argument('--prob_flip', type=float, default=0.0)
    parser.add_argument('--preact', action='store_true', default=False, help="pre-activaton")
    parser.add_argument('--chain_interval', action='store_true', default=False, help="")
    parser.add_argument('--weighted', action='store_true', default=False, help="eloc weighted sum")
    parser.add_argument('--accept_min', type=float, default=0.0)
    parser.add_argument('--last_linear_only', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--clip_grad', type=float, default=-1)
    parser.add_argument('--conv', type=str, default='pattern')
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--sublattice', action='store_true', default=False)
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--norm', type=str, default='layer')
    parser.add_argument('--last_conv', action='store_true', default=False)
    parser.add_argument('--lr_decay_step', type=int, default=20000)
    # parser.add_argument('--milestones', type=int, nargs='+', default=[4000, 8000, 12000, 16000], help='Steps to decay lr')
    parser.add_argument('--milestones', type=int, nargs='+', default=None, help='Steps to decay lr')
    parser.add_argument('--gamma', type=float, default=0.1, help="Step lr decay rate")
    parser.add_argument('--save_stable_avg', action='store_true', help="Whether to compare average recent energies when saving stable checkpoints.")

    args = parser.parse_args()

    print(args)

    seed = 42  # 10086

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    ### load data and get undirected unique edge index list
    global data
    datapath = os.path.join(args.data_dir, args.dataname + '.pt')
    #     print(datapath)
    data = torch.load(datapath)
    if args.dataname.endswith('honeycomb_lattice'):
        print('Honeycomb')
        idx_list = get_undirected_idx_list(data, gen_idx=True)
    else:
        idx_list = get_undirected_idx_list(data, periodic=False, square=False)
    # print("Undirected edge number: ", len(idx_list))

    global num_spin
    num_spin = args.num_spin
    num_hidden = args.num_spin
    if args.J2 != 0:
        J2 = args.J2
        print('J2 is not 0')
    else:
        print('J2 is 0')
        J2 = None



    ### get hamiltonian matrix and ground truth energy
    E_exact = None
    if num_spin <= 16:
        H = tensor_prod_graph_Heisenberg(data, n=num_spin)
        # e_vals, e_vecs = np.linalg.eigh(H)
        [energy, state] = eigsh(H, k=1, which='SA')
        E_exact = energy  # e_vals[0]
        print('Exact energy: {}'.format(E_exact))

    ### visualize the loss history
    energy_list, precision_list = [], []

    def _update_curve(energy, precision, save_dir):
        energy_list.append(energy)
        precision_list.append(precision)
        if len(energy_list) % 10 == 0:
            fig = plt.figure()
            plt.errorbar(np.arange(1, len(energy_list) + 1), energy_list, yerr=precision_list, capsize=3)
            # dashed line for exact energy
            if E_exact is not None:
                plt.axhline(E_exact, ls='--')
            #     plt.show()
            fig.savefig(save_dir + 'energy.png')
            plt.close(fig)

    # params = {
    #     'num_spins': args.num_spin,
    #     'hidden': args.emb_dim,
    #     'dropout': args.drop_ratio,
    #     'depth': args.num_layers
    # }

    if args.GPU:
        # device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device("cpu")




    if args.model == 'cnn2d-se-hex':
        ansatz = CNN2D_SE_Hex(data.num_nodes, num_hidden, filters_in=1, filters=args.filters_size, kernel_size=args.kernel_size, num_blocks=args.num_blocks,
                                first_kernel_size=args.first_kernel_size, non_local=args.non_local, conv=args.conv, sublattice=args.sublattice,
                                device=device, norm=args.norm, last_conv=args.last_conv, preact=args.preact, remove_SE=args.remove_SE)
    elif args.model == 'cnn2d-se-hex-108':
        ansatz = CNN2D_SE_Hex_108(data.num_nodes, num_hidden, filters_in=1, filters=args.filters_size, kernel_size=3, num_blocks=args.num_blocks,
                                    non_local=args.non_local, conv=args.conv)
    elif args.model == 'cnn2d-se-kagome':
        ansatz = CNN2D_SE_Kagome(data.num_nodes, num_hidden, filters_in=1, filters=args.filters_size,
                                    kernel_size=3, num_blocks=args.num_blocks, non_local=args.non_local, mode=args.mode, preact=args.preact,
                                    conv=args.conv, sublattice=args.sublattice, device=device, act=args.act, norm=args.norm,
                                    last_conv=args.last_conv, remove_SE=args.remove_SE, only_nonlocal=False)
    elif args.model == 'cnn2d-se-kagome-108':
        ansatz = CNN2D_SE_Kagome_108(data.num_nodes, num_hidden, filters_in=1, filters=args.filters_size, kernel_size=3, num_blocks=args.num_blocks,
                                        non_local=args.non_local, mode=args.mode, preact=args.preact, conv=args.conv, remove_SE=args.remove_SE)
    # conv='nn.conv2d' for square and conv='HoneycombConv2d_v5' for honeycomb
    elif args.model == 'cnn2d-se':
        ansatz = CNN2D_SE_2(data.num_nodes, num_hidden, filters_in=1, filters=args.filters_size, kernel_size=args.kernel_size, non_local=args.non_local, conv_name=args.conv, 
                          num_blocks=args.num_blocks, preact=args.preact, aggr=args.aggr, act=args.act, use_sublattice=args.use_sublattice, norm=args.norm, no_se=args.remove_SE)
    else:
        raise ValueError('Invalid model type')



    if args.optim == 'it_swo' or args.optim == 'hybrid':
        energy_phi = heisenberg_loc_it_swo_J1_J2
    else:
        energy_phi = None

    ## initialize random generator
    generator = RandomGenerator(args.batch_size, args.num_spin, accept_min=args.accept_min, accept_max=1)
    model = VMCKernel(data, heisenberg_loc_batch_fast_J1_J2, ansatz=ansatz.to(device), energy_phi=energy_phi)

    if args.optim == 'it_swo' or args.optim == 'hybrid':
        ansatz_phi = deepcopy(model.ansatz)

    optimizer = torch.optim.Adam(model.ansatz.parameters(), betas=(0.9, 0.99), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

    if args.milestones is not None:
        scheduler = MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)



    if args.test:
        save_dir = os.path.join(args.save_dir, args.savefolder, args.model, '')
        print("Testing using stable best checkpoint!")
        start = time.time()
        checkpoint_dir = os.path.join(args.checkpoint_dir, args.savefolder, args.model, '')
        checkpoint_path = checkpoint_dir + 'checkpoint_stable_best.pt'
        print("Stable best checkpoint path: ", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        print("Training energy for stable best model: %.4f, step: %d" % (checkpoint['energy'], checkpoint['step']))
        ansatz.load_state_dict(checkpoint['model_state_dict'])
        model = VMCKernel(data=data, energy_loc=heisenberg_loc_batch_fast_J1_J2, ansatz=ansatz.to(device))
        test(model=model, num_spin=num_spin, idx_list=idx_list, J2=J2, device=device, batch_size=args.batch_size,
             prob_flip=args.prob_flip, generator=generator, save_dir=save_dir, checkpoint_step=checkpoint['step'])
        print("Testing time: ", time.time() - start)

        print("Testing using last checkpoint!")
        start = time.time()
        checkpoint_dir = os.path.join(args.checkpoint_dir, args.savefolder, args.model, '')
        checkpoint_path = checkpoint_dir + 'checkpoint.pt'
        print("Checkpoint path: ", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        print("Training energy for last model: %.4f, step: %d" % (checkpoint['energy'], checkpoint['step']))
        print("Training energy: ", checkpoint['energy'])
        ansatz.load_state_dict(checkpoint['model_state_dict'])
        model = VMCKernel(data=data, energy_loc=heisenberg_loc_batch_fast_J1_J2, ansatz=ansatz.to(device))
        test(model=model, num_spin=num_spin, idx_list=idx_list, J2=J2, device=device, batch_size=args.batch_size, 
             prob_flip=args.prob_flip, generator=generator, save_dir=save_dir, checkpoint_step=checkpoint['step'])
        print("Testing time: ", time.time() - start)

        print("Testing using best checkpoint!")
        start = time.time()
        checkpoint_dir = os.path.join(args.checkpoint_dir, args.savefolder, args.model, '')
        checkpoint_path = checkpoint_dir + 'checkpoint_best.pt'
        print("Best checkpoint path: ", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        print("Training energy for best model: %.4f, step: %d" % (checkpoint['energy'], checkpoint['step']))
        ansatz.load_state_dict(checkpoint['model_state_dict'])
        model = VMCKernel(data=data, energy_loc=heisenberg_loc_batch_fast_J1_J2, ansatz=ansatz.to(device))
        test(model=model, num_spin=num_spin, idx_list=idx_list, J2=J2, device=device, batch_size=args.batch_size,
             prob_flip=args.prob_flip, generator=generator, save_dir=save_dir, checkpoint_step=checkpoint['step'])
        print("Testing time: ", time.time() - start)



        return

    num_params = sum(p.numel() for p in model.ansatz.parameters())
    print(f'#Params: {num_params}')

    if args.log_dir != '':
        log_dir = os.path.join(args.log_dir, args.savefolder, args.model, '')
        writer = SummaryWriter(log_dir=log_dir)

    t0 = time.time()
    first_batch = True
    sample_batch = None
    step_total = 0
    pre_energy = 1000
    save_best = False
    save_stable_best = False
    window_size = 10
    lowest_energy = 1000
    lowest_energy_step = 0
    lowest_stable_energy = 1000
    lowest_stable_energy_step = 0

    checkpoint_dict = OrderedDict()
    recent_energies = []
    recent_checkpoints = []

    for epoch in range(args.epochs):
        steps = int(200000 / args.batch_size)
        for step in range(steps):
            step_total = step + epoch * steps
            if step_total > 0:
                first_batch = False

            if args.optim == 'energy':
                energy, precision, sample_batch = train(step_total, model, num_spin, idx_list, optimizer, J2, first_batch=first_batch,
                                                        sample_batch=sample_batch, device=device, batch_size=args.batch_size,
                                                        prob_flip=args.prob_flip, chain_interval=args.chain_interval,
                                                        weighted=args.weighted, generator=generator, clip_grad=args.clip_grad)
            elif args.optim == 'it_swo':
                if step_total % 30 == 0:
                    ansatz_phi.load_state_dict(model.ansatz.state_dict())
                energy, precision, sample_batch = train_it_swo(model, num_spin, idx_list, optimizer, J2, first_batch=first_batch, 
                                                               sample_batch=sample_batch, ansatz_phi=ansatz_phi, device=device, batch_size=args.batch_size,
                                                               generator=generator, clip_grad=args.clip_grad)
            elif args.optim == 'hybrid':
                if step_total < 2000:
                    energy, precision, sample_batch = train(model, num_spin, idx_list, optimizer, J2,
                                                            first_batch=first_batch,
                                                            sample_batch=sample_batch, device=device,
                                                            batch_size=args.batch_size,
                                                            prob_flip=args.prob_flip,
                                                            chain_interval=args.chain_interval,
                                                            weighted=args.weighted, generator=generator,
                                                            clip_grad=args.clip_grad)
                else:
                    if step_total % 30 == 0:
                        ansatz_phi.load_state_dict(model.ansatz.state_dict())
                    energy, precision, sample_batch = train_it_swo(model, num_spin, idx_list, optimizer, J2,
                                                                   first_batch=first_batch,
                                                                   sample_batch=sample_batch, ansatz_phi=ansatz_phi,
                                                                   device=device, batch_size=args.batch_size,
                                                                   generator=generator, clip_grad=args.clip_grad)

            t1 = time.time()


            if energy < lowest_energy:
                if abs(energy - pre_energy) < 0.005:
                    lowest_energy = energy
                    lowest_energy_step = step_total
                    save_best = True
            # if (energy - pre_energy) > 0.1 or (energy - lowest_energy) > 0.1:
            #     # raise Exception("Stop training!")
            #     print('Stop training!')
            #     sys.exit(0)

            checkpoint = {'step': step_total, 'model_state_dict': model.ansatz.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'scheduler_state_dict': scheduler.state_dict(), 'energy': energy, 'num_params': num_params}

            if not args.save_stable_avg:
                if len(checkpoint_dict) < window_size:
                    checkpoint_dict[energy.item()] = checkpoint
                else:
                    checkpoint_dict[energy.item()] = checkpoint
                    checkpoint_dict.popitem(last=False)

                if np.std(list(checkpoint_dict.keys())) < 0.0015 and len(checkpoint_dict) == window_size:
                    checkpoint_keys = list(checkpoint_dict.keys())
                    idx = np.min(checkpoint_keys)
                    checkpoint_stable_best = checkpoint_dict[idx]
                    if lowest_stable_energy > checkpoint_stable_best['energy']:
                        lowest_stable_energy = checkpoint_stable_best['energy']
                        lowest_stable_energy_step = checkpoint_stable_best['step']
                        save_stable_best = True

            else:
                recent_energies.append(energy.item())
                recent_checkpoints.append(checkpoint)
                if len(recent_energies) > window_size:
                    recent_energies.pop(0)
                    recent_checkpoints.pop(0)

                recent_avg_energy = np.mean(recent_energies)
                if recent_avg_energy <= lowest_avg_energy and np.std(recent_energies) <= 0.0015:
                    lowest_avg_energy = recent_avg_energy
                    best_idx = np.argmin(recent_energies)
                    lowest_stable_energy = recent_energies[best_idx]
                    lowest_stable_energy_step = step_total - len(recent_energies) + best_idx + 1
                    save_stable_best = True

            print('Step %d, E = %.4f, elapse = %.4f, lowest_E = %.4f, lowest_E_step = %d, '
                  'lowest_stable_E = %.4f, lowest_stable_E_step = %d' % (step_total, energy,
                    t1 - t0, lowest_energy, lowest_energy_step, lowest_stable_energy, lowest_stable_energy_step))

            pre_energy = energy

            t0 = time.time()

            if args.log_dir != '':
                if E_exact is None:
                    writer.add_scalars('energy', {'pred': energy}, step_total)
                else:
                    writer.add_scalars('energy', {'pred': energy, 'gt': E_exact}, step_total)

            # if args.save_dir != '':
            #     save_dir = os.path.join(args.save_dir, args.savefolder, args.model, '')
            #     if not os.path.exists(save_dir):
            #         os.makedirs(save_dir)
            #     # _update_curve(energy, precision, save_dir)
            #     energy_save_dir = os.path.join(save_dir, 'graph_energy_list.npy')
            #     precision_save_dir = os.path.join(save_dir, 'graph_precision_list.npy')
            #     np.save(energy_save_dir, energy_list)
            #     np.save(precision_save_dir, precision_list)

            if args.checkpoint_dir != '':

                checkpoint_dir = os.path.join(args.checkpoint_dir, args.savefolder, args.model, '')
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                torch.save(checkpoint, checkpoint_dir + 'checkpoint.pt')

                if save_best:
                    save_best = False
                    torch.save(checkpoint, checkpoint_dir + 'checkpoint_best.pt')

                if save_stable_best:
                    save_stable_best = False
                    torch.save(checkpoint_stable_best, checkpoint_dir + 'checkpoint_stable_best.pt')


            # if optimizer.param_groups[0]['lr'] > 0.00001:
            #     scheduler.step()
            scheduler.step()


if __name__ == "__main__":
    main()