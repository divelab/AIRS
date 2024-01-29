import numpy as np
import os
import sys
sys.path.append("./protein_autoencoder/")
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.nn import radius_graph
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import argparse
from torch_geometric.data import Data
from utils import RMSD
from model import ProAuto


device = torch.device('cuda:0')


def get_ground_truth(train_set):
    aa_train_list = []
    edge_dist_train_list = []
    seq_len_train_list = []
    pos_x_train_list = []
    pos_y_train_list = []
    pos_z_train_list = []
    bond_angle_train_list = []
    torsion_angle_train_list = []

    for i in range(len(train_set)):
        protein_mask = train_set[i].protein_mask
        coords_ca = train_set[i].coords_ca[protein_mask]
        aa_type = train_set[i].x[protein_mask]
        aa_train_list.append(aa_type)
        dist = (coords_ca[0:-1] - coords_ca[1:]).pow(2).sum(-1).sqrt()
        edge_dist_train_list.append(dist)
        seq_len_train_list.append(coords_ca.shape[0])

        # x y z position
        pos_x_train_list.append(coords_ca[:, 0])
        pos_y_train_list.append(coords_ca[:, 1])
        pos_z_train_list.append(coords_ca[:, 2])

        # bond angle
        r_ji = coords_ca[1:-1] - coords_ca[0:-2]
        r_jk = coords_ca[1:-1] - coords_ca[2:]
        e_ji = r_ji / r_ji.pow(2).sum(-1).sqrt().view(-1, 1)
        e_jk = r_jk / r_jk.pow(2).sum(-1).sqrt().view(-1, 1)
        cos_phi = (e_ji * e_jk).sum(-1)
        bond_angle = torch.acos(cos_phi)  # * (180 / 3.1415926)
        bond_angle_train_list.append(bond_angle)

        # torsion angle
        v1 = coords_ca[1:-2] - coords_ca[0:-3]  # r_ji
        v2 = coords_ca[2:-1] - coords_ca[1:-2]  # r_kj
        v3 = coords_ca[3:] - coords_ca[2:-1]  # r_lk
        v1 = v1 / v1.norm(dim=1, keepdim=True)
        v2 = v2 / v2.norm(dim=1, keepdim=True)
        v3 = v3 / v3.norm(dim=1, keepdim=True)
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))

        torsion_angle = torch.nan_to_num(torch.atan2(b, a))
        torsion_angle_train_list.append(torsion_angle)

    aa_train = torch.cat(aa_train_list, dim=0)
    edge_dist_train = torch.cat(edge_dist_train_list, dim=0)
    seq_len_train = torch.from_numpy(np.array(seq_len_train_list))

    pos_x_train = torch.cat(pos_x_train_list, dim=0)
    pos_y_train = torch.cat(pos_y_train_list, dim=0)
    pos_z_train = torch.cat(pos_z_train_list, dim=0)
    bond_angle_train = torch.cat(bond_angle_train_list, dim=0)
    torsion_angle_train = torch.cat(torsion_angle_train_list, dim=0)

    gt = {'aa': aa_train,
          'edge_dist': edge_dist_train,
          'seq_len': seq_len_train,
          'pos_x': pos_x_train,
          'pos_y': pos_y_train,
          'pos_z': pos_z_train,
          'bond_angle': bond_angle_train,
          'torsion_angle': torsion_angle_train}

    return gt

def get_decoded(model, latent_data):
    model.eval()
    batch = torch.zeros((32,), dtype=torch.int32).to(device)
    aa_list = []
    edge_dist_list = []
    seq_len_list = []
    pos_x_list = []
    pos_y_list = []
    pos_z_list = []
    bond_angle_list = []
    torsion_angle_list = []

    count = 0
    for i in tqdm(range(latent_data['h'].shape[0])):
        with torch.no_grad():
            coords_ca_pred, h = model.decoder(latent_data['x'][i].double(), latent_data['h'][i].double(), batch, None)
            pad_pred = model.sigmoid(model.mlp_padding(h))
            aa_pred = model.mlp_residue(h)

        if torch.where(pad_pred > 0.5)[0].shape[0] == 0:
            # print("generated length < 50")
            continue
        idx = torch.where(pad_pred > 0.5)[0][0]
        if idx < 50:
            # print("generated length < 50")
            continue
        coords_ca = coords_ca_pred[0:idx]
        aa_type = torch.argmax(aa_pred[0:idx], dim=1)

        aa_list.append(aa_type)
        dist = (coords_ca[0:-1] - coords_ca[1:]).pow(2).sum(-1).sqrt()
        edge_dist_list.append(dist)
        seq_len_list.append(idx.cpu().item())

        pos_x_list.append(coords_ca[:, 0])
        pos_y_list.append(coords_ca[:, 1])
        pos_z_list.append(coords_ca[:, 2])

        # bond angle
        r_ji = coords_ca[1:-1] - coords_ca[0:-2]
        r_jk = coords_ca[1:-1] - coords_ca[2:]
        e_ji = r_ji / r_ji.pow(2).sum(-1).sqrt().view(-1, 1)
        e_jk = r_jk / r_jk.pow(2).sum(-1).sqrt().view(-1, 1)
        cos_phi = (e_ji * e_jk).sum(-1)
        bond_angle = torch.acos(cos_phi)  # * (180 / 3.1415926)
        bond_angle_list.append(bond_angle)

        # torsion angle
        v1 = coords_ca[1:-2] - coords_ca[0:-3]  # r_ji
        v2 = coords_ca[2:-1] - coords_ca[1:-2]  # r_kj
        v3 = coords_ca[3:] - coords_ca[2:-1]  # r_lk
        v1 = v1 / v1.norm(dim=1, keepdim=True)
        v2 = v2 / v2.norm(dim=1, keepdim=True)
        v3 = v3 / v3.norm(dim=1, keepdim=True)
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))

        torsion_angle = torch.nan_to_num(torch.atan2(b, a))
        torsion_angle_list.append(torsion_angle)

        count += 1

    print(f"{count} proteins with length > 50")
    assert len(aa_list) > 0
    aa = torch.cat(aa_list, dim=0)
    edge_dist = torch.cat(edge_dist_list, dim=0)
    seq_len = torch.from_numpy(np.array(seq_len_list))

    pos_x = torch.cat(pos_x_list, dim=0)
    pos_y = torch.cat(pos_y_list, dim=0)
    pos_z = torch.cat(pos_z_list, dim=0)
    bond_angle_decode = torch.cat(bond_angle_list, dim=0)
    torsion_angle_decode = torch.cat(torsion_angle_list, dim=0)

    reconstructed = {'aa': aa,
                     'edge_dist': edge_dist,
                     'seq_len': seq_len,
                     'pos_x': pos_x,
                     'pos_y': pos_y,
                     'pos_z': pos_z,
                     'bond_angle': bond_angle_decode,
                     'torsion_angle': torsion_angle_decode}

    return reconstructed

def plot_histogram(gt, reconstructed, save_path, key_name, bins=None):
    colors = ['blue', 'orange']
    labels = ['true', 'sample']

    title_font = {'fontsize': 30}
    axis_font = {'fontsize': 30}
    tick_font = {'fontsize': 20}


    # plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(figsize=(9,8))
    if key_name == 'aa':
        n_train, _, _ = ax.hist(gt.numpy(), bins=np.arange(0, 20 + 1.1, 1),
                                 weights=None, cumulative=False, bottom=None, density=True,
                                 histtype=u'bar', align=u'left', orientation=u'vertical',
                                 rwidth=1, log=False, color='blue', label='true', stacked=False, alpha=0.6)
        n, _, _ = ax.hist(reconstructed.cpu().numpy(), bins=np.arange(0, 20 + 1.1, 1),
                           weights=None, cumulative=False, bottom=None, density=True,
                           histtype=u'bar', align=u'left', orientation=u'vertical',
                           rwidth=1, log=False, color='orange', label='AE reconstructed', stacked=False, alpha=0.6)
        ax.legend()
        ax.set_title('Amini acid type')
        ax.set_xlabel('Type')
        ax.set_ylabel('Density')
        # plt.xticks(**tick_font)
        # plt.yticks(**tick_font)

    else:
        if key_name in ['bond_angle', 'edge_dist', 'torsion_angle', 'pos_x', 'pos_y', 'pos_z']:
            _, _, _ = ax.hist(gt.numpy(), bins=bins, density=True, color='blue', alpha=0.6,
                            label='Test')
            _, _, _ = ax.hist(reconstructed.cpu().numpy(), bins=bins, density=True, color='orange', alpha=0.6,
                            label='Generated')
            ax.legend(**axis_font)

        if key_name == 'bond_angle':
            ax.set_title(f'Bond angle', **title_font)
            ax.set_xlabel(f'Angle (rad)', **axis_font)
            ax.set_ylabel('Density', **axis_font)
            plt.xticks(**tick_font)
            plt.yticks(**tick_font)
        elif key_name == 'edge_dist':
            ax.set_title(f'CA-CA distance', **title_font)
            ax.set_xlabel(f'Distance (Å)', **axis_font)
            ax.set_ylabel('Density', **axis_font)
            plt.xticks(**tick_font)
            plt.yticks(**tick_font)
        elif key_name == 'torsion_angle':
            ax.set_title(f'Torsion angle', **title_font)
            ax.set_xlabel(f'Angle (rad)', **axis_font)
            ax.set_ylabel('Density', **axis_font)
            plt.xticks(**tick_font)
            plt.yticks(**tick_font)

    # print(save_path)
    fig.savefig(os.path.join(save_path, f'{key_name}.pdf'))

def plot_latent_distribution(latent_data, latent_train_data, save_path):
    ###############################################

    title_font = {'fontsize': 30}
    axis_font = {'fontsize': 30}
    tick_font = {'fontsize': 20}

    latent_edge_dist_list = []
    latent_edge_dist_train_list = []

    for i in tqdm(range(latent_data['h'].shape[0])):
        x = latent_data['x'][i]
        dist = (x[0:-1] - x[1:]).pow(2).sum(-1).sqrt()
        latent_edge_dist_list.append(dist)
    latent_edge_dist = torch.cat(latent_edge_dist_list, dim=0)

    for i in tqdm(range(len(latent_train_data))):
        x = latent_train_data[i].coords
        dist = (x[0:-1] - x[1:]).pow(2).sum(-1).sqrt()
        latent_edge_dist_train_list.append(dist)
    latent_edge_train_dist = torch.cat(latent_edge_dist_train_list, dim=0)

    fig, ax = plt.subplots(figsize=(12,8))
    _, _, _ = ax.hist(latent_edge_train_dist.cpu().numpy(), bins=np.arange(2, 30, 0.1), density=True, color='blue',
                       alpha=0.6, label='latent train')
    _, _, _ = ax.hist(latent_edge_dist.cpu().numpy(), bins=np.arange(2, 30, 0.1), density=True, color='orange',
                       alpha=0.6, label='latent sample')

    ax.legend(**axis_font)
    ax.set_title('Latent edge distance', **title_font)
    ax.set_xlabel('Distance (Å)', **axis_font)
    ax.set_ylabel('Density', **axis_font)
    plt.xticks(**tick_font)
    plt.yticks(**tick_font)
    fig.savefig(os.path.join(save_path, 'latent_edge_distance.pdf'))

    #################################################
    latent_pos_x_list = []
    latent_pos_x_train_list = []

    for i in tqdm(range(latent_data['h'].shape[0])):
        pos_x = latent_data['x'][i][:, 0]
        latent_pos_x_list.append(pos_x)
    latent_pos_x = torch.cat(latent_pos_x_list, dim=0)

    for i in tqdm(range(len(latent_train_data))):
        pos_x = latent_train_data[i].coords[:, 0]
        latent_pos_x_train_list.append(pos_x)
    latent_pos_x_train = torch.cat(latent_pos_x_train_list, dim=0)

    fig, ax = plt.subplots(figsize=(12,8))
    _, _, _ = ax.hist(latent_pos_x_train.cpu().numpy(), bins=np.arange(-50, 50, 0.1), density=True, color='blue',
                       alpha=0.6, label='latent train')
    _, _, _ = ax.hist(latent_pos_x.cpu().numpy(), bins=np.arange(-50, 50, 0.1), density=True, color='orange', alpha=0.6,
                       label='latent sample')

    ax.legend(**axis_font)
    ax.set_title('Latent position in x direction', **title_font)
    ax.set_xlabel('Position (Å)', **axis_font)
    ax.set_ylabel('Density', **axis_font)
    plt.xticks(**tick_font)
    plt.yticks(**tick_font)
    fig.savefig(os.path.join(save_path, 'latent_pos_x.pdf'))

    ###################################################
    latent_h_0_list = []
    latent_h_0_train_list = []

    # import pdb; pdb.set_trace()
    for i in tqdm(range(latent_data['h'].shape[0])):
        h_0 = latent_data['h'][i][:, 0]
        latent_h_0_list.append(h_0)
    latent_h_0 = torch.cat(latent_h_0_list, dim=0)

    for i in tqdm(range(len(latent_train_data))):
        h_0 = latent_train_data[i].h[:, 0]
        latent_h_0_train_list.append(h_0)
    latent_h_0_train = torch.cat(latent_h_0_train_list, dim=0)

    fig, ax = plt.subplots(figsize=(12,8))
    _, _, _ = ax.hist(latent_h_0_train.cpu().numpy(), bins=np.arange(-5, 5, 0.1), density=True, color='blue',
                       alpha=0.6, label='latent train')
    _, _, _ = ax.hist(latent_h_0.cpu().numpy(), bins=np.arange(-5, 5, 0.1), density=True, color='orange', alpha=0.6,
                       label='latent sample')

    ax.legend(**axis_font)
    ax.set_title('Latent embeddings in first dimension', **title_font)
    ax.set_xlabel('Embedding value', **axis_font)
    ax.set_ylabel('Density', **axis_font)
    plt.xticks(**tick_font)
    plt.yticks(**tick_font)
    fig.savefig(os.path.join(save_path, 'latent_h_dim0.pdf'))


def analyze(model, loader=None, save_path=None, latent_data=None):

    # load dataset
    # if loader == None:
    print('Loading data...')
    test_set = torch.load(os.path.join('./data/', 'PDB_data_128_Test_complete.pt'))

    # get ground truth and reconstructed information
    gt = get_ground_truth(test_set)
    decoded = get_decoded(model, latent_data)


    savedir_decoded = os.path.join(save_path, 'diffusion_decoded')

    if not os.path.exists(savedir_decoded):
        os.makedirs(savedir_decoded)

    # plot histogram
    for key in decoded.keys():
        if key in ['pos_x', 'pos_y', 'pos_z']:
            # bins = np.arange(-80, 80, 0.1)
            bins = np.arange(-50, 50, 0.1)
        elif key == 'bond_angle':
            bins = np.arange(0, 3.14, 0.05)
        elif key == 'torsion_angle':
            bins = np.arange(-3.2, 3.2, 0.05)
        elif key == 'edge_dist':
            bins = np.arange(0, 6, 0.1)
        else:
            continue

        plot_histogram(gt[key], decoded[key], savedir_decoded, key, bins)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein generation")
    parser.add_argument('--model_path', type=str, default='', help='model path')
    parser.add_argument("--working_dir", type=str, default="./", help="working directory for logs, saved models, etc.")
    parser.add_argument('--suffix', type=str, default='', help='')
    parser.add_argument('--save_folder', type=str, default='', help='')
    parser.add_argument('--diffusion_train_data', type=str, default='', help='data name')
    parser.add_argument('--diffusion_generate_data', type=str, default='', help='data name')

    args = parser.parse_args()

    params = {
        "mp_steps": 4,
        "layers": 2,
        "num_types": 27,
        "type_dim": 32,
        "hidden_dim": 32,
        "out_node_dim": 32,
        "in_edge_dim": 32,
        "output_pad_dim": 1,
        "output_res_dim": 20,
        "pooling": True,
        "up_mlp": False,
        "residual": True,
        "noise": False,
        "transpose": True,
        "attn": True,
        "stride": 2,
        "kernel": 3,
        "padding": 1
    }


    latent_data = torch.load(f'./diffusion/outputs/{args.diffusion_generate_data}/latent_data.pt')

    latent_data['h'] = torch.tensor(latent_data['h'], device=device)
    latent_data['x'] = torch.tensor(latent_data['x'], device=device)

    latent_train_data = torch.load(f'./data/{args.diffusion_train_data}.pt')

    savedir_difflatent = os.path.join(args.working_dir, args.model_path, args.save_folder, 'diffusion_latent')
    if not os.path.exists(savedir_difflatent):
        os.makedirs(savedir_difflatent)
    plot_latent_distribution(latent_data, latent_train_data, savedir_difflatent)

    print('Loading model...')
    model = ProAuto(**params).double().to(device)
        
    suffix = 'trained_models/checkpoint_bst_rmsd.pt' if args.suffix == '' else args.suffix
    model.load_state_dict(torch.load(os.path.join(args.working_dir, args.model_path, suffix))['model_state_dict'])

    savedir = os.path.join(args.working_dir, args.model_path, args.save_folder)

    analyze(model, loader=None, save_path=savedir, latent_data=latent_data)

