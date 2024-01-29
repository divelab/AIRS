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
from datasets_config import pdb_protein
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
import time

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

def encoder(model, batched_data):
    x, coords_ca, edge_index, batch = batched_data.x, batched_data.coords_ca, batched_data.edge_index, batched_data.batch

    h = model.residue_type_embedding(x.squeeze(1).long()).to(device)

    # encoder
    emb_coords_ca, emb_h, batched_data, edge_index = model.encoder(coords_ca, h, edge_index, batch, batched_data)

    return emb_coords_ca, emb_h, model.mlp_mu_h(emb_h), model.mlp_sigma_h(emb_h)

def get_reconstructed(model, data_set):
    dataloader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=0)

    model.eval()

    aa_list = []
    edge_dist_list = []
    seq_len_list = []
    pos_x_list = []
    pos_y_list = []
    pos_z_list = []
    bond_angle_list = []
    torsion_angle_list = []

    emb_coords_list = []
    emb_h_list = []
    mu_h_list = []
    sigma_h_list = []

    aa_train_list = []
    coords_train_list = []
    coords_reconstructed_list = []

    for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
        if step == 100:
            break

        batch.coords_ca = batch.coords_ca.double()
        batch = batch.to(device)
        with torch.no_grad():

            coords_ca_pred, aa_pred, pad_pred, _, _ = model(batch)

            emb_coords_ca, emb_h, mu_h, sigma_h = encoder(model, batch)


        emb_coords_list.append(emb_coords_ca.cpu().numpy())
        emb_h_list.append(emb_h.cpu().numpy())
        mu_h_list.append(mu_h.cpu().numpy())
        sigma_h_list.append(sigma_h.cpu().numpy())


        if torch.where(pad_pred > 0.5)[0].shape[0] != 0:
            idx = torch.where(pad_pred > 0.5)[0][0]
        else:
            continue

        coords_train_list.append(batch.coords_ca[batch.protein_mask].cpu())
        aa_train_list.append(batch.x.squeeze()[batch.protein_mask].cpu())

        coords_ca = coords_ca_pred[0:idx]
        aa_type = torch.argmax(aa_pred[0:idx], dim=1)

        coords_reconstructed_list.append(coords_ca.cpu().numpy())
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

    aa_origin = torch.cat(aa_list, dim=0)
    edge_dist_origin = torch.cat(edge_dist_list, dim=0)
    seq_len_origin = torch.from_numpy(np.array(seq_len_list))

    pos_x = torch.cat(pos_x_list, dim=0)
    pos_y = torch.cat(pos_y_list, dim=0)
    pos_z = torch.cat(pos_z_list, dim=0)
    bond_angle = torch.cat(bond_angle_list, dim=0)
    torsion_angle = torch.cat(torsion_angle_list, dim=0)

    emb_coords = np.concatenate(emb_coords_list, axis=0)
    emb_h = np.concatenate(emb_h_list, axis=0)
    mu_h = np.concatenate(mu_h_list, axis=0)
    sigma_h = np.concatenate(sigma_h_list, axis=0)
    z_h = mu_h + np.exp(sigma_h / 2) * np.random.randn(*mu_h.shape)

    reconstructed = {'aa': aa_origin,
                     'edge_dist': edge_dist_origin,
                     'seq_len': seq_len_origin,
                     'pos_x': pos_x,
                     'pos_y': pos_y,
                     'pos_z': pos_z,
                     'bond_angle': bond_angle,
                     'torsion_angle': torsion_angle}

    latent = {
        'emb_coords': emb_coords,
        'emb_h': emb_h,
        'mu_h': mu_h,
        'sigma_h': sigma_h,
        'z_h': z_h,
    }

    reconstructed_protein = {
        'coords': coords_reconstructed_list,
        'aa': aa_list
    }

    train_protein = {
        'coords': coords_train_list,
        'aa': aa_train_list
    }

    return reconstructed, latent, reconstructed_protein, train_protein



def get_decoded(model, latent_data, num_samples=100, gaussian_h=False):
    print('num_samples: ', num_samples)
    model.eval()
    batch = torch.zeros((32,), dtype=torch.int32).to(device)
    aa_list = []
    coords_list = []
    seq_len_list = []


    count = 0
    num_decoded_proteins = 0

    start = time.time()
    for i in tqdm(range(latent_data['h'].shape[0])):
        num_decoded_proteins += 1
        with torch.no_grad():
            if gaussian_h:
                h = torch.randn_like(latent_data['h'][i]).double() * 0.7
                coords_ca_pred, h = model.decoder(latent_data['x'][i].double(), h, batch, None)
            else:
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
        coords_list.append(coords_ca.cpu().numpy())

        seq_len_list.append(idx.cpu().item())

        count += 1
        # print("count: ", count)

        if count == num_samples:
            break

    print(f"Total time for decoding {num_decoded_proteins} proteins: {time.time() - start}")
    print(f"{count} proteins with length > 50")
    assert count == num_samples

    generated_protein = {
        'coords': coords_list,
        'aa': aa_list,
        'seq_len': seq_len_list
    }

    return generated_protein

def get_decoded_batch(model, latent_data, num_samples=100, gaussian_h=False):
    print('num_samples: ', num_samples)
    model.eval()
    batch = torch.zeros((32,), dtype=torch.int32).to(device)
    aa_list = []
    # edge_dist_list = []
    coords_list = []
    seq_len_list = []

    batch_size = 32

    dataset = [Data(x=latent_data['x'][i], emb_h=latent_data['h'][i]) for i in range(latent_data['h'].shape[0])]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    count = 0
    num_decoded_proteins = 0

    start = time.time()
    for i, batch_data in enumerate(tqdm(dataloader, desc="Iteration")):
        num_decoded_proteins += (batch_size)

        with torch.no_grad():

            coords_ca_pred, h = model.decoder(batch_data.x.double(), batch_data.emb_h.double(), batch_data.batch, None)

            pad_pred = model.sigmoid(model.mlp_padding(h))
            aa_pred = model.mlp_residue(h)

    print(f"Total time for decoding {num_decoded_proteins} proteins: {time.time() - start}")
    print(f"{count} proteins with length > 50")
    assert count == num_samples

    generated_protein = {
        'coords': coords_list,
        'aa': aa_list,
        'seq_len': seq_len_list
    }

    return generated_protein

def get_interpolated(model, data_set):

    train_loader = DataLoader(data_set, batch_size=2, shuffle=False, num_workers=0)

    model.eval()

    aa_list = []
    edge_dist_list = []
    seq_len_list = []
    coords_list = []
    coords_train_list = []

    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        if step == 100:
            break
        batch.coords_ca = batch.coords_ca.double()
        batch = batch.to(device)
        with torch.no_grad():

            # encoder
            emb_coords_ca, emb_h, mu_h, sigma_h = encoder(model, batch)

            # interpolate two latent representation
            z_h = mu_h + torch.exp(sigma_h / 2) * torch.randn_like(mu_h)
            z_h = z_h.view(2, 32, 32)
            interp_h = z_h[0] * 0.9 + z_h[1] * 0.1
            emb_coords_ca = emb_coords_ca.view(2, 32, 3)
            interp_ca = emb_coords_ca[0] * 0.9 + emb_coords_ca[1] * 0.1


            # decoder
            batch = torch.zeros((32,), dtype=torch.int32).to(device)
            coords_ca_pred, h = model.decoder(interp_ca, interp_h, batch, None)
            pad_pred = model.sigmoid(model.mlp_padding(h))
            aa_pred = model.mlp_residue(h)

        idx = torch.where(pad_pred > 0.5)[0][0]
        if idx < 50:
            continue
        coords_ca = coords_ca_pred[0:idx]
        aa_type = torch.argmax(aa_pred[0:idx], dim=1)


        if aa_type.max() > 19:
            continue

        aa_list.append(aa_type)
        dist = (coords_ca[0:-1] - coords_ca[1:]).pow(2).sum(-1).sqrt()
        edge_dist_list.append(dist)
        coords_list.append(coords_ca.cpu().numpy())
        seq_len_list.append(idx.cpu().item())

    aa = torch.cat(aa_list, dim=0)
    edge_dist = torch.cat(edge_dist_list, dim=0)
    seq_len = np.array(seq_len_list)

    interpolated_protein = {
        'coords': coords_list,
        'aa': aa_list
    }

    return interpolated_protein

def save_aa_coords(protein, savedir=None):

    idx2aa = {v: pdb_protein['amino_acid_abbr'][k] for k, v in pdb_protein['aa2idx'].items()}

    aa_list = protein['aa']
    coords_list = protein['coords']

    aa_seq_list = []
    for aa in aa_list:
        aa = aa.cpu().numpy()
        seq = [idx2aa[idx] for idx in aa]
        aa_seq_list.append("".join(seq))

    with open(os.path.join(savedir, "sequence.fasta"), 'w') as file:
        for idx, aa_seq in enumerate(aa_seq_list):
            file.write(f">seq{idx} \n")
            file.write(aa_seq + " \n")

    with open(os.path.join(savedir, "seq_len.txt"), 'w') as file:
        for idx, aa_seq in enumerate(aa_seq_list):
            file.write(f">seq{idx} {len(aa_seq)} \n")

    savedir = os.path.join(savedir, "pdbs", "")
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for i in range(len(coords_list)):
        fname = os.path.join(savedir, f"generated_{i}.pdb")
        write_coords_to_pdb(coords_list[i], fname)

def write_coords_to_pdb(coords: np.ndarray, out_fname: str) -> str:
    """
    Write the coordinates to the given pdb fname
    """
    # Create a new PDB file using biotite
    # https://www.biotite-python.org/tutorial/target/index.html#creating-structures
    # assert len(coords) % 3 == 0

    atoms = []
    for i, ca_coord in enumerate(coords):

        atom = struc.Atom(
            ca_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i + 1,
            res_name="GLY",
            atom_name="CA",
            element="C",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )

        atoms.extend([atom])
    full_structure = struc.array(atoms)


    sink = PDBFile()
    sink.set_structure(full_structure)
    sink.write(out_fname)
    return out_fname

def analyze(model, loader=None, save_path=None, latent_data=None, num_samples=100, gaussian_h=False):

    # load dataset
    # if loader == None:
    print('Loading data...')
    test_set = torch.load(os.path.join('./data/', 'PDB_data_128_Test_complete.pt'))

    savedir = os.path.join(save_path, 'scTM')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # save ground truth seq and structure
    # save reconstructed seq and structure
    print('Get reconstruction structure...')
    _, _, reconstructed_protein, train_protein = get_reconstructed(model, test_set)
    save_path = os.path.join(savedir, 'ground_truth')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_aa_coords(train_protein, save_path)

    save_path = os.path.join(savedir, 'reconstructed')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_aa_coords(reconstructed_protein, save_path)


    # save generated seq and structure
    print('Get generation structure...')

    start = time.time()
    generated_protein = get_decoded(model, latent_data, num_samples=num_samples, gaussian_h=gaussian_h)
    print("decode time: {:.6}".format(time.time() - start))
    save_path = os.path.join(savedir, 'generated')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_aa_coords(generated_protein, save_path)


    # save perturbed/interpolated latent graph decoded seq and structure

    # print('Get interpolation structure...')
    # interpolated_protein = get_interpolated(model, test_set)
    # save_path = os.path.join(savedir, 'interpolated')
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # save_aa_coords(interpolated_protein, save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein generation")
    parser.add_argument('--model_path', type=str, default='', help='model path')
    parser.add_argument("--working_dir", type=str, default="./", help="working directory for logs, saved models, etc.")
    parser.add_argument('--suffix', type=str, default='', help='')
    parser.add_argument('--save_folder', type=str, default='', help='')
    parser.add_argument('--diffusion_train_data', type=str, default='', help='data name')
    parser.add_argument('--diffusion_generate_data', type=str, default='', help='data path')
    parser.add_argument('--num_samples', type=int, default=100, help='number of samples')
    parser.add_argument('--gaussian_h', type=eval, default=False, help='sample h from gaussian')

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


    print('Loading model...')
    model = ProAuto(**params).double().to(device)

    suffix = 'trained_models/checkpoint_bst_rmsd.pt' if args.suffix == '' else args.suffix
    model.load_state_dict(torch.load(os.path.join(args.working_dir, args.model_path, suffix))['model_state_dict'])

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n# Params: {num_params}")

    savedir = os.path.join(args.working_dir, args.model_path, args.save_folder)

    analyze(model, loader=None, save_path=savedir, latent_data=latent_data, num_samples=args.num_samples, gaussian_h=args.gaussian_h)




