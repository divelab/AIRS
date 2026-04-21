import os
import argparse

import torch
from tqdm import tqdm
# import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
# from plot_utils import Dipole, Efield
import math
import importlib

import yaml
import torch_scatter

from orbevo.eval_utils import tfn_to_abacus_mol, normalize_density, get_dipole, Absorption, Freq2eV

def get_model(cfg, **kwargs):
    model_cls = getattr(importlib.import_module('orbevo.models'), cfg['model']['name'])
    if cfg['model']['args'] is not None:
        net = model_cls(**cfg['model']['args'], **kwargs)
    else:
        net = model_cls(**kwargs)
    return net


def rollout(model, batch, time_cond=8, time_future=8, onestep=False, disable_tqdm=False):
    # Onestep
    # for test_id, data in tqdm(enumerate(test_ds)):
    #     data_torch = torch.load(f'{data_root}/raw/{test_inds[test_id]:05d}.pt')

    # batch = collate_fn([data_pyg])
    batch = {k: v.to(device) for k, v in batch.items()}

    use_amp = device.startswith("cuda")
    autocast_device_type = "cuda" if use_amp else "cpu"
    with torch.no_grad():
        efield_clone = batch['molecule_data'].efield.clone()

        preds = []
        rollout_steps = math.ceil(100 / time_future)
        for i in tqdm(range(rollout_steps), disable=disable_tqdm):
            batch['molecule_data'].efield = efield_clone[:, time_future * 10 * i : time_future * 10 * i + (time_cond + time_future) * 10]
            if batch['molecule_data'].efield.shape[1] < (time_cond + time_future) * 10:
                # pad
                pad_len = (time_cond + time_future) * 10 - batch['molecule_data'].efield.shape[1]
                batch['molecule_data'].efield = torch.cat(
                    [batch['molecule_data'].efield, torch.zeros(efield_clone.shape[0], pad_len).to(efield_clone.device)], dim=1
                )

            with torch.autocast(device_type=autocast_device_type, enabled=use_amp):
                pred = model(batch)

            preds.append(pred['delta_coef_t_norm'])
            if onestep:
                batch['state_data'].delta_coef_cond = batch['state_data'].delta_coef_target[i * time_future : (i + 1) * time_future]
            else:
                batch['state_data'].delta_coef_cond = pred['delta_coef_t_norm']

    preds = torch.cat(preds, dim=0)[:100]
    targets = batch['state_data'].delta_coef_target[:100]

    return preds, targets


def scaled_l2(pred, target, graph_batch, state_ind_batch):
    err = pred - target
    err_norm = err.flatten(start_dim=2).norm(dim=-1)  # T, N
    err_norm = torch_scatter.scatter_sum(err_norm ** 2, graph_batch, dim=1).sqrt()  # T, num_ebands
    err_norm = torch.sum(err_norm ** 2, dim=0).sqrt()  # num_ebands
    err_norm = torch_scatter.scatter_mean(err_norm, state_ind_batch, dim=0)  # batch_size
    target_norm = target.flatten(start_dim=2).norm(dim=-1)  # T, N
    target_norm = torch_scatter.scatter_sum(target_norm ** 2, graph_batch, dim=1).sqrt()  # T, num_ebands
    target_norm = torch.sum(target_norm ** 2, dim=0).sqrt()  # num_ebands
    target_norm = torch_scatter.scatter_mean(target_norm, state_ind_batch, dim=0)  # batch_size
    return err_norm / target_norm


def compute_dipole_absorption_nrmse(coef_0_mol, preds_mol, targets_mol, mat_r, occ, efield):
    # Dipole
    coef_0 = coef_0_mol.unsqueeze(0).cpu()
    mat_r = mat_r.unsqueeze(0)
    # occ = batch['molecule_data'].occ[batch['state_data'].state_ind_batch == 0].cpu()
    dipole_0 = get_dipole(coef_0, torch.zeros_like(coef_0), mat_r, occ)
    dipole_target = get_dipole(coef_0, targets_mol.cpu().unsqueeze(0), mat_r, occ) - dipole_0
    dipole_pred = get_dipole(coef_0, preds_mol.cpu().unsqueeze(0), mat_r, occ) - dipole_0

    dipole_err_all = (dipole_pred - dipole_target).flatten().norm() / dipole_target.flatten().norm()
    dipole_err_z = (dipole_pred - dipole_target)[:, :, 2].flatten().norm() / dipole_target[:, :, 2].flatten().norm()

    # Absorption
    Abs = Absorption(length=101, dt=0.05)
    efield_fft = np.fft.fft(Abs.padding(torch.cat([torch.tensor([0.]), efield[9::10]])))   
    dipole_target_fft = np.fft.fft(Abs.padding(torch.cat([torch.tensor([0.]), dipole_target[0, :, 2].real])))
    alpha_target = np.abs((dipole_target_fft / efield_fft).imag)
    dipole_pred_fft = np.fft.fft(Abs.padding(torch.cat([torch.tensor([0.]), dipole_pred[0, :, 2].real])))
    alpha_pred = np.abs((dipole_pred_fft / efield_fft).imag)

    Ndim = 101 * 10
    index=np.linspace(0, Ndim - 1, Ndim)
    energies = Freq2eV * index / 0.05 / len(index)
    x_data = energies

    absorption_err = torch.tensor((alpha_pred - alpha_target)[x_data < 20]).norm() / torch.tensor(alpha_target[x_data < 20]).norm()

    return dipole_err_all, dipole_err_z, absorption_err


def find_config_path(ckpt_path, override_path=None):
    if override_path is not None:
        return override_path

    run_root = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
    candidates = [
        os.path.join(run_root, "config", "config.yaml"),
        os.path.join(run_root, ".hydra", "config.yaml"),
        os.path.join(run_root, "config.yaml"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Could not locate config.yaml near ckpt path: {ckpt_path}. "
        "Tried run_root/config/config.yaml, run_root/.hydra/config.yaml, and run_root/config.yaml."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['MDA', 'QM9', 'QM9_ood'], required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--split', choices=['val', 'test'], default='test', required=False)
    args = parser.parse_args()

    from orbevo.datasets.pyg_dataset import TDDFTv2_pyg, collate_fn, keep_qm9_good_after_split

    config_path = find_config_path(args.ckpt_path, args.config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(f'Loaded config from {config_path}')
    print(yaml.dump(config, default_flow_style=False))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    rms_0 = config['dataset']['rms_0']
    rms_t = config['dataset']['rms_t']

    config['dataset']['name'] = args.dataset
    print(f"Evaluating on dataset: {config['dataset']['name']}")

    # Get data
    if config['dataset']['name'] == 'QM9':
        data_root = config['dataset'].get('root', 'data/QM9_tddft')
        dataset =  TDDFTv2_pyg(root=data_root, transform=None, time_start=0, time_cond=config['time_cond'], time_future=100, T=100,
                            rms_0=rms_0, rms_t=rms_t)

        perm_inds= np.random.default_rng(seed=34).permutation(np.arange(5000))
        if args.split == 'test':
            test_inds = torch.tensor(perm_inds[4500 :])
        elif args.split == 'val':
            test_inds = torch.tensor(perm_inds[4000 : 5000])
        else:
            raise ValueError('Unknown split')
        test_inds = keep_qm9_good_after_split(config['dataset']['name'], test_inds)
        
        test_ds = torch.utils.data.Subset(dataset, test_inds)
        test_loader = DataLoader(test_ds, batch_size=20, shuffle=False, num_workers=10, collate_fn=collate_fn)
    elif config['dataset']['name'] == 'MDA':
        # Use explicit test split root for MDA evaluation if provided; fallback to default path.
        data_root = config['dataset'].get('val_test_root', 'data/MDA_test_tddftv2')
        dataset =  TDDFTv2_pyg(root=data_root, transform=None, time_start=0, time_cond=config['time_cond'], time_future=100, T=100,
                            rms_0=rms_0, rms_t=rms_t)
        test_inds = torch.arange(len(dataset))
        test_ds = dataset
        test_loader = DataLoader(test_ds, batch_size=20, shuffle=False, num_workers=10, collate_fn=collate_fn)
    elif config['dataset']['name'] == 'QM9_ood':
        data_root = config['dataset'].get('root', 'data/QM9_tddft')
        dataset =  TDDFTv2_pyg(root=data_root, transform=None, time_start=0, time_cond=config['time_cond'], time_future=100, T=100,
                               rms_0=rms_0, rms_t=rms_t)

        # Get the directory of the current file
        split_inds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'qm9_ood_split_inds.pt')
        all_inds = torch.load(split_inds_path)
        print(f'Loaded split inds from {split_inds_path}')
        # train_inds = all_inds['train']
        if args.split == 'test':
            test_inds = all_inds['test']
            # if args.dataset == 'qm9_id_ood_intersect':
            #     # Test overlap sets
            #     perm_inds= np.random.default_rng(seed=34).permutation(np.arange(5000))
            #     test_inds_rand = torch.tensor(perm_inds[4500 :])
            #     cat_inds, counts = torch.cat([test_inds, test_inds_rand]).unique(return_counts=True)
            #     test_inds = cat_inds[torch.where(counts.gt(1))] # intersection

        elif args.split == 'val':
            test_inds = all_inds['val']
            # if args.dataset == 'qm9_id_ood_intersect':
            #     # Test overlap sets
            #     perm_inds= np.random.default_rng(seed=34).permutation(np.arange(5000))
            #     test_inds_rand = torch.tensor(perm_inds[4000 : 4500])
            #     cat_inds, counts = torch.cat([test_inds, test_inds_rand]).unique(return_counts=True)
            #     test_inds = cat_inds[torch.where(counts.gt(1))] # intersection
        else:
            raise ValueError('Unknown split')
        test_inds = keep_qm9_good_after_split(config['dataset']['name'], test_inds)
        
        test_ds = torch.utils.data.Subset(dataset, test_inds)
        test_loader = DataLoader(test_ds, batch_size=20, shuffle=False, num_workers=10, collate_fn=collate_fn)
    else:
        raise ValueError
    
    # Get model
    model = get_model(
        config, 
        time_cond=config['time_cond'], 
        time_future=config['time_future'], 
        avg_num_nodes=config['dataset']['avg_num_nodes'], 
        avg_degree=config['dataset']['avg_degree']
    )

    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    print('epcoh:', ckpt['epoch'], ', iter:', ckpt['global_iter'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model = model.to(device)

    # Onestep
    onestep_err_list = []
    for batch in tqdm(test_loader):
        preds_onestep, targets = rollout(model=model,
                                        batch=batch,
                                        time_cond=config['time_cond'],
                                        time_future=config['time_future'],
                                        onestep=True,
                                        disable_tqdm=True)
        err = preds_onestep - targets
        atomwise_l2mae = err.flatten(start_dim=2).norm(dim=-1).mean(0)  # N
        molwise_l2mae = torch_scatter.scatter_mean(atomwise_l2mae, index=batch['state_data'].mol_batch, dim=0)
        onestep_err_list.append(molwise_l2mae)

    onestep_l2mae = torch.cat(onestep_err_list).mean() * config['dataset']['rms_t']
    print(f'onestep l2mae: {onestep_l2mae}')

    # Rollout
    rollout_err_list = []
    rollout_sl2_list = {8 : [], 16 : [], 32 : [], 64 : [], 100 : []} # {time_length: err}

    dipole_err_all_list = []
    dipole_err_z_list = []
    absorption_err_list = []
    test_idx_offset = 0
    for batch in tqdm(test_loader):
    # for test_id, data_pyg in tqdm(enumerate(test_ds)):
        # data_torch = torch.load(f'{data_root}/raw/{test_inds[test_id]:05d}.pt')
        preds_rollout, targets = rollout(model=model,
                                        batch=batch,
                                        time_cond=config['time_cond'],
                                        time_future=config['time_future'],
                                        onestep=False,
                                        disable_tqdm=True)
        
        err = preds_rollout - targets
        atomwise_l2mae = err.flatten(start_dim=2).norm(dim=-1).mean(0)  # N
        molwise_l2mae = torch_scatter.scatter_mean(atomwise_l2mae, index=batch['state_data'].mol_batch, dim=0)
        rollout_err_list.append(molwise_l2mae)

        for time_length in [8, 16, 32, 64, 100]:
            sl2_err = scaled_l2(pred=preds_rollout[:time_length], 
                                target=targets[:time_length], 
                                graph_batch=batch['state_data'].batch, 
                                state_ind_batch=batch['state_data'].state_ind_batch)
            rollout_sl2_list[time_length].append(sl2_err)

        batch_size = max(batch['molecule_data'].batch).item() + 1

        for i in range(batch_size):
            test_idx = i + test_idx_offset
            data_torch = torch.load(f'{data_root}/raw/{test_inds[test_idx]:05d}.pt')

            # Dipole
            coef_0_mol = tfn_to_abacus_mol(coef=batch['state_data'].coef_0[:, batch['state_data'].mol_batch == i, :, :],
                                abacus_to_tfn=dataset.abacus_to_tfn, 
                                atom_types=batch['molecule_data'].atom_type[batch['molecule_data'].batch == i], 
                                num_bands=batch['state_data'].num_states[i].item())
            coef_0_mol *= config['dataset']['rms_0']

            preds_mol = tfn_to_abacus_mol(coef=preds_rollout[:, batch['state_data'].mol_batch == i, :, :],
                                        abacus_to_tfn=dataset.abacus_to_tfn,
                                        atom_types=batch['molecule_data'].atom_type[batch['molecule_data'].batch == i],
                                        num_bands=batch['state_data'].num_states[i].item())
            preds_mol *= config['dataset']['rms_t']

            # preds_norm = preds_mol
            preds_mol_norm = normalize_density(coef_0_mol=coef_0_mol, 
                                        preds_mol=preds_mol,
                                        mat_S=data_torch['mat_S'].to(device)
                                        )

            targets_mol = tfn_to_abacus_mol(coef=targets[:, batch['state_data'].mol_batch == i, :, :],
                                        abacus_to_tfn=dataset.abacus_to_tfn,
                                        atom_types=batch['molecule_data'].atom_type[batch['molecule_data'].batch == i], 
                                        num_bands=batch['state_data'].num_states[i].item())
            targets_mol *= config['dataset']['rms_t']

            efield = data_torch['efield'][:1000, 1]

            dipole_err_all, dipole_err_z, absorption_err = compute_dipole_absorption_nrmse(
                coef_0_mol=coef_0_mol,
                preds_mol=preds_mol_norm,
                targets_mol=targets_mol,
                mat_r=data_torch['mat_r'],
                occ=batch['molecule_data'].occ[batch['molecule_data'].occ_batch == i].cpu(),
                efield=efield
            )

            dipole_err_all_list.append(dipole_err_all)
            dipole_err_z_list.append(dipole_err_z)
            absorption_err_list.append(absorption_err)

        test_idx_offset += batch_size

    rollout_l2mae = torch.cat(rollout_err_list).mean() * config['dataset']['rms_t']
    rollout_sl2 = {}
    for time_length in rollout_sl2_list:
        rollout_sl2[time_length] = torch.cat(rollout_sl2_list[time_length]).mean()
    print(f'Rollout l2mae: {rollout_l2mae}')
    print(f'Rollout scaled l2: {rollout_sl2}')
    dipole_err_all = torch.tensor(dipole_err_all_list).mean()
    dipole_err_z = torch.tensor(dipole_err_z_list).mean()
    absorption_err = torch.tensor(absorption_err_list).mean()
    print(f'Dipole all nrmse: {dipole_err_all}')
    print(f'Dipole z nrmse: {dipole_err_z}')
    print(f'Absorption nrmse: {absorption_err}')
