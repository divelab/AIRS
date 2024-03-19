# the test scripts currently run over the molecule one by one
# causing the evaluation needs a long time

#!/usr/bin/env python3
import os
import time
import torch
import hydra
import pyscf
import logging
import numpy as np
from tqdm import tqdm
from models import QHNet
from torch_scatter import scatter_sum
from datasets import QH9Stable, QH9Dynamic
from argparse import Namespace
from torch_geometric.loader import DataLoader
from load_pretrained_models import load_pretrained_model_parameters
logger = logging.getLogger()


convention_dict = {
    'pyscf_631G': Namespace(
        atom_to_orbitals_map={1: 'ss', 6: 'ssspp', 7: 'ssspp', 8: 'ssspp', 9: 'ssspp'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd':
                          [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1], 6: [0, 1, 2, 3, 4], 7:  [0, 1, 2, 3, 4],
            8:  [0, 1, 2, 3, 4], 9:  [0, 1, 2, 3, 4]
        },
    ),
    'pyscf_def2svp': Namespace(
        atom_to_orbitals_map={1: 'ssp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd', 9: 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2], 6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5], 9: [0, 1, 2, 3, 4, 5]
        },
    ),
    'back2pyscf': Namespace(
        atom_to_orbitals_map={1: 'ssp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd', 9: 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
        1: [0, 1, 2], 6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5],
        8: [0, 1, 2, 3, 4, 5], 9: [0, 1, 2, 3, 4, 5]
        }
    )
}


def criterion(outputs, target, names):
    error_dict = {}
    for key in names:
        if key == 'orbital_coefficients':
            "The shape if [batch, total_orb, num_occ_orb]."
            error_dict[key] = torch.cosine_similarity(outputs[key], target[key], dim=1).abs().mean()
        elif key in ['diagonal_hamiltonian', 'non_diagonal_hamiltonian']:
            diff_blocks = outputs[key] - target[key]
            mae_blocks = torch.sum(torch.abs(diff_blocks) * target[f"{key}_mask"], dim=[1, 2])
            count_sum_blocks = torch.sum(target[f"{key}_mask"], dim=[1, 2])
            if key == 'non_diagonal_hamiltonian':
                row = target.edge_index_full[0]
                batch = target.batch[row]
            else:
                batch = target.batch
            mae_blocks = scatter_sum(mae_blocks, batch)
            count_sum_blocks = scatter_sum(count_sum_blocks, batch)
            error_dict[key + '_mae'] = (mae_blocks / count_sum_blocks).mean()
        else:
            diff = outputs[key] - target[key]
            mae  = torch.mean(torch.abs(diff))
            error_dict[key]  = mae
    return error_dict


def matrix_transform(matrices, atoms, convention='pyscf_def2svp'):
    conv = convention_dict[convention]
    orbitals = ''
    orbitals_order = []
    for a in atoms:
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a]]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append(np.array(map_idx) + offset)
        transform_signs.append(np.array(map_sign))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = np.concatenate(transform_indices).astype(np.int32)
    transform_signs = np.concatenate(transform_signs)

    transform_indices = torch.from_numpy(transform_indices).to(matrices.device).type(torch.long)
    transform_signs = torch.from_numpy(transform_signs).to(matrices.device)

    matrices_new = matrices[...,transform_indices, :]
    matrices_new = matrices_new[...,:, transform_indices]
    matrices_new = matrices_new * transform_signs[:, None]
    matrices_new = matrices_new * transform_signs[None, :]
    return matrices_new


def cal_orbital_and_energies(overlap_matrix, full_hamiltonian):
    eigvals, eigvecs = torch.linalg.eigh(overlap_matrix)
    eps = 1e-8 * torch.ones_like(eigvals)
    eigvals = torch.where(eigvals > 1e-8, eigvals, eps)
    frac_overlap = eigvecs / torch.sqrt(eigvals).unsqueeze(-2)

    Fs = torch.bmm(torch.bmm(frac_overlap.transpose(-1, -2), full_hamiltonian), frac_overlap)
    orbital_energies, orbital_coefficients = torch.linalg.eigh(Fs)
    orbital_coefficients = torch.bmm(frac_overlap, orbital_coefficients)
    return orbital_energies, orbital_coefficients


@torch.no_grad()
def test_over_dataset(test_data_loader, model, device, default_type):
    model.eval()
    total_error_dict = {'total_items': 0}
    loss_weights = {
        'hamiltonian': 1.0,
        'diagonal_hamiltonian': 1.0,
        'non_diagonal_hamiltonian': 1.0,
        'orbital_energies': 1.0,
        "orbital_coefficients": 1.0,
        'HOMO': 1.0, 'LUMO': 1.0, 'GAP': 1.0,
    }
    total_time = 0
    total_graph = 0
    for valid_batch_idx, batch in tqdm(enumerate(test_data_loader)):
        batch = post_processing(batch, default_type)
        batch = batch.to(device)
        tic = time.time()
        outputs = model(batch)
        outputs['hamiltonian'] = model.build_final_matrix(
            batch, outputs['hamiltonian_diagonal_blocks'], outputs['hamiltonian_non_diagonal_blocks'])

        batch.hamiltonian = model.build_final_matrix(
            batch, batch[0].diagonal_hamiltonian, batch[0].non_diagonal_hamiltonian)

        outputs['hamiltonian'] = outputs['hamiltonian'].type(torch.float64)
        outputs['hamiltonian'] = matrix_transform(
            outputs['hamiltonian'], batch.atoms.cpu().squeeze().numpy(), convention='back2pyscf')
        batch.hamiltonian = batch.hamiltonian.type(torch.float64)
        batch.hamiltonian = matrix_transform(
            batch.hamiltonian, batch.atoms.cpu().squeeze().numpy(), convention='back2pyscf')

        duration = time.time() - tic
        total_graph = total_graph + batch.ptr.shape[0] - 1
        total_time = duration + total_time
        mol = pyscf.gto.Mole()
        t = [[batch.atoms[atom_idx].cpu().item(), batch.pos[atom_idx].cpu().numpy()]
             for atom_idx in range(batch.num_nodes)]
        mol.build(verbose=0, atom=t, basis='def2svp', unit='ang')

        overlap = torch.from_numpy(mol.intor("int1e_ovlp")).unsqueeze(0)
        # overlap = matrix_transform(
        #     overlap, batch.atoms.cpu().squeeze().numpy(), convention='pyscf_def2svp')
        overlap = overlap.to(device)

        outputs['orbital_energies'], outputs['orbital_coefficients'] = \
            cal_orbital_and_energies(overlap, outputs['hamiltonian'])
        batch.orbital_energies, batch.orbital_coefficients = \
            cal_orbital_and_energies(overlap, batch['hamiltonian'])

        # here it only considers the occupied orbitals
        # pay attention that the last dimension here corresponds to the different orbitals.
        num_orb = int(batch.atoms[batch.ptr[0]: batch.ptr[1]].sum() / 2)
        pred_HOMO = outputs['orbital_energies'][:, num_orb-1]
        gt_HOMO = batch.orbital_energies[:, num_orb-1]
        pred_LUMO = outputs['orbital_energies'][:, num_orb]
        gt_LUMO = batch.orbital_energies[:, num_orb]
        outputs['HOMO'], outputs['LUMO'], outputs['GAP'] = pred_HOMO, pred_LUMO, pred_LUMO - pred_HOMO
        batch.HOMO, batch.LUMO, batch.GAP = gt_HOMO, gt_LUMO, gt_LUMO - gt_HOMO

        outputs['orbital_energies'], outputs['orbital_coefficients'], \
        batch.orbital_energies, batch.orbital_coefficients = \
            outputs['orbital_energies'][:, :num_orb], outputs['orbital_coefficients'][:, :, :num_orb], \
            batch.orbital_energies[:, :num_orb], batch.orbital_coefficients[:, :, :num_orb]

        # batch.hamiltonian_diagonal_blocks, batch.hamiltonian_non_diagonal_blocks = \
        #     batch.diagonal_hamiltonian, batch.non_diagonal_hamiltonian
        outputs['diagonal_hamiltonian'], outputs['non_diagonal_hamiltonian'] = \
            outputs['hamiltonian_diagonal_blocks'], outputs['hamiltonian_non_diagonal_blocks']
        error_dict = criterion(outputs, batch, loss_weights)

        for key in error_dict.keys():
            if key in total_error_dict.keys():
                total_error_dict[key] += error_dict[key].item() * batch.hamiltonian.shape[0]
            else:
                total_error_dict[key] = error_dict[key].item() * batch.hamiltonian.shape[0]
        total_error_dict['total_items'] += batch.hamiltonian.shape[0]

    for key in total_error_dict.keys():
        if  key != 'total_items':
            total_error_dict[key] = total_error_dict[key] / total_error_dict['total_items']
    return total_error_dict


@hydra.main(config_path='config', config_name='config')
def main(conf):
    default_type = torch.float32
    torch.set_default_dtype(default_type)

    logger.info(conf)
    torch.set_default_dtype(default_type)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    root_path = os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-3]))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{conf.device}")
    else:
        device = torch.device('cpu')

    # load dataset
    logger.info(f"loading {conf.datasets.dataset_name}...")
    if conf.datasets.dataset_name == 'QH9Stable':
        dataset = QH9Stable(os.path.join(root_path, 'datasets'), split=conf.datasets.split)
    elif conf.datasets.dataset_name == 'QH9Dynamic':
        dataset = QH9Dynamic(os.path.join(root_path, 'datasets'), split=conf.datasets.split, version=conf.datasets.version)
    test_dataset = dataset[dataset.test_mask]

    test_data_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=conf.datasets.num_workers,
        pin_memory=conf.datasets.pin_memory)
    print("data loaded.")

    # define model
    model = QHNet(
        in_node_features=1,
        sh_lmax=4,
        hidden_size=128,
        bottle_hidden_size=32,
        num_gnn_layers=5,
        max_radius=15,
        num_nodes=10,
        radius_embed_dim=16
    )

    # load the pretrained model parameters automatically or load your model parameters
    if conf.use_pretrained:
        model = load_pretrained_model_parameters(model, conf.datasets.dataset_name, dataset, conf.pretrained_model_parameter_dir)
    else:
        path_to_the_saved_models = conf.trained_model
        state_dict = torch.load(path_to_the_saved_models)['state_dict']
        model.load_state_dict(state_dict)

    model.set(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"the number of parameters in this model is {num_params}.")
    errors = test_over_dataset(test_data_loader, model, device, default_type)
    msg = f"dataset {conf.datasets.dataset_name} {errors['total_items']}: "
    for key in errors.keys():
        if key != 'total_num':
            msg = msg + f" {key}: {errors[key]:.8f}"
    logger.info(msg)


def post_processing(batch, default_type):
    for key in batch.keys():
        if torch.is_floating_point(batch[key]):
            batch[key] = batch[key].type(default_type)
    return batch


if __name__ == '__main__':
    main()
