import os
import time

import torch
import hydra
import logging
import pyscf
from pyscf import dft
import numpy as np
from tqdm import tqdm

from models import QHNet
from datasets import QH9Stable, QH9Dynamic
from torchvision.transforms import Compose
from torch_geometric.loader import DataLoader

logger = logging.getLogger()


def cal_orbital_and_energies(overlap_matrix, full_hamiltonian):
    eigvals, eigvecs = torch.linalg.eigh(overlap_matrix)
    eps = 1e-8 * torch.ones_like(eigvals)
    eigvals = torch.where(eigvals > 1e-8, eigvals, eps)
    frac_overlap = eigvecs / torch.sqrt(eigvals).unsqueeze(-2)

    Fs = torch.bmm(torch.bmm(frac_overlap.transpose(-1, -2), full_hamiltonian), frac_overlap)
    orbital_energies, orbital_coefficients = torch.linalg.eigh(Fs)
    orbital_coefficients = torch.bmm(frac_overlap, orbital_coefficients)
    return orbital_energies, orbital_coefficients



from argparse import Namespace
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
        },
    ),
}

def matrix_transform(hamiltonian, atoms, convention='pyscf_def2svp'):
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
    transform_indices = np.concatenate(transform_indices).astype(np.int)
    transform_signs = np.concatenate(transform_signs)

    hamiltonian_new = hamiltonian[...,transform_indices, :]
    hamiltonian_new = hamiltonian_new[...,:, transform_indices]
    hamiltonian_new = hamiltonian_new * transform_signs[:, None]
    hamiltonian_new = hamiltonian_new * transform_signs[None, :]

    return hamiltonian_new


def to_atom(atom_idx):
    atom_name_dict = {1: 'H', 6: 'O', 7: 'N', 8: 'F'}
    ret_atom_name = []
    for atom in atom_idx:
        ret_atom_name.append(atom_name_dict[atom.item() if isinstance(atom, torch.Tensor) else atom])
    return ret_atom_name


def build_matrix(mol, dm0=None, batch=None):
    scf_eng = dft.RKS(mol)
    scf_eng.xc = 'b3lyp'
    scf_eng.basis = 'def2svp'
    scf_eng.grids.level = 3
    if dm0 is not None:
        dm0 = dm0.astype('float64')
    scf_eng.kernel(dm0 = dm0)
    num_cycle = scf_eng.total_cycle
    return num_cycle


def test_over_dataset(test_data_loader, model, num_examples, device, default_type):
    total_time = 0
    total_ratio = 0
    count = 0
    for batch_idx, batch in tqdm(enumerate(test_data_loader)):
        if batch_idx >= num_examples:
            break
        
        mol = pyscf.gto.Mole()
        t = [[batch.atoms[atom_idx].cpu().item(), batch.pos[atom_idx].cpu().numpy()]
             for atom_idx in range(batch.num_nodes)]
        mol.build(verbose=0, atom=t, basis='def2svp', unit='ang')
        overlap_pyscf = torch.from_numpy(mol.intor("int1e_ovlp")).unsqueeze(0)
        batch.pos = batch.pos
        with torch.no_grad():
            batch = post_processing(batch, default_type)
            batch = batch.to(device)
            outputs = model(batch)
        hamiltonian = model.build_final_matrix(
            batch, outputs['hamiltonian_diagonal_blocks'], outputs['hamiltonian_non_diagonal_blocks'])
        hamiltonian = hamiltonian.type(torch.double).cpu()
        hamiltonian_pyscf = matrix_transform(
            hamiltonian, batch.atoms.cpu().squeeze().numpy(), convention='back2pyscf')

        orbital_energies, orbital_coefficients = \
            cal_orbital_and_energies(overlap_pyscf, hamiltonian_pyscf)
        num_orb = int(batch.atoms[batch.ptr[0]: batch.ptr[1]].sum() / 2)
        orbital_coefficients = orbital_coefficients.squeeze()
        dm0 = orbital_coefficients[:, :num_orb].matmul(orbital_coefficients[:, :num_orb].T) * 2
        dm0 = dm0.cpu().numpy()

        build_matrix_w_dm = build_matrix(mol, dm0, batch)
        tic = time.time()
        build_matrix_wo_dm = build_matrix(mol, None, batch)

        ratio = build_matrix_w_dm / build_matrix_wo_dm
        duration = time.time() - tic
        total_time = duration + total_time
        total_ratio = total_ratio + ratio
        count = count + 1
    total_time = total_time / count
    total_ratio = total_ratio / count
    return total_time, total_ratio


@hydra.main(config_path='config', config_name='config')
def main(conf):
    logger.info(conf)
    torch.set_default_dtype(torch.float32)
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
        dataset = QH9Dynamic(os.path.join(root_path, 'datasets'), split=conf.datasets.split)
    test_dataset = dataset[dataset.test_mask]
    
    test_data_loader = DataLoader(
        test_dataset[:50], batch_size=1, shuffle=False,
        num_workers=conf.datasets.num_workers, pin_memory=conf.datasets.pin_memory)

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

    model.set(device)

    # load model from the path
    path_to_the_saved_models = conf.trained_model
    state_dict = torch.load(path_to_the_saved_models)['state_dict']
    model.load_state_dict(state_dict)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"the number of parameters in this model is {num_params}.")

    total_time, total_ratio = test_over_dataset(test_data_loader, model, 50, device, default_type=torch.float32)
    msg = f"total_time: {total_time}, total_ratio:{total_ratio} "
    logger.info(msg)


def post_processing(batch, default_type):
    for key in batch.keys:
        if torch.is_floating_point(batch[key]):
            batch[key] = batch[key].type(default_type)
    return batch


if __name__ == '__main__':
    main()
