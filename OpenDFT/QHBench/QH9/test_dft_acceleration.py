# This scripts needs to run DFT for evaluation, and the process needs a long time
import os
NUM_THREADS = 6
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)
os.environ['MKL_NUM_THREADS'] = str(NUM_THREADS)
os.environ['NUMEXPR_NUM_THREADS'] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)

import sys
import torch
import hydra
import logging
import pyscf
from pyscf import dft
import numpy as np
from tqdm import tqdm
from pip._internal.exceptions import InstallationError
import subprocess

from models import QHNet
from datasets import QH9Stable, QH9Dynamic
from load_pretrained_models import load_pretrained_model_parameters
from torch_geometric.loader import DataLoader
logger = logging.getLogger()
NUM_EXAMPLES = 50


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
            1: [0, 1], 6: [0, 1, 2, 3, 4], 7: [0, 1, 2, 3, 4],
            8: [0, 1, 2, 3, 4], 9: [0, 1, 2, 3, 4]
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
        orbitals += conv.atom_to_orbitals_map[a.item()]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a.item()]]

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

    hamiltonian_new = hamiltonian[..., transform_indices, :]
    hamiltonian_new = hamiltonian_new[..., :, transform_indices]
    hamiltonian_new = hamiltonian_new * transform_signs[:, None]
    hamiltonian_new = hamiltonian_new * transform_signs[None, :]

    return hamiltonian_new


def to_atom(atom_idx):
    atom_name_dict = {1: 'H', 6: 'O', 7: 'N', 8: 'F'}
    ret_atom_name = []
    for atom in atom_idx:
        ret_atom_name.append(atom_name_dict[atom.item() if isinstance(atom, torch.Tensor) else atom])
    return ret_atom_name


def get_total_cycles(envs):
    setattr(envs['mf'], 'total_cycle', envs['cycle'])
    if envs['mf'].gt is not None:
        if np.mean(np.abs(envs['fock'] - envs['mf'].gt)) < envs['mf'].error_level and \
                envs['mf'].achieve_error_flag is False:
            setattr(envs['mf'], 'achieve_error_flag', True)
            setattr(envs['mf'], 'achieve_error_cycle', envs['cycle'])


def build_matrix(mol, dm0=None, error_level=None, Hamiltonian_gt=None):
    scf_eng = dft.RKS(mol)
    scf_eng.gt = Hamiltonian_gt
    scf_eng.total_cycle = None
    scf_eng.achieve_error_cycle = None
    scf_eng.achieve_error_flag = False
    scf_eng.error_level = error_level

    scf_eng.xc = 'b3lyp'
    scf_eng.basis = 'def2svp'
    scf_eng.grids.level = 3
    scf_eng.callback = get_total_cycles
    if dm0 is not None:
        dm0 = dm0.astype('float64')
    scf_eng.kernel(dm0=dm0)
    num_cycle = scf_eng.total_cycle
    if hasattr(scf_eng, 'achieve_error_cycle'):
        achieve_error_cycle = scf_eng.achieve_error_cycle
    else:
        achieve_error_cycle = None
    return num_cycle, achieve_error_cycle


def test_over_dataset_model_pred(test_data_loader, model, device, default_type):
    num_cycles_model_pred = []
    error_level_list = []
    for batch_idx, batch in tqdm(enumerate(test_data_loader)):
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

        hamiltonian_pyscf_gt = model.build_final_matrix(
            batch, batch['diagonal_hamiltonian'], batch['non_diagonal_hamiltonian'])
        hamiltonian_pyscf_gt = hamiltonian_pyscf_gt.cpu()
        hamiltonian_pyscf_gt = matrix_transform(
            hamiltonian_pyscf_gt, batch.atoms.cpu().squeeze().numpy(), convention='back2pyscf')
        error_level = (hamiltonian_pyscf - hamiltonian_pyscf_gt).abs().mean()

        orbital_energies, orbital_coefficients = \
            cal_orbital_and_energies(overlap_pyscf, hamiltonian_pyscf)
        num_orb = int(batch.atoms[batch.ptr[0]: batch.ptr[1]].sum() / 2)
        orbital_coefficients = orbital_coefficients.squeeze()
        dm0 = orbital_coefficients[:, :num_orb].matmul(orbital_coefficients[:, :num_orb].T) * 2
        dm0 = dm0.cpu().numpy()

        build_matrix_w_dm, _ = build_matrix(mol, dm0)
        num_cycles_model_pred.append(build_matrix_w_dm)
        error_level_list.append(error_level)
    return num_cycles_model_pred, error_level_list


def test_over_dataset_DFT(test_data_loader, model, guessing_method_name='minao', error_level_list=None):
    num_cycles_guessing_initalization = []
    achieve_error_cycle_list = []
    for batch_idx, batch in tqdm(enumerate(test_data_loader)):
        error_level = error_level_list[batch_idx].numpy()
        mol = pyscf.gto.Mole()
        t = [[batch.atoms[atom_idx].cpu().item(), batch.pos[atom_idx].cpu().numpy()]
             for atom_idx in range(batch.num_nodes)]
        mol.build(verbose=0, atom=t, basis='def2svp', unit='ang')

        if guessing_method_name.lower() == '1e':
            dm0 = pyscf.scf.hf.init_guess_by_1e(mol)
        elif guessing_method_name.lower() == 'minao':
            dm0 = pyscf.scf.hf.init_guess_by_minao(mol)
        else:
            raise NotImplementedError

        batch = batch.to(model.device)
        hamiltonian = model.build_final_matrix(
            batch, batch['diagonal_hamiltonian'], batch['non_diagonal_hamiltonian'])
        hamiltonian = hamiltonian.cpu()
        hamiltonian_pyscf_gt = matrix_transform(
            hamiltonian, batch.atoms.cpu().squeeze().numpy(), convention='back2pyscf').numpy()

        build_matrix_w_dm, achieve_error_cycle = build_matrix(
            mol, dm0, error_level, Hamiltonian_gt=hamiltonian_pyscf_gt)
        num_cycles_guessing_initalization.append(build_matrix_w_dm)
        achieve_error_cycle_list.append(achieve_error_cycle)

    return num_cycles_guessing_initalization, achieve_error_cycle_list


def get_lowest_achievable_ratio(num_cycles):
    lowest_achievable_ratio_list = [1 / num_cycles[example_idx] for example_idx in range(len(num_cycles))]
    return lowest_achievable_ratio_list


def get_error_level_ratio(error_level_cycles, total_cycles):
    error_level_ratio_list = []
    for idx in range(len(error_level_cycles)):
        error_level_ratio_list.append(error_level_cycles[idx] / total_cycles[idx])
    return error_level_ratio_list


def get_optimization_ratio(num_cycles_model_pred, num_cycles_minao_guessing_initalization):
    DFT_optimization_ratio = [num_cycles_model_pred[idx] / num_cycles_minao_guessing_initalization[idx]
                              for idx in range(len(num_cycles_model_pred))]
    return DFT_optimization_ratio


def get_stable_dataset_split(root_path):
    processed_random = \
        os.path.join(root_path, 'datasets', 'QH9Stable', 'processed', 'processed_QH9Stable_random.pt')
    processed_ood = \
        os.path.join(root_path, 'datasets', 'QH9Stable', 'processed', 'processed_QH9Stable_size_ood.pt')
    split_idx_iid_test_mask = torch.load(processed_random)[2]
    split_idx_ood_test_mask = torch.load(processed_ood)[2]
    # test_data_mask = np.logical_and(split_idx_iid_test_mask, split_idx_ood_test_mask)
    # test_data_indices = np.where(test_data_mask)[0]
    test_data_indices = np.intersect1d(split_idx_iid_test_mask, split_idx_ood_test_mask)
    rng = np.random.default_rng(43)
    test_data_indices_sampled = rng.choice(test_data_indices, size=NUM_EXAMPLES, replace=False)
    return test_data_indices_sampled


def get_dynamic_dataset_split(processed_dir):
    processed_mol = \
        os.path.join(processed_dir, 'processed_QH9Dynamic_mol.pt')
    processed_geometry = \
        os.path.join(processed_dir, 'processed_QH9Dynamic_geometry.pt')
    split_idx_mol_test_mask = torch.load(processed_mol)[2]
    split_idx_geometry_test_mask = torch.load(processed_geometry)[2]
    # test_data_mask = np.logical_and(split_idx_mol_test_mask, split_idx_geometry_test_mask)
    # test_data_indices = np.where(test_data_mask)[0]
    test_data_indices = np.intersect1d(split_idx_mol_test_mask, split_idx_geometry_test_mask)
    rng = np.random.default_rng(43)
    test_data_indices_sampled = rng.choice(test_data_indices, size=NUM_EXAMPLES, replace=False)
    return test_data_indices_sampled


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
        test_data_indices_sampled = get_stable_dataset_split(root_path)
        test_dataset = dataset[test_data_indices_sampled]



    elif conf.datasets.dataset_name == 'QH9Dynamic':
        dataset = QH9Dynamic(os.path.join(root_path, 'datasets'), split=conf.datasets.split, version=conf.datasets.version)
        test_data_indices_sampled = get_dynamic_dataset_split(dataset.processed_dir)
        test_dataset = dataset[test_data_indices_sampled]


    if conf.datasets.dataset_name == 'QH9Dynamic' and conf.datasets.version == '100k':
        required_pyscf_version = '2.3.0'
    else:
        required_pyscf_version = '2.2.1'

    # Check the pyscf version
    if not pyscf.__version__ == required_pyscf_version:
        print(f"We install the corresponding pyscf version automatically to make sure the version is correct")
        try:
            subprocess.Popen([sys.executable, "-m", "pip", "install", f"pyscf=={required_pyscf_version}"]).wait()
        except:
            print("Automatical installation failed. Please install the corresponding package in current environment.")
            raise InstallationError(f"Installation error occurred: pyscf with version {required_pyscf_version}")


    # we need to generate the index mask, it should maintain two
    test_data_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=conf.datasets.num_workers,
        pin_memory=conf.datasets.pin_memory
    )
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
        # load model from the path
        path_to_the_saved_models = conf.trained_model
        state_dict = torch.load(path_to_the_saved_models)['state_dict']
        model.load_state_dict(state_dict)

    model.set(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"the number of parameters in this model is {num_params}.")

    num_cycles_model_pred, error_level_list = \
        test_over_dataset_model_pred(
            test_data_loader, model, device, default_type=torch.float32)

    num_cycles_minao_guessing_initalization, achieve_error_cycle_list_minao = \
        test_over_dataset_DFT(
            test_data_loader, model,
            guessing_method_name='minao', error_level_list=error_level_list)

    num_cycles_1e_guessing_initalization, achieve_error_cycle_list_1e = \
        test_over_dataset_DFT(
            test_data_loader, model,
            guessing_method_name='1e', error_level_list=error_level_list)

    num_cycles_minao_guessing_initalization_lowest_ratio = \
        get_lowest_achievable_ratio(num_cycles_minao_guessing_initalization)
    num_cycles_1e_guessing_initalization_lowest_ratio = \
        get_lowest_achievable_ratio(num_cycles_1e_guessing_initalization)

    optimization_ratio_minao = get_optimization_ratio(
        num_cycles_model_pred, num_cycles_minao_guessing_initalization)
    optimization_ratio_1e = get_optimization_ratio(
        num_cycles_model_pred, num_cycles_1e_guessing_initalization)


    error_level_optimization_ratio_minao = get_optimization_ratio(
        achieve_error_cycle_list_minao, num_cycles_minao_guessing_initalization)
    error_level_optimization_ratio_1e = get_optimization_ratio(
        achieve_error_cycle_list_1e, num_cycles_1e_guessing_initalization)


    logger.info(
        f"num_cycles_minao_guessing_initalization_lowest_ratio is {num_cycles_minao_guessing_initalization_lowest_ratio}.")
    logger.info(
        f"num_cycles_1e_guessing_initalization_lowest_ratio is {num_cycles_1e_guessing_initalization_lowest_ratio}.")

    logger.info(f"optimization_ratio_minao is {optimization_ratio_minao}.")
    logger.info(f"optimization_ratio_1e is {optimization_ratio_1e}.")
    logger.info(f"error_level_optimization_ratio_minao is {error_level_optimization_ratio_minao}.")
    logger.info(f"error_level_optimization_ratio_1e is {error_level_optimization_ratio_1e}.")
    logger.info(f"================================")
    logger.info(
        f"num_cycles_minao_guessing_initalization_lowest_ratio: mean is {np.mean(num_cycles_minao_guessing_initalization_lowest_ratio)}, std is {np.std(num_cycles_minao_guessing_initalization_lowest_ratio)}.")
    logger.info(
        f"num_cycles_1e_guessing_initalization_lowest_ratio: mean is {np.mean(num_cycles_1e_guessing_initalization_lowest_ratio)}, std is {np.std(num_cycles_1e_guessing_initalization_lowest_ratio)}.")
    logger.info(
        f"optimization_ratio_minao: mean is {np.mean(optimization_ratio_minao)}, std is {np.std(optimization_ratio_minao)}.")
    logger.info(
        f"optimization_ratio_1e: mean is {np.mean(optimization_ratio_1e)}, std is {np.std(optimization_ratio_1e)}.")
    logger.info(
        f"error_level_optimization_ratio_minao: mean is {np.mean(error_level_optimization_ratio_minao)}, std is {np.std(error_level_optimization_ratio_minao)}.")
    logger.info(
        f"error_level_optimization_ratio_1e: mean is {np.mean(error_level_optimization_ratio_1e)}, std is {np.std(error_level_optimization_ratio_1e)}.")


def post_processing(batch, default_type):
    for key in batch.keys:
        if torch.is_floating_point(batch[key]):
            batch[key] = batch[key].type(default_type)
    return batch


if __name__ == '__main__':
    main()
