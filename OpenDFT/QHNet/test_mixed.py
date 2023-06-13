#!/usr/bin/env python3
import os
import torch
import hydra
import logging
from tqdm import tqdm
from models import get_model
from torch_geometric.loader import DataLoader
from ori_dataset import Mixed_MD17_DFT, get_mask
logger = logging.getLogger()


def criterion(outputs, target, names):
    error_dict = {}
    for key in names:
        if key == 'orbital_coefficients':
            error_dict[key] = torch.cosine_similarity(outputs[key], target[key]).abs().mean()
        else:
            diff = outputs[key] - target[key]
            mae  = torch.mean(torch.abs(diff))
            error_dict[key]  = mae
    return error_dict


def cal_orbital_and_energies(overlap_matrix, full_hamiltonian):
    eigvals, eigvecs = torch.linalg.eigh(overlap_matrix)
    eps = 1e-8 * torch.ones_like(eigvals)
    eigvals = torch.where(eigvals > 1e-8, eigvals, eps)
    frac_overlap = eigvecs / torch.sqrt(eigvals).unsqueeze(-2)

    Fs = torch.bmm(torch.bmm(frac_overlap.transpose(-1, -2), full_hamiltonian), frac_overlap).to('cpu')
    orbital_energies, orbital_coefficients = torch.linalg.eigh(Fs)
    orbital_coefficients = torch.bmm(frac_overlap, orbital_coefficients)
    return orbital_energies, orbital_coefficients


@torch.no_grad()
def test_over_dataset(test_data_loader, model, device, default_type):
    model.eval()
    total_error_dict = {'total_items': 0}
    loss_weights = {'hamiltonian': 1.0, 'orbital_energies': 1.0, "orbital_coefficients": 1.0}
    for valid_batch_idx, batch in tqdm(enumerate(test_data_loader)):
        batch = post_processing(batch, default_type)
        batch = batch.to(device)
        outputs = model(batch, keep_blocks=True)

        batch.hamiltonian = model.build_final_matrix(batch,
            batch['hamiltonian_diagonal_blocks'], batch['hamiltonian_non_diagonal_blocks'])
        batch.overlap = model.build_final_matrix(batch,
            batch['overlap_diagonal_blocks'], batch['overlap_non_diagonal_blocks'])
        outputs['hamiltonian'] = model.build_final_matrix(batch,
            outputs['hamiltonian_diagonal_blocks'], outputs['hamiltonian_non_diagonal_blocks'])
        outputs['overlap'] = model.build_final_matrix(batch,
            outputs['overlap_diagonal_blocks'], outputs['overlap_non_diagonal_blocks'])
        for key in outputs.keys():
            outputs[key] = outputs[key].to('cpu')
        batch = batch.to('cpu')

        outputs['orbital_energies'], outputs['orbital_coefficients'] = \
            cal_orbital_and_energies(batch['overlap'], outputs['hamiltonian'])
        batch.orbital_energies, batch.orbital_coefficients = \
            cal_orbital_and_energies(batch['overlap'], batch['hamiltonian'])

        # here it only considers the occupied orbitals
        num_orb = int(batch.atoms[batch.ptr[0]: batch.ptr[1]].sum() / 2)
        outputs['orbital_energies'], outputs['orbital_coefficients'], \
        batch.orbital_energies, batch.orbital_coefficients = \
            outputs['orbital_energies'][:, :num_orb], outputs['orbital_coefficients'][:, :num_orb], \
            batch.orbital_energies[:, :num_orb], batch.orbital_coefficients[:, :num_orb]
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
    if conf.data_type == 'float64':
        default_type = torch.float64
    else:
        default_type = torch.float32

    logger.info(conf)
    torch.set_default_dtype(default_type)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    root_path = os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-3]))
    # determine whether GPU is used for training
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{conf.device}")
    else:
        device = torch.device('cpu')

    # load dataset(s)
    logger.info(f"loading {conf.dataset.dataset_name}...")
    dataset = Mixed_MD17_DFT(
        os.path.join('/data/haiyang/QC_matrix/equiwave', 'dataset'),
        name=conf.dataset.dataset_name,
        transform=get_mask
    )
    test_dataset = dataset[dataset.test_mask]
    g = torch.Generator()
    g.manual_seed(0)
    test_data_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        num_workers=conf.dataset.num_workers, pin_memory=conf.dataset.pin_memory, generator=g)

    # define model
    model = get_model(conf.model)
    model.set(device)

    # load model from the path
    model_path = os.path.join(root_path, "outputs", "2023-01-22", "22-12-56", "results.pt")
    model.load_state_dict(torch.load(model_path)['state_dict'])

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"the number of parameters in this model is {num_params}.")
    errors = test_over_dataset(test_data_loader, model, device, default_type)
    msg = f"dataset {conf.dataset.dataset_name} {errors['total_items']}: "
    for key in errors.keys():
        if key != 'total_num':
            msg = msg + f"{key}: {errors[key]:.8f}"
    logger.info(msg)


def post_processing(batch, default_type):
    for key in batch.keys:
        if torch.is_floating_point(batch[key]):
            batch[key] = batch[key].type(default_type)
    return batch


if __name__ == '__main__':
    main()
