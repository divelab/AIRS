#!/usr/bin/env python3
import os
import time

import torch
import hydra
import logging

from models import get_model
from torchvision.transforms import Compose
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ori_dataset import MD17_DFT, random_split, get_mask
logger = logging.getLogger()


def criterion(outputs, target, names):
    error_dict = {}
    for key in names:
        if key == 'orbital_coefficients':
            error_dict[key] = torch.cosine_similarity(outputs[key], target[key], dim=1).abs().mean()
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
    total_time = 0
    total_graph = 0
    for valid_batch_idx, batch in tqdm(enumerate(test_data_loader)):
        batch = post_processing(batch, default_type)
        batch = batch.to(device)
        tic = time.time()
        outputs = model(batch)
        duration = time.time() - tic
        total_graph = total_graph + batch.ptr.shape[0] - 1
        total_time = duration + total_time
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
            outputs['orbital_energies'][:, :num_orb], outputs['orbital_coefficients'][:, :, :num_orb], \
            batch.orbital_energies[:, :num_orb], batch.orbital_coefficients[:, :, :num_orb]
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

    # root_path = os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-3]))
    root_path = '/data/haiyang/QC_matrix/equiwave'
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{conf.device}")
    else:
        device = torch.device('cpu')

    # load dataset(s)
    logger.info(f"loading {conf.dataset.dataset_name}...")
    dataset = MD17_DFT(
        os.path.join(root_path, 'dataset'),
        name=conf.dataset.dataset_name,
        transform=Compose([get_mask])
    )

    train_dataset, valid_dataset, test_dataset = \
        random_split(dataset,
                     [conf.dataset.num_train, conf.dataset.num_valid,
                      len(dataset)-(conf.dataset.num_train+conf.dataset.num_valid)],
                     seed=conf.split_seed)

    g = torch.Generator()
    g.manual_seed(0)
    test_data_loader = DataLoader(
        valid_dataset, batch_size=64, shuffle=False,
        num_workers=conf.dataset.num_workers, pin_memory=conf.dataset.pin_memory, generator=g)

    # define model
    model = get_model(conf.model)

    # load model from the path
    model_path = conf.model_path
    old_state_dict = torch.load(model_path)['state_dict']
    new_state_dict = model.state_dict()
    for param_name in new_state_dict.keys():
        all_key_match = True
        if param_name in old_state_dict.keys():
            new_state_dict[param_name] = old_state_dict[param_name]
        else:
            all_key_match = False
    msg = "all key matched." if all_key_match is True else "some key is not matched."
    print(msg)
    model.load_state_dict(new_state_dict)

    torch.save({
        "state_dict": model.cpu().state_dict(),
        "eval": torch.load(model_path)['eval'],
        "batch_idx": torch.load(model_path)['batch_idx']
    },
        os.path.join(root_path, f"{conf.dataset.dataset_name}_RLROP.pt")
    )

    model.set(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"the number of parameters in this model is {num_params}.")
    errors = test_over_dataset(test_data_loader, model, device, default_type)
    msg = f"dataset {conf.dataset.dataset_name} {errors['total_items']}: "
    for key in errors.keys():
        if key != 'total_num':
            msg = msg + f"{key}: {errors[key]:.8f}"
    logger.info(msg)


def post_processing(batch, default_type):
    if 'overlap' in batch.keys:
        batch.overlap = batch.overlap.view(
            batch.overlap.shape[0] // batch.overlap.shape[1],
            batch.overlap.shape[1], batch.overlap.shape[1])
    if 'hamiltonian' in batch.keys:
        batch.hamiltonian = batch.hamiltonian.view(
            batch.hamiltonian.shape[0] // batch.hamiltonian.shape[1],
            batch.hamiltonian.shape[1], batch.hamiltonian.shape[1])

    for key in batch.keys:
        if torch.is_floating_point(batch[key]):
            batch[key] = batch[key].type(default_type)
    return batch


if __name__ == '__main__':
    main()
