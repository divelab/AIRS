#!/usr/bin/env python3
import os
import torch
import hydra
import logging
from models import get_model
from torch_geometric.loader import DataLoader
from ori_dataset import Mixed_MD17_DFT, get_mask
from torch_ema import ExponentialMovingAverage
from transformers import get_polynomial_decay_schedule_with_warmup
logger = logging.getLogger()
torch.multiprocessing.set_sharing_strategy('file_system')


def criterion(outputs, target, loss_weights):
    error_dict = {}
    keys = loss_weights.keys()
    # the diagonal and non-diagonal should be considered with the mask
    try:
        for key in keys:
            diff_diagonal = outputs[f'{key}_diagonal_blocks']-target[f'{key}_diagonal_blocks']
            mse_diagonal  = torch.sum(diff_diagonal**2 * target[f"{key}_diagonal_block_masks"])
            mae_diagonal  = torch.sum(torch.abs(diff_diagonal) * target[f"{key}_diagonal_block_masks"])
            count_sum_diagonal =  torch.sum(target[f"{key}_diagonal_block_masks"])

            diff_non_diagonal = outputs[f'{key}_non_diagonal_blocks']-target[f'{key}_non_diagonal_blocks']
            mse_non_diagonal  = torch.sum(diff_non_diagonal**2 * target[f"{key}_non_diagonal_block_masks"])
            mae_non_diagonal  = torch.sum(torch.abs(diff_non_diagonal) * target[f"{key}_non_diagonal_block_masks"])
            count_sum_non_diagonal =  torch.sum(target[f"{key}_non_diagonal_block_masks"])

            mae = (mae_diagonal / count_sum_diagonal + mae_non_diagonal / count_sum_non_diagonal)
            mse = (mse_diagonal / count_sum_diagonal + mse_non_diagonal / count_sum_non_diagonal)

            error_dict[key+'_mae']  = mae
            error_dict[key+'_rmse'] = torch.sqrt(mse)
            error_dict[key + '_diagonal_mae'] = mae_diagonal / count_sum_diagonal
            error_dict[key + '_non_diagonal_mae'] = mae_non_diagonal / count_sum_non_diagonal
            loss = mse + mae
            error_dict[key] = loss
            if 'loss' in error_dict.keys():
                error_dict['loss'] = error_dict['loss'] + loss_weights[key] * loss
            else:
                error_dict['loss'] = loss_weights[key] * loss
    except:
        import pdb; pdb.set_trace()
    return error_dict


def train_one_batch(conf, batch, model, optimizer):
    loss_weights = {'hamiltonian': 1.0}
    outputs = model(batch, keep_blocks=True)
    errors = criterion(outputs, batch, loss_weights=loss_weights)
    optimizer.zero_grad()
    errors['loss'].backward()
    if conf.dataset.use_gradient_clipping:
        torch.nn.utils.clip_grad_norm_(model.parameters(), conf.dataset.clip_norm)
    optimizer.step()
    return errors


@torch.no_grad()
def validation_dataset(valid_data_loader, model, device, default_type):
    model.eval()
    total_error_dict = {'total_items': 0}
    loss_weights = {'hamiltonian': 1.0}
    for valid_batch_idx, batch in enumerate(valid_data_loader):
        batch = post_processing(batch, default_type)
        batch = batch.to(device)
        outputs = model(batch, keep_blocks=True)
        error_dict = criterion(outputs, batch, loss_weights)

        for key in error_dict.keys():
            if key not in ['total_items', 'loss']:
                if key in total_error_dict.keys():
                    total_error_dict[key] += error_dict[key].item() * (batch.ptr.shape[0] - 1)
                else:
                    total_error_dict[key] = error_dict[key].item() * (batch.ptr.shape[0] - 1)
        total_error_dict['total_items'] += (batch.ptr.shape[0] - 1)
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

    torch.set_default_dtype(default_type)
    logger.info(conf)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    root_path = os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-3]))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{conf.device}")
    else:
        device = torch.device('cpu')

    assert conf.dataset.dataset_name == 'all', "Please set dataset name to all."
    logger.info(f"loading {conf.dataset.dataset_name}...")
    dataset = Mixed_MD17_DFT(
        os.path.join(root_path, 'dataset'),
        name='all',
        transform=get_mask
    )

    train_dataset = dataset[dataset.train_mask]
    valid_dataset = dataset[dataset.val_mask]
    test_dataset = dataset[dataset.test_mask]

    g = torch.Generator()
    g.manual_seed(0)
    train_data_loader = DataLoader(
        train_dataset, batch_size=conf.dataset.train_batch_size, shuffle=True,
        num_workers=conf.dataset.num_workers, pin_memory=conf.dataset.pin_memory, generator=g)
    val_data_loader = DataLoader(
        valid_dataset, batch_size=conf.dataset.train_batch_size, shuffle=False,
        num_workers=conf.dataset.num_workers, pin_memory=conf.dataset.pin_memory)
    train_iterator = iter(train_data_loader)

    # define model
    model = get_model(conf.model)
    model.set(device)

    logger.info(model)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"the number of parameters in this model is {num_params}.")

    #choose optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=conf.dataset.learning_rate,
        betas=(0.99, 0.999),
        amsgrad=False)

    ema = ExponentialMovingAverage(model.parameters(), decay=0.99)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer, num_warmup_steps=conf.warmup_step, num_training_steps=conf.num_training_steps,
        lr_end = conf.end_lr, power = 1.0, last_epoch = -1)

    model.train()
    epoch = 0
    best_eval_result = 1
    for batch_idx in range(conf.num_training_steps + 1000):
        try:
            batch = next(train_iterator)
            batch = post_processing(batch, default_type)
        except StopIteration:
            epoch += 1
            train_iterator = iter(train_data_loader)
            batch = next(train_iterator)
            batch = post_processing(batch, default_type)

        batch = batch.to(device)
        errors = train_one_batch(conf, batch, model, optimizer)
        scheduler.step()
        if conf.ema_start_epoch > -1 and epoch > conf.ema_start_epoch:
            ema.update()

        if batch_idx % conf.dataset.train_batch_interval == 0:
            logger.info(f"Train: Epoch {epoch} {batch_idx} hamiltonian: {errors['hamiltonian_mae']:.8f}")
            logger.info(f"hamiltonian: diagonal/non diagonal :{errors['hamiltonian_diagonal_mae']:.8f}, "
                        f"{errors['hamiltonian_non_diagonal_mae']:.8f}")

        if batch_idx % conf.dataset.validation_batch_interval == 0:
            logger.info(f"Evaluating on epoch {epoch}")
            if conf.ema_start_epoch > -1 and epoch > conf.ema_start_epoch:
                with ema.average_parameters():
                    errors = validation_dataset(val_data_loader, model, device, default_type)
                    if errors['hamiltonian_mae'] < best_eval_result:
                        best_eval_result = errors['hamiltonian_mae']
                        torch.save({"state_dict": model.cpu().state_dict(), "eval": errors,
                                    "batch_idx": batch_idx}, "results.pt")
                    if batch_idx in [26000, 52000] or batch_idx % 30000 == 0:
                        torch.save({"state_dict": model.cpu().state_dict(), "eval": errors,
                                    "batch_idx": batch_idx}, f"results_{batch_idx}.pt")
                    model = model.to(device)
            else:
                errors = validation_dataset(val_data_loader, model, device, default_type)

            logger.info(f"Epoch {epoch} batch_idx {batch_idx} with hamiltonian "
                        f"{errors['hamiltonian_mae']:.8f}.")
            logger.info(f"hamiltonian: diagonal/non diagonal :{errors['hamiltonian_diagonal_mae']:.8f}, "
                        f"{errors['hamiltonian_non_diagonal_mae']:.8f}")


def post_processing(batch, default_type):
    for key in batch.keys:
        if torch.is_floating_point(batch[key]):
            batch[key] = batch[key].type(default_type)
    return batch


if __name__ == '__main__':
    main()
