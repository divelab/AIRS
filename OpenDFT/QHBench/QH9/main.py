import os
import torch
import hydra
import logging

from models import QHNet
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_sum

from datasets import QH9Stable, QH9Dynamic
from torch_ema import ExponentialMovingAverage
from transformers import get_polynomial_decay_schedule_with_warmup
logger = logging.getLogger()


def criterion(outputs, target, loss_weights):
    error_dict = {}
    keys = loss_weights.keys()
    try:
        for key in keys:
            row = target.edge_index[0]
            edge_batch = target.batch[row]
            diff_diagonal = outputs[f'{key}_diagonal_blocks']-target[f'diagonal_{key}']
            mse_diagonal  = torch.sum(diff_diagonal**2 * target[f"diagonal_{key}_mask"], dim=[1, 2])
            mae_diagonal  = torch.sum(torch.abs(diff_diagonal) * target[f"diagonal_{key}_mask"], dim=[1, 2])
            count_sum_diagonal =  torch.sum(target[f"diagonal_{key}_mask"], dim=[1, 2])
            mse_diagonal = scatter_sum(mse_diagonal, target.batch)
            mae_diagonal = scatter_sum(mae_diagonal, target.batch)
            count_sum_diagonal = scatter_sum(count_sum_diagonal, target.batch)

            diff_non_diagonal = outputs[f'{key}_non_diagonal_blocks']-target[f'non_diagonal_{key}']
            mse_non_diagonal  = torch.sum(diff_non_diagonal**2 * target[f"non_diagonal_{key}_mask"], dim=[1, 2])
            mae_non_diagonal  = torch.sum(torch.abs(diff_non_diagonal) * target[f"non_diagonal_{key}_mask"], dim=[1, 2])
            count_sum_non_diagonal =  torch.sum(target[f"non_diagonal_{key}_mask"], dim=[1, 2])
            mse_non_diagonal = scatter_sum(mse_non_diagonal, edge_batch)
            mae_non_diagonal = scatter_sum(mae_non_diagonal, edge_batch)
            count_sum_non_diagonal = scatter_sum(count_sum_non_diagonal, edge_batch)

            mae = ((mae_diagonal + mae_non_diagonal) / (count_sum_diagonal + count_sum_non_diagonal)).mean()
            mse = ((mse_diagonal + mse_non_diagonal) / (count_sum_diagonal + count_sum_non_diagonal)).mean()

            error_dict[key+'_mae']  = mae
            error_dict[key+'_rmse'] = torch.sqrt(mse)
            error_dict[key + '_diagonal_mae'] = (mae_diagonal / count_sum_diagonal).mean()
            error_dict[key + '_non_diagonal_mae'] = (mae_non_diagonal / count_sum_non_diagonal).mean()
            loss = mse + mae
            error_dict[key] = loss
            if 'loss' in error_dict.keys():
                error_dict['loss'] = error_dict['loss'] + loss_weights[key] * loss
            else:
                error_dict['loss'] = loss_weights[key] * loss
    except Exception as exc:
        raise exc
    return error_dict


def train_one_batch(conf, batch, model, optimizer):
    loss_weights = {'hamiltonian': 1.0}
    outputs = model(batch)
    errors = criterion(outputs, batch, loss_weights=loss_weights)
    optimizer.zero_grad()
    errors['loss'].backward()
    if conf.datasets.use_gradient_clipping:
        torch.nn.utils.clip_grad_norm_(model.parameters(), conf.datasets.clip_norm)
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
        outputs = model(batch)
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
    default_type = torch.float32
    torch.set_default_dtype(default_type)
    logger.info(conf)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # root_path = '/data/meng/QC_features'
    root_path = os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-3]))
    # determine whether GPU is used for training
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{conf.device}")
    else:
        device = torch.device('cpu')

    # load dataset(s)
    logger.info(f"loading {conf.datasets.dataset_name}...")
    if conf.datasets.dataset_name == 'QH9Stable':
        dataset = QH9Stable(os.path.join(root_path, 'datasets'), split=conf.datasets.split)
    elif conf.datasets.dataset_name == 'QH9Dynamic':
        dataset = QH9Dynamic(os.path.join(root_path, 'datasets'), split=conf.datasets.split)

    train_dataset = dataset[dataset.train_mask]
    valid_dataset = dataset[dataset.val_mask]
    test_dataset = dataset[dataset.test_mask]

    g = torch.Generator()
    g.manual_seed(0)
    train_data_loader = DataLoader(
        train_dataset, batch_size=conf.datasets.train_batch_size, shuffle=True,
        num_workers=conf.datasets.num_workers, pin_memory=conf.datasets.pin_memory, generator=g)
    val_data_loader = DataLoader(
        valid_dataset, batch_size=conf.datasets.valid_batch_size, shuffle=False,
        num_workers=conf.datasets.num_workers, pin_memory=conf.datasets.pin_memory)
    test_data_loader = DataLoader(
        test_dataset, batch_size=conf.datasets.test_batch_size, shuffle=False,
        num_workers=conf.datasets.num_workers, pin_memory=conf.datasets.pin_memory)
    train_iterator = iter(train_data_loader)
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

    logger.info(model)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"the number of parameters in this model is {num_params}.")

    #choose optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=conf.datasets.learning_rate,
        betas=(0.99, 0.999),
        amsgrad=False)

    ema = ExponentialMovingAverage(model.parameters(), decay=0.99)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer, num_warmup_steps=conf.datasets.warmup_steps,
        num_training_steps=conf.datasets.total_steps,
        lr_end = conf.datasets.lr_end, power = 1.0, last_epoch = -1)

    model.train()
    epoch = 0
    best_val_result = 1
    final_train_result = 1
    final_test_result = 1
    final_test_diagonal = 1
    final_test_nondiagonal = 1
    for batch_idx in range(conf.datasets.total_steps+10000):
        try:
            batch = next(train_iterator)
            batch = post_processing(batch, default_type)
        except StopIteration:
            epoch += 1
            train_iterator = iter(train_data_loader)
            continue

        batch = batch.to(device)
        errors = train_one_batch(conf, batch, model, optimizer)
        scheduler.step()
        if conf.ema_start_epoch > -1 and epoch > conf.ema_start_epoch:
            ema.update()

        if batch_idx % conf.datasets.train_batch_interval == 0:
            logger.info(f"Train: Epoch {epoch} {batch_idx} hamiltonian: {errors['hamiltonian_mae']:.8f}.")
            logger.info(f"hamiltonian: diagonal/non diagonal :{errors['hamiltonian_diagonal_mae']:.8f}, "
                        f"{errors['hamiltonian_non_diagonal_mae']:.8f}, lr: {optimizer.param_groups[0]['lr']}.")

        if batch_idx % conf.datasets.validation_batch_interval == 0:
            logger.info(f"Evaluating on epoch {epoch}")
            if conf.ema_start_epoch > -1 and epoch > conf.ema_start_epoch:
                logger.info("with ema")
                with ema.average_parameters():
                    val_errors = validation_dataset(val_data_loader, model, device, default_type)
                    if val_errors['hamiltonian_mae'] < best_val_result:
                        best_val_result = val_errors['hamiltonian_mae']
                        test_errors = validation_dataset(test_data_loader, model, device, default_type)
                        final_train_result = errors['hamiltonian_mae']
                        final_test_result = test_errors['hamiltonian_mae']
                        final_test_diagonal = test_errors['hamiltonian_diagonal_mae']
                        final_test_nondiagonal = test_errors['hamiltonian_non_diagonal_mae']
                        torch.save({"state_dict": model.cpu().state_dict(), "eval": errors,
                                    "batch_idx": batch_idx}, "results_best.pt")
                    if batch_idx % 30000 == 0:
                        torch.save({"state_dict": model.cpu().state_dict(), "eval": errors,
                                    "batch_idx": batch_idx}, f"results_{batch_idx}.pt")
                    model = model.to(device)
            else:
                val_errors = validation_dataset(val_data_loader, model, device, default_type)
                if val_errors['hamiltonian_mae'] < best_val_result:
                    best_val_result = val_errors['hamiltonian_mae']
                    test_errors = validation_dataset(test_data_loader, model, device, default_type)
                    final_train_result = errors['hamiltonian_mae']
                    final_test_result = test_errors['hamiltonian_mae']
                    final_test_diagonal = test_errors['hamiltonian_diagonal_mae']
                    final_test_nondiagonal = test_errors['hamiltonian_non_diagonal_mae']
                    torch.save({"state_dict": model.cpu().state_dict(), "eval": errors,
                                "batch_idx": batch_idx}, "results_best.pt")
                if batch_idx % 30000 == 0:
                    torch.save({"state_dict": model.cpu().state_dict(), "eval": errors,
                                "batch_idx": batch_idx}, f"results_{batch_idx}.pt")
                model = model.to(device)

            logger.info(f"Epoch {epoch} batch_idx {batch_idx} with hamiltonian "
                        f"{errors['hamiltonian_mae']:.8f}.")
            logger.info(f"hamiltonian: diagonal/non diagonal :{errors['hamiltonian_diagonal_mae']:.8f}, "
                        f"{errors['hamiltonian_non_diagonal_mae']:.8f}.")
            logger.info("-------------------------")
            logger.info(f"best val hamiltonian so far: {best_val_result:.8f}.")
            logger.info(f"final train hamiltonian so far: {final_train_result:.8f}," f"final test hamiltonian so far: {final_test_result:.8f}," f"final test hamiltonian so far: diagonal/non diagonal :{final_test_diagonal:.8f}, "
                        f"{final_test_nondiagonal:.8f}.")
            logger.info("=========================")


def post_processing(batch, default_type):
    for key in batch.keys:
        if torch.is_tensor(batch[key]) and torch.is_floating_point(batch[key]):
            batch[key] = batch[key].type(default_type)
    return batch


if __name__ == '__main__':
    main()
