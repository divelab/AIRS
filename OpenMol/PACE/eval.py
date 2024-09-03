import time
import torch


def evaluate(model, loader, loss_fn, ema, device):
    if ema is not None:
        with ema.average_parameters():
            eval_metrics = run_eval(
                model=model,
                loss_fn=loss_fn,
                loader=loader,
                device=device,
            )
    else:
        eval_metrics = run_eval(
            model=model,
            loss_fn=loss_fn,
            loader=loader,
            device=device,
        )

    return eval_metrics


def run_eval(model, loader, loss_fn, device):
    start_time = time.time()
    model.eval()

    total_loss = 0
    delta_es_list, delta_fs_list = [], []
    delta_es_per_atom_list = []
    for batch in loader:
        batch = batch.to(device)

        output = model(batch, training=False)
        loss = loss_fn(gt_batch=batch, pred=output)

        total_loss += loss.detach().cpu().item()

        delta_es_list.append(batch.y.detach().cpu() - output["energy"].detach().cpu())
        # delta_es_per_atom_list.append(
        #         ((batch.y - output["energy"]) / (batch.ptr[1:] - batch.ptr[:-1])).detach().cpu()
        #     )
        if "force" in output:
            delta_fs_list.append(
                batch.force.detach().cpu() - output["force"].detach().cpu()
            )

    delta_es = torch.cat(delta_es_list, dim=0)
    mae_e = torch.mean(torch.abs(delta_es)).item()
    rmse_e = torch.sqrt(torch.mean(torch.square(delta_es))).item()
    # delta_es_per_atom = torch.cat(delta_es_per_atom_list, dim=0)
    # rmse_e_per_atom = torch.sqrt(torch.mean(torch.square(delta_es_per_atom))).item()
    if len(delta_fs_list) > 0:
        delta_fs = torch.cat(delta_fs_list, dim=0)
        mae_f = torch.mean(torch.abs(delta_fs)).item()
        rmse_f = torch.sqrt(torch.mean(torch.square(delta_fs))).item()
    else:
        mae_f = 0
        rmse_f = 0

    eval_dict = {
        "loss": total_loss / len(loader),
        "mae_e": mae_e,
        "mae_f": mae_f,
        "rmse_e": rmse_e,
        # "rmse_e_per_atom": rmse_e_per_atom,
        "rmse_f": rmse_f,
        "time": time.time() - start_time,
    }

    return eval_dict
