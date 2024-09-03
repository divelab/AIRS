import time
from tqdm import tqdm


def train(model, loader, loss_fn, optimizer, ema, device):
    model.train()

    total_loss = 0
    # for batch in tqdm(loader):
    for batch in loader:
        start_time = time.time()

        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        output = model(batch, training=True)
        loss = loss_fn(gt_batch=batch, pred=output)
        loss.backward()
        optimizer.step()

        if ema is not None:
            ema.update()

        total_loss += loss.detach().cpu().item()

    train_dict = {
        "loss": total_loss / len(loader),
        "time": time.time() - start_time,
    }

    return train_dict
