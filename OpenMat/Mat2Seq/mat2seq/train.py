"""
Adapted from:
https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/train.py
"""
import sys
sys.path.append(".")
import os
from dataclasses import dataclass
from typing import Union
import math
import time
from tqdm import tqdm

from crystallm import parse_config
from omegaconf import OmegaConf
import numpy as np
import torch
import pickle
from contextlib import nullcontext
from torch.utils.data.dataloader import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from mat2seq import (
    GPT,
    GPTConfig,
    CinDataset
)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def run_DDP(rank, world_size, args):
    setup(rank, world_size)
    run(args, rank)
    cleanup()


@dataclass
class TrainDefaults:
    out_dir: str = "out"  # the path to the folder where the model checkpoints will be stored
    eval_interval: int = 250  # how often to evaluate against the validation set
    log_interval: int = 1  # how often to print to
    eval_iters_train: int = 200
    eval_iters_val: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = False  # if True, always save a checkpoint after each eval
    init_from: str = "scratch"  # 'scratch' or 'resume'

    # data
    dataset: str = ""  # the path to the folder containing the .bin files with encoded tokens
    gradient_accumulation_steps: int = 1  # abandoned, used to simulate larger batch sizes
    batch_size: int = 32  # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size: int = 768  # context of up to `block_size` previous characters
    num_workers: int = 0

    # model
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 768
    dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False  # do we use bias inside LayerNorm and Linear layers?

    # AdamW optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 600000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95  # make a bit bigger because number of tokens per iter is small
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for; not super necessary potentially
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # system
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "bfloat16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster
    underrep_p: float = 0.0
    validate: bool = False  # whether to evaluate the model using the validation set
    dist: bool = False  # parallelize across multiple GPUs


def run(C, rank=None):
    print("Using configuration:")
    print(OmegaConf.to_yaml(C))

    if C.dist:
        print(f"Distributed training...")

    print(f"Creating {C.out_dir}...")
    os.makedirs(C.out_dir, exist_ok=True)

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in C.device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if not C.dataset:
        raise Exception("The 'dataset' option is required and cannot be empty")

    # train_data = np.memmap(os.path.join(C.dataset, "train.bin"), dtype=np.uint16, mode="r")
    with open(os.path.join(C.dataset, "train.pkl"), "rb") as f_train:
        train_data = pickle.load(f_train)
    # val_data = np.memmap(os.path.join(C.dataset, "val.bin"), dtype=np.uint16, mode="r") if C.validate else None
    with open(os.path.join(C.dataset, "val.pkl"), "rb") as f_val:
        val_data = pickle.load(f_val)
    train_dataset = CinDataset(train_data.astype(np.int64))
    valid_dataset = CinDataset(val_data.astype(np.int64))

    iter_num = 0
    best_val_loss = 1e9

    meta_path = os.path.join(C.dataset, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"Found vocab_size = {meta_vocab_size} (inside {meta_path})")

    model_args = dict(n_layer=C.n_layer, n_head=C.n_head, n_embd=C.n_embd, block_size=C.block_size,
                      bias=C.bias, vocab_size=None, dropout=C.dropout)
    if C.init_from == "scratch":
        print("Initializing a new model from scratch...")
        if meta_vocab_size is None:
            print("Defaulting to vocab_size of 371...")
        model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 371
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif C.init_from == "resume":
        print(f"Resuming training from {C.out_dir}...")
        ckpt_path = os.path.join(C.out_dir, "690000_ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=C.device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training;
        #  the rest of the attributes (e.g. dropout) can stay as desired
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    # crop down the model block size if desired, using model surgery
    if C.block_size < model.config.block_size:
        model.crop_block_size(C.block_size)
        model_args["block_size"] = C.block_size  # so that the checkpoint will have the right value
    # model.to(C.device)
    if C.dist:
        C.device = f'cuda:{rank}'
        model = model.to(C.device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[C.device])
    elif torch.cuda.is_available():
        C.device = f'cuda:0'  #torch.cuda.current_device()
        model = torch.nn.DataParallel(model).to(C.device)

    # initialize a GradScaler; if enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(C.dtype == "float16"))
    raw_model = model.module if hasattr(model, "module") else model
    optimizer = raw_model.configure_optimizers(C.weight_decay, C.learning_rate, (C.beta1, C.beta2))
    # optimizer = model.configure_optimizers(C.weight_decay, C.learning_rate, (C.beta1, C.beta2))
    if C.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])

    if C.compile:
        print("Compiling the model (takes a ~minute)...")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        losses = []
        if C.dist:
            sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            loader = DataLoader(train_dataset, shuffle=False, pin_memory=True,
                                batch_size=C.batch_size,
                                sampler=sampler)
        else:
            loader = DataLoader(train_dataset, shuffle=True, pin_memory=True,
                                batch_size=C.batch_size,
                                num_workers=C.num_workers)
        pbar = enumerate(loader)
        for it, (input_ids, targets) in pbar:
            input_ids = input_ids.to(C.device)
            targets = targets.to(C.device)
            logits, loss = model(input_ids, targets)
            loss = loss.mean()
            losses.append(loss.item())
            out["train"] = float(np.mean(losses))

        losses = []
        if C.dist:
            sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
            loader = DataLoader(valid_dataset, shuffle=False, pin_memory=True,
                                batch_size=C.batch_size,
                                sampler=sampler)
        else:
            loader = DataLoader(valid_dataset, shuffle=True, pin_memory=True,
                                batch_size=C.batch_size,
                                num_workers=C.num_workers)
        pbar = enumerate(loader)
        for it, (input_ids, targets) in pbar:
            input_ids = input_ids.to(C.device)
            targets = targets.to(C.device)
            logits, loss = model(input_ids, targets)
            loss = loss.mean()
            losses.append(loss.item())
            out["val"] = float(np.mean(losses))
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < C.warmup_iters:
            return C.learning_rate * it / C.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > C.lr_decay_iters:
            return C.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - C.warmup_iters) / (C.lr_decay_iters - C.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return C.min_lr + coeff * (C.learning_rate - C.min_lr)

    # training loop
    # X, Y = get_batch("train")

    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    running_mfu = -1.0
    epoch = int(-1)
    if C.dist:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        loader = DataLoader(train_dataset, shuffle=False, pin_memory=True,
                            batch_size=C.batch_size,
                            sampler=sampler)
    else:
        loader = DataLoader(train_dataset, shuffle=True, pin_memory=True,
                            batch_size=C.batch_size,
                            num_workers=C.num_workers)
    while True:
        epoch += 1
        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        pbar = tqdm(enumerate(loader), total=len(loader))
        print('epoch:', epoch)
        if C.dist:
            sampler.set_epoch(epoch)
        for it, (input_ids, targets) in pbar:
            input_ids = input_ids.to(C.device)
            targets = targets.to(C.device)
            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num) if C.decay_lr else C.learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # for micro_step in range(C.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(input_ids, targets)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
            if (it + 1) % C.gradient_accumulation_steps == 0:
                # clip the gradient
                if C.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), C.grad_clip)
                # step the optimizer and scaler if training in fp16
                scaler.step(optimizer)
                scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)
                iter_num += 1
                local_iter_num += 1


            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % C.log_interval == 0:
                lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
                if local_iter_num >= 5:  # let the training loop settle a bit
                    raw_model = model.module if hasattr(model, "module") else model
                    mfu = raw_model.estimate_mfu(C.batch_size, dt)  # C.batch_size * C.gradient_accumulation_steps
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")


            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % C.eval_interval == 0:
                if C.validate and ((iter_num+1) % (10000*C.eval_interval) == 0):
                    losses = estimate_loss()
                    print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                else:
                    losses = {"train": 10.0, "val": 10.0}
                if (C.validate and losses["val"] < best_val_loss) or C.always_save_checkpoint:
                    best_val_loss = losses["val"]  # if C.validate else 0.
                    if iter_num > 0:
                        if C.dist:
                            if C.device == 'cuda:0':
                                raw_model = model.module if hasattr(model, "module") else model
                                checkpoint = {
                                    "model": raw_model.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "model_args": model_args,
                                    "iter_num": iter_num,
                                    "best_val_loss": best_val_loss,
                                    "config": dict(C),
                                }
                                print("saving checkpoint with validation loss %f" % losses["val"])
                                os.makedirs(C.out_dir, exist_ok=True)
                                torch.save(checkpoint, os.path.join(C.out_dir, f"{iter_num}_ckpt.pt"))
                        else:
                            raw_model = model.module if hasattr(model, "module") else model
                            checkpoint = {
                                "model": raw_model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "model_args": model_args,
                                "iter_num": iter_num,
                                "best_val_loss": best_val_loss,
                                "config": dict(C),
                            }
                            print("saving checkpoint with validation loss %f" % losses["val"])
                            os.makedirs(C.out_dir, exist_ok=True)
                            torch.save(checkpoint, os.path.join(C.out_dir, f"{iter_num}_ckpt.pt"))

        if iter_num == 0 and C.eval_only:
            break

        # termination conditions
        if iter_num > C.max_iters:
            if C.dist:
                if C.device == 'cuda:0':
                    raw_model = model.module if hasattr(model, "module") else model
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": dict(C),
                    }
                    print(f"saving checkpoint to {C.out_dir}...")
                    os.makedirs(C.out_dir, exist_ok=True)
                    torch.save(checkpoint, os.path.join(C.out_dir, "ckpt_final.pt"))
            else:
                raw_model = model.module if hasattr(model, "module") else model
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": dict(C),
                }
                print(f"saving checkpoint to {C.out_dir}...")
                os.makedirs(C.out_dir, exist_ok=True)
                torch.save(checkpoint, os.path.join(C.out_dir, "ckpt_final.pt"))

            break


if __name__ == "__main__":
    C = parse_config(TrainDefaults)

    if C.dist:
        world_size = torch.cuda.device_count()
        print(f"world_size: {world_size}")
        mp.spawn(run_DDP,
                 args=(world_size, C),
                 nprocs=world_size,
                 join=True)
    else:
        run(C)

