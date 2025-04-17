import math
import logging
from tqdm import tqdm
import numpy as np
import copy
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler
import re
import pandas as pd
from rdkit import Chem

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_tokens = 375e2 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e7 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    run_name = None
    num_workers = 0 # for DataLoader
    load_checkpoint_path = None

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, stoi=None, itos=None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.tokens = 0
        self.device = 'cpu'
        self.stoi = stoi
        self.itos = itos
        print('dist:', config.dist)
        if config.dist:
            self.device = config.rank
            self.model = self.model.to(self.device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device])
        elif torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)


    def save_checkpoint(self, epoch, model, best_loss, optimizer, tokens, scaler, save_path):
        raw_model = model.module if hasattr(model, "module") else model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'tokens': tokens,
            'best_loss': best_loss,
        }
        if self.config.dist:
            if self.device == 0:
                torch.save(checkpoint, save_path)
        else:
            torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, load_path, optimizer, scaler):
        checkpoint = torch.load(load_path, map_location='cuda')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.tokens = checkpoint['tokens']
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        return checkpoint['epoch'], checkpoint['best_loss']

    def train(self, wandb):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        scaler = GradScaler()
        if config.load_checkpoint_path is not None:
            print(f'resuming training from {config.load_checkpoint_path}...')
            start_epoch, best_loss = self.load_checkpoint(config.load_checkpoint_path, optimizer, scaler)
        else:
            start_epoch = -1
            best_loss = float('inf')
            self.tokens = 0

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            if self.config.dist:
                sampler = torch.utils.data.distributed.DistributedSampler(data)
                loader = DataLoader(data, shuffle=False, pin_memory=True,
                                    batch_size=config.batch_size,
                                    sampler=sampler)
            else:
                loader = DataLoader(data, shuffle=True, pin_memory=True,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (input_ids, targets, condition_split_id) in pbar:
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                condition_split_id = condition_split_id.to(self.device)

                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(is_train):
                        logits, loss = model(input_ids, targets=targets, condition_split_id=condition_split_id)
                        loss = loss.mean()
                        losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    if config.lr_decay:
                        self.tokens += (targets >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    if (it + epoch*len(loader)) % 500 == 0:
                        print(f"step_train_loss: {loss} train_step: {it + epoch*len(loader)}, learning_rate: {lr}")
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
    
            if is_train:
                return float(np.mean(losses))
            if not is_train:
                test_loss = float(np.mean(losses))
                print("test loss: %f", test_loss)
                return test_loss

        for epoch in range(start_epoch+1, config.max_epochs):
            train_loss = run_epoch('train')
            print(f"epoch_train_loss: {train_loss}, epoch: {epoch + 1}")
            if ((epoch+1) >= self.config.save_start_epoch and (epoch+1) % self.config.save_interval_epoch == 0) or epoch == config.max_epochs - 1:
                ckpt_path = f'../cond_gpt/weights/{self.config.run_name}_ep{epoch+1}.pt'
                print(f'Saving at latest epoch {epoch + 1}: {ckpt_path}')
                if self.config.dist:
                    if self.device == 0:
                        self.save_checkpoint(epoch, model, best_loss, optimizer, self.tokens, scaler, ckpt_path)
                else:
                    self.save_checkpoint(epoch, model, best_loss, optimizer, self.tokens, scaler, ckpt_path)

        return None
