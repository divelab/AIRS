import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler

import re
import pandas as pd



logger = logging.getLogger(__name__)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, stoi=None, itos=None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
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
            # self.model = self.model.to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        if self.config.dist:
            if self.device == 0:
                torch.save(raw_model.state_dict(), self.config.ckpt_path)
        else:
            torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self, wandb):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        scaler = GradScaler()

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            # self.save_checkpoint()
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
            for it, (input_ids, targets) in pbar:

                # place data on the correct device
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                # forward the model
                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(is_train):
                        logits, loss, _ = model(input_ids, targets=targets)  # prop=prop, scaffold=scaffold)
                        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (targets >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
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

                    # report progress
                    if (it + epoch*len(loader)) % 500 == 0:
                        print(f"step_train_loss: {loss} train_step: {it + epoch*len(loader)}, learning_rate: {lr}")
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
    
            if is_train:
                return float(np.mean(losses))

            if not is_train:
                test_loss = float(np.mean(losses))
                print("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        molecules = []

        for epoch in range(config.max_epochs):

            train_loss = run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            print(f"epoch_valid_loss: {test_loss}, epoch_train_loss: {train_loss}, epoch: {epoch + 1}")

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                print(f'Saving at epoch {epoch + 1}')
                self.save_checkpoint()

            if ((epoch+1) >= self.config.save_start_epoch and (epoch+1) % self.config.save_interval_epoch == 0) or epoch == config.max_epochs - 1:
                print(f'Saving at latest epoch {epoch + 1}')
                last_model = self.model.module if hasattr(self.model, "module") else self.model
                ckpt_path = f'../cond_gpt/weights/{self.config.run_name}_ep{epoch+1}.pt'
                if self.config.dist:
                    if self.device == 0:
                        logger.info("saving %s", ckpt_path)
                        torch.save(last_model.state_dict(), ckpt_path)
                else:
                    logger.info("saving %s", ckpt_path)
                    torch.save(last_model.state_dict(), ckpt_path)

        if self.config.generate:
            df = pd.DataFrame(molecules, columns = ['molecule', 'smiles', 'epoch'])
            return df

        return None
