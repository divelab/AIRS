"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler

from utils import check_novelty, sample
from utils import get_mol
import re
import pandas as pd
from rdkit import Chem
import os
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e2 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e7 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    run_name = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, stoi=None, itos=None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        if self.config.resume:
            ckpt = torch.load(self.config.resume_ckpt_path)
            print(f"Resuming from epoch {ckpt['epoch']}")
            self.model.load_state_dict(ckpt['model_state_dict'])

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

    def save_checkpoint(self, epoch, best=False):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if best:
            logger.info("saving %s", self.config.best_ckpt_path)
            ckpt_path = self.config.best_ckpt_path
        else:    
            logger.info("saving %s", self.config.ckpt_path)
            ckpt_path = self.config.ckpt_path
        checkpoint = {'epoch': epoch, 'model_state_dict': raw_model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
                'tokens': self.tokens, 'scaler_state_dict': self.scaler.state_dict()}
        if self.config.dist:
            if self.device == 0:
                torch.save(checkpoint, ckpt_path)
        else:
            torch.save(checkpoint, ckpt_path)

    def train(self, wandb):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        self.optimizer = raw_model.configure_optimizers(config)
        self.scaler = GradScaler()

        # [' ', '#', '(', ')', '-', '1', '2', '3', '4', '5', '6', '<', '=', 'B', 'C', 'F', 'H', 'N', 'O', 'S', '[', ']', 'c', 'l', 'n', 'o', 'r', 's']
        # ['#', '(', ')', '-', '1', '2', '3', '4', '5', '6', '<', '=', 'Br', 'C', 'Cl', 'F', 'N', 'O', 'S', '[H]', '[nH]', 'c', 'n', 'o', 's']

        if self.config.resume:
            ckpt = torch.load(self.config.resume_ckpt_path)
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
            self.tokens = ckpt['tokens']
            self.resume_epoch = ckpt['epoch']


        def run_epoch(split, epoch):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            self.save_checkpoint(epoch)
            if self.config.dist:
                sampler = torch.utils.data.distributed.DistributedSampler(data)
                loader = DataLoader(data, shuffle=False, pin_memory=True,
                                    batch_size=config.batch_size,
                                    sampler=sampler)
            else:
                loader = DataLoader(data, shuffle=True, pin_memory=True,
                                    batch_size=config.batch_size,
                                    # batch_size=2,
                                    # num_workers=0)
                                    num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            # import pdb; pdb.set_trace()
            for it, (input_ids, targets, condition_split_id, protein_padded_embedding, protein_embedding_mask) in pbar:
            # for it, (input_ids, targets, condition_split_id) in pbar:
                # import pdb; pdb.set_trace()
                # place data on the correct device
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                condition_split_id = condition_split_id.to(self.device)
                
                if protein_padded_embedding is not None:
                    if torch.isnan(protein_padded_embedding).any():
                        raise ValueError("protein_padded_embedding has nan")
                    protein_padded_embedding = protein_padded_embedding.to(self.device)
                if protein_embedding_mask is not None:
                    if torch.isnan(protein_embedding_mask).any():
                        raise ValueError("protein_embedding_mask has nan")
                    protein_embedding_mask = protein_embedding_mask.to(self.device)
                
                # protein_padded_embedding = None
                # protein_embedding_mask = None
                

                # forward the model
                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(is_train):
                        logits, loss, _ = model(input_ids, 
                                                targets=targets, 
                                                condition_split_id=condition_split_id,
                                                protein_padded_embedding=protein_padded_embedding,
                                                protein_embedding_mask=protein_embedding_mask)
                        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

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
                        for param_group in self.optimizer.param_groups:
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

        resume_epoch = 0
        if self.config.resume:
            resume_epoch = self.resume_epoch
        for epoch in range(resume_epoch, config.max_epochs):

            train_loss = run_epoch('train', epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('test', epoch)

            print(f"epoch_valid_loss: {test_loss}, epoch_train_loss: {train_loss}, epoch: {epoch + 1}")
            
            if self.config.tensorboard_writer is not None:
                if self.config.dist:
                    if self.device == 0:
                        self.config.tensorboard_writer.add_scalar('train_loss', train_loss, epoch)
                        self.config.tensorboard_writer.add_scalar('test_loss', test_loss, epoch)
                else:
                    self.config.tensorboard_writer.add_scalar('train_loss', train_loss, epoch)
                    self.config.tensorboard_writer.add_scalar('test_loss', test_loss, epoch)
                              
            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                print(f'Saving at epoch {epoch + 1}')
                self.save_checkpoint(epoch, best=True)

            if ((epoch+1) >= self.config.save_start_epoch and (epoch+1) % self.config.save_interval_epoch == 0) or epoch == config.max_epochs - 1:
                print(f'Saving at latest epoch {epoch + 1}')
                last_model = self.model.module if hasattr(self.model, "module") else self.model
                # ckpt_path = f'./cond_gpt/weights/{self.config.run_name}_ep{epoch+1}.pt'
                ckpt_path = os.path.join(self.config.ckpt_folder, f"{self.config.run_name}_ep{epoch+1}.pt")
                checkpoint = {'epoch': epoch, 'model_state_dict': last_model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
                'tokens': self.tokens,  'scaler_state_dict': self.scaler.state_dict()}
                
                
                if self.config.dist:
                    if self.device == 0:
                        logger.info("saving %s", ckpt_path)
                        # torch.save(last_model.state_dict(), ckpt_path)
                        torch.save(checkpoint, ckpt_path)
                else:
                    logger.info("saving %s", ckpt_path)
                    # torch.save(last_model.state_dict(), ckpt_path)
                    torch.save(checkpoint, ckpt_path)


            if self.config.generate:
                pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
                regex = re.compile(pattern)
                context = "C"
                for i in range(2):
                    x = torch.tensor([self.stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(512, 1).to('cuda')
                    p = None
                    sca = None
                    y = sample(model, x, self.config.block_size, temperature=0.8, sample=True, top_k=10, prop = p, scaffold = sca)
                    for gen_mol in y:
                        completion = ''.join([self.itos[int(i)] for i in gen_mol])
                        completion = completion.replace('<', '')
                        mol = get_mol(completion)
                        if mol:
                            smiles = Chem.MolToSmiles(mol)
                            molecules.append((mol, smiles, epoch))

        if self.config.generate:
            df = pd.DataFrame(molecules, columns = ['molecule', 'smiles', 'epoch'])
            return df

        return None
