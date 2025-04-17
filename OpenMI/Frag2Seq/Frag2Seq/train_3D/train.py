import pandas as pd
import argparse
from utils import set_seed
import numpy as np
import wandb
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import Mol3DDataset, SimpleTokenizer, SubChTokenizer, NewMol3DDataset
import math
import re
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

import time
from torch.utils.tensorboard import SummaryWriter

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def load_dataset_from_files(root_path, split, ids):
    dataset = []
    for id in range(ids):
        with open(root_path+'_'+split+'_'+str(id)+'.txt', 'r') as file:
            dataset.extend(file.readlines())
            print('loaded dataset from '+root_path+'_'+split+'_'+str(id)+'.txt')
    return dataset

def load_tokenizer(tokenizer_path,max_length, support_rag=False):
    tokenizer = SimpleTokenizer(max_length, support_rag=support_rag)  # Update max_length if needed
    tokenizer.load_vocab(tokenizer_path)
    return tokenizer


def run_DDP(rank, world_size, args):
    setup(rank, world_size)
    run(args, rank)
    cleanup()


def run(args, rank=None):
    set_seed(45)
    # wandb.init(project="lig_gpt", name=args.run_name)
    os.environ["WANDB_MODE"] = "dryrun"

    print("making tokenizer")

    max_len = args.max_len
    print("tokenizer:")
    tokenizer_path = args.output_tokenizer_dir
    if not os.path.isdir(tokenizer_path):
        os.makedirs(tokenizer_path)
    tokenizer_path = args.output_tokenizer_dir + "/vocab.json"
    print(tokenizer_path)
    if os.path.exists(tokenizer_path):
        print(f"The file '{tokenizer_path}' exists.")
        tokenizer = load_tokenizer(tokenizer_path, max_len, support_rag=args.rag)
    else:
        tokenizer = SimpleTokenizer(max_length=max_len, support_rag=args.rag)
        if args.tokenizer == 'subch':
            tokenizer = SubChTokenizer(max_length=max_len)

        tokenizer.fit_on_file(args.root_path + '.txt')
        tokenizer.fit_on_file(args.root_path + '_val.txt')
        if args.conditions_path is not None:
            tokenizer.fit_on_file(args.conditions_path + '.txt')
            tokenizer.fit_on_file(args.conditions_path + '_val.txt')

        if args.pre_root_path is not None:
            tokenizer.fit_on_file(args.pre_root_path + '.txt')
            if not os.path.exists(args.pre_root_path + '_val.txt'):
                with open(args.pre_root_path + '.txt', 'r') as f:
                    train_seq = f.readlines()
                last_100_lines = train_seq[-100:]
                with open(args.pre_root_path + '_val.txt', 'w') as file:
                    file.writelines(last_100_lines)
            tokenizer.fit_on_file(args.pre_root_path + '_val.txt')
        
        if args.rag_db_path is not None:
            tokenizer.fit_on_lmdb(args.rag_db_path)
            tokenizer.fit_on_lmdb(args.rag_db_val_path)
            tokenizer.fit_on_lmdb(args.rag_db_test_path)
            

        tokenizer.save_vocab(tokenizer_path)
        print("tokenizer saved")

    print(tokenizer.get_vocab())  # Print vocabulary
    vocab_size = tokenizer.get_vocab_size()

    print("making dataset")

    if args.pretrain:
        # switch to pretrain dataset
        args.root_path = args.pre_root_path
        
    with open(args.root_path + '.txt', 'r') as file:
        train_data = file.readlines()
    file.close()

    with open(args.root_path + '_val.txt', 'r') as file:
        val_data = file.readlines()
    file.close()

    if args.conditions_path is not None:
        print("loading conditions")
        with open(args.conditions_path + '.txt', 'r') as file:
            conditions_data = file.readlines()
        file.close()
        with open(args.conditions_path + '_val.txt', 'r') as file:
            conditions_data_val = file.readlines()
    else:
        conditions_data = None
        conditions_data_val = None
    if args.conditions_split_id_path is not None:
        print("loading conditions split id")
        with open(args.conditions_split_id_path + '.txt', 'r') as file:
            conditions_split_id = file.readlines()
        file.close()
        with open(args.conditions_split_id_path + '_val.txt', 'r') as file:
            conditions_split_id_val = file.readlines()
    else:
        conditions_split_id = None
        conditions_split_id_val = None


    if args.ESM_protein:
        train_dataset = NewMol3DDataset(train_data, tokenizer, max_len, conditions_data, conditions_split_id, 
                                        db_path=args.protein_embedding_path)
        valid_dataset = NewMol3DDataset(val_data, tokenizer, max_len, conditions_data_val, conditions_split_id_val, 
                                        db_path=args.protein_embedding_val_path)
    else:
        train_dataset = Mol3DDataset(train_data, tokenizer, max_len, conditions_data, conditions_split_id)
        valid_dataset = Mol3DDataset(val_data, tokenizer, max_len, conditions_data_val, conditions_split_id_val)
    
    
    print(f"train dataset size: {len(train_dataset)}")
    print(f"val dataset size: {len(valid_dataset)}")

    if args.conditions_path is not None or args.conditions_split_id_path is not None:
        isconditional = True
    else:
        isconditional = False

    print("loading model")
    if args.model == 'gpt':
        # mconf = GPTConfig(vocab_size, max_len, num_props=args.num_props,  # args.num_props,
        #                   n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, scaffold=args.scaffold,
        #                   scaffold_maxlen=max_len, lstm=args.lstm, lstm_layers=args.lstm_layers, isconditional=isconditional, 
        #                   mode=args.mode, ESM_protein=args.ESM_protein, rag=args.rag, alpha=args.alpha)
        mconf = GPTConfig(vocab_size, max_len,
                          n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,isconditional=isconditional, 
                          mode=args.mode, ESM_protein=args.ESM_protein, rag=args.rag, alpha=args.alpha)
        model = GPT(mconf)
    else:
        raise ValueError("model is not supported")
        
    if args.pre_model_path is not None:
        print("loading pretrained model: ", args.pre_model_path)
        model_path = args.pre_model_path
        model.load_state_dict(torch.load(model_path), strict=False)
    print('total params:', sum(p.numel() for p in model.parameters()))
    
    cur_time = time.strftime("%Y%m%d_%H%M")
    args.cur_time = cur_time
    checkpoint_folder = os.path.join('./checkpoint/', cur_time + args.run_name)
    # os.makedirs(f'./cond_gpt/weights/', exist_ok=True)
    if args.dist:
        if rank == 0:
            os.makedirs(f'{checkpoint_folder}', exist_ok=True)
    else:
        os.makedirs(f'{checkpoint_folder}', exist_ok=True)
        
    checkpoint_path = os.path.join(checkpoint_folder, f'{args.run_name}.pt')
    checkpoint_best_path = os.path.join(checkpoint_folder, f'{args.run_name}_best.pt')
    
    log_dir = os.path.join('./log/', cur_time + args.run_name)
    if not os.path.exists(log_dir):

        if args.dist:
            if rank == 0:
                os.makedirs(log_dir)
        else:
            os.makedirs(log_dir)
        
    else:
        # shutil.rmtree(log_dir)
        for f in os.listdir(log_dir):
            os.remove(os.path.join(log_dir, f))
    
    writer = SummaryWriter(log_dir=log_dir)
    
    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                          lr_decay=True, warmup_tokens=0.1 * len(train_data) * max_len,
                          final_tokens=args.max_epochs * len(train_data) * max_len,
                          num_workers=args.num_workers, ckpt_path=checkpoint_path, best_ckpt_path=checkpoint_best_path,
                          run_name=args.run_name, block_size=max_len, generate=False, save_start_epoch=args.save_start_epoch,
                          grad_norm_clip=args.grad_norm_clip,
                          save_interval_epoch=args.save_interval_epoch, dist=args.dist, rank=rank, 
                          ckpt_folder=checkpoint_folder, tensorboard_writer=writer,
                          resume=args.resume, resume_ckpt_path=args.resume_ckpt_path)
    trainer = Trainer(model, train_dataset, valid_dataset, tconf)  # , train_dataset.stoi, train_dataset.itos)
    df = trainer.train(wandb)
    if args.dist:
        if rank == 0:
            df.to_csv(f'{args.run_name}.csv', index=False)
    else:
        df.to_csv(f'{args.run_name}.csv', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
    parser.add_argument('--data_name', type=str, default='',
                        help="name of the dataset to train on", required=False)
    parser.add_argument('--model', type=str, default='gpt',
                        help="name of the model", required=False)
    parser.add_argument('--tokenizer', type=str, default='simple',
                        help="name of the tokenizer", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=768,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=60,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=32,
                        help="batch size", required=False)
    parser.add_argument('--num_workers', type=int, default=12,
                        help="number of workers for data loaders", required=False)
    parser.add_argument('--save_start_epoch', type=int, default=120,
                        help="save model start epoch", required=False)
    parser.add_argument('--save_interval_epoch', type=int, default=10,
                        help="save model epoch interval", required=False)
    parser.add_argument('--learning_rate', type=float,
                        default=4e-4, help="learning rate", required=False)
    parser.add_argument('--max_len', type=int, default=512,
                        help="max_len", required=False)
    parser.add_argument('--grad_norm_clip', type=float, default=1.0,
                        help="gradient norm clipping. smaller values mean stronger normalization.", required=False)
    parser.add_argument('--auto_fp16to32', action='store_true',
                        default=False, help='Auto casting fp16 tensors to fp32 when necessary')
    parser.add_argument('--pre_root_path', default=None,
                        help="Path to the pretrain data directory", required=False)
    parser.add_argument('--pre_model_path', default=None,
                        help="Path to the pretrain model", required=False)
    parser.add_argument('--root_path', default='',
                        help="Path to the root data directory", required=False)
    parser.add_argument('--output_tokenizer_dir', default='',
                        help="Path to the saved tokenizer directory", required=False)
    parser.add_argument('--conditions_path', default=None,
                        help="Path to the generation condition", required=False)
    parser.add_argument('--conditions_split_id_path', default=None,
                        help="Path to the conditions_split_id", required=False)
    parser.add_argument('--dist', action='store_true',
                        default=False, help='use torch.distributed to train the model in parallel')
    
    
    parser.add_argument('--protein_embedding_path', default='',
                    help="Path to the train database of pre-calculated protein embedding", required=False)
    parser.add_argument('--protein_embedding_val_path', default='',
                    help="Path to the validation database of pre-calculated protein embedding", required=False)
    parser.add_argument('--mode', default='concat',
                    help="mode to incorporate protein embedding, ['concat', 'cross']", required=False)
    parser.add_argument('--ESM_protein', action='store_true',
                        default=False, help='use ESM protein embedding')
    parser.add_argument('--pretrain', action='store_true',
                        default=False, help='whether to pretrain the model')
    parser.add_argument('--rag_db_path', type=str, default=None,
                        help='path to the train rag database')
    parser.add_argument('--rag_db_val_path', type=str, default=None,
                        help='path to the validation rag database')
    parser.add_argument('--rag_db_test_path', type=str, default=None,
                        help='path to the test rag database')
    parser.add_argument('--rag', action='store_true',
                        default=False, help='whether to use rag')
    parser.add_argument('--resume', action='store_true',
                        default=False, help='whether to resume training')
    parser.add_argument('--resume_ckpt_path', type=str, default=None,
                    help='path to the checkpoint for resume training')
    parser.add_argument('--alpha', type=float, default=1.0, help='weight of loss for retrieved part')

    args = parser.parse_args()

    
    if args.dist:
        world_size = torch.cuda.device_count()
        # import pdb; pdb.set_trace()
        mp.spawn(run_DDP,
                 args=(world_size, args),
                 nprocs=world_size,
                 join=True)
    else:
        run(args)
