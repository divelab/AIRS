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
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler
# from transformers import AutoTokenizer
from model import GPT, GPTConfig
# from trainer import Trainer, TrainerConfig
from dataset import Mol3DDataset, SimpleTokenizer, SubChTokenizer
import math
import re
import json
import random
from collections import Counter
from tqdm import tqdm

from utils import sample
from rdkit import Chem

import lmdb
import pickle
import gzip


def load_model(model_path, config, model_name):
    if model_name == 'gpt':
        model = GPT(config)

    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        # model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


def load_tokenizer(tokenizer_path,max_length,tokenizer_type='simple'):
    tokenizer = SimpleTokenizer(max_length)
    if tokenizer_type == 'subch':
        tokenizer = SubChTokenizer(max_length)
    tokenizer.load_vocab(tokenizer_path)
    return tokenizer


def get_first_token_distribution(train_data_path):
    with open(train_data_path, 'r') as file:
        data = file.readlines()
    first_tokens = [line.strip().split()[0] for line in data if len(line.strip())]
    token_counts = Counter(first_tokens)
    total = sum(token_counts.values())
    token_probs = {token: count / total for token, count in token_counts.items()}
    return token_probs


def sample_first_token(token_probs):
    tokens, probs = zip(*token_probs.items())
    first_token = random.choices(tokens, weights=probs)[0]
    return first_token

def top_k_logits(logits, k):
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).expand_as(logits)
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

def sample_from_logits(logits, top_k, temp=1.0):
    logits = top_k_logits(logits, top_k)
    probabilities = F.softmax(logits / temp, dim=-1)
    next_token = torch.multinomial(probabilities, 1, replacement=True)
    return next_token

def generate_sample(model, tokenizer, first_tokens, max_length, top_k=100, temp=1.0, beam_size=None, 
                    protein_embedding_mask=None,
                    protein_padded_embedding=None):
    model.eval()
    first_token_ids = [tokenizer.generation_encode(token) for token in first_tokens]
    input_ids = pad_sequence([torch.tensor(ids, dtype=torch.long) for ids in first_token_ids], batch_first=True, padding_value=tokenizer.vocab["<pad>"]).cuda()
    # if beam_size is not None:
    #     return beam_search(model, input_ids, max_length, beam_size, top_k)
    init_len = input_ids.size(1)
    with torch.no_grad():
        for _ in range(max_length - init_len):
            # import pdb; pdb.set_trace()
            output = model(input_ids,
                           protein_embedding_mask = protein_embedding_mask.cuda(),
                           protein_padded_embedding = protein_padded_embedding.cuda())
            next_token_logits = output[0][:, -1, :]
            next_token = sample_from_logits(next_token_logits, top_k, temp=temp)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check if all sequences in the batch have reached the end token or pad token
            if all(token.item() in [tokenizer.vocab["</s>"], tokenizer.vocab["<pad>"]] for token in next_token):
                break

    return [tokenizer.decode(ids) for ids in input_ids]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')

    parser.add_argument('--scaffold', action='store_true',
                        default=False, help='condition on scaffold')
    parser.add_argument('--lstm', action='store_true',
                        default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--data_name', type=str, default='',
                        help="name of the dataset to train on", required=False)
    parser.add_argument('--props', nargs="+", default=['qed'],
                        help="properties to be used for condition", required=False)
    parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=768,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=30,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=30,
                        help="batch size", required=False)
    parser.add_argument('--sample_repeats', type=int, default=12000,
                        help="number of generate samples", required=False)
    parser.add_argument('--top_k', type=int, default=100,
                        help="top_k for sampling", required=False)
    parser.add_argument('--temp', type=float, default=1.0,
                        help="sampling temperature", required=False)
    parser.add_argument('--tokenizer', type=str, default='simple',
                        help="name of the tokenizer", required=False)
    parser.add_argument('--num_workers', type=int, default=8,
                        help="number of workers for data loaders", required=False)
    parser.add_argument('--learning_rate', type=float,
                        default=1e-4, help="learning rate", required=False)
    parser.add_argument('--lstm_layers', type=int, default=0,
                        help="number of layers in lstm", required=False)
    parser.add_argument('--max_len', type=int, default=512,
                        help="max_len", required=False)
    parser.add_argument('--epoch', type=int, default=0,
                        help="saved model epoch", required=False)
    parser.add_argument('--root_path', default='',
                        help="Path to the root data directory", required=False)
    parser.add_argument('--output_tokenizer_dir', default='',
                        help="Path to the saved tokenizer directory", required=False)
    parser.add_argument('--conditions_path', default=None,
                        help="Path to the generation condition", required=False)
    parser.add_argument('--save_path', default='',
                        help="Path to the generation file dir", required=False)
    parser.add_argument('--mode', default='concat',
                help="mode to incorporate protein embedding, ['concat', 'cross']", required=False)
    parser.add_argument('--ESM_protein', action='store_true',
                        default=False, help='use ESM protein embedding')
    parser.add_argument('--test_lmdb_path', default='',
                        help="Path to the test data lmdb", required=False)
    parser.add_argument('--model_root_path', default='',
                        help="root path to the trained model", required=False)
    parser.add_argument('--seed', type=int, default=56,
                        help="seed", required=False)
    parser.add_argument('--model_name', type=str, default='gpt',
                        help="name of the model", required=False)

    args = parser.parse_args()

    set_seed(args.seed)

    # wandb.init(project="lig_gpt", name=args.run_name)
    os.environ["WANDB_MODE"] = "dryrun"

    max_len = args.max_len

    tokenizer_path = args.output_tokenizer_dir + "/vocab.json"
    # tokenizer.save_vocab(tokenizer_path)
    print(tokenizer_path)
    print("tokenizer:")
    tokenizer = load_tokenizer(tokenizer_path, max_len, args.tokenizer)
    print(tokenizer.get_vocab())  # Print vocabulary
    vocab_size = tokenizer.get_vocab_size()


    if args.conditions_path is not None:
        isconditional = True
    else:
        isconditional = False

    if args.model_name == 'gpt':
        mconf = GPTConfig(vocab_size, max_len, num_props=args.num_props,  # args.num_props,
                            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, scaffold=args.scaffold,
                            scaffold_maxlen=max_len, lstm=args.lstm, lstm_layers=args.lstm_layers,
                            isconditional=isconditional, ESM_protein=args.ESM_protein, mode=args.mode)
    else:
        raise ValueError("model is not supported")


    print("loading model")
    if args.epoch == 0:
        model_path = os.path.join(args.model_root_path, f'{args.run_name}.pt')
        gen_name = f'{args.run_name}'
    else:
        model_path = os.path.join(args.model_root_path, f'{args.run_name}_ep{args.epoch}.pt')
        gen_name = f'{args.run_name}_ep{args.epoch}_top{args.top_k}_temp{args.temp}'
    train_data_path = args.root_path + '.txt'

    # Load model and tokenizer
    model = load_model(model_path, mconf, args.model_name).cuda()
    print("loaded model")
    print('total params:', sum(p.numel() for p in model.parameters()))


    # Get first token distribution
    train_data_path = args.root_path + '.txt'
    token_probs = get_first_token_distribution(train_data_path)

    
    #################################
    ## load test database
    #################################
    env = lmdb.open(
        args.test_lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    
    txn = env.begin()
    
    keys = list(txn.cursor().iternext(values=False))
    
    num_test_data = len(keys)


    num_samples = args.sample_repeats  # Number of samples to generate for each first token

    os.makedirs(args.save_path, exist_ok=True)
    print(f'Saving samples to: {args.save_path}')
    # Generate samples for each first token and write to separate files
    import time
    start = time.time()
    time_list = []
    for idx, key in enumerate(keys):
        substart = time.time()
        # print(f'Generating samples for: {first_token}')
        key = int(idx).to_bytes(4, byteorder="big")

        datapoint_pickled_compressed = txn.get(key=key)
        protein_embedding_dict = pickle.loads(gzip.decompress(datapoint_pickled_compressed))
        if protein_embedding_dict['padded_embedding'] is None:
            print(f'Completed {idx} with {len(samples)} samples')
            print("Create empty file!!!")
            with open(f'{args.save_path}/gen_mol_{idx}.txt', 'w') as f:
                print("Create empty file!!!")
            continue
        protein_padded_embedding = protein_embedding_dict['padded_embedding']
        protein_embedding_mask = protein_embedding_dict['mask']
        
        # import pdb; pdb.set_trace()
        protein_padded_embedding = protein_padded_embedding.unsqueeze(0).expand(num_samples, -1,-1)
        protein_embedding_mask = protein_embedding_mask.unsqueeze(0).expand(num_samples,-1)
        # import pdb; pdb.set_trace()
        
        batch_first_tokens = [sample_first_token(token_probs) for _ in range(num_samples)]
        # batch_first_tokens = [first_token for _ in range(num_samples)]
        samples = generate_sample(model, 
                                  tokenizer, 
                                  batch_first_tokens, 
                                  max_len, 
                                  top_k=args.top_k, 
                                  temp=args.temp, 
                                  beam_size=None,
                                  protein_embedding_mask=protein_embedding_mask,
                                  protein_padded_embedding=protein_padded_embedding)
        # first_token_ids = torch.tensor([tokenizer.generation_encode(first_token) for _ in range(num_samples)],
        #                                dtype=torch.long).to(device)
        # out = generate(
        #     first_token_ids)  # Ensure this function supports batched input and generates a sequence for each input
        # samples = [tokenizer.decode(ids) for ids in out.sequences.cpu()]
        print(f'Completed {idx} with {len(samples)} samples')
        print(f'Sub-time taken: {time.time() - substart}')
        time_list.append(time.time() - substart)
        print(samples[0])
        with open(f'{args.save_path}/gen_mol_{idx}.txt', 'a') as f:
            for sample, first_token in zip(samples, batch_first_tokens):
                if sample.startswith(first_token):
                    gen_sample = sample[:]
                    f.write(gen_sample + '\n')
                else:
                    print(f"{sample} does not start with {first_token}")
    
    print(f'Time taken: {time.time() - start}')
    print("mean: ", np.array(time_list).mean(), " std: ", np.array(time_list).std())