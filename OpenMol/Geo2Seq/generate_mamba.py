import argparse
import time
import json
import random
from collections import Counter
from dataset import Mol3DDataset, SimpleTokenizer
import argparse
from utils import set_seed
import numpy as np
import wandb
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from mamba import MambaLMHeadModel, MambaConfig
from functools import partial
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

def load_tokenizer(tokenizer_path,max_length):
    tokenizer = SimpleTokenizer(max_length)  # Update max_length if needed
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
    # in moses dataset, on average, there are only 5 molecules per scaffold
    parser.add_argument('--scaffold', action='store_true',
                        default=False, help='condition on scaffold')
    parser.add_argument('--lstm', action='store_true',
                        default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--data_name', type=str, default='moses2',
                        help="name of the dataset to train on", required=False)
    # parser.add_argument('--property', type=str, default = 'qed', help="which property to use for condition", required=False)
    parser.add_argument('--props', nargs="+", default=['qed'],
                        help="properties to be used for condition", required=False)
    parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
    # parser.add_argument('--prop1_unique', type=int, default = 0, help="unique values in that property", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=768,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=30,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=80,
                        help="batch size", required=False)
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
    parser.add_argument('--root_path', default='QM9_seq/spherical_seq',
                        help="Path to the root data directory", required=False)
    parser.add_argument('--output_tokenizer_dir', default='spherical_seq/tokenizer',
                        help="Path to the saved tokenizer directory", required=False)
    parser.add_argument('--conditions_path', default=None,
                        help="Path to the generation condition", required=False)
    parser.add_argument("--model-name", type=str, default="state-spaces/mamba-130m")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--promptlen", type=int, default=100)
    parser.add_argument("--genlen", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--topp", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=12000)

    args = parser.parse_args()

    set_seed(45)

    os.environ["WANDB_MODE"] = "dryrun"

    max_len = args.max_len
    tokenizer_path = args.output_tokenizer_dir + "/vocab.json"
    print(tokenizer_path)
    print("tokenizer:")
    tokenizer = load_tokenizer(tokenizer_path,max_len)
    print(tokenizer.get_vocab())  # Print vocabulary
    vocab_size = tokenizer.get_vocab_size()

    repeats = args.repeats
    device = "cuda"
    dtype = torch.float16

    print("loading model")
    # model = GPT(mconf)
    if args.epoch == 0:
        model_path = f'cond_gpt/weights/{args.run_name}.pt'
        gen_name = f'{args.run_name}'
    else:
        model_path = f'cond_gpt/weights/{args.run_name}_ep{args.epoch}.pt'
        gen_name = f'{args.run_name}_ep{args.epoch}_top{args.topk}_temp{args.temperature}'
    train_data_path = args.root_path + '.txt'
    if args.conditions_path is not None:
        conditions_path = args.conditions_path + '.txt'

    # Load model and tokenizer
    mamba_config = MambaConfig(d_model=args.n_embd, n_layer=args.n_layer, vocab_size=vocab_size,
                               num_props=args.num_props, scaffold=args.scaffold)
    model = MambaLMHeadModel(mamba_config)
    model.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'], strict=True)
    model.to(device)
    model.eval()
    print("loaded model")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


    # Get first token distribution
    if args.conditions_path is not None:
        token_probs = get_first_token_distribution(conditions_path)
    else:
        token_probs = get_first_token_distribution(train_data_path)

    generate = partial(model.generate,
                 max_length=max_len,
                 cg=True,
                 return_dict_in_generate=True,
                 output_scores=True,
                 enable_timing=False,
                 temperature=args.temperature,
                 top_k=args.topk,
                 top_p=args.topp,
                 repetition_penalty=args.repetition_penalty,
                 )
    # out = fn()
    samples = []
    batch_size = args.batch_size
    num_batches = args.repeats // batch_size
    torch.cuda.synchronize()
    start = time.time()
    for i in range(num_batches):
        first_tokens = [sample_first_token(token_probs) for _ in range(batch_size)]
        first_token_ids = torch.tensor([tokenizer.generation_encode(token) for token in first_tokens]).to(device)
        out = generate(first_token_ids)
        batch_samples = [tokenizer.decode(ids) for ids in out.sequences.cpu()]
        # batch_samples = [tokenizer.decode(ids[len(first_token_ids[0]):]) for ids in out.sequences.cpu()]
        samples.extend(batch_samples)
        if i % 5 == 0:
            print(f'Generated {i * batch_size} samples')
            print(batch_samples[0])
    torch.cuda.synchronize()
    print(f"model prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.0f}ms")

    # Save samples
    with open('generated_samples_'+gen_name+'.txt', 'w') as file:
        for sample in samples:
            file.write(sample + '\n')









