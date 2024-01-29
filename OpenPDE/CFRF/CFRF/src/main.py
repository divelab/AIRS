import os
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import random
import time
import argparse
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    torch.autograd.set_detect_anomaly(True)
    # parse arguments
    parser = argparse.ArgumentParser(description="PDE super resolution")

    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')


    parser.add_argument("--experiment", type=str, default="Supervised", help="Supervised, LRinput_freq")

    # model hyperparameters
    parser.add_argument("--emb_dim", type=int, default=16, help="dimensionality of hidden layers")
    parser.add_argument("--layers", type=int, default=3, help="number of layers")
    parser.add_argument("--head", type=int, default=2, help="number of attention heads for transformer")
    parser.add_argument('--enc_modes', action='store_true', default=False, help='Encode modes for mode transformer')
    parser.add_argument('--no_layer_norm', action='store_true', help='layer norm for transformer')
    parser.add_argument("--dropout", type=float, default=0, help="dropout probability for transformer")
    parser.add_argument('--encoding', type=str, default=None, help="Type of positional encoding for transformer; one of ['complex', 'stack']")
    parser.add_argument('--injection', type=str, default=None, help="Type of positional encoding injection for transformer; one of ['add', 'cat']")
    parser.add_argument('--projection', type=str, default=None, help="Projection to use in physical space for the transformer")
    parser.add_argument('--activation', type=str, default="CReLU", help="Activation for the transformer")
    parser.add_argument('--postLN', action='store_true', help='use post layernorm')
    parser.add_argument('--Hermitian', action='store_true', help='use Hermitian property to learn only the half modes')
    parser.add_argument('--ring', action='store_true', help='only learn the ring area of modes')
    parser.add_argument('--complex_pos_enc', action='store_true', help='if use the complex position encoding')
    

    # training
    parser.add_argument("--model", type=str, default="SRNet", help="SRNet, CNN")
    parser.add_argument("--mode", type=str, default="train", help="one of 'train' or 'test'")
    parser.add_argument("--lr_init", type=float, default = 1e-3, help="initial learning rate")
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs to train")
    parser.add_argument('--cosine_scheduler', action='store_true', default=False, help='Use a cosine scheduler for training')
    parser.add_argument("--batch_size", type=int, default=128, help="input batch size for training")
    parser.add_argument("--sw", type=int, default=3, help="sliding window")
    parser.add_argument('--residual', action='store_true', default=False, help='predict residual')
    parser.add_argument("--upsample", type=str, default="nearest", help="upsampling method")
    parser.add_argument("--freq_center_size", type=int, default=5, help="frequency mask offset")
    parser.add_argument("--freeze_size", type=int, default=5, help="freeze center modes")
    parser.add_argument("--frozen_layers", type=int, default=0, help="Number of layers in which modes should be frozen for SRFNO")
    parser.add_argument("--domain_size", type=str, default='1', help="domain size")
    parser.add_argument('--content_loss', action='store_true', default=False, help='')
    parser.add_argument('--divergence', action='store_true', default=False, help='')
    parser.add_argument('--scheduler', action='store_true', default=False, help='')
    parser.add_argument('--supervise_train_subset', action='store_true', default=False, help='')
    parser.add_argument('--subset', action='store_true', default=False, help='')
    parser.add_argument('--unsupervised', action='store_true', help='Unsupervised training in the supervised experiment')
    parser.add_argument('--unsupervise_finetune', action='store_true', default=False, help='')
    parser.add_argument('--supervise_finetune', action='store_true', default=False, help='')
    parser.add_argument('--same_model_finetune', action='store_true', default=False, help='')
    parser.add_argument("--content_loss_weights", type=float, default=1., help="")
    parser.add_argument('--finetune_net1', action='store_true', default=False, help='')
    parser.add_argument("--data_frac", type=str, default='', help="")
    parser.add_argument("--choose_best_model", type=str, default='mse', help="")
    parser.add_argument("--reference_model", type=str, default='SRCNN', help="")
    parser.add_argument("--supervised_loss", type=str, default='mse', help="")
    parser.add_argument('--gradient_loss', action='store_true', default=False, help='')

    parser.add_argument("--field_dim", type=int, default=1, help="dimention of state")
    parser.add_argument("--PDE", type=str, default='iCFD', help="PDE equation name")

    parser.add_argument('--seperate_prediction', action='store_true', default=False, help='predict each field seperately')

    parser.add_argument("--LR_resolution", type=int, default=32, help="resolution of LR data")
    parser.add_argument("--HR_resolution", type=int, default=32, help="resolution of HR data")
    

    # data
    parser.add_argument("--dataname", type=str, default="", help="data")
    parser.add_argument("--num_workers", type=int, default=4, help="num of data loader workers")
    parser.add_argument("--data_path", type=str, default="../data/", help="path to data")
    parser.add_argument("--scale_factor", type=int, default=4, help="downsampling scale factor")

    # directory
    parser.add_argument("--working_dir", type=str, default="../", help="working directory for logs, saved models, etc.")
    parser.add_argument("--suffix", type=str, default="", help="optional suffix added to working_dir")
    parser.add_argument("--log_dir", type=str, default="../log", help="tensorboard log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="trained_models", help="directory to save checkpoint in working directory")
    # parser.add_argument("--saved_model_dir", type=str, default=None, help="directory with checkpoint.pt")
    parser.add_argument("--savedir", type=str, default="/save_eval", help="directory to save predicted HR solution")
    parser.add_argument("--experiment_dir", type=str, default="", help="")
    parser.add_argument("--test_dir", type=str, default="", help="")

    # import pdb; pdb.set_trace()
    args = parser.parse_args()

    if args.domain_size == '2pi':
        args.L = 2 * math.pi
    elif args.domain_size == '1':
        args.L = 1
    else:
        raise ValueError(f"{args.domain_size} domain size not recognized")

    print(args)
    cur_time = time.strftime("%Y%m%d_%H%M")
    args.cur_time = cur_time

    # create the working dir
    if args.debug:
        args.working_dir = os.path.join(args.working_dir, args.suffix+"_debug")
    else:
        args.working_dir = os.path.join(args.working_dir, cur_time + args.suffix)
    if args.mode == 'train':
        os.makedirs(args.working_dir, exist_ok=True)

        # write arguments to txt
        with open(os.path.join(args.working_dir, 'args.txt'), 'w') as f:
            f.write(str(args))
    elif args.mode == 'test':
        print("Set test dir as the working dir!!!")
        args.working_dir = os.path.join('../', args.test_dir)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)


    if args.experiment == "LRinput_freq":
        from experiment_LRinput_frequency_PDEloss import Experiment
    elif args.experiment == "Supervised":
        print("Use supervised setting!!!")
        from experiment_supervised import Experiment
    else:
        assert False, "Experiment not found"


    experiment = Experiment(args)
    experiment.run()
    

if __name__ == "__main__":
    main()
