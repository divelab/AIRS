#!/bin/sh

GPU=0

CUDA_VISIBLE_DEVICES=${GPU} python ../protein_autoencoder/main.py --dataname='AFPDB_data_128_complete' --data_path='../data/' \
--mode='train' --lr_init=1e-3 --layers=2 --epochs=200 --batch_size=100 --suffix='protein_autoencoder' \
--transpose --attn --edgeloss_weight=0.5 --kl_weight=1e-4


