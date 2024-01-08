#!/bin/sh

GPU=1

CUDA_VISIBLE_DEVICES=${GPU} python ../src/main_fast_new.py --GPU --J2=0.2 --dataname='98_node_honeycomb_lattice'\
 --num_spin=98 --data_dir='../dataset/' --savefolder='98_honeycomb_J02' --save_dir='../result/Heisenberg/'\
 --checkpoint_dir='../checkpoints/Heisenberg/' --log_dir='../log/Heisenberg/' --model='cnn2d-se-2' \
 --conv='HoneycombConv2d_v5' --num_blocks=4 --non_local --preact --emb_dim=64 --kernel_size=3 --act='relu' --optim energy --epochs=200 --batch_size 100 \
 --lr=7e-4 --milestones 10000 16000 22000 --gamma=0.1 --clip_grad=1
 