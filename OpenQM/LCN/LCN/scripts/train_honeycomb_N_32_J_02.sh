#!/bin/sh

GPU=4

CUDA_VISIBLE_DEVICES=${GPU} python ../src/main_fast_new.py --GPU --J2=0.2 --dataname='32_node_honeycomb_lattice'\
 --num_spin=32 --data_dir='../dataset/' --savefolder='32_honeycomb_J02' --save_dir='../result/Heisenberg/'\
 --checkpoint_dir='../checkpoints/Heisenberg/' --log_dir='../log/Heisenberg/' --model='cnn2d-se' \
 --conv='HoneycombConv2d_v5' --num_blocks=2 --non_local --preact --emb_dim=64 --kernel_size=3 --act='relu' --optim energy --epochs=200 --batch_size 500 \
 --lr=1e-3 --milestones 20000 40000 60000 --gamma=0.1 --clip_grad=1
