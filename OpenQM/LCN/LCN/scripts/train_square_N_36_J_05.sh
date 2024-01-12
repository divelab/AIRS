#!/bin/sh

GPU=1

CUDA_VISIBLE_DEVICES=${GPU} python ../src/main_fast_new.py --GPU --J2=0.5 --dataname='36_node_square_lattice'\
 --num_spin=36 --data_dir='../dataset/' --savefolder='36_square_J05' --save_dir='../result/Heisenberg/'\
 --checkpoint_dir='../checkpoints/Heisenberg/' --log_dir='../log/Heisenberg/' --model='cnn2d-se' \
 --conv='nn.Conv2d' --num_blocks=2 --non_local --preact --emb_dim=64 --kernel_size=3 --act='relu' --optim energy --epochs=200 --batch_size 500 \
 --lr=1e-3 --milestones 20000 40000 60000 --gamma=0.1 --clip_grad=1
