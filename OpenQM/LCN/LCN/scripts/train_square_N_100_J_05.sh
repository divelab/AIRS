#!/bin/sh

GPU=1

CUDA_VISIBLE_DEVICES=${GPU} python ../src/main_fast_new.py --GPU --J2=0.5 --dataname='100_node_square_lattice'\
 --num_spin=100 --data_dir='../dataset/' --savefolder='100_square_J05' --save_dir='../result/Heisenberg/'\
 --checkpoint_dir='../checkpoints/Heisenberg/' --log_dir='../log/Heisenberg/' --model='cnn2d-se-2' \
 --conv='nn.Conv2d' --num_blocks=4 --non_local --preact --emb_dim=64 --kernel_size=3 --act='relu' --optim energy --epochs=200 --batch_size 200 \
 --lr=5e-4 --milestones 8000 12000 16000 --gamma=0.1 --clip_grad=1
