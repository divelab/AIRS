#!/bin/sh

GPU=7

CUDA_VISIBLE_DEVICES=${GPU} python ../src/main_fast_new.py --GPU --J2=-0.02 --lr=0.001 --dataname='36_kagome_lattice'\
 --savefolder='36_kagome_lattice_pbc_J2_-002_f_64_non_local_preact_regularconv_V2_clip_grad_1_lrdecay_4000' --data_dir='../dataset/' --save_dir='../result/Heisenberg/' \
 --checkpoint_dir='../checkpoints/Heisenberg/' --log_dir='../log/Heisenberg/' --model='cnn2d-se-kagome' \
 --num_layers=3 --emb_dim=32 --epochs=2000 --num_spin=36 --batch_size=1000 --non_local --preact --conv='regular-v2' --weight_decay=2e-4 --clip_grad=1 \
 --lr_decay_step=4000
