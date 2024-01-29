#!/bin/sh

GPU=1

CUDA_VISIBLE_DEVICES=${GPU} python ../src/main_fast_new.py --GPU --J2=-0 --lr=0.001 --dataname='108_kagome_lattice'\
 --savefolder='108_kagome_lattice_pbc_J2_0_f_64_non_local_preact_regularconv_clip_grad_1_lrdecay_4000' --data_dir='../dataset/' --save_dir='../result/Heisenberg/' \
 --checkpoint_dir='../checkpoints/Heisenberg/' --log_dir='../log/Heisenberg/' --model='cnn2d-se-kagome-108' \
 --num_layers=3 --emb_dim=32 --epochs=2000 --num_spin=108 --batch_size=200 --non_local --conv='regular' --weight_decay=2e-4 --clip_grad=1 \
 --lr_decay_step=4000
