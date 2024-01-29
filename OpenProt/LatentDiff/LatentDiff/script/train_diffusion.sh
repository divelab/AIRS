#!/bin/sh

GPU=3,4,5,6

CUDA_VISIBLE_DEVICES=${GPU} python ../diffusion/main_protein.py --dp=True --batch_size=2048 --n_epochs=16000 --exp_name=latent_diffusion \
--latent_dataname=<set this variable> --test_epochs=20 \
--diffusion_steps=1000 \