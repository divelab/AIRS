#!/bin/sh

gpu=(4 0 2)

suffix_list=(
    "LRinput_supervised_SRCNN_bicubic_init_s_64_L_2pi_unsupervise_finetune"
)

experiment_dir_list=(
    "path to the root folder of the trained model"
)

data_frac_list=(
    "1/16"
)

for ((i=0; i<1; i++)); do
    (
        CUDA_VISIBLE_DEVICES=${gpu[i]} \
        python ../src/main.py \
        --model='ModeFormer' \
        --batch_size=16 \
        --PDE='iCFD' \
        --scale_factor=4 \
        --epochs=100 \
        --suffix=${suffix_list[i]} \
        --experiment='LRinput_freq' \
        --upsample='bicubic' \
        --freq_center_size=28 \
        --domain_size='2pi' \
        --emb_dim=32 \
        --sw=7 \
        --unsupervise_finetune \
        --content_loss \
        --residual \
        --experiment_dir=${experiment_dir_list[i]} \
        --data_frac=${data_frac_list[i]} \
        --enc_modes \
        --encoding='stack' \
        --injection='cat' \
        --activation='zReLU' \
        --postLN \
        --ring \
        --freeze_size=8 \
    ) #&
done