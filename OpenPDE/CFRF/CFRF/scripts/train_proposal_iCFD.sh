#!/bin/sh

gpu=(2 2 2)

suffix_list=(
    "LRinput_supervised_SRCNN_bicubic_init_s_64_L_2pi"
)

data_frac_list=(
    "1/16"
)

for ((i=0; i<1; i++)); do
    (
        CUDA_VISIBLE_DEVICES=${gpu[i]} \
        python ../src/main.py \
        --model='SRCNN' \
        --batch_size=32 \
        --PDE='iCFD' \
        --scale_factor=4 \
        --epochs=200 \
        --suffix=${suffix_list[i]} \
        --experiment='Supervised' \
        --upsample='bicubic' \
        --freq_center_size=28 \
        --freeze_size=8 \
        --domain_size='2pi' \
        --emb_dim=32 \
        --subset \
        --data_frac=${data_frac_list[i]} \
    ) #&
done