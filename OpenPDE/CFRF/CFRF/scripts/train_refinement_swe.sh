#!/bin/sh

gpu=(4 0 2)

PDE='swe'
field_dim=3
scale_factor=2

HR_resolution=128
LR_resolution=$((HR_resolution / scale_factor))
reference_model='SRCNN'

content_loss_weights=0.1

suffix_list=(
    "${PDE}_LRinput_supervised_${reference_model}_bicubic_init_s_${LR_resolution}to${HR_resolution}_contentlossweights${content_loss_weights}_unsupervise_finetune"
)

experiment_dir_list=(
    "path to the root folder of the trained model"
)


data_frac_list=(
    "1/20"
)

for ((i=0; i<1; i++)); do
    (
        CUDA_VISIBLE_DEVICES=${gpu[i]} \
        python ../src/main.py \
        --model='ModeFormer' \
        --batch_size=8 \
        --scale_factor=${scale_factor} \
        --epochs=100 \
        --suffix=${suffix_list[i]} \
        --experiment='LRinput_freq' \
        --upsample='bicubic' \
        --freq_center_size=28 \
        --domain_size='2pi' \
        --emb_dim=32 \
        --sw=3 \
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
        --freeze_size=5 \
        --PDE=${PDE} \
        --field_dim=${field_dim} \
        --seperate_prediction \
        --reference_model=${reference_model} \
        --content_loss_weights=${content_loss_weights} \
        --LR_resolution=${LR_resolution} \
        --HR_resolution=${HR_resolution} \
    )  # &
done