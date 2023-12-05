#!/bin/sh

gpu=(5 0 2)

PDE='diff-react'
field_dim=2
scale_factor=8
sw=4

HR_resolution=128
LR_resolution=$((HR_resolution / scale_factor))
reference_model='SRCNN'

content_loss_weights=0.5
freeze_modes=8
freq_center_size=36
emb_dim=64

suffix_list=(
    "${PDE}_LRinput_supervised_${reference_model}_bicubic_init_s_${LR_resolution}to${HR_resolution}_sw_${sw}_unsupervise_finetune_contentlossweights${content_loss_weights}_freeze${freeze_modes}_center${freq_center_size}_embdim${emb_dim}"
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
        --batch_size=4 \
        --scale_factor=${scale_factor} \
        --epochs=100 \
        --suffix=${suffix_list[i]} \
        --experiment='LRinput_freq' \
        --upsample='bicubic' \
        --freq_center_size=${freq_center_size} \
        --domain_size='2pi' \
        --emb_dim=${emb_dim} \
        --sw=${sw} \
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
        --freeze_size=${freeze_modes} \
        --PDE=${PDE} \
        --field_dim=${field_dim} \
        --seperate_prediction \
        --reference_model=${reference_model} \
        --content_loss_weights=${content_loss_weights} \
        --LR_resolution=${LR_resolution} \
        --HR_resolution=${HR_resolution} \
        --mode='test' \
        --test_dir='/mnt/data/shared/congfu/PDE/unsupervised_SR/20230808_1638diff-react_LRinput_supervised_SRCNN_bicubic_init_s_16to128_sw_4_1|20_train_data_unsupervise_finetune_FreqNetV5_postLN_ring_contentlossweights0.5_freeze8_center36_embdim64'
    )  # &
done