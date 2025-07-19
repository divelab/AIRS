# bert base

ROOT_PATH=""
model_path="$ROOT_PATH/outputs/enhancers_mel/train_inf_bert_base_step1_lr1e-4/checkpoints/val/fbd.ckpt"


python -m train \
    experiment=hg38/enhancer_mel_debug \
    wandb.mode=online \
    wandb.group=Enhancer_Mel \
    wandb.name=genfull_bert_base_temp \
    hydra.run.dir="./outputs/enhancers_mel/genfull_bert_base_temp" \
    task=DNA_masked_gen \
    model=bert \
    dataset.load_prob=True \
    model.config.block_size=500 \
    model.config.generation_step=full \
    model.config.greedy_gen=False \
    model.config.n_layer=12 \
    model.config.n_head=12 \
    model.config.n_embd=768 \
    train.only_test=True \
    train.only_test_model_path=$model_path




