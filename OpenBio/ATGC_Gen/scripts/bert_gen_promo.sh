# bert base

ROOT_PATH=""
model_path="$ROOT_PATH/outputs/promoter/train_inf_bert_base_step1/checkpoints/val/sei_promoter.ckpt"



python -m train \
    experiment=hg38/promoter \
    wandb.mode=online \
    wandb.group=Promoter \
    wandb.name=genfull_bert_base_temp \
    hydra.run.dir="./outputs/promoter/genfull_bert_base_temp" \
    task=DNA_masked_gen \
    model=bert \
    dataset.load_prob=True \
    model.config.block_size=1024 \
    model.config.generation_step=full \
    model.config.greedy_gen=False \
    model.config.n_layer=12 \
    model.config.n_head=12 \
    model.config.n_embd=768 \
    dataset.batch_size=16 \
    train.only_test=True \
    train.only_test_model_path=$model_path

