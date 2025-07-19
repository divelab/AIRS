python -m train \
    experiment=hg38/enhancer_mel_debug \
    wandb.mode=online \
    wandb.group=Enhancer_Mel \
    wandb.name=train_inf_bert_base_step1_lr1e-4 \
    hydra.run.dir="./outputs/enhancers_mel/train_inf_bert_base_step1_lr1e-4" \
    task=DNA_masked_gen \
    model=bert \
    dataset.load_prob=True \
    model.config.block_size=500 \
    train.remove_test_loader_in_eval=False \
    trainer.limit_val_batches=1.0 \
    trainer.check_val_every_n_epoch=1 \
    model.config.generation_step=1 \
    optimizer.lr=1e-4 \
    model.config.n_layer=12 \
    model.config.n_head=12 \
    model.config.n_embd=768

