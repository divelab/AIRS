# bert base
python -m train \
    experiment=hg38/promoter \
    wandb.mode=online \
    wandb.group=Promoter \
    wandb.name=train_inf_bert_base_step1 \
    hydra.run.dir=./outputs/promoter/train_inf_bert_base_step1 \
    task=DNA_masked_gen \
    model=bert \
    dataset.load_prob=True \
    model.config.block_size=1024 \
    train.remove_test_loader_in_eval=False \
    trainer.limit_val_batches=1.0 \
    trainer.check_val_every_n_epoch=1 \
    model.config.generation_step=1 \
    model.config.n_layer=12 \
    model.config.n_head=12 \
    model.config.n_embd=768 \
    dataset.batch_size=16

