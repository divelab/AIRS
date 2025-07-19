ROOT_PATH=""
model_path="$ROOT_PATH/outputs/chipseq/train_inf_gpt_small_gm12878/checkpoints/val/sei.ckpt"


python -m train \
    experiment=hg38/chipseq \
    wandb.mode=online \
    wandb.group=Chipseq \
    wandb.name=gen_gpt_small_gm12878 \
    hydra.run.dir="./outputs/chipseq/gen_gpt_small_gm12878" \
    model.config.n_layer=12 \
    model.config.n_head=12 \
    model.config.n_embd=768 \
    dataset.batch_size=16 \
    train.only_test=True \
    train.only_test_model_path=$model_path




