# gpt

ROOT_PATH=""
model_path="$ROOT_PATH/outputs/promoter/train_inf_gpt/checkpoints/val/sei_promoter.ckpt"

python -m train \
    experiment=hg38/promoter \
    wandb.mode=online \
    wandb.group=PromoterGen \
    wandb.name=gen_gpt \
    hydra.run.dir=./outputs/promoter/gen_gpt \
    dataset.batch_size=16 \
    train.only_test=True \
    train.only_test_model_path=$model_path
