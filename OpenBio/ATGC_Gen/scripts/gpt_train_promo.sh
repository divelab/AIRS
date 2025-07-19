# gpt base
python -m train \
    experiment=hg38/promoter \
    wandb.mode=online \
    wandb.group=Promoter \
    wandb.name=train_inf_gpt \
    hydra.run.dir=./outputs/promoter/train_inf_gpt


