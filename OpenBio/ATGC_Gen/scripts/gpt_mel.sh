# gpt base
python -m train \
    experiment=hg38/enhancer_mel \
    wandb.mode=online \
    wandb.group=Enhancer_Mel \
    wandb.name=gpt \
    hydra.run.dir=./outputs/enhancers_mel/gpt


