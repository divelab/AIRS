# gpt base
python -m train \
    experiment=hg38/enhancer_flybrain \
    wandb.mode=online \
    wandb.group=Enhancer_Fb \
    wandb.name=gpt \
    hydra.run.dir=./outputs/enhancers_fb/gpt


