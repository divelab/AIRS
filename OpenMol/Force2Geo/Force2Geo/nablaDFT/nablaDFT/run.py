"""Usage: python run.py --config-name path-to-config"""

import hydra
from nablaDFT.pipelines import run
from nablaDFT.utils import init_wandb, load_envs
from omegaconf import DictConfig


@hydra.main(config_path="./config", config_name=None, version_base="1.2")
def main(config: DictConfig):
    load_envs()
    init_wandb()
    run(config)


if __name__ == "__main__":
    main()
