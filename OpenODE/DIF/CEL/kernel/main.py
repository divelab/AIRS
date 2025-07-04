import pathlib

import jax
print(jax.devices())
from CEL.utils.config_reader import config_summoner
from CEL.utils.args import args_parser
import torch
from CEL.data.data_manager import load_dataset, load_dataloader
from CEL.networks.model_manager import load_model
from CEL.experiments.exp_manager import load_experiment
from CEL.utils.register import register
from dataclasses import dataclass
from jsonargparse import set_docstring_parse_options
import jsonargparse
from jsonargparse import ArgumentParser, ActionConfigFile
from CEL.experiments.soft_intervention_exp import ExpClass
from CEL.data.data_manager import MyDataset
from CEL.definitions import ROOT_DIR
from argparse import ArgumentParser as origArgumentParser
import yaml
import wandb
from typing import List
from hashlib import sha1
import os

# import subprocess
# print(subprocess.check_output(['which', 'ptxas']))

def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 't', 'yes', 'y')
    raise ValueError("Input must be a string or a boolean")


class CEL:
    def __init__(self, exp: ExpClass, name: str = None, tags: List[str] = None):
        self.exp = exp
        self.name = name
        self.tags = tags

def dump_config(cfg):
    yaml.add_representer(pathlib.Path, lambda dumper, data: dumper.represent_scalar('tag:yaml.org,2002:str', str(data)))
    yaml.add_representer(pathlib.PosixPath,
                         lambda dumper, data: dumper.represent_scalar('tag:yaml.org,2002:str', str(data)))
    yaml.add_representer(jsonargparse._util.Path,
                         lambda dumper, data: dumper.represent_scalar('tag:yaml.org,2002:str', str(data)))
    return yaml.dump(cfg.as_dict(), sort_keys=True)



if __name__ == '__main__':
    # main()

    # --- pre parser ---
    pre_parser = origArgumentParser()
    pre_parser.add_argument('--config', type=pathlib.Path, help='Config file')
    pre_cfg, unk = pre_parser.parse_known_args()
    real_path = ROOT_DIR / pre_cfg.config
    simple_config = yaml.load(real_path.read_text(), Loader=yaml.FullLoader)
    experiment_name = simple_config['exp']['class_path']

    # --- main parser ---
    parser = ArgumentParser(parser_mode='omegaconf')
    # -- A wrapper for the experiment class: No nested key --
    parser.add_class_arguments(CEL, sub_configs=True)
    # parser.add_subclass_arguments(ExpClass, 'exp')
    if 'aphynity' in experiment_name.lower():
        parser.link_arguments('exp.dataloader.dataset', 'exp.init_args.net.init_args.dataset', apply_on='instantiate')
    # parser.link_arguments('exp.loader["train"].dataset.input_length', 'exp.init_args.model.init_args.input_length', apply_on='instantiate')
    parser.add_argument('--post_config', action=ActionConfigFile, help='Config file')
    NEW_RUN = to_bool(os.environ.get('NEW_RUN', True))
    if NEW_RUN:
        # parser.default_config_files = [str(real_path)]
        cfg = parser.parse_args(['--post_config', str(real_path)] + unk)
        # cfg = parser.parse_path(real_path)

        # cfg = parser.parse_args()
        dumped_config = dump_config(cfg)
        print(dumped_config)
        exp_hash = sha1(dumped_config.encode()).hexdigest()
        dict_cfg = cfg.as_dict()

        cfg = parser.instantiate_classes(cfg)
        if cfg.exp.wandb_logger:
            wandb.init(project='Invariant_Phy', name=eval(f"f{repr(cfg.name)}") if cfg.name is not None else None, tags=cfg.tags, config=dict_cfg | {'exp_hash': exp_hash})
        print(cfg.exp, '\nExperiment sha1:', exp_hash)
        # cfg.exp: ExpClass
        cfg.exp(exp_hash=exp_hash, dumped_config=dumped_config)
    else:
        # exp_hash = '94e86c2a86f72572b3e9a890a25cc209b29a7bcd'
        exp_hash = "c51dcc8b487e981eb8d4a1aed55c564d47df9444"
        checkpoint_path = pathlib.Path(os.environ['STORAGE_DIR']) / 'exp' / exp_hash
        config_file = checkpoint_path / 'config.yml'
        cfg = parser.parse_args(['--post_config', str(config_file)] + unk)
        dict_cfg = cfg.as_dict()
        orig_dataset_root = pathlib.Path(cfg.exp.init_args.dataloader.init_args.dataset.init_args.root)
        rel_dataset_root = pathlib.Path(*orig_dataset_root.parts[-2:])
        new_dataset_root = pathlib.Path(os.environ['STORAGE_DIR']) / rel_dataset_root
        cfg.exp.init_args.dataloader.init_args.dataset.init_args.root = str(new_dataset_root)
        cfg = parser.instantiate_classes(cfg)

        if cfg.exp.wandb_logger:
            wandb.init(project='Invariant_Phy', name=eval(f"f{repr(cfg.name)}") if cfg.name is not None else None,
                       tags=cfg.tags, config=dict_cfg | {'exp_hash': exp_hash})

        cfg.exp.explain_in_pysr(exp_hash=exp_hash, epoch=None)