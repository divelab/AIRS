from pdearena.configs.config import Config
from pdearena.configs import (
    Coal,
    Blast
)
from importlib import import_module
import os

config_prefixes = {
    Coal: "coal",
    Blast: "blast",
}
CONFIGS = dict()
CONFIG_PATH = "pdearena.configs"

def get_config_names(path: str, configs: dict):
    modules = os.listdir(path)
    for module in modules:
        module_path = os.path.join(path, module)
        if os.path.isfile(module_path):
            if module.endswith(".py"):
                if path not in configs:
                    configs[path] = []
                configs[path].append(module)
        else:
            get_config_names(module_path, configs)
    return configs

for config_class, prefix in config_prefixes.items():
    class_name = config_class.__name__.split(".")[-1]
    config_paths = get_config_names(config_class.__path__[0], dict())
    for config_path, modules in config_paths.items():
        # https://stackoverflow.com/a/34939934/10965084 (import_module)
        parent_module = CONFIG_PATH + config_path.replace(os.path.sep, ".").split(CONFIG_PATH)[-1]
        for module in modules:
            assert module.endswith(".py"), module
            config_name = module[:-3]
            if config_name != "config":
                config = import_module(f"{parent_module}.{config_name}").config
                config_name = f"{prefix}_{config_name}"
                assert config_name not in CONFIGS, config_name
                CONFIGS[config_name] = config


def config_str(config: Config):
    return "\n".join([f"{arg_name}: {arg}" for arg_name, arg in config.model_dump().items()])

def print_configs(name_only: bool=True):
    for config_name, config in CONFIGS.items():
        if name_only:
            print(config_name)
        else:
            print(f"\n\n{config_name}:\n{config_str(config)}")

def get_config(name: str, verbose: bool=True):
    if validate_config(name):
        return CONFIGS[name]
    else:
        if verbose:
            print_configs()
        raise ValueError(f"Config {name} not found. See available configs above.")

def validate_config(name: str):
    return name in CONFIGS

if __name__ == "__main__":
    print_configs(name_only=True)