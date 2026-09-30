import os
import signal
from pathlib import Path
from typing import Optional

import dotenv
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf


# This is the function that will be called when the timeout happens
def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")


# This is a decorator to add timeout functionality
def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set the signal handler and a 5-second alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)

            try:
                # This try/except clause is used to catch the TimeoutError
                # and to turn off the alarm in case the function finished
                # before the timeout
                result = func(*args, **kwargs)
            except TimeoutError as te:
                print(te, seconds)
                return None
            finally:
                # Disable the alarm
                signal.alarm(0)
            return result

        return wrapper

    return decorator


def get_env(env_name: str, default: Optional[str] = None) -> str:
    """
    Safely read an environment variable.
    Raises errors if it is not defined or it is empty.

    :param env_name: the name of the environment variable
    :param default: the default (optional) value for the environment variable

    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        if default is None:
            raise KeyError(f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value


def _resolve_dotenv_path(env_file: Optional[str]) -> Path:
    """Resolve `.env` for this repo; avoid python-dotenv walking into a sibling checkout."""
    if env_file is not None and env_file != ".env":
        return Path(env_file)

    for key in ("PROJECT_ROOT", "REPO_ROOT"):
        root = os.environ.get(key)
        if root:
            candidate = Path(root) / ".env"
            if candidate.is_file():
                return candidate

    # Package lives at <repo>/scdp/common/system.py
    candidate = Path(__file__).resolve().parents[2] / ".env"
    if candidate.is_file():
        return candidate

    found = dotenv.find_dotenv(".env", raise_error_if_not_found=True)
    return Path(found)


def load_envs(env_file: Optional[str] = ".env") -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    Prefers ``PROJECT_ROOT`` / ``REPO_ROOT`` already set in the environment (``.env``
    or the Slurm recipe) so a second editable install does not override paths via
    find_dotenv.
    """
    env_path = _resolve_dotenv_path(env_file)
    print(env_path)
    # Keep variables already exported by the shell / Slurm script.
    dotenv.load_dotenv(dotenv_path=env_path, override=False)


STATS_KEY: str = "stats"

# Adapted from https://github.com/hobogalaxy/lightning-hydra-template/blob/6bf03035107e12568e3e576e82f83da0f91d6a11/src/utils/template_utils.py#L125
def log_hyperparameters(
    cfg: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionally saves:
        - sizes of train, val, test dataset
        - number of trainable model parameters
    Args:
        cfg (DictConfig): [description]
        model (pl.LightningModule): [description]
        trainer (pl.Trainer): [description]
    """
    hparams = OmegaConf.to_container(cfg, resolve=True)

    # save number of model parameters
    hparams[f"{STATS_KEY}/params_total"] = sum(p.numel() for p in model.parameters())
    hparams[f"{STATS_KEY}/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams[f"{STATS_KEY}/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # DDP non-zero ranks use logger=False (see train.py); skip hyperparam logging there.
    logger = trainer.logger
    if logger is None or logger is False:
        return

    logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
    logger.log_hyperparams = lambda params: None


# Load environment variables
load_envs()

# Set the cwd to the project root
PROJECT_ROOT: Path = Path(get_env("PROJECT_ROOT"))
# DATA_DIR: Path = Path(get_env("DATA_DIR"))
if PROJECT_ROOT.exists():
    os.chdir(PROJECT_ROOT)
