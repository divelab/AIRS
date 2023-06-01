"""Pydantic model for default configuration and validation."""

import subprocess
from typing import Optional, Union
import os
from pydantic import root_validator

# vfrom pydantic import Field, root_validator, validator
from pydantic.typing import Literal

# from typing import List
from models.base import BaseSettings
from models.potnet import PotNetConfig

try:
    VERSION = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    )
except Exception as exp:
    VERSION = "NA"
    pass


FEATURESET_SIZE = {"basic": 11, "atomic_number": 1, "cfid": 438, "cgcnn": 92}


TARGET_ENUM = Literal[
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "mbj_bandgap",
    "optb88vdw_total_energy",
    "ehull",
    "gap pbe",
    "e_form",
    "e_hull",
    "formation_energy_per_atom",
    "band_gap",
    "bulk modulus",
    "shear modulus",
    "energy_per_atom",
    "target",
]


class TrainingConfig(BaseSettings):
    """Training config defaults and validation."""

    version: str = VERSION

    # dataset configuration
    dataset: Literal[
        "dft_3d",
        "megnet",
    ] = "dft_3d"
    target: TARGET_ENUM = "formation_energy_peratom"
    atom_features: Literal["basic", "atomic_number", "cfid", "cgcnn"] = "cgcnn"
    id_tag: Literal["jid", "id"] = "jid"

    # logging configuration

    # training configuration
    random_seed: Optional[int] = 123
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    n_train: Optional[int] = None
    train_ratio: Optional[float] = 0.8
    val_ratio: Optional[float] = 0.1
    test_ratio: Optional[float] = 0.1
    epochs: int = 500
    batch_size: int = 64
    weight_decay: float = 0.0
    learning_rate: float = 1e-3
    warmup_steps: int = 2000
    criterion: Literal["mse", "l1", "poisson"] = "mse"
    optimizer: Literal["adamw", "sgd"] = "adamw"
    scheduler: Literal["onecycle", "step", "none"] = "onecycle"
    pin_memory: bool = False
    write_checkpoint: bool = True
    write_predictions: bool = True
    store_outputs: bool = True
    progress: bool = True
    log_tensorboard: bool = False
    num_workers: int = 8
    normalize: bool = False
    euclidean: bool = False
    keep_data_order: bool = False
    cutoff: float = 8.0
    max_neighbors: int = 12
    infinite_funcs = ["zeta", "zeta", "exp"]
    infinite_params = [3.0, 0.5, 3.0]
    R: int = 5
    n_early_stopping: Optional[int] = None
    output_dir: str = ""
    process_dir: str = "processed"
    cache_dir: str = "cache"
    checkpoint_dir: str = "checkpoints"

    # model configuration
    model: Union[
        PotNetConfig,
    ] = PotNetConfig(name="potnet")

    @root_validator()
    def set_input_size(cls, values):
        """Automatically configure node feature dimensionality."""
        print(values)
        values["model"].atom_input_features = FEATURESET_SIZE[
            values["atom_features"]
        ]

        return values