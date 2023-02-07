"""Pydantic model for default configuration and validation."""
"""Implementation based on the template of ALIGNN."""

import subprocess
from typing import Optional, Union
import os
from pydantic import root_validator

# vfrom pydantic import Field, root_validator, validator
from pydantic.typing import Literal
from matformer.utils import BaseSettings
from matformer.models.pyg_att import MatformerConfig

# from typing import List

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
    "slme",
    "magmom_oszicar",
    "spillage",
    "kpoint_length_unit",
    "encut",
    "optb88vdw_total_energy",
    "epsx",
    "epsy",
    "epsz",
    "mepsx",
    "mepsy",
    "mepsz",
    "max_ir_mode",
    "min_ir_mode",
    "n-Seebeck",
    "p-Seebeck",
    "n-powerfact",
    "p-powerfact",
    "ncond",
    "pcond",
    "nkappa",
    "pkappa",
    "ehull",
    "exfoliation_energy",
    "dfpt_piezo_max_dielectric",
    "dfpt_piezo_max_eij",
    "dfpt_piezo_max_dij",
    "gap pbe",
    "e_form",
    "e_hull",
    "energy_per_atom",
    "formation_energy_per_atom",
    "band_gap",
    "e_above_hull",
    "mu_b",
    "bulk modulus",
    "shear modulus",
    "elastic anisotropy",
    "U0",
    "HOMO",
    "LUMO",
    "R2",
    "ZPVE",
    "omega1",
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U",
    "H",
    "G",
    "Cv",
    "A",
    "B",
    "C",
    "all",
    "target",
    "max_efg",
    "avg_elec_mass",
    "avg_hole_mass",
    "_oqmd_band_gap",
    "_oqmd_delta_e",
    "_oqmd_stability",
    "edos_up",
    "pdos_elast",
    "bandgap",
    "energy_total",
    "net_magmom",
    "b3lyp_homo",
    "b3lyp_lumo",
    "b3lyp_gap",
    "b3lyp_scharber_pce",
    "b3lyp_scharber_voc",
    "b3lyp_scharber_jsc",
    "log_kd_ki",
    "max_co2_adsp",
    "min_co2_adsp",
    "lcd",
    "pld",
    "void_fraction",
    "surface_area_m2g",
    "surface_area_m2cm3",
    "indir_gap",
    "f_enp",
    "final_energy",
    "energy_per_atom",
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
    neighbor_strategy: Literal["k-nearest", "voronoi", "pairwise-k-nearest"] = "k-nearest"
    id_tag: Literal["jid", "id", "_oqmd_entry_id"] = "jid"

    # logging configuration

    # training configuration
    random_seed: Optional[int] = 123
    classification_threshold: Optional[float] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    n_train: Optional[int] = None
    train_ratio: Optional[float] = 0.8
    val_ratio: Optional[float] = 0.1
    test_ratio: Optional[float] = 0.1
    target_multiplication_factor: Optional[float] = None
    epochs: int = 300
    batch_size: int = 64
    weight_decay: float = 0
    learning_rate: float = 1e-2
    filename: str = "sample"
    warmup_steps: int = 2000
    criterion: Literal["mse", "l1", "poisson", "zig"] = "mse"
    optimizer: Literal["adamw", "sgd"] = "adamw"
    scheduler: Literal["onecycle", "none", "step"] = "onecycle"
    pin_memory: bool = False
    save_dataloader: bool = False
    write_checkpoint: bool = True
    write_predictions: bool = True
    store_outputs: bool = True
    progress: bool = True
    log_tensorboard: bool = False
    standard_scalar_and_pca: bool = False
    use_canonize: bool = True
    num_workers: int = 2
    cutoff: float = 8.0
    max_neighbors: int = 12
    keep_data_order: bool = False
    distributed: bool = False
    n_early_stopping: Optional[int] = None  # typically 50
    output_dir: str = os.path.abspath(".")  # typically 50
    matrix_input: bool = False
    pyg_input: bool = False
    use_lattice: bool = False
    use_angle: bool = False

    # model configuration
    model = MatformerConfig(name="matformer")
    print(model)
    @root_validator()
    def set_input_size(cls, values):
        """Automatically configure node feature dimensionality."""
        values["model"].atom_input_features = FEATURESET_SIZE[
            values["atom_features"]
        ]

        return values
