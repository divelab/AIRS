from enum import Enum
import os
from typing import Dict

import torch

import hienet._keys as KEY
from hienet.nn.activation import ShiftedSoftPlus

HIENET_VERSION = '1.0.0'


_prefix = os.path.abspath(f'{os.path.dirname(__file__)}')
HIENET_0 = f'{_prefix}/../checkpoints/HIENet-V3.pth'


IMPLEMENTED_RADIAL_BASIS = ['bessel', 'rbf', 'orbit_rbf']
IMPLEMENTED_CUTOFF_FUNCTION = ['poly_cut', 'XPLOR']
# TODO: support None. This became difficult because of paralell model
IMPLEMENTED_SELF_CONNECTION_TYPE = ['nequip', 'linear']

IMPLEMENTED_SHIFT = ['per_atom_energy_mean', 'elemwise_reference_energies']
IMPLEMENTED_SCALE = ['force_rms', 'per_atom_energy_std', 'elemwise_force_rms']

SUPPORTING_METRICS = ['RMSE', 'ComponentRMSE', 'MAE', 'Loss']
SUPPORTING_ERROR_TYPES = [
    'TotalEnergy',
    'Energy',
    'Force',
    'Stress',
    'Stress_GPa',
    'TotalLoss',
]

IMPLEMENTED_MODEL = ['E3_equivariant_model']

# string input to real torch function
ACTIVATION = {
    'relu': torch.nn.functional.relu,
    'silu': torch.nn.functional.silu,
    'tanh': torch.tanh,
    'abs': torch.abs,
    'ssp': ShiftedSoftPlus,
    'sigmoid': torch.sigmoid,
    'elu': torch.nn.functional.elu,
}
ACTIVATION_FOR_EVEN = {
    'ssp': ShiftedSoftPlus,
    'silu': torch.nn.functional.silu,
}
ACTIVATION_FOR_ODD = {'tanh': torch.tanh, 'abs': torch.abs}
ACTIVATION_DICT = {'e': ACTIVATION_FOR_EVEN, 'o': ACTIVATION_FOR_ODD}


# to avoid torch script to compile torch_geometry.data
AtomGraphDataType = Dict[str, torch.Tensor]


class LossType(Enum):
    ENERGY = 'energy'  # eV or eV/atom
    FORCE = 'force'  # eV/A
    STRESS = 'stress'  # kB


def error_record_condition(x):
    if type(x) is not list:
        return False
    for v in x:
        if type(v) is not list or len(v) != 2:
            return False
        if v[0] not in SUPPORTING_ERROR_TYPES:
            return False
        if v[0] == 'TotalLoss':
            continue
        if v[1] not in SUPPORTING_METRICS:
            print('w')
            return False
    return True


DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG = {
    KEY.IRREPS_MANUAL: False,
    KEY.INV_FEATURES: False,
    KEY.USE_COMF_EMBEDDING: False,
    KEY.NUM_INVARIANT_CONV: 2,
    KEY.SH: '1x0e + 1x1e + 1x2e',
    KEY.USE_EDGE_CONV: False,
    KEY.TRIPLET_FEATURES: 32,
    KEY.LMAX: 1,
    KEY.LMAX_EDGE: -1,  # -1 means lmax_edge = lmax
    KEY.LMAX_NODE: -1,  # -1 means lmax_node = lmax
    KEY.IS_PARITY: True,
    KEY.USE_CGCNN_EMBEDDING: False,
    KEY.RADIAL_BASIS: {
        KEY.RADIAL_BASIS_NAME: 'bessel',
    },
    KEY.CUTOFF_FUNCTION: {
        KEY.CUTOFF_FUNCTION_NAME: 'poly_cut',
    },
    KEY.ACTIVATION_RADIAL: 'silu',
    KEY.CUTOFF: 4.5,
    KEY.EMA_DECAY: 0,
    KEY.CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS: [64, 64],
    KEY.NUM_CONVOLUTION: 3,
    KEY.ACTIVATION_SCARLAR: {'e': 'silu', 'o': 'tanh'},
    KEY.ACTIVATION_GATE: {'e': 'silu', 'o': 'tanh'},
    #KEY.AVG_NUM_NEIGH: True,  # deprecated
    #KEY.TRAIN_AVG_NUM_NEIGH: False,  # deprecated
    KEY.CONV_DENOMINATOR: 'avg_num_neigh',
    KEY.TRAIN_DENOMINTAOR: False,
    KEY.TRAIN_SHIFT_SCALE: False,  
    #KEY.OPTIMIZE_BY_REDUCE: True,  # deprecated, always True
    KEY.USE_BIAS_IN_LINEAR: False,
    KEY.READOUT_AS_FCN: False,
    # Applied af readout as fcn is True
    KEY.READOUT_FCN_HIDDEN_NEURONS: [30, 30],
    KEY.READOUT_FCN_ACTIVATION: 'relu',
    KEY.SELF_CONNECTION_TYPE: 'nequip',
    KEY._NORMALIZE_SPH: True,
}


# Basically, "If provided, it should be type of ..."
MODEL_CONFIG_CONDITION = {
    KEY.USE_COMF_EMBEDDING: bool,
    KEY.LMAX: int,
    KEY.SH: str,
    KEY.LMAX_EDGE: int,
    KEY.LMAX_NODE: int,
    KEY.IS_PARITY: bool,
    KEY.TRIPLET_FEATURES: int,
    KEY.USE_EDGE_CONV: bool,
    KEY.NUM_INVARIANT_CONV: int,
    KEY.USE_CGCNN_EMBEDDING: bool,
    KEY.RADIAL_BASIS: {
        KEY.RADIAL_BASIS_NAME: lambda x: x in IMPLEMENTED_RADIAL_BASIS,
    },
    KEY.CUTOFF_FUNCTION: {
        KEY.CUTOFF_FUNCTION_NAME: lambda x: x in IMPLEMENTED_CUTOFF_FUNCTION,
    },
    KEY.CUTOFF: float,
    KEY.EMA_DECAY: float,
    KEY.NUM_CONVOLUTION: int,
    KEY.CONV_DENOMINATOR: lambda x: isinstance(x, float) or x in ["avg_num_neigh", "sqrt_avg_num_neigh"],
    KEY.CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS: list,
    KEY.TRAIN_SHIFT_SCALE: bool,
    KEY.TRAIN_DENOMINTAOR: bool,
    KEY.USE_BIAS_IN_LINEAR: bool,
    KEY.READOUT_AS_FCN: bool,
    KEY.READOUT_FCN_HIDDEN_NEURONS: list,
    KEY.READOUT_FCN_ACTIVATION: str,
    KEY.ACTIVATION_RADIAL: str,
    KEY.SELF_CONNECTION_TYPE: lambda x: x in IMPLEMENTED_SELF_CONNECTION_TYPE,
    KEY._NORMALIZE_SPH: bool,
}


def model_defaults(config):
    defaults = DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG

    if KEY.READOUT_AS_FCN not in config:
        config[KEY.READOUT_AS_FCN] = defaults[KEY.READOUT_AS_FCN]
    if config[KEY.READOUT_AS_FCN] is False:
        defaults.pop(KEY.READOUT_FCN_ACTIVATION, None)
        defaults.pop(KEY.READOUT_FCN_HIDDEN_NEURONS, None)

    return defaults



DEFAULT_DATA_CONFIG = {
    KEY.DTYPE: 'single',
    KEY.DATA_FORMAT: 'ase',
    KEY.DATA_FORMAT_ARGS: {},
    KEY.SAVE_DATASET: False,
    KEY.SAVE_BY_LABEL: False,
    KEY.SAVE_BY_TRAIN_VALID: False,
    KEY.RATIO: 0.1,
    KEY.BATCH_SIZE: 6,
    KEY.PREPROCESS_NUM_CORES: 1,
    #KEY.USE_SPECIES_WISE_SHIFT_SCALE: False,
    KEY.SHIFT: "per_atom_energy_mean",
    KEY.SCALE: "force_rms",
    KEY.DATA_SHUFFLE: True,
}

DATA_CONFIG_CONDITION = {
    KEY.DTYPE: str,
    KEY.DATA_FORMAT: str,
    KEY.DATA_FORMAT_ARGS: dict,
    KEY.SAVE_DATASET: str,
    KEY.SAVE_BY_LABEL: bool,
    KEY.SAVE_BY_TRAIN_VALID: bool,
    KEY.RATIO: float,
    KEY.BATCH_SIZE: int,
    KEY.PREPROCESS_NUM_CORES: int,
    #KEY.USE_SPECIES_WISE_SHIFT_SCALE: bool,
    KEY.SHIFT: lambda x: type(x) in [float, list] or x in IMPLEMENTED_SHIFT,
    KEY.SCALE: lambda x: type(x) in [float, list] or x in IMPLEMENTED_SCALE,
    KEY.DATA_SHUFFLE: bool,
    KEY.SAVE_DATASET: str,
}


def data_defaults(config):
    defaults = DEFAULT_DATA_CONFIG
    if KEY.LOAD_VALIDSET in config:
        defaults.pop(KEY.RATIO, None)
    return defaults


DEFAULT_TRAINING_CONFIG = {
    KEY.RANDOM_SEED: 1,
    KEY.EPOCH: 300,
    KEY.ENERGY_LOSS: 'mse',
    KEY.FORCE_LOSS: 'mse',
    KEY.STRESS_LOSS: 'mse',
    KEY.OPTIMIZER: 'adam',
    KEY.SCHEDULER: 'exponentiallr',
    KEY.DROPOUT: 0.0,
    KEY.DROPOUT_ATTN: 0.0,
    KEY.USE_DENOISING: False,
    KEY.ENERGY_WEIGHT: 1.0,
    KEY.FORCE_WEIGHT: 0.1,
    KEY.STRESS_WEIGHT: 1e-6,  # SIMPLE-NN default
    KEY.PER_EPOCH: 5,
    KEY.USE_TESTSET: False,
    KEY.USE_FULL_TRAINING: False,
    KEY.CONTINUE: {
        KEY.CHECKPOINT: False,
        KEY.RESET_OPTIMIZER: False,
        KEY.RESET_SCHEDULER: False,
        KEY.RESET_EPOCH: False,
        KEY.USE_STATISTIC_VALUES_OF_CHECKPOINT: True,
    },
    KEY.CSV_LOG: 'log_denom.csv',
    KEY.NUM_WORKERS: 0,
    KEY.IS_TRACE_STRESS: False,
    KEY.IS_TRAIN_STRESS: True,
    KEY.TRAIN_SHUFFLE: True,
    KEY.ERROR_RECORD: [
        ['Energy', 'RMSE'],
        ['Force', 'RMSE'],
        ['Stress', 'RMSE'],
        ['TotalLoss', 'None'],
    ],
    KEY.BEST_METRIC: 'TotalLoss',
}


TRAINING_CONFIG_CONDITION = {
    KEY.RANDOM_SEED: int,
    KEY.EPOCH: int,
    KEY.USE_DENOISING: bool,
    KEY.DROPOUT: float,
    KEY.DROPOUT_ATTN: float,
    KEY.ENERGY_WEIGHT: float,
    KEY.FORCE_WEIGHT: float,
    KEY.STRESS_WEIGHT: float,
    KEY.USE_FULL_TRAINING: bool,
    KEY.USE_TESTSET: None,  # Not used
    KEY.NUM_WORKERS: None,  # Not used
    KEY.PER_EPOCH: int, 
    KEY.CONTINUE: {
        KEY.CHECKPOINT: str,
        KEY.RESET_OPTIMIZER: bool,
        KEY.RESET_SCHEDULER: bool,
        KEY.RESET_EPOCH: bool,
        KEY.USE_STATISTIC_VALUES_OF_CHECKPOINT: bool,
    },
    KEY.IS_TRACE_STRESS: bool,  # Not used
    KEY.IS_TRAIN_STRESS: bool,
    KEY.TRAIN_SHUFFLE: bool,
    KEY.ERROR_RECORD: error_record_condition,
    KEY.BEST_METRIC: str,
    KEY.CSV_LOG: str,
}


def train_defaults(config):
    defaults = DEFAULT_TRAINING_CONFIG
    if KEY.IS_TRAIN_STRESS not in config:
        config[KEY.IS_TRAIN_STRESS] = defaults[KEY.IS_TRAIN_STRESS]
    if not config[KEY.IS_TRAIN_STRESS]:
        defaults.pop(KEY.STRESS_WEIGHT, None)
    return defaults
