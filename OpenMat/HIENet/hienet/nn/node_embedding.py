from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional
from ase.symbols import symbols2numbers
from e3nn.util.jit import compile_mode

import hienet._keys as KEY
from hienet._const import AtomGraphDataType
#from jarvis.core.specie import get_node_attributes, atomic_numbers_to_symbols
import numpy as np


# TODO: put this to model_build and do not preprocess data by onehot
@compile_mode('script')
class CGCNNEmbedding(nn.Module):
    """
    x : tensor of shape (N, 1)
    x_after : tensor of shape (N, num_classes)
    It overwrite data_key_x
    and saves input to data_key_save and output to data_key_additional
    I know this is strange but it is for compatibility with previous version
    and to specie wise shift scale work
    ex) [0 1 1 0] -> [[1, 0] [0, 1] [0, 1] [1, 0]] (num_classes = 2)
    """

    def __init__(
        self,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_save: str = KEY.ATOM_TYPE,
        data_key_additional: str = KEY.NODE_ATTR,  # additional output
        features="cgcnn"
    ):
        super().__init__()
        self.key_x = data_key_x
        self.key_save = data_key_save
        self.key_additional_output = data_key_additional

        element_atomic_numbers = {
            0: 89,   # Ac
            1: 47,   # Ag
            2: 13,   # Al
            3: 18,   # Ar
            4: 33,   # As
            5: 79,   # Au
            6: 5,    # B
            7: 56,   # Ba
            8: 4,    # Be
            9: 83,   # Bi
            10: 35,  # Br
            11: 6,   # C
            12: 20,  # Ca
            13: 48,  # Cd
            14: 58,  # Ce
            15: 17,  # Cl
            16: 27,  # Co
            17: 24,  # Cr
            18: 55,  # Cs
            19: 29,  # Cu
            20: 66,  # Dy
            21: 68,  # Er
            22: 63,  # Eu
            23: 9,   # F
            24: 26,  # Fe
            25: 31,  # Ga
            26: 64,  # Gd
            27: 32,  # Ge
            28: 1,   # H
            29: 2,   # He
            30: 72,  # Hf
            31: 80,  # Hg
            32: 67,  # Ho
            33: 53,  # I
            34: 49,  # In
            35: 77,  # Ir
            36: 19,  # K
            37: 36,  # Kr
            38: 57,  # La
            39: 3,   # Li
            40: 71,  # Lu
            41: 12,  # Mg
            42: 25,  # Mn
            43: 42,  # Mo
            44: 7,   # N
            45: 11,  # Na
            46: 41,  # Nb
            47: 60,  # Nd
            48: 10,  # Ne
            49: 28,  # Ni
            50: 93,  # Np
            51: 8,   # O
            52: 76,  # Os
            53: 15,  # P
            54: 91,  # Pa
            55: 82,  # Pb
            56: 46,  # Pd
            57: 61,  # Pm
            58: 59,  # Pr
            59: 78,  # Pt
            60: 94,  # Pu
            61: 37,  # Rb
            62: 75,  # Re
            63: 45,  # Rh
            64: 44,  # Ru
            65: 16,  # S
            66: 51,  # Sb
            67: 21,  # Sc
            68: 34,  # Se
            69: 14,  # Si
            70: 62,  # Sm
            71: 50,  # Sn
            72: 38,  # Sr
            73: 73,  # Ta
            74: 65,  # Tb
            75: 43,  # Tc
            76: 52,  # Te
            77: 90,  # Th
            78: 22,  # Ti
            79: 81,  # Tl
            80: 69,  # Tm
            81: 92,  # U
            82: 23,  # V
            83: 74,  # W
            84: 54,  # Xe
            85: 39,  # Y
            86: 70,  # Yb
            87: 30,  # Zn
            88: 40   # Zr
        }
        element_atomic_numbers = torch.tensor(list(element_atomic_numbers.values()))

        self.register_buffer("element_atomic_numbers", element_atomic_numbers)

        self.features = features


    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        inp = data[self.key_x]

        atomic_nums = self.element_atomic_numbers[inp]
        embd = torch.tensor(np.array([get_node_attributes(species, atom_features=self.features) for species in atomic_numbers_to_symbols(atomic_nums.tolist())]), device=inp.device)
        embd = embd.float()

        data[self.key_x] = embd
        if self.key_additional_output is not None:
            data[self.key_additional_output] = embd
        if self.key_save is not None:
            data[self.key_save] = inp
        return data


# Might be more efficient to add this to the preprocessing step
@compile_mode('script')
class OnehotEmbedding(nn.Module):
    """
    x : tensor of shape (N, 1)
    x_after : tensor of shape (N, num_classes)
    It overwrite data_key_x
    and saves input to data_key_save and output to data_key_additional
    I know this is strange but it is for compatibility with previous version
    and to specie wise shift scale work
    ex) [0 1 1 0] -> [[1, 0] [0, 1] [0, 1] [1, 0]] (num_classes = 2)
    """

    def __init__(
        self,
        num_classes: int,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_save: str = KEY.ATOM_TYPE,
        data_key_additional: str = KEY.NODE_ATTR,  # additional output
    ):
        super().__init__()
        self.num_classes = num_classes
        self.key_x = data_key_x
        self.key_save = data_key_save
        self.key_additional_output = data_key_additional

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        inp = data[self.key_x]
        embd = torch.nn.functional.one_hot(inp, self.num_classes)
        embd = embd.float()
        data[self.key_x] = embd
        if self.key_additional_output is not None:
            data[self.key_additional_output] = embd
        if self.key_save is not None:
            data[self.key_save] = inp
        return data


def get_type_mapper_from_specie(specie_list: List[str]):
    """
    from ['Hf', 'O']
    return {72: 0, 16: 1}
    """
    specie_list = sorted(specie_list)
    type_map = {}
    unique_counter = 0
    for specie in specie_list:
        atomic_num = symbols2numbers(specie)[0]
        if atomic_num in type_map:
            continue
        type_map[atomic_num] = unique_counter
        unique_counter += 1
    return type_map


# deprecated
def one_hot_atom_embedding(
    atomic_numbers: List[int], type_map: Dict[int, int]
):
    """
    atomic numbers from ase.get_atomic_numbers
    type_map from get_type_mapper_from_specie()
    """
    num_classes = len(type_map)
    try:
        type_numbers = torch.LongTensor(
            [type_map[num] for num in atomic_numbers]
        )
    except KeyError as e:
        raise ValueError(f'Atomic number {e.args[0]} is not expected')
    embd = torch.nn.functional.one_hot(type_numbers, num_classes)
    embd = embd.to(torch.get_default_dtype())

    return embd
