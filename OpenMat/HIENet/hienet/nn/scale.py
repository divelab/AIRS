from typing import List

import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

import hienet._keys as KEY
from hienet._const import AtomGraphDataType


@compile_mode('script')
class Rescale(nn.Module):
    """
    Scaling and shifting energy (and automatically force and stress)
    """

    def __init__(
        self,
        shift: float,
        scale: float,
        data_key_in=KEY.SCALED_ATOMIC_ENERGY,
        data_key_out=KEY.ATOMIC_ENERGY,
        train_shift_scale: bool = False,
    ):
        super().__init__()
        self.shift = nn.Parameter(
            torch.FloatTensor([shift]), requires_grad=train_shift_scale
        )
        self.scale = nn.Parameter(
            torch.FloatTensor([scale]), requires_grad=train_shift_scale
        )
        self.key_input = data_key_in
        self.key_output = data_key_out

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = data[self.key_input] * self.scale + self.shift

        return data


@compile_mode('script')
class SpeciesWiseRescale(nn.Module):
    """
    Scaling and shifting energy (and automatically force and stress)
    """

    def __init__(
        self,
        shift: List[float],
        scale: List[float],
        data_key_in=KEY.SCALED_ATOMIC_ENERGY,
        data_key_out=KEY.ATOMIC_ENERGY,
        data_key_indicies=KEY.ATOM_TYPE,
        train_shift_scale: bool = False,
    ):
        super().__init__()
        self.shift = nn.Parameter(
            torch.FloatTensor(shift), requires_grad=train_shift_scale
        )
        self.scale = nn.Parameter(
            torch.FloatTensor(scale), requires_grad=train_shift_scale
        )
        self.key_input = data_key_in
        self.key_output = data_key_out
        self.key_indicies = data_key_indicies

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        indicies = data[self.key_indicies]
        data[self.key_output] = data[self.key_input] * self.scale[
            indicies
        ].view(-1, 1) + self.shift[indicies].view(-1, 1)

        return data
