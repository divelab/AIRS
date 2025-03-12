import warnings
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

import hienet._keys as KEY
from hienet._const import AtomGraphDataType


@compile_mode('script')
class AtomGraphSequential(nn.Sequential):
    """
    same as nn.Sequential but with type notation
    see
    https://github.com/pytorch/pytorch/issues/52588
    """

    def __init__(
        self,
        modules: Dict[str, nn.Module],
        cutoff: float = 0.0,
        type_map: Dict[int, int] = {-1: -1},
    ):
        if type(modules) != OrderedDict:
            modules = OrderedDict(modules)
        self.cutoff = cutoff
        self.type_map = type_map
        if cutoff == 0.0:
            warnings.warn('cutoff is 0.0 or not given', UserWarning)
        if type_map == {-1: -1}:
            warnings.warn('type_map is not given', UserWarning)

        super().__init__(modules)

    def set_is_batch_data(self, flag: bool):
        # whether given data is batched or not some module have to change
        # its behavior. checking whether data is batched or not inside
        # forward function make problem harder when make it into torchscript
        for module in self:
            try:  # Easier to ask for forgiveness than permission.
                module._is_batch_data = flag
            except AttributeError:
                pass

    def get_irreps_in(self, modlue_name: str, attr_key: str = 'irreps_in'):
        tg_module = self._modules[modlue_name]
        for m in tg_module.modules():
            try:
                return repr(m.__getattribute__(attr_key))
            except AttributeError:
                pass
        return None

    def prepand_module(self, key: str, module: nn.Module):
        self._modules.update({key: module})
        self._modules.move_to_end(key, last=False)

    def replace_module(self, key: str, module: nn.Module):
        self._modules.update({key: module})

    def delete_module_by_key(self, key: str):
        if key in self._modules.keys():
            del self._modules[key]

    def to_onehot_idx(self, data: AtomGraphDataType) -> AtomGraphDataType:
        """
        User must call this function first before the forward
        if the data is not one-hot encoded
        """
        if self.type_map is {-1: -1}:
            raise ValueError('type_map is not set')
        device = data[KEY.NODE_FEATURE].device
        data[KEY.NODE_FEATURE] = torch.LongTensor(
            [self.type_map[z.item()] for z in data[KEY.NODE_FEATURE]]
        ).to(device)

        return data

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        """
        type_map is a dict of {atomic_number: one_hot_idx}
        """
        for module in self:
            data = module(data)
        return data
