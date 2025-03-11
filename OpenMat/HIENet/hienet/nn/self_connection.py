import torch.nn as nn
from e3nn.o3 import FullyConnectedTensorProduct, Irreps, Linear
from e3nn.util.jit import compile_mode

import hienet._keys as KEY
from hienet._const import AtomGraphDataType


@compile_mode('script')
class SelfConnectionIntro(nn.Module):
    """
    do TensorProduct of x and some data(here attribute of x)
    and save it (to concatenate updated x at SelfConnectionOutro)
    """

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_operand: Irreps,
        irreps_out: Irreps,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_operand: str = KEY.NODE_ATTR,
        **kwargs,  # for compatibility
    ):
        super().__init__()

        self.fc_tensor_product = FullyConnectedTensorProduct(
            irreps_x, irreps_operand, irreps_out
        )
        self.key_x = data_key_x
        self.key_operand = data_key_operand

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[KEY.SELF_CONNECTION_TEMP] = self.fc_tensor_product(
            data[self.key_x], data[self.key_operand]
        )
        return data


@compile_mode('script')
class SelfConnectionLinearIntro(nn.Module):
    """
    Linear style self connection update
    """

    def __init__(
        self,
        irreps_x: Irreps,
        irreps_out: Irreps,
        data_key_x: str = KEY.NODE_FEATURE,
        **kwargs,  # for compatibility
    ):
        super().__init__()
        self.linear = Linear(irreps_x, irreps_out)
        self.key_x = data_key_x

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[KEY.SELF_CONNECTION_TEMP] = self.linear(data[self.key_x])
        return data


@compile_mode('script')
class SelfConnectionOutro(nn.Module):
    """
    do TensorProduct of x and some data(here attribute of x)
    and save it (to concatenate updated x at SelfConnectionOutro)
    """

    def __init__(
        self,
        data_key_x: str = KEY.NODE_FEATURE,
    ):
        super().__init__()
        self.key_x = data_key_x

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_x] = data[self.key_x] + data[KEY.SELF_CONNECTION_TEMP]
        del data[KEY.SELF_CONNECTION_TEMP]
        return data
