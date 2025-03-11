import torch
import torch.nn as nn
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps, Linear
from e3nn.util.jit import compile_mode
from torch_scatter import scatter

import hienet._keys as KEY
from hienet._const import AtomGraphDataType


@compile_mode('script')
class IrrepsLinear(nn.Module):
    """
    wrapper class of e3nn Linear to operate on AtomGraphData
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        data_key_in: str,
        data_key_out: str = None,
        **e3nn_linear_params,
    ):
        super().__init__()
        self.key_input = data_key_in
        if data_key_out is None:
            self.key_output = data_key_in
        else:
            self.key_output = data_key_out

        self.linear = Linear(irreps_in, irreps_out, **e3nn_linear_params)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = self.linear(data[self.key_input])
        return data


class DynamicIrrepsLinear(nn.Module):
    def __init__(self, in_irreps, out_irreps, use_dynamic_irreps=True, **kwargs):
        super().__init__()

        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.use_dynamic_irreps = use_dynamic_irreps

        self.linear = eComfIrrepsLinear(self.in_irreps, self.out_irreps, **kwargs)
        
        if self.use_dynamic_irreps:
            self.gate = nn.Parameter(torch.ones(self.out_irreps.dim))
            self.conditioning_layer = nn.Linear(self.in_irreps.dim, self.out_irreps.dim)
    
    def forward(self, x):
        out = self.linear(x)
        
        if self.use_dynamic_irreps:
            # Dynamically adjust the gates based on input features
            dynamic_gate = torch.sigmoid(self.conditioning_layer(x.mean(dim=0)))  # Example conditioning
            out = out * (self.gate * dynamic_gate)
        
        return out


@compile_mode('script')
class eComfIrrepsLinear(nn.Module):
    """
    wrapper class of e3nn Linear to operate on AtomGraphData
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        **e3nn_linear_params,
    ):
        super().__init__()

        self.linear = Linear(irreps_in, irreps_out, **e3nn_linear_params)

    def forward(self, features_in):
        return self.linear(features_in)


@compile_mode('script')
class AtomReduce(nn.Module):
    """
    atomic energy -> total energy
    constant is multiplied to data
    """

    def __init__(
        self,
        data_key_in: str,
        data_key_out: str,
        reduce='sum',
        constant: float = 1.0,
    ):
        super().__init__()

        self.key_input = data_key_in
        self.key_output = data_key_out
        self.constant = constant
        self.reduce = reduce

        # controlled by the upper most wrapper 'AtomGraphSequential'
        self._is_batch_data = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if self._is_batch_data:
            data[self.key_output] = (
                scatter(
                    data[self.key_input],
                    data[KEY.BATCH],
                    dim=0,
                    reduce=self.reduce,
                )
                * self.constant
            )
            data[self.key_output] = data[self.key_output].squeeze(1)
        else:
            data[self.key_output] = (
                torch.sum(data[self.key_input]) * self.constant
            )

        return data


@compile_mode('script')
class FCN_e3nn(nn.Module):
    """
    wrapper class of e3nn FullyConnectedNet
    """

    def __init__(
        self,
        irreps_in: Irreps,  # confirm it is scalar & input size
        dim_out: int,
        hidden_neurons,
        activation,
        data_key_in: str,
        data_key_out: str = None,
        **e3nn_params,
    ):
        super().__init__()
        self.key_input = data_key_in
        self.irreps_in = irreps_in
        if data_key_out is None:
            self.key_output = data_key_in
        else:
            self.key_output = data_key_out

        for _, irrep in irreps_in:
            assert irrep.is_scalar()
        inp_dim = irreps_in.dim

        self.fcn = FullyConnectedNet(
            [inp_dim] + hidden_neurons + [dim_out],
            activation,
            **e3nn_params,
        )

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = self.fcn(data[self.key_input])
        return data


def get_linear(
    in_features,
    out_features,
    activation = None,
    dropout = 0.0,
    **e3nn_linear_params
):
    """
    Build a linear layer with optional activation and dropout.
    """
    layers = [Linear(in_features, out_features, **e3nn_linear_params)]
    if activation:
        layers.append(build_activation(activation))
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)



class IrrepsDropoutLinear(nn.Module):
    """
    wrapper class of e3nn Linear to operate on AtomGraphData
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        data_key_in: str,

        data_key_out: str = None,
        activation = None,
        dropout = 0.0,
        **e3nn_linear_params,
    ):
        super().__init__()
        self.key_input = data_key_in
        if data_key_out is None:
            self.key_output = data_key_in
        else:
            self.key_output = data_key_out

        self.linear = get_linear(
            irreps_in, irreps_out, activation, dropout, **e3nn_linear_params
        )

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.key_output] = self.linear(data[self.key_input])
        return data