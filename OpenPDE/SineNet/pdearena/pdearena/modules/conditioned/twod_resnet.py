# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from pdearena.modules.activations import ACTIVATION_REGISTRY

from .condition_utils import ConditionedBlock, EmbedSequential, fourier_embedding
from .fourier_cond import SpectralConv2d


class FourierBasicBlock(ConditionedBlock):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        cond_channels: int,
        stride: int = 1,
        modes1: int = 16,
        modes2: int = 16,
        activation: str = "gelu",
        norm: bool = False,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")
        assert not norm
        self.fourier1 = SpectralConv2d(in_planes, planes, cond_channels, modes1=self.modes1, modes2=self.modes2)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True)
        self.fourier2 = SpectralConv2d(planes, planes, cond_channels, modes1=self.modes1, modes2=self.modes2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True)
        self.cond_emb = nn.Linear(cond_channels, planes)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        x1 = self.fourier1(x, emb)
        x2 = self.conv1(x)
        emb_out = self.cond_emb(emb)
        while len(emb_out.shape) < len(x2.shape):
            emb_out = emb_out[..., None]

        out = self.activation(x1 + x2 + emb_out)
        x1 = self.fourier2(out, emb)
        x2 = self.conv2(out)
        out = x1 + x2
        out = self.activation(out)
        return out

class DilatedBasicBlock(ConditionedBlock):
    """Basic block for Dilated ResNet

    Args:
        in_planes (int): number of input channels
        planes (int): number of output channels
        stride (int, optional): stride of the convolution. Defaults to 1.
        activation (str, optional): activation function. Defaults to "relu".
        norm (bool, optional): whether to use group normalization. Defaults to True.
        num_groups (int, optional): number of groups for group normalization. Defaults to 1.
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        cond_channels: int,
        padding_mode='zeros',
        stride: int = 1,
        activation: str = "relu",
        norm: bool = True,
        num_groups: int = 1,
    ):
        super().__init__()

        self.dilation = [1, 2, 4, 8, 4, 2, 1]
        dilation_layers = []
        cond_emb = []
        for i, dil in enumerate(self.dilation):
            if i < len(self.dilation) - 1:
                cond_emb.append(nn.Linear(cond_channels, planes))
            dilation_layers.append(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=3,
                    stride=stride,
                    dilation=dil,
                    padding=dil,
                    bias=True,
                    padding_mode=padding_mode
                )
            )
        self.dilation_layers = nn.ModuleList(dilation_layers)
        self.cond_emb = nn.ModuleList(cond_emb)
        self.norm_layers = nn.ModuleList(
            nn.GroupNorm(num_groups, num_channels=planes) if norm else nn.Identity() for dil in self.dilation
        )
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        out = x
        for i, (layer, norm) in enumerate(zip(self.dilation_layers, self.norm_layers)):                
            out = self.activation(layer(norm(out)))
            if i < len(self.cond_emb):
                emb_out = self.cond_emb[i](emb)
                while len(emb_out.shape) < len(out.shape):
                    emb_out = emb_out[..., None]
                out = out + emb_out
        return out + x

class ResNet(nn.Module):
    padding = 9

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        block: nn.Module,
        num_blocks: list,
        time_history: int,
        time_future: int,
        hidden_channels: int = 64,
        activation: str = "gelu",
        norm: bool = False,
        diffmode: bool = False,
        usegrid: bool = False,
        param_conditioning: Optional[str] = None,
    ):
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.diffmode = diffmode
        self.usegrid = usegrid
        self.in_planes = hidden_channels
        self.param_conditioning = param_conditioning
        insize = time_history * (self.n_input_scalar_components + self.n_input_vector_components * 2)
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

        time_embed_dim = hidden_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_channels, time_embed_dim),
            self.activation,
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        if self.param_conditioning is not None:
            if self.param_conditioning == "scalar":
                self.pde_emb = nn.Sequential(
                    nn.Linear(hidden_channels, time_embed_dim),
                    self.activation,
                    nn.Linear(time_embed_dim, time_embed_dim),
                )
            else:
                raise NotImplementedError(f"Param conditioning {self.param_conditioning} not implemented")

        if self.usegrid:
            insize += 2
        self.conv_in1 = nn.Conv2d(
            insize,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv2d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out1 = nn.Conv2d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv2d(
            self.in_planes,
            time_future * (self.n_output_scalar_components + self.n_output_vector_components * 2),
            kernel_size=1,
            bias=True,
        )

        self.layers = nn.ModuleList(
            [
                self._make_layer(
                    block,
                    self.in_planes,
                    time_embed_dim,
                    num_blocks[i],
                    stride=1,
                    activation=activation,
                    norm=norm,
                )
                for i in range(len(num_blocks))
            ]
        )

    def _make_layer(
        self,
        block: nn.Module,
        planes: int,
        cond_channels: int,
        num_blocks: int,
        stride: int,
        activation: str,
        norm: bool = True,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    cond_channels=cond_channels,
                    stride=stride,
                    activation=activation,
                    norm=norm,
                )
            )
            self.in_planes = planes * block.expansion
        return EmbedSequential(*layers)

    def __repr__(self):
        return "ResNet"

    def forward(self, x: torch.Tensor, time, z=None) -> torch.Tensor:
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C

        emb = self.time_embed(fourier_embedding(time, self.in_planes))
        if z is not None:
            if self.param_conditioning == "scalar":
                emb = emb + self.pde_emb(fourier_embedding(z, self.in_planes))
        # prev = x.float()
        x = self.activation(self.conv_in1(x.float()))
        x = self.activation(self.conv_in2(x.float()))

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for layer in self.layers:
            x = layer(x, emb)

        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)

        if self.diffmode:
            raise NotImplementedError("diffmode")
            # x = x + prev[:, -1:, ...].detach()
        return x.reshape(
            orig_shape[0], -1, (self.n_output_scalar_components + self.n_output_vector_components * 2), *orig_shape[3:]
        )
