import torch
from torch import nn
from functools import partial
from .activations import ACTIVATION_REGISTRY


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode, num_groups=1, norm: bool = True, activation="gelu") -> None:
        super().__init__()
        self.activation = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        if norm:
            self.norm1 = nn.GroupNorm(num_groups, out_channels)
            self.norm2 = nn.GroupNorm(num_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        h = self.activation(self.norm1(self.conv1(x)))
        h = self.activation(self.norm2(self.conv2(h)))
        return h


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, in0_channels, padding_mode, num_blocks, residual, pool, avg, num_groups=1, norm: bool = True, activation="gelu", first=False, disentangle=True) -> None:
        super().__init__()
        self.channels = in_channels, out_channels
        self.residual = residual
        self.num_blocks = num_blocks
        self.first = first
        self.disentangle = disentangle
        if pool:
            self.down = nn.AvgPool2d(2) if avg else nn.MaxPool2d(2)
        else:
            raise NotImplemented
            self.down = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode)
        self.conv = nn.ModuleList()
        for block in range(num_blocks):
            in_c = (in_channels + in0_channels * (not first and disentangle)) if block == 0 else out_channels
            self.conv.append(ConvBlock(in_c, out_channels, padding_mode, num_groups, norm, activation))
        if residual:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1) # if in_channels != out_channels else nn.Identity()

    def forward(self, dx: torch.Tensor, h: torch.Tensor, down=True):
        if down:
            if self.disentangle:
                dx = self.down(dx)
            h = self.down(h)
            h0 = h.clone()
            if not self.first and self.disentangle: 
                h = torch.cat([h, dx], dim=1)
        for block in range(self.num_blocks):
            h = self.conv[block](h)
            if self.residual:
                h = h + (self.shortcut(h0) if block == 0 else h0)
                h0 = h.clone()
        return dx, h

class circular_interpolate(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
    
    def forward(self, x):
        x = torch.nn.functional.pad(x, (2, 2, 2, 2), mode='circular')
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode=self.mode)
        return x[..., 4:-4, 4:-4]
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, mult, padding_mode, num_blocks, residual, interp, interp_mode, num_groups=1, norm: bool = True, activation="gelu") -> None:
        super().__init__()
        self.channels = in_channels, out_channels
        self.residual = residual
        self.num_blocks = num_blocks
        if interp:
            in0_conv = in_channels + out_channels # int(in_channels // mult)
            in0_resid = in_channels
            self.up = circular_interpolate(mode=interp_mode) if padding_mode == "circular" else partial(torch.nn.functional.interpolate, scale_factor=2, mode=interp_mode)
        else:
            raise NotImplementedError
            in0_conv = 2 * int(in_channels // mult)
            in0_resid = int(in_channels // mult)
            self.up = nn.ConvTranspose2d(int(in_channels), int(in_channels // mult), kernel_size=2, stride=2)
        self.conv = nn.ModuleList()
        for block in range(num_blocks):
            self.conv.append(ConvBlock(in0_conv if block == 0 else out_channels, out_channels, padding_mode, num_groups, norm, activation))
        if residual:
            self.shortcut = nn.Conv2d(in0_resid, out_channels, 1) # if (in0_resid != out_channels or mult == 1) else nn.Identity()


    def forward(self, x1: torch.Tensor, x2: torch.Tensor, up=True):
        if up:
            h = self.up(x1)
            x = h.clone()
            h = torch.cat([x2, h], dim=1)
        else:
            h = x1
            x = h.clone()
        for block in range(self.num_blocks):
            h = self.conv[block](h)
            if self.residual:
                h = h + (self.shortcut(x) if block == 0 else x)
                x = h.clone()
        return h

class sinenet(nn.Module):
    """
    Args:
        n_input_scalar_components (int): Number of scalar components in the model
        n_input_vector_components (int): Number of vector components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        n_output_vector_components (int): Number of output vector components in the model
        time_history (int): Number of time steps in the input.
        time_future (int): Number of time steps in the output.
        hidden_channels (int): Number of channels in the hidden layers.
        activation (str): Activation function to use. One of ["gelu", "relu", "silu"].
    """

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        time_history: int,
        time_future: int,
        hidden_channels: int,
        padding_mode: str,
        activation="gelu",
        num_layers=4,
        num_waves=2,
        num_blocks=1,
        norm=True,
        mult=2,
        residual=True,
        wave_residual=True,
        disentangle=True,
        down_pool=True,
        avg_pool=True,
        up_interpolation=True,
        interpolation_mode='bicubic',
        par1=None
    ) -> None:
        super().__init__()
        print(padding_mode)
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.time_history = time_history
        self.time_future = time_future
        self.hidden_channels = hidden_channels
        self.activation = ACTIVATION_REGISTRY.get(activation, None)
        self.residual = wave_residual
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")
        self.num_layers = num_layers

        insize = time_history * (self.n_input_scalar_components + self.n_input_vector_components * 2)
        n_channels = hidden_channels
        self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=3, padding=1, padding_mode=padding_mode)

        self.num_waves = num_waves
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        down_args = dict(norm=norm, activation=activation, residual=residual, padding_mode=padding_mode, num_blocks=num_blocks, disentangle=disentangle, pool=down_pool, avg=avg_pool, in0_channels=n_channels)
        up_args = dict(norm=norm, activation=activation, mult=mult, residual=residual, padding_mode=padding_mode, num_blocks=num_blocks, interp=up_interpolation, interp_mode=interpolation_mode)
        for _ in range(self.num_waves):
            self.down.append(
                nn.ModuleList(
                    [
                        Down(n_channels, int(n_channels * mult), **down_args, first=True),
                        Down(int(n_channels * mult), int(n_channels * mult ** 2), **down_args),
                        Down(int(n_channels * mult ** 2), int(n_channels * mult ** 3), **down_args),
                        Down(int(n_channels * mult ** 3), int(n_channels * mult ** 4), **down_args),
                    ]))
            self.up.append(
                nn.ModuleList(
                    [
                        Up(int(n_channels * mult ** 4), int(n_channels * mult ** 3), **up_args),
                        Up(int(n_channels * mult ** 3), int(n_channels * mult ** 2), **up_args),
                        Up(int(n_channels * mult ** 2), int(n_channels * mult), **up_args),
                        Up(int(n_channels * mult), n_channels, **up_args),
                    ]))

        out_channels = time_future * (self.n_output_scalar_components + self.n_output_vector_components * 2)
        self.final = nn.Conv2d(n_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)

        par_ct = sum(par.numel() for par in self.parameters())
        print(f"# par: {par_ct}, M={mult}" + (f", diff: {par1 - par_ct}" if par1 else ""))
        print("Channels: " + "->".join([str(ch) for ch in [n_channels, int(n_channels * mult), int(n_channels * mult ** 2), int(n_channels * mult ** 3), int(n_channels * mult ** 4)]]))

    def forward(self, x):
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])
        x = self.image_proj(x)

        for stack in range(self.num_waves):
            x0 = x.clone()
            xs = [x]
            dx = x
            for i in range(self.num_layers):
                dx, h = self.down[stack][i](dx, xs[-1])
                xs.append(h)
            x = xs.pop(-1)
            for i in range(self.num_layers):
                x = self.up[stack][i](x, xs.pop(-1))
            if self.residual:
                x = x0 + x

        x = self.final(x)
        return x.reshape(
            orig_shape[0], -1, (self.n_output_scalar_components + self.n_output_vector_components * 2), *orig_shape[3:]
        )