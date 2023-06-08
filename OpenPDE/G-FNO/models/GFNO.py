import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from utils import grid

# ----------------------------------------------------------------------------------------------------------------------
# GFNO2d
# ----------------------------------------------------------------------------------------------------------------------
class GConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, first_layer=False, last_layer=False,
                 spectral=False, Hermitian=False, reflection=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reflection = reflection
        self.rt_group_size = 4
        self.group_size = self.rt_group_size * (1 + reflection)
        assert kernel_size % 2 == 1, "kernel size must be odd"
        dtype = torch.cfloat if spectral else torch.float
        self.kernel_size_Y = kernel_size
        self.kernel_size_X = kernel_size // 2 + 1 if Hermitian else kernel_size
        self.Hermitian = Hermitian
        if first_layer or last_layer:
            self.W = nn.Parameter(torch.empty(out_channels, 1, in_channels, self.kernel_size_Y, self.kernel_size_X, dtype=dtype))
        else:
            if self.Hermitian:
                self.W = nn.ParameterDict({
                    'y0_modes': torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, self.kernel_size_X - 1, 1, dtype=dtype)),
                    'yposx_modes': torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, self.kernel_size_Y, self.kernel_size_X - 1, dtype=dtype)),
                    '00_modes': torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, 1, 1, dtype=torch.float))
                })
            else:
                self.W = nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, self.kernel_size_Y, self.kernel_size_X, dtype=dtype))
        self.first_layer = first_layer
        self.last_layer = last_layer
        self.B = nn.Parameter(torch.empty(1, out_channels, 1, 1)) if bias else None
        self.eval_build = True
        self.reset_parameters()
        self.get_weight()

    def reset_parameters(self):
        if self.Hermitian:
            for v in self.W.values():
                nn.init.kaiming_uniform_(v, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.B is not None:
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def get_weight(self):

        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return

        if self.Hermitian:
            self.weights = torch.cat([self.W["y0_modes"], self.W["00_modes"].cfloat(), self.W["y0_modes"].flip(dims=(-2, )).conj()], dim=-2)
            self.weights = torch.cat([self.weights, self.W["yposx_modes"]], dim=-1)
            self.weights = torch.cat([self.weights[..., 1:].conj().rot90(k=2, dims=[-2, -1]), self.weights], dim=-1)
        else:
            self.weights = self.W[:]

        if self.first_layer or self.last_layer:

            # construct the weight
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1)

            # apply each of the group elements to the corresponding repetition
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k].rot90(k=k, dims=[-2, -1])

            # apply each the reflection group element to the rotated kernels
            if self.reflection:
                self.weights[:, self.rt_group_size:] = self.weights[:, :self.rt_group_size].flip(dims=[-2])

            # collapse out_channels and group1 dimensions for use with conv2d
            if self.first_layer:
                self.weights = self.weights.view(-1, self.in_channels, self.kernel_size_Y, self.kernel_size_Y)
                if self.B is not None:
                    self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)
            else:
                self.weights = self.weights.transpose(2, 1).reshape(self.out_channels, -1, self.kernel_size_Y, self.kernel_size_Y)
                self.bias = self.B

        else:

            # construct the weight
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1, 1)

            # apply elements in the rotation group
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k - 1].rot90(dims=[-2, -1])

                if self.reflection:
                    self.weights[:, k] = torch.cat([self.weights[:, k, :, self.rt_group_size - 1].unsqueeze(2),
                                                    self.weights[:, k, :, :(self.rt_group_size - 1)],
                                                    self.weights[:, k, :, (self.rt_group_size + 1):],
                                                    self.weights[:, k, :, self.rt_group_size].unsqueeze(2)], dim=2)
                else:
                    self.weights[:, k] = torch.cat([self.weights[:, k, :, -1].unsqueeze(2), self.weights[:, k, :, :-1]], dim=2)

            if self.reflection:
                # apply elements in the reflection group
                self.weights[:, self.rt_group_size:] = torch.cat(
                    [self.weights[:, :self.rt_group_size, :, self.rt_group_size:],
                     self.weights[:, :self.rt_group_size, :, :self.rt_group_size]], dim=3).flip([-2])

            # collapse out_channels / groups1 and in_channels/groups2 dimensions for use with conv2d
            self.weights = self.weights.view(self.out_channels * self.group_size, self.in_channels * self.group_size,
                                             self.kernel_size_Y, self.kernel_size_Y)
            if self.B is not None:
                self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)

        if self.Hermitian:
            self.weights = self.weights[..., -self.kernel_size_X:]

    def forward(self, x):

        self.get_weight()

        # output is of shape (batch * out_channels, number of group elements, ny, nx)
        x = nn.functional.conv2d(input=x, weight=self.weights)

        # add the bias
        if self.B is not None:
            x = x + self.bias
        return x


class GSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, reflection=False):
        super(GSpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.conv = GConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2 * modes - 1,
                            reflection=reflection, bias=False, spectral=True, Hermitian=True)
        self.get_weight()

    # Building the weight
    def get_weight(self):
        self.conv.get_weight()
        self.weights = self.conv.weights.transpose(0, 1)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # get the index of the zero frequency and construct weight
        freq0_y = (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0).nonzero().item()
        self.get_weight()

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.fftshift(torch.fft.rfft2(x), dim=-2)
        x_ft = x_ft[..., (freq0_y - self.modes + 1):(freq0_y + self.modes), :self.modes]

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.weights.shape[0], x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[..., (freq0_y - self.modes + 1):(freq0_y + self.modes), :self.modes] = \
            self.compl_mul2d(x_ft, self.weights)

        # Return to physical space
        x = torch.fft.irfft2(torch.fft.ifftshift(out_ft, dim=-2), s=(x.size(-2), x.size(-1)))

        return x

class GMLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, reflection=False, last_layer=False):
        super(GMLP2d, self).__init__()
        self.mlp1 = GConv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, reflection=reflection)
        self.mlp2 = GConv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, reflection=reflection,
                            last_layer=last_layer)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class GNorm(nn.Module):
    def __init__(self, width, group_size):
        super().__init__()
        self.group_size = group_size
        self.norm = torch.nn.InstanceNorm3d(width)

    def forward(self, x):
        x = x.view(x.shape[0], -1, self.group_size, x.shape[-2], x.shape[-1])
        x = self.norm(x)
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        return x

class GFNO2d(nn.Module):
    def __init__(self, num_channels, modes, width, initial_step, reflection, grid_type):
        super(GFNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes = modes
        self.width = width

        self.grid = grid(twoD=True, grid_type=grid_type)
        self.p = GConv2d(in_channels=num_channels * initial_step + self.grid.grid_dim, out_channels=self.width, kernel_size=1,
                         reflection=reflection, first_layer=True)  # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = GSpectralConv2d(in_channels=self.width, out_channels=self.width, modes=self.modes, reflection=reflection)
        self.conv1 = GSpectralConv2d(in_channels=self.width, out_channels=self.width, modes=self.modes, reflection=reflection)
        self.conv2 = GSpectralConv2d(in_channels=self.width, out_channels=self.width, modes=self.modes, reflection=reflection)
        self.conv3 = GSpectralConv2d(in_channels=self.width, out_channels=self.width, modes=self.modes, reflection=reflection)
        self.mlp0 = GMLP2d(in_channels=self.width, out_channels=self.width, mid_channels=self.width, reflection=reflection)
        self.mlp1 = GMLP2d(in_channels=self.width, out_channels=self.width, mid_channels=self.width, reflection=reflection)
        self.mlp2 = GMLP2d(in_channels=self.width, out_channels=self.width, mid_channels=self.width, reflection=reflection)
        self.mlp3 = GMLP2d(in_channels=self.width, out_channels=self.width, mid_channels=self.width, reflection=reflection)
        self.w0 = GConv2d(in_channels=self.width, out_channels=self.width, kernel_size=1, reflection=reflection)
        self.w1 = GConv2d(in_channels=self.width, out_channels=self.width, kernel_size=1, reflection=reflection)
        self.w2 = GConv2d(in_channels=self.width, out_channels=self.width, kernel_size=1, reflection=reflection)
        self.w3 = GConv2d(in_channels=self.width, out_channels=self.width, kernel_size=1, reflection=reflection)
        self.norm = GNorm(self.width, group_size=4 * (1 + reflection))
        self.q = GMLP2d(in_channels=self.width, out_channels=num_channels, mid_channels=self.width * 4, reflection=reflection,
                        last_layer=True)  # output channel is 1: u(x, y)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], -1)
        x = self.grid(x)
        x = x.permute(0, 3, 1, 2)
        x = self.p(x)

        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x.unsqueeze(-2)

# ----------------------------------------------------------------------------------------------------------------------
# GFNO3d
# ----------------------------------------------------------------------------------------------------------------------
class GConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_size_T, bias=True, first_layer=False, last_layer=False,
                 spectral=False, Hermitian=False, reflection=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reflection = reflection
        self.rt_group_size = 4
        self.group_size = self.rt_group_size * (1 + reflection)
        assert kernel_size % 2 == 1, "kernel size must be odd"
        dtype = torch.cfloat if spectral else torch.float
        self.kernel_size_Y = kernel_size
        self.kernel_size_X = kernel_size // 2 + 1 if Hermitian else kernel_size
        self.kernel_size_T_full = kernel_size_T
        self.kernel_size_T = kernel_size_T // 2 + 1 if Hermitian else kernel_size_T
        self.Hermitian = Hermitian
        if first_layer or last_layer:
            self.W = nn.Parameter(torch.empty(out_channels, 1, in_channels, self.kernel_size_Y, self.kernel_size_X,
                                  self.kernel_size_T, dtype=dtype))
        else:
            if self.Hermitian:
                self.W = nn.ParameterDict({
                    'y00_modes':torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size,
                                                               self.kernel_size_X - 1, 1, 1, dtype=torch.cfloat)),
                    'yposx0_modes': torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size,
                                                                   self.kernel_size_Y, self.kernel_size_X - 1, 1,
                                                                   dtype=torch.cfloat)),
                    '000_modes':torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, 1, 1, 1)),
                    'yxpost_modes':torch.nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size,
                                                                  self.kernel_size_Y, self.kernel_size_Y,
                                                                  self.kernel_size_T - 1, dtype=torch.cfloat))
                })
            else:
                self.W = nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, self.kernel_size_Y,
                                                  self.kernel_size_X, self.kernel_size_T, dtype=dtype))
        self.first_layer = first_layer
        self.last_layer = last_layer
        self.B = nn.Parameter(torch.empty(1, out_channels, 1, 1, 1)) if bias else None
        self.eval_build = True
        self.reset_parameters()
        self.get_weight()

    def reset_parameters(self):
        if self.Hermitian:
            for v in self.W.values():
                nn.init.kaiming_uniform_(v, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.B is not None:
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def get_weight(self):

        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return

        if self.Hermitian:
            self.weights = torch.cat([self.W['y00_modes'].conj().flip((-3,)), self.W["000_modes"], self.W["y00_modes"]],
                                     dim=-3)
            self.weights = torch.cat([self.W['yposx0_modes'].conj().rot90(k=2, dims=[-3, -2]), self.weights,
                                      self.W['yposx0_modes']], dim=-2)
            self.weights = torch.cat([self.W['yxpost_modes'].conj().rot90(k=2, dims=[-3, -2]).flip((-1,)), self.weights,
                                      self.W['yxpost_modes']], dim=-1)
        else:
            self.weights = self.W[:]

        if self.first_layer or self.last_layer:

            # construct the weight
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1, 1)

            # apply each of the group elements to the corresponding repetition
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k].rot90(k=k, dims=[-3, -2])

            # apply each the reflection group element to the rotated kernels
            if self.reflection:
                self.weights[:, self.rt_group_size:] = self.weights[:, :self.rt_group_size].flip(dims=[-3])

            # collapse out_channels and group1 dimensions for use with conv2d
            if self.first_layer:
                self.weights = self.weights.view(-1, self.in_channels, self.kernel_size_Y, self.kernel_size_Y,
                                                 self.kernel_size_T)
                if self.B is not None:
                    self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)
            else:
                self.weights = self.weights.transpose(2, 1).reshape(self.out_channels, -1, self.kernel_size_Y,
                                                                    self.kernel_size_Y, self.kernel_size_T)
                self.bias = self.B

        else:

            # construct the weight
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1, 1, 1)

            # apply elements in the rotation group
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k - 1].rot90(dims=[-3, -2])

                if self.reflection:
                    self.weights[:, k] = torch.cat([self.weights[:, k, :, self.rt_group_size - 1].unsqueeze(2),
                                                    self.weights[:, k, :, :(self.rt_group_size - 1)],
                                                    self.weights[:, k, :, (self.rt_group_size + 1):],
                                                    self.weights[:, k, :, self.rt_group_size].unsqueeze(2)], dim=2)
                else:
                    self.weights[:, k] = torch.cat([self.weights[:, k, :, -1].unsqueeze(2), self.weights[:, k, :, :-1]],
                                                   dim=2)

            if self.reflection:
                # apply elements in the reflection group
                self.weights[:, self.rt_group_size:] = torch.cat(
                    [self.weights[:, :self.rt_group_size, :, self.rt_group_size:],
                     self.weights[:, :self.rt_group_size, :, :self.rt_group_size]], dim=3).flip([-3])

            # collapse out_channels / groups1 and in_channels/groups2 dimensions for use with conv3d
            self.weights = self.weights.view(self.out_channels * self.group_size, self.in_channels * self.group_size,
                                             self.kernel_size_Y, self.kernel_size_Y, self.kernel_size_T_full)
            if self.B is not None:
                self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)

        if self.Hermitian:
            self.weights = self.weights[..., -self.kernel_size_T:]

    def forward(self, x):

        self.get_weight()

        # output is of shape (batch * out_channels, number of group elements, ny, nx)
        x = nn.functional.conv3d(input=x, weight=self.weights)

        # add the bias
        if self.B is not None:
            x = x + self.bias
        return x

class GSpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, time_modes, reflection):
        super(GSpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.time_modes = time_modes
        self.conv = GConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=2 * modes - 1,
                            kernel_size_T=2 * time_modes - 1, reflection=reflection, bias=False, spectral=True,
                            Hermitian=True)
        self.get_weight()

    # Building the weight
    def get_weight(self):
        self.conv.get_weight()
        self.weights = self.conv.weights.transpose(0, 1)

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # get the index of the zero frequency and construct weight
        freq0_x = (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0).nonzero().item()
        freq0_y = (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-3])) == 0).nonzero().item()
        self.get_weight()

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.fftshift(torch.fft.rfftn(x, dim=[-3, -2, -1]), dim=[-3, -2])
        x_ft = x_ft[..., (freq0_y - self.modes + 1):(freq0_y + self.modes),
               (freq0_x - self.modes + 1):(freq0_x + self.modes),
               :self.time_modes]

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.weights.shape[0], x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[..., (freq0_y - self.modes + 1):(freq0_y + self.modes),
        (freq0_x - self.modes + 1):(freq0_x + self.modes), :self.time_modes] = self.compl_mul3d(x_ft, self.weights)

        # Return to physical space
        x = torch.fft.irfftn(torch.fft.ifftshift(out_ft, dim=[-3, -2]), s=(x.size(-3), x.size(-2), x.size(-1)))

        return x

class GMLP3d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, reflection=False, last_layer=False):
        super(GMLP3d, self).__init__()
        self.mlp1 = GConv3d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, kernel_size_T=1,
                            reflection=reflection)
        self.mlp2 = GConv3d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, kernel_size_T=1,
                            reflection=reflection, last_layer=last_layer)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class GFNO3d(nn.Module):
    def __init__(self, num_channels, modes, time_modes, width, initial_step, reflection, grid_type, time_pad=False):
        super(GFNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes = modes
        self.time_modes = time_modes
        self.width = width

        self.time_pad = time_pad
        self.padding = 6  # pad the domain if input is non-periodic

        self.grid = grid(twoD=False, grid_type=grid_type)
        self.p = GConv3d(in_channels=num_channels * initial_step + self.grid.grid_dim, out_channels=self.width, kernel_size=1,
                         kernel_size_T=1, reflection=reflection, first_layer=True)  # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = GSpectralConv3d(in_channels=self.width, out_channels=self.width, modes=self.modes,
                                     time_modes=self.time_modes, reflection=reflection)
        self.conv1 = GSpectralConv3d(in_channels=self.width, out_channels=self.width, modes=self.modes,
                                     time_modes=self.time_modes, reflection=reflection)
        self.conv2 = GSpectralConv3d(in_channels=self.width, out_channels=self.width, modes=self.modes,
                                     time_modes=self.time_modes, reflection=reflection)
        self.conv3 = GSpectralConv3d(in_channels=self.width, out_channels=self.width, modes=self.modes,
                                     time_modes=self.time_modes, reflection=reflection)
        self.mlp0 = GMLP3d(in_channels=self.width, out_channels=self.width, mid_channels=self.width, reflection=reflection)
        self.mlp1 = GMLP3d(in_channels=self.width, out_channels=self.width, mid_channels=self.width, reflection=reflection)
        self.mlp2 = GMLP3d(in_channels=self.width, out_channels=self.width, mid_channels=self.width, reflection=reflection)
        self.mlp3 = GMLP3d(in_channels=self.width, out_channels=self.width, mid_channels=self.width, reflection=reflection)
        self.w0 = GConv3d(in_channels=self.width, out_channels=self.width, kernel_size=1, kernel_size_T=1, reflection=reflection)
        self.w1 = GConv3d(in_channels=self.width, out_channels=self.width, kernel_size=1, kernel_size_T=1, reflection=reflection)
        self.w2 = GConv3d(in_channels=self.width, out_channels=self.width, kernel_size=1, kernel_size_T=1, reflection=reflection)
        self.w3 = GConv3d(in_channels=self.width, out_channels=self.width, kernel_size=1, kernel_size_T=1, reflection=reflection)
        self.q = GMLP3d(in_channels=self.width, out_channels=num_channels, mid_channels=self.width * 4,
                        reflection=reflection, last_layer=True)  # output channel is 1: u(x, y)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3], -1)
        x = self.grid(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.p(x)

        if self.time_pad:
            x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        if self.time_pad:
            x = x[..., :-self.padding]

        x = self.q(x)
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        return x.unsqueeze(-2)