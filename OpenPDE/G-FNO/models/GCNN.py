import torch.nn.functional as F
import torch
import torch.nn as nn
from models.GFNO import GConv2d, GConv3d, GMLP2d, GMLP3d, GNorm
from functools import partial
# ----------------------------------------------------------------------------------------------------------------------
# GCNN2d
# ----------------------------------------------------------------------------------------------------------------------
class GCNN2d(nn.Module):
    def __init__(self, num_channels, width, initial_step, reflection):
        super(GCNN2d, self).__init__()

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

        self.kernel_size = 3
        assert self.kernel_size % 2 == 1, "Kernel size should be odd"
        self.padding = (self.kernel_size - 1) // 2
        self.pad = partial(torch.nn.functional.pad, pad=[self.padding] * 4, mode="circular")

        self.width = width

        self.p = GConv2d(in_channels=num_channels * initial_step + 1, out_channels=self.width, kernel_size=1,
                         reflection=reflection, first_layer=True)  # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = GConv2d(in_channels=self.width, out_channels=self.width, kernel_size=self.kernel_size, reflection=reflection)
        self.conv1 = GConv2d(in_channels=self.width, out_channels=self.width, kernel_size=self.kernel_size, reflection=reflection)
        self.conv2 = GConv2d(in_channels=self.width, out_channels=self.width, kernel_size=self.kernel_size, reflection=reflection)
        self.conv3 = GConv2d(in_channels=self.width, out_channels=self.width, kernel_size=self.kernel_size, reflection=reflection)
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
        grid = self.get_grid(x.shape).to(x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = self.p(x)

        x1 = self.norm(self.conv0(self.pad(self.norm(x))))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv1(self.pad(self.norm(x))))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.pad(self.norm(x))))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv3(self.pad(self.norm(x))))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x.unsqueeze(-2)

    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x).reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y).reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        midpt = 0.5
        gridx = (gridx - midpt) ** 2
        gridy = (gridy - midpt) ** 2
        return gridx + gridy

# ----------------------------------------------------------------------------------------------------------------------
# GCNN3d
# ----------------------------------------------------------------------------------------------------------------------
class GCNN3d(nn.Module):
    def __init__(self, num_channels, width, initial_step, reflection):
        super(GCNN3d, self).__init__()

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

        self.kernel_size = 3
        assert self.kernel_size % 2 == 1, "Kernel size should be odd"
        self.padding = (self.kernel_size - 1) // 2
        self.pad = partial(torch.nn.functional.pad, pad=[0, 0] + [self.padding] * 4, mode="circular")
        self.pad0 = partial(torch.nn.functional.pad, pad=[self.padding] * 2)

        self.width = width

        self.p = GConv3d(in_channels=num_channels * initial_step + 2, out_channels=self.width, kernel_size=1,
                         kernel_size_T=1, reflection=reflection, first_layer=True)  # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = GConv3d(in_channels=self.width, out_channels=self.width, kernel_size=self.kernel_size, kernel_size_T=self.kernel_size, reflection=reflection)
        self.conv1 = GConv3d(in_channels=self.width, out_channels=self.width, kernel_size=self.kernel_size, kernel_size_T=self.kernel_size, reflection=reflection)
        self.conv2 = GConv3d(in_channels=self.width, out_channels=self.width, kernel_size=self.kernel_size, kernel_size_T=self.kernel_size, reflection=reflection)
        self.conv3 = GConv3d(in_channels=self.width, out_channels=self.width, kernel_size=self.kernel_size, kernel_size_T=self.kernel_size, reflection=reflection)
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
        grid = self.get_grid(x.shape).to(x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.p(x)

        x1 = self.conv0(self.pad0(self.pad(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(self.pad0(self.pad(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(self.pad0(self.pad(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(self.pad0(self.pad(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        return x.unsqueeze(-2)

    def get_grid(self, shape):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x).reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.linspace(0, 1, size_y).reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.linspace(0, 1, size_z).reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        midpt = 0.5
        gridx = (gridx - midpt) ** 2
        gridy = (gridy - midpt) ** 2
        return torch.cat((gridx + gridy, gridz), dim=-1)
