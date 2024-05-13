import torch
import torch.nn as nn
import torch.nn.functional as F
from models.FNO import MLP2d, MLP3d
from utils import grid

################################################################
# 2d radial fourier layers
################################################################
class radialSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, reflection):
        super(radialSpectralConv2d, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.reflection = reflection
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.scale = 1 / (in_channels * out_channels)
        self.dtype = torch.float

        if reflection:
            # get indices of lower triangular part of a matrix that is of shape (modes x modes)
            self.inds_lower = torch.tril_indices(self.modes + 1, self.modes + 1)

            # init weights
            self.W = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.inds_lower.shape[1], dtype=self.dtype))
        else:
            # lower center component of weights; [in_channels, out_channels, modes, 1]
            self.W_LC = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes + 1, 1, dtype=self.dtype))

            # lower right component of weights; [in_channels, out_channels, modes, modes]
            self.W_LR = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, self.modes, dtype=self.dtype))
        self.eval_build = True
        self.get_weight()

    # Building the weight
    def get_weight(self):

        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return

        if self.reflection:

            # construct the lower right part of the parameter matrix; this matrix is symmetric
            W_LR = torch.zeros(self.in_channels, self.out_channels, self.modes + 1, self.modes + 1,
                               dtype=self.dtype).to(self.W.device)
            W_LR[..., self.inds_lower[0], self.inds_lower[1]] = self.W
            W_LR.transpose(-1, -2)[..., self.inds_lower[0], self.inds_lower[1]] = self.W

            # construct the right part of the parameter matrix
            self.weights = torch.cat([W_LR[..., 1:, :].flip((-2)), W_LR], dim=-2).cfloat()

        else:

            # Build the right half of the weight by first constructing the lower and upper parts of the right half
            W_LR = torch.cat([self.W_LC[:, :, 1:], self.W_LR], dim=-1)
            W_UR = torch.cat([self.W_LC.flip(-2), W_LR.rot90(dims=[-2, -1])], dim=-1)
            self.weights = torch.cat([W_UR, W_LR], dim=-2).cfloat()

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
        x_ft = x_ft[..., (freq0_y - self.modes):(freq0_y + self.modes + 1), :(self.modes + 1)]

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[..., (freq0_y - self.modes):(freq0_y + self.modes + 1), :(self.modes + 1)] = \
            self.compl_mul2d(x_ft, self.weights)

        # Return to physical space
        x = torch.fft.irfft2(torch.fft.ifftshift(out_ft, dim=-2), s=(x.size(-2), x.size(-1)))

        return x

class radialNO2d(nn.Module):
    def __init__(self, num_channels, modes, width, initial_step, reflection, grid_type):
        super(radialNO2d, self).__init__()

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
        self.act = nn.ReLU()
        self.norm = nn.InstanceNorm2d(width)

        self.modes = modes
        self.width = width

        self.grid = grid(twoD=True, grid_type=grid_type)
        self.p = nn.Linear(initial_step * num_channels + self.grid.grid_dim, self.width)  # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = radialSpectralConv2d(self.width, self.width, self.modes, reflection)
        self.conv1 = radialSpectralConv2d(self.width, self.width, self.modes, reflection)
        self.conv2 = radialSpectralConv2d(self.width, self.width, self.modes, reflection)
        self.conv3 = radialSpectralConv2d(self.width, self.width, self.modes, reflection)
        self.mlp0 = MLP2d(self.width, self.width, self.width)
        self.mlp1 = MLP2d(self.width, self.width, self.width)
        self.mlp2 = MLP2d(self.width, self.width, self.width)
        self.mlp3 = MLP2d(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP2d(self.width, num_channels, self.width * 4)  # output channel is 1: u(x, y)

    def forward(self, x):  # , grid):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], -1)
        x = self.grid(x)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.act(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.act(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.act(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x.unsqueeze(-2)

################################################################
# 3d radial fourier layers
################################################################
class radialSpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, time_modes, reflection):
        super(radialSpectralConv3d, self).__init__()
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.reflection = reflection
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.time_modes = time_modes
        self.scale = 1 / (in_channels * out_channels)
        self.dtype = torch.float

        if reflection:
            # get indices of lower triangular part of a matrix that is of shape (modes x modes)
            self.inds_lower = torch.tril_indices(self.modes, self.modes)

            # init weights
            self.W = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.inds_lower.shape[1],
                                                          self.time_modes, dtype=self.dtype))

        else:
            # lower center component of weights; [in_channels, out_channels, modes, 1, 2 * modes - 1 (time dimension)]
            self.W_LC = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, 1, self.time_modes,
                                                             dtype=self.dtype))

            # lower right component of weights; [in_channels, out_channels, modes, modes]
            self.W_LR = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes - 1, self.modes - 1,
                                                             self.time_modes, dtype=self.dtype))

        self.eval_build = True
        self.get_weight()

    # Building the weight
    def get_weight(self):

        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return

        if self.reflection:

            # construct the lower right part of the parameter matrix; this matrix is symmetric
            W_LR = torch.zeros(self.in_channels, self.out_channels, self.modes, self.modes,
                               self.time_modes, dtype=self.dtype).to(self.W.device)
            W_LR[..., self.inds_lower[0], self.inds_lower[1], :] = self.W
            W_LR.transpose(-2, -3)[..., self.inds_lower[0], self.inds_lower[1], :] = self.W

            # construct the right part of the parameter matrix
            W_R = torch.cat([W_LR[..., 1:, :, :].flip((-3)), W_LR], dim=-3)
        else:

            # Build the right half of the weight by first constructing the lower and upper parts of the right half
            W_LR = torch.cat([self.W_LC[:, :, 1:], self.W_LR], dim=-2)
            W_UR = torch.cat([self.W_LC.flip(-3), W_LR.rot90(dims=[-3, -2])], dim=-2)
            W_R = torch.cat([W_UR, W_LR], dim=-3)

        # Build the left half of the weight using the right half and combine the two pieces to get the result
        self.weights = torch.cat([W_R[..., 1:, :].flip(dims=[-3, -2]), W_R], dim=-2).cfloat()

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
               (freq0_x - self.modes + 1):(freq0_x + self.modes), :self.time_modes]

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[..., (freq0_y - self.modes + 1):(freq0_y + self.modes),
        (freq0_x - self.modes + 1):(freq0_x + self.modes), :self.time_modes] = \
            self.compl_mul3d(x_ft, self.weights)

        # Return to physical space
        x = torch.fft.irfftn(torch.fft.ifftshift(out_ft, dim=[-3, -2]), s=(x.size(-3), x.size(-2), x.size(-1)))

        return x

class radialNO3d(nn.Module):
    def __init__(self, num_channels, modes, time_modes, width, initial_step, reflection, grid_type, time_pad=False):
        super(radialNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 3 locations (u(t-10, x, y, z), ..., u(t-1, x, y, z),  x, y, z)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """
        self.act = nn.ReLU()

        self.modes = modes
        self.time_modes = time_modes
        self.width = width

        self.time_pad = time_pad
        self.padding = 6  # pad the domain if input is non-periodic

        self.grid = grid(twoD=False, grid_type=grid_type)
        self.p = nn.Linear(initial_step * num_channels + self.grid.grid_dim, self.width)  # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = radialSpectralConv3d(self.width, self.width, self.modes, self.time_modes, reflection)
        self.conv1 = radialSpectralConv3d(self.width, self.width, self.modes, self.time_modes, reflection)
        self.conv2 = radialSpectralConv3d(self.width, self.width, self.modes, self.time_modes, reflection)
        self.conv3 = radialSpectralConv3d(self.width, self.width, self.modes, self.time_modes, reflection)
        self.mlp0 = MLP3d(self.width, self.width, self.width)
        self.mlp1 = MLP3d(self.width, self.width, self.width)
        self.mlp2 = MLP3d(self.width, self.width, self.width)
        self.mlp3 = MLP3d(self.width, self.width, self.width)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.q = MLP3d(self.width, num_channels, self.width * 4)  # output channel is 1: u(x, y)

    def forward(self, x):  # , grid):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3], -1)
        x = self.grid(x)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)

        if self.time_pad:
            x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.act(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.act(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.act(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        if self.time_pad:
            x = x[..., :-self.padding]

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 3, 4, 1)
        return x