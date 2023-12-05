import torch.nn.functional as F
import torch
import torch.nn as nn
import math

# ----------------------------------------------------------------------------------------------------------------------
# Baseline FNO: code from https://github.com/neural-operator/fourier_neural_operator (master branch)
# ----------------------------------------------------------------------------------------------------------------------
################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        # self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, 2 * self.modes - 1, self.modes, dtype=torch.cfloat))

        # weightsR = torch.empty(in_channels, out_channels, 2 * self.modes - 1, self.modes)
        # weightsC = torch.empty(in_channels, out_channels, 2 * self.modes - 1, self.modes)
        # nn.init.kaiming_uniform_(weightsR, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(weightsC, a=math.sqrt(5))
        # weights = weightsR + 1j * weightsC
        weights = self.scale * torch.rand(in_channels, out_channels, 2 * self.modes - 1, self.modes, dtype=torch.cfloat)
        self.weights = nn.Parameter(weights)

        # for modes in range(1, 1001):
        #     assert len(torch.fft.fftfreq(modes)) == (2 * (len(torch.fft.rfftfreq(modes))) - (1 + ((modes % 2) == 0)))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # get the index of the zero frequency
        modes = torch.fft.fftshift(torch.fft.fftfreq(x.shape[-1]))
        freq0 = (modes == 0).nonzero().item()
        assert freq0 == x.shape[-1] // 2
        assert modes[freq0] == 0
        assert modes[(freq0 - self.modes + 1):(freq0 + self.modes)][[0,-1]] @ torch.ones(2) == 0, "frequencies don't match"

        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.fftshift(torch.fft.rfft2(x), dim=-2)
        out_ft = torch.zeros_like(x_ft)

        # remove high-frequency modes
        x_ft = x_ft[..., (freq0 - self.modes + 1):(freq0 + self.modes), :self.modes]

        # Multiply relevant Fourier modes
        out_ft[..., (freq0 - self.modes + 1):(freq0 + self.modes), :self.modes] = self.compl_mul2d(x_ft, self.weights)

        # Return to physical space
        x = torch.fft.irfft2(torch.fft.ifftshift(out_ft, dim=-2), s=(x.shape[-2], x.shape[-2]))

        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes, width, initial_step, nlayers, scale, residual, upsample='bicubic'):
        super(FNO2d, self).__init__()

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

        self.residual = residual

        self.nlayers = nlayers

        self.scale = scale
        self.initial_step = initial_step

        self.upsampler = nn.Upsample(scale_factor=scale, mode=upsample)
        self.p = nn.Linear(initial_step + 2, self.width)  # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.convs = torch.nn.ModuleList([])
        self.mlps = torch.nn.ModuleList([])
        self.w = torch.nn.ModuleList([])
        for _ in range(self.nlayers):
            self.convs.append(SpectralConv2d(self.width, self.width, self.modes))
            self.mlps.append(MLP(self.width, self.width, self.width))
            self.w.append(nn.Conv2d(self.width, self.width, 1))
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, initial_step, self.width * 4)  # output channel is 1: u(x, y)

    def forward(self, x):

        in_shape = x.shape # B, T, S, S
        assert in_shape[1] == self.initial_step, "Wrong input shape"

        x_int = self.upsampler(x)
        x = x_int.clone()
        x = x.permute(0, 2, 3, 1) # B, S, S, T

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2) # B, T, S, S
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        for layer, (conv, mlp, w) in enumerate(zip(self.convs, self.mlps, self.w)):
            x1 = self.norm(conv(self.norm(x)))
            x1 = mlp(x1)
            x2 = w(x)
            x = x1 + x2
            if layer < self.nlayers - 1:
                x = F.gelu(x)

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        if self.residual:
            x = x + x_int

        assert x.shape[1] == self.initial_step and x.shape[2] == self.scale * in_shape[2] and x.shape[3] == self.scale * in_shape[3], "Wrong output shape"
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x).reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y).reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)