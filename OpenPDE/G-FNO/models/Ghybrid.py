import torch.nn

from models.GFNO import *
from models.FNO import *
import torch.nn as nn

# ----------------------------------------------------------------------------------------------------------------------
# Ghybrid2d
# ----------------------------------------------------------------------------------------------------------------------
class Ghybrid2d(nn.Module):
    def __init__(self, num_channels, modes, Gwidth, width, initial_step, reflection, n_equiv):
        super(Ghybrid2d, self).__init__()
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
        self.Gwidth = Gwidth
        self.width = width

        self.n_equiv = n_equiv
        self.rt_group_size = 4
        self.group_size = self.rt_group_size * (1 + reflection)

        self.p = GConv2d(in_channels=num_channels * initial_step + 1, out_channels=self.Gwidth, kernel_size=1,
                         reflection=reflection, first_layer=True)  # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        assert n_equiv in [1, 2, 3], "Number of equivariant layers should be 1, 2, or 3"

        self.spectral_convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.convs = nn.ModuleList()

        for layer in range(n_equiv):
            self.spectral_convs.append(GSpectralConv2d(in_channels=self.Gwidth, out_channels=self.Gwidth, modes=self.modes, reflection=reflection))
            self.mlps.append(GMLP2d(in_channels=self.Gwidth, out_channels=self.Gwidth, mid_channels=self.Gwidth, reflection=reflection))
            self.convs.append(GConv2d(in_channels=self.Gwidth, out_channels=self.Gwidth, kernel_size=1, reflection=reflection))

        for layer in range(4 - n_equiv):
            in_width = self.Gwidth * self.group_size if layer == 0 else self.width
            self.spectral_convs.append(SpectralConv2d(in_width, self.width, self.modes, self.modes))
            self.mlps.append(MLP2d(self.width, self.width, self.width))
            self.convs.append(nn.Conv2d(in_width, self.width, 1))

        self.Gnorm = GNorm(self.Gwidth, group_size=4 * (1 + reflection))
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP2d(self.width, num_channels, self.width * 4)  # output channel is 1: u(x, y)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], -1)
        grid = self.get_grid(x.shape).to(x.device)
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = self.p(x)
        norm = self.Gnorm

        for layer in range(4):
            x1 = norm(self.spectral_convs[layer](norm(x)))
            x1 = self.mlps[layer](x1)
            x2 = self.convs[layer](x)
            x = x1 + x2
            if layer < 3:
                x = F.gelu(x)
            if layer == self.n_equiv - 1:
                norm = self.norm

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