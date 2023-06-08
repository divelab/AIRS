import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import escnn
import timeit

# ----------------------------------------------------------------------------------------------------------------------
# GFNO2d
# ----------------------------------------------------------------------------------------------------------------------
class GSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, basis, reflection=False):
        super(GSpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.n_modes = 2 * self.modes - 1
        self.group_size = (1 + reflection) * 4
        self.basis = basis

        self.coef = nn.Parameter(torch.empty(len(self.basis), out_channels, 1, in_channels, 1, 1, 1))
        nn.init.kaiming_uniform_(self.coef, a=math.sqrt(5))

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

        if self.coef.device != self.basis.device:
            self.basis = self.basis.to(self.coef.device)

        self.weights = torch.empty(self.out_channels, self.group_size, self.in_channels, self.group_size, self.n_modes, self.modes, dtype=torch.cfloat).to(self.coef.device)

        # in and out channel loop
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                self.weights[i, :, j] = (self.coef[:, i, :, j] * self.basis).sum(0)

        self.weights = self.weights.view(self.group_size * self.out_channels, self.group_size * self.in_channels, self.n_modes, self.modes)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (out_channel, in_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,oixy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # get the index of the zero frequency and construct weight
        freq0_y = (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0).nonzero().item()
        self.get_weight()
        weights = nn.Parameter(self.weights)

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.fftshift(torch.fft.rfft2(x), dim=-2)
        x_ft = x_ft[..., (freq0_y - self.modes + 1):(freq0_y + self.modes), :self.modes]

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.weights.shape[0], x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[..., (freq0_y - self.modes + 1):(freq0_y + self.modes), :self.modes] = \
            self.compl_mul2d(x_ft, weights)

        # Return to physical space
        x = torch.fft.irfft2(torch.fft.ifftshift(out_ft, dim=-2), s=(x.size(-2), x.size(-1)))

        return x

class GMLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, reflection=False, last_layer=False):
        super(GMLP2d, self).__init__()
        r2_act = escnn.gspaces.flipRot2dOnR2(N=4) if reflection else escnn.gspaces.rot2dOnR2(N=4)
        out_rep = r2_act.trivial_repr if last_layer else r2_act.regular_repr
        self.feat_type_in = escnn.nn.FieldType(r2_act, in_channels * [r2_act.regular_repr])
        self.feat_type_mid = escnn.nn.FieldType(r2_act, mid_channels * [r2_act.regular_repr])
        self.feat_type_out = escnn.nn.FieldType(r2_act, out_channels * [out_rep])
        self.mlp1 = escnn.nn.R2Conv(self.feat_type_in, self.feat_type_mid, kernel_size=1)
        self.mlp2 = escnn.nn.R2Conv(self.feat_type_mid, self.feat_type_out, kernel_size=1)

    def forward(self, x):
        x = escnn.nn.GeometricTensor(x, self.feat_type_in)
        x = self.mlp1(x).tensor
        x = F.gelu(x)
        x = escnn.nn.GeometricTensor(x, self.feat_type_mid)
        x = self.mlp2(x).tensor
        return x

class conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, reflection=False, first_layer=False):
        super(conv1x1, self).__init__()
        r2_act = escnn.gspaces.flipRot2dOnR2(N=4) if reflection else escnn.gspaces.rot2dOnR2(N=4)
        in_rep = r2_act.trivial_repr if first_layer else r2_act.regular_repr
        self.feat_type_in = escnn.nn.FieldType(r2_act, in_channels * [in_rep])
        self.feat_type_out = escnn.nn.FieldType(r2_act, out_channels * [r2_act.regular_repr])
        self.mlp1 = escnn.nn.R2Conv(self.feat_type_in, self.feat_type_out, kernel_size=1)

    def forward(self, x):
        x = escnn.nn.GeometricTensor(x, self.feat_type_in)
        x = self.mlp1(x).tensor
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

class GFNO2d_steer(nn.Module):
    def __init__(self, num_channels, modes, width, initial_step, reflection, input_size=None):
        super(GFNO2d_steer, self).__init__()

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
        input_size = input_size if input_size else self.modes
        self.get_basis(input_size=input_size, reflection=reflection)

        self.p = conv1x1(in_channels=num_channels * initial_step + 1, out_channels=self.width,
                         reflection=reflection, first_layer=True)  # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = GSpectralConv2d(in_channels=self.width, out_channels=self.width, modes=self.modes, basis=self.basis, reflection=reflection)
        self.conv1 = GSpectralConv2d(in_channels=self.width, out_channels=self.width, modes=self.modes, basis=self.basis, reflection=reflection)
        self.conv2 = GSpectralConv2d(in_channels=self.width, out_channels=self.width, modes=self.modes, basis=self.basis, reflection=reflection)
        self.conv3 = GSpectralConv2d(in_channels=self.width, out_channels=self.width, modes=self.modes, basis=self.basis, reflection=reflection)
        self.mlp0 = GMLP2d(in_channels=self.width, out_channels=self.width, mid_channels=self.width, reflection=reflection)
        self.mlp1 = GMLP2d(in_channels=self.width, out_channels=self.width, mid_channels=self.width, reflection=reflection)
        self.mlp2 = GMLP2d(in_channels=self.width, out_channels=self.width, mid_channels=self.width, reflection=reflection)
        self.mlp3 = GMLP2d(in_channels=self.width, out_channels=self.width, mid_channels=self.width, reflection=reflection)
        self.w0 = conv1x1(in_channels=self.width, out_channels=self.width, reflection=reflection)
        self.w1 = conv1x1(in_channels=self.width, out_channels=self.width, reflection=reflection)
        self.w2 = conv1x1(in_channels=self.width, out_channels=self.width, reflection=reflection)
        self.w3 = conv1x1(in_channels=self.width, out_channels=self.width, reflection=reflection)
        self.norm = GNorm(self.width, group_size=4 * (1 + reflection))
        self.q = GMLP2d(in_channels=self.width, out_channels=num_channels, mid_channels=self.width * 4,
                        reflection=reflection, last_layer=True) # output channel is 1: u(x, y)

    def get_basis(self, input_size, reflection):
        start = timeit.default_timer()
        print("Building basis...")
        if input_size % 2 == 0:
            input_size += 1
        r2_act = escnn.gspaces.flipRot2dOnR2(N=4) if reflection else escnn.gspaces.rot2dOnR2(N=4)
        feat_type = escnn.nn.FieldType(r2_act, [r2_act.regular_repr])
        with torch.no_grad():
            conv = escnn.nn.R2Conv(feat_type, feat_type, kernel_size=input_size).cuda()
            base_exp = conv._basisexpansion
            b_exp = getattr(base_exp, "block_expansion_('regular', 'regular')")
            group_size = 8 if reflection else 4
            self.basis = b_exp.sampled_basis.detach().cpu().reshape(-1, group_size, group_size, input_size, input_size)
        torch.cuda.empty_cache()
        self.basis = torch.fft.fftshift(torch.fft.rfft2(torch.fft.ifftshift(self.basis, dim=[-2, -1]), dim=[-2, -1]), dim=-2)
        freq0_y = (torch.fft.fftshift(torch.fft.fftfreq(input_size)) == 0).nonzero().item()
        self.basis = self.basis[..., (freq0_y - self.modes + 1):(freq0_y + self.modes), :self.modes]
        print(f"Built basis; {timeit.default_timer() - start}s elapsed")

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], -1)
        grid = self.get_grid(x.shape).to(x.device)
        x = torch.cat((x, grid), dim=-1)
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

    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x).reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y).reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        midpt = 0.5
        gridx = (gridx - midpt) ** 2
        gridy = (gridy - midpt) ** 2
        return gridx + gridy