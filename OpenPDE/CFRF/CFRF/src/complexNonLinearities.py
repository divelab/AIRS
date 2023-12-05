import torch
import torch.nn as nn

def get_activation(activation):
    if activation == "CReLU":
        return complexReLU()
    if activation == "modReLU":
        return modReLU()
    if activation == "zReLU":
        return zReLU()
    raise ValueError(f"Activation {activation} not recognized")

class modReLU(nn.Module): # Unitary evolution recurrent neural networks
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(1) * 1e-1)

    def forward(self, z):
        z_mag = (z.real**2 + z.imag**2)**0.5
        return ((z_mag + self.bias) >= 0).float() * (1 + self.bias / z_mag) * z

class complexReLU(nn.Module):
    def forward(self, x):
        return torch.relu(x.real) + 1.j * torch.relu(x.imag)

class zReLU(nn.Module):
    def forward(self, z):
        z_angle = z.angle()
        mask = (0 < z_angle) * (z_angle < torch.pi/2)
        return z * mask
