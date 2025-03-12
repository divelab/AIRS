import torch
from torch.nn import Linear


class DenoisingBlock(torch.nn.Module):
    """
    Force Block: Output block computing the per atom forces

    Args:
        num_channels (int):         Number of channels
        num_sphere_samples (int):   Number of samples used to approximate the integral on the sphere
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        num_channels,
        num_sphere_samples,
        act,
    ):
        super(ForceBlock, self).__init__()
        self.num_channels = num_channels
        self.num_sphere_samples = num_sphere_samples
        self.act = act

        self.fc1 = Linear(self.num_channels, self.num_channels)
        self.fc2 = Linear(self.num_channels, self.num_channels)
        self.fc3 = Linear(self.num_channels, 1, bias=False)

    def forward(self, x_pt, sphere_points):
        # x_pt are the values of the channels sampled at different points on the sphere
        x_pt = self.act(self.fc1(x_pt))
        x_pt = self.act(self.fc2(x_pt))
        x_pt = self.fc3(x_pt)
        x_pt = x_pt.view(-1, self.num_sphere_samples, 1)
        forces = x_pt * sphere_points.view(1, self.num_sphere_samples, 3)
        forces = torch.sum(forces, dim=1) / self.num_sphere_samples

        return forces