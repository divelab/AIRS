import torch


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class RadialFunction(torch.nn.Module):
    """
        1.  Contruct a radial function (linear layers + layer normalization + SiLU) given a list of channels
        2.  If `use_rad_l_parametrization` == True and `use_expand` == True, all the m components 
            within a type-L vector will share the same weight.
            We expand the outputs so that they can be directly multiplied with SO(3) features.
        3.  If `use_rad_l_parametrization` == False and `use_expand` == True, the +=m within the same 
            type-L vector will share the same weight while different m will have different weights. 
            Thus, this will have more parameters than 3.
            Different from 2., We expand the outputs so that they can be directly multiplied with SO(2) features.
        4.  If `use_expand` == False, we simply return the outputs of radial functions.
    """
    def __init__(self, channels_list, lmax=None, mmax=None, use_rad_l_parametrization=True, use_expand=True):
        super().__init__()
        modules = []
        input_channels = channels_list[0]
        for i in range(len(channels_list)):
            if i == 0:
                continue

            modules.append(torch.nn.Linear(input_channels, channels_list[i], bias=True))
            input_channels = channels_list[i]

            if i == len(channels_list) - 1:
                break

            modules.append(torch.nn.LayerNorm(channels_list[i]))
            modules.append(torch.nn.SiLU())

        self.net = torch.nn.Sequential(*modules)

        self.lmax = lmax
        self.mmax = mmax
        self.use_rad_l_parametrization = use_rad_l_parametrization
        self.use_expand = use_expand

        if self.use_expand:
            if not self.use_rad_l_parametrization:
                expand_index = []
                offset = 0
                for m in range(self.mmax + 1):
                    index = torch.arange((self.lmax + 1 - m))
                    index = index + offset
                    expand_index.append(index)
                    if m > 0:
                        expand_index.append(index)    # +- m
                    offset = offset + len(index)
                expand_index = torch.cat(expand_index, dim=0)
                expand_index = expand_index.long()
                self.register_buffer('expand_index', expand_index)
                self.num_m_components = offset
                assert channels_list[-1] % self.num_m_components == 0
            else:
                assert self.lmax == self.mmax
                expand_index = torch.zeros([((self.lmax + 1) ** 2)]).long()
                start_idx = 0
                for l in range(self.lmax + 1):
                    length = 2 * l + 1
                    expand_index[start_idx : (start_idx + length)] = l
                    start_idx = start_idx + length
                self.register_buffer('expand_index', expand_index)
                assert channels_list[-1] % (self.lmax + 1) == 0


    def forward(self, inputs):
        outputs = self.net(inputs)
        if self.use_expand:
            if not self.use_rad_l_parametrization:
                # Convert to the format that can be directly multiplied with SO(2) features
                outputs = outputs.view(outputs.shape[0], self.num_m_components, -1)
            else:
                # Convert to the format that can be directly multiplied with SO(3) features
                outputs = outputs.view(outputs.shape[0], (self.lmax + 1), -1)
            outputs = torch.index_select(outputs, dim=1, index=self.expand_index)    
        return outputs