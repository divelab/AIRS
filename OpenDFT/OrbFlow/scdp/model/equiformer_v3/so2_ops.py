import torch
import math


class SO2MLinear(torch.nn.Module):
    """
        Perform an SO(2) linear operation to features corresponding to +- m

        Args:
            m (int):                    Order of the spherical harmonic coefficients
            num_in_channels (int):      Number of input channels
            num_out_channels (int):     Number of output channels
            lmax (int):                 Maximum degrees (l)
            mmax (int):                 Maximum order (m)
    """
    def __init__(
        self,
        m,
        num_in_channels,
        num_out_channels,
        lmax,
        mmax
    ):
        super(SO2MLinear, self).__init__()

        self.m = m
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.lmax = lmax
        self.mmax = mmax

        num_m_components = self.lmax - self.m + 1
        assert num_m_components > 0

        self.in_features = num_m_components * self.num_in_channels
        self.out_features = num_m_components * self.num_out_channels
        
        self.fc = torch.nn.Linear(
            self.in_features,
            (2 * self.out_features),
            bias=False
        )
        self.fc.weight.data.mul_(1 / math.sqrt(2))


    def forward(self, x_m, concat_outputs=True):
        x_m = self.fc(x_m)
        x_r = x_m.narrow(2, 0, self.out_features)
        x_i = x_m.narrow(2, self.out_features, self.out_features)
        x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1) #x_r[:, 0] - x_i[:, 1]
        x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1) #x_r[:, 1] + x_i[:, 0]
        #x_r, x_i = torch.chunk(x_m, chunks=2, dim=2)
        #x_r_0, x_r_1 = torch.chunk(x_r, chunks=2, dim=1)
        #x_i_0, x_i_1 = torch.chunk(x_i, chunks=2, dim=1)
        #x_m_r = x_r_0 - x_i_1
        #x_m_i = x_r_1 + x_i_0
        x_out = (x_m_r, x_m_i)
        if concat_outputs:
            x_out = torch.cat(x_out, dim=1)
        return x_out


class SO2Linear(torch.nn.Module):
    """
        Perform SO(2) linear operations to all m (orders) components

        Args:
            num_in_channels (int):      Number of input channels
            num_out_channels (int):     Number of output channels
            lmax (int):                 Maximum degrees (l)
            mmax (int):                 Maximum order (m)
            extra_m0_out_channels (int):    If not None, return `outputs` (torch.Tensor) and `extra_m0_features` (torch.Tensor).
    """
    def __init__(
        self,
        num_in_channels,
        num_out_channels,
        lmax,
        mmax,
        extra_m0_out_channels=None
    ):
        super(SO2Linear, self).__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.lmax = lmax
        self.mmax = mmax
        self.extra_m0_out_channels = extra_m0_out_channels

        # for m = 0
        num_in_channels_m0 = (self.lmax + 1) * self.num_in_channels
        num_out_channels_m0 = (self.lmax + 1) * self.num_out_channels
        if self.extra_m0_out_channels is not None:
            self.num_channels_m0_list = [self.extra_m0_out_channels, num_out_channels_m0]
            num_out_channels_m0 = num_out_channels_m0 + self.extra_m0_out_channels
        self.fc_m0 = torch.nn.Linear(num_in_channels_m0, num_out_channels_m0)

        # SO(2) linear for non-zero m
        self.so2_m_linear = torch.nn.ModuleList()
        for m in range(1, self.mmax + 1):
            self.so2_m_linear.append(
                SO2MLinear(
                    m,
                    self.num_in_channels,
                    self.num_out_channels,
                    self.lmax,
                    self.mmax,
                )
            )


    def forward(self, x):
        """
            1.  `x` shape: [num_edges, num_m_components, num_channels]
            2.  We assume the layout of m components is (0, 0, ...), (1, 1, ...), ...
        """
        num_edges = x.shape[0]
        outputs = []

        # Compute m=0 coefficients separately since they only have real values (no imaginary)
        x_m0 = x.narrow(1, 0, (self.lmax + 1))
        x_m0 = x_m0.reshape(num_edges, -1)
        x_m0 = self.fc_m0(x_m0)

        x_m0_extra = None
        # extract extra m0 features
        if self.extra_m0_out_channels is not None:
            x_m0_extra, x_m0 = torch.split(x_m0, self.num_channels_m0_list, dim=1)

        x_m0 = x_m0.view(num_edges, -1, self.num_out_channels)
        outputs.append(x_m0)

        # Compute the values for the m > 0 coefficients
        offset = self.lmax + 1
        for m in range(1, self.mmax + 1):
            x_m = x.narrow(1, offset, 2 * (self.lmax + 1 - m))
            offset = offset + 2 * (self.lmax + 1 - m)
            x_m = x_m.reshape(num_edges, 2, -1)
            """
            x_m = self.so2_m_linear[m - 1](x_m, concat_outputs=True)
            x_m = x_m.view(num_edges, -1, self.num_out_channels)
            out.append(x_m)
            """
            # Replace the original one with the followings to prevent one `torch.cat()` for each m > 0
            x_m = self.so2_m_linear[m - 1](x_m, concat_outputs=False)
            x_m_pos, x_m_neg = x_m[0], x_m[1]
            x_m_pos = x_m_pos.view(num_edges, -1, self.num_out_channels)
            x_m_neg = x_m_neg.view(num_edges, -1, self.num_out_channels)
            outputs.append(x_m_pos)
            outputs.append(x_m_neg)
            
        outputs = torch.cat(outputs, dim=1)

        if self.extra_m0_out_channels is not None:
            return outputs, x_m0_extra
        else:
            return outputs