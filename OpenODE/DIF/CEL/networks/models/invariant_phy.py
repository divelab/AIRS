import torch
from CEL.networks.models.meta_model import ModelClass
from torch.autograd import Function
from itertools import chain

class InvariantPhy(torch.nn.Module, ModelClass):
    def __init__(
            self,
            y_channels: int,
            Wc_channels: int,
            hidden_channels: int,
            depth_enc: int,
            inv_type: str,
            num_envs: int,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.depth_enc = depth_enc
        self.inv_enc = torch.nn.Sequential(
            *([
                torch.nn.Linear(y_channels + Wc_channels, hidden_channels // 4),
                torch.nn.BatchNorm1d(hidden_channels // 4),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels // 4, hidden_channels // 2),
                torch.nn.BatchNorm1d(hidden_channels // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels // 2, hidden_channels),
            ] + list(chain(*[[
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            ] for _ in range(depth_enc)])))
        )

        self.env_enc = torch.nn.Sequential(
            *([
                torch.nn.Linear(y_channels + Wc_channels, hidden_channels // 4),
                torch.nn.BatchNorm1d(hidden_channels // 4),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels // 4, hidden_channels // 2),
                torch.nn.BatchNorm1d(hidden_channels // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels // 2, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            ]) + list(chain(*[[
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            ] for _ in range(depth_enc)]))
        )

        self.combine_mlp = torch.nn.Sequential(
            *[
                torch.nn.Linear(hidden_channels * 2, hidden_channels, bias=False),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels, bias=False),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, y_channels, bias=False),
            ]
        )

        self.inv_mlp = torch.nn.Sequential(
            *[
                torch.nn.Linear(hidden_channels, y_channels),
            ]
        )
        self.inv_type = inv_type
        if inv_type == 'DANN':
            self.inv_eclassifier = torch.nn.Sequential(
                *[
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.BatchNorm1d(hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.BatchNorm1d(hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.BatchNorm1d(hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, num_envs),
                ]
            )

    def forward(self, y, Wc, **kwargs):
        inv_z = self.inv_enc(torch.cat([y, Wc[:, None]], dim=-1))
        env_z = self.env_enc(torch.cat([y, Wc[:, None]], dim=-1))
        inv_dy = self.inv_mlp(inv_z)
        final_dy = self.combine_mlp(torch.cat([inv_z, env_z], dim=-1))
        if self.inv_type == 'DANN' and self.training:
            inv_z_reverse = GradientReverseLayerF.apply(inv_z, kwargs.get('alpha') * kwargs.get('lambda_reverse'))
            inv_eoutput = self.inv_eclassifier(inv_z_reverse)
            return final_dy, (inv_dy, inv_eoutput)
        return final_dy, inv_dy

class GradientReverseLayerF(Function):
    r"""
    Gradient reverse layer for DANN algorithm.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        r"""
        gradient forward propagation

        Args:
            ctx (object): object of the GradientReverseLayerF class
            x (Tensor): feature representations
            alpha (float): the GRL learning rate

        Returns (Tensor):
            feature representations

        """
        ctx.alpha = alpha
        return x.view_as(x)  # * alpha

    @staticmethod
    def backward(ctx, grad_output):
        r"""
        gradient backpropagation step

        Args:
            ctx (object): object of the GradientReverseLayerF class
            grad_output (Tensor): raw backpropagation gradient

        Returns (Tensor):
            backpropagation gradient

        """
        output = grad_output.neg() * ctx.alpha
        return output, None