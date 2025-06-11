from torch import nn
import torch 
import math

ACTIVATION_REGISTRY = {
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
}

class SinusoidalEmbedding(nn.Module):
    # https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/90e21b5a36908a305f9dfa04a8e8ddc4d602f2b5/labml_nn/diffusion/ddpm/unet.py#L44-L83
    def __init__(self, n_channels: int, spacing: float):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        self.spacing = spacing

    def forward(self, x: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        x = x / self.spacing
        half_dim = self.n_channels // 2
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return emb

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module