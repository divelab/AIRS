# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from pdearena.utils.activations import SinusoidalEmbedding

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, use_emb, cond_channels: int = None, use_scale_shift_norm: bool = False, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_emb = use_emb
        if use_emb:
            cond_out_channels = 2 * dim if use_scale_shift_norm else dim
            self.use_scale_shift_norm = use_scale_shift_norm
            self.cond_emb = nn.Sequential(
                nn.Linear(cond_channels, 2 * dim),
                self.act,
                nn.Linear(2 * dim, cond_out_channels),
            )



    def forward(self, x, emb):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.add_emb(h=x, emb=emb)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

    def add_emb(self, h, emb):
        if emb is None:
            assert not self.use_emb
            return h
        else:
            assert self.use_emb
        emb_out = self.cond_emb(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift  # where we do -1 or +1 doesn't matter
        else:
            h = h + emb_out

        return h

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, args, max_pool: bool = False,
                 sinusoidal_embedding: bool = False,
                 sine_spacing: float = 1e-4,
                 use_scale_shift_norm: bool = False,
                 num_param_conditioning: int = 0,
                 num_classes=1, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()
        self.max_pool = max_pool
        self.n_input_fields = args.n_input_fields
        self.n_output_fields = args.n_output_fields
        self.time_history = args.time_history
        self.time_future = args.time_future
        self.num_param_conditioning = num_param_conditioning

        if sinusoidal_embedding:
            embed_dim = min(dims)
            get_embed_fn = lambda: SinusoidalEmbedding(
                n_channels=embed_dim, 
                spacing=sine_spacing
            )
        else:
            embed_dim = 1
            get_embed_fn = lambda: nn.Identity()

        if num_param_conditioning > 0:
            self.use_emb = True
            self.pde_embed = nn.ModuleList()
            for _ in range(num_param_conditioning):
                self.pde_embed.append(get_embed_fn())
        else:
            self.use_emb = False

        cond_embed_dim = embed_dim * num_param_conditioning

        
        in_chans = self.time_history * self.n_input_fields

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            down_fn = nn.Sequential(
                nn.Conv2d(dims[i], dims[i+1], kernel_size=3, padding=1), 
                nn.MaxPool2d(kernel_size=2, stride=2)
            ) if max_pool else nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    down_fn,
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.ModuleList(
                [Block(
                    dim=dims[i], 
                    use_emb=self.use_emb, 
                    cond_channels=cond_embed_dim, 
                    use_scale_shift_norm=use_scale_shift_norm, 
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x, emb):
        for i in range(4):
            x = self.downsample_layers[i](x)
            for conv in self.stages[i]:
                x = conv(x, emb)
        x = x.flatten(2)
        if self.max_pool:
            x = x.max(dim=-1).values
        else:
            x = x.mean(dim=-1)
        x = self.norm(x)
        return x

    def forward(self, batch):
        x = batch.x
        assert x.dim() == 5
        x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C
        z = batch.z
        emb = self.embed(z=z)
        x = self.forward_features(x, emb)
        x = self.head(x).flatten()
        return x

    def embed(
        self,
        z: torch.Tensor
    ):
        if not self.use_emb:
            return None

        assert z is not None
        z_list = z.chunk(self.num_param_conditioning, dim=1)
        embs = []
        for i, z in enumerate(z_list):
            z = z.flatten()
            embs.append(self.pde_embed[i](z))
        
        embs = [emb.view(len(emb), -1) for emb in embs]
        emb = torch.cat(embs, dim=1)
        return emb

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x