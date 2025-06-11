# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import math
from pdearena.utils.activations import (
    ACTIVATION_REGISTRY, 
    SinusoidalEmbedding
)
from pdearena.configs.config import Config
from torch_geometric.data import Batch
from torch.utils.checkpoint import checkpoint

# Largely based on https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
# MIT License
# Copyright (c) 2020 Varuna Jayasiri

class ResidualBlock(nn.Module):
    """Wide Residual Blocks used in modern Unet architectures.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        cond_channels (int): Number of channels in the conditioning vector.
        activation (str): Activation function to use.
        norm (bool): Whether to use normalization.
        n_groups (int): Number of groups for group normalization.
        use_scale_shift_norm (bool): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        padding_mode: str,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), padding_mode=padding_mode)
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        cond_out_channels = 2 * out_channels if use_scale_shift_norm else out_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.cond_emb = nn.Sequential(
            nn.Linear(cond_channels, 2 * out_channels),
            self.activation,
            nn.Linear(2 * out_channels, cond_out_channels),
        )
        

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        # First convolution layer
        h = self.conv1(self.activation(self.norm1(x)))
        emb_out = self.cond_emb(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift  # where we do -1 or +1 doesn't matter
            h = self.conv2(self.activation(h))
        else:
            h = h + emb_out
            # Second convolution layer
            h = self.conv2(self.activation(self.norm2(h)))
        # Add the shortcut connection and return
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """Attention block This is similar to [transformer multi-head
    attention](https://arxiv.org/abs/1706.03762).

    Args:
        n_channels: the number of channels in the input
        n_heads:  the number of heads in multi-head attention
        d_k: the number of dimensions in each head
        n_groups: the number of groups for [group normalization][torch.nn.GroupNorm]

    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: Optional[int] = None, n_groups: int = 1):
        """ """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k**-0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res


class DownBlock(nn.Module):
    """Down block This combines `ResidualBlock` and `AttentionBlock`.

    These are used in the first half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        cond_channels (int): Number of channels in the conditioning vector.
        has_attn (bool): Whether to use attention block
        activation (nn.Module): Activation function
        norm (bool): Whether to use normalization
        use_scale_shift_norm (bool): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        padding_mode: str,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        self.res = ResidualBlock(
            in_channels,
            out_channels,
            cond_channels,
            padding_mode=padding_mode,
            activation=activation,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        x = self.res(x, emb)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """Up block This combines `ResidualBlock` and `AttentionBlock`.

    These are used in the second half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        cond_channels (int): Number of channels in the conditioning vector.
        has_attn (bool): Whether to use attention block
        activation (str): Activation function
        norm (bool): Whether to use normalization
        use_scale_shift_norm (bool): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        padding_mode: str,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(
            in_channels + out_channels,
            out_channels,
            cond_channels,
            padding_mode=padding_mode,
            activation=activation,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.res(x, emb)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """Middle block It combines a `ResidualBlock`, `AttentionBlock`, followed by another
    `ResidualBlock`.

    This block is applied at the lowest resolution of the U-Net.

    Args:
        n_channels (int): Number of channels in the input and output.
        cond_channels (int): Number of channels in the conditioning vector.
        has_attn (bool, optional): Whether to use attention block. Defaults to False.
        activation (str): Activation function to use. Defaults to "gelu".
        norm (bool, optional): Whether to use normalization. Defaults to False.
        use_scale_shift_norm (bool, optional): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`). Defaults to False.
    """

    def __init__(
        self,
        n_channels: int,
        cond_channels: int,
        padding_mode: str,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        self.res1 = ResidualBlock(
            n_channels,
            n_channels,
            cond_channels,
            padding_mode=padding_mode,
            activation=activation,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        self.attn = AttentionBlock(n_channels) if has_attn else nn.Identity()
        self.res2 = ResidualBlock(
            n_channels,
            n_channels,
            cond_channels,
            padding_mode=padding_mode,
            activation=activation,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, emb)
        x = self.attn(x)
        x = self.res2(x, emb)
        return x


class Upsample(nn.Module):
    r"""Scale up the feature map by $2 \times$"""

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Downsample(nn.Module):
    r"""Scale down the feature map by $\frac{1}{2} \times$"""

    def __init__(self, n_channels: int, padding_mode: str):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1), padding_mode=padding_mode)

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Unet(nn.Module):
    """Modern U-Net architecture

    This is a modern U-Net architecture with wide-residual blocks and spatial attention blocks

    Args:
        n_input_scalar_components (int): Number of scalar components in the model
        n_input_vector_components (int): Number of vector components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        n_output_vector_components (int): Number of output vector components in the model
        time_history (int): Number of time steps in the input
        time_future (int): Number of time steps in the output
        hidden_channels (int): Number of channels in the hidden layers
        activation (str): Activation function to use
        norm (bool): Whether to use normalization
        ch_mults (list): List of channel multipliers for each resolution
        is_attn (list): List of booleans indicating whether to use attention blocks
        mid_attn (bool): Whether to use attention block in the middle block
        n_blocks (int): Number of residual blocks in each resolution
        param_conditioning (Optional[str]): Type of conditioning to use. Defaults to None.
        use_scale_shift_norm (bool): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`). Defaults to False.
        use1x1 (bool): Whether to use 1x1 convolutions in the initial and final layers

    Note:
        Currently, only `scalar` parameter conditioning is supported.
    """

    def __init__(
        self,
        args: Config,
        hidden_channels: int,
        sinusoidal_embedding: bool = False,
        sine_spacing: float=None,
        use_dt: bool = True,
        activation="gelu",
        norm: bool = False,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        num_param_conditioning: int = 0,
        diffusion_time_conditioning: bool = False,
        use_scale_shift_norm: bool = False,
        use1x1: bool = False,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.use_checkpoint = args.use_gradient_checkpoint
        self.n_input_fields = args.n_input_fields
        self.n_output_fields = args.n_output_fields
        self.time_history = args.time_history
        self.time_future = args.time_future
        self.hidden_channels = hidden_channels
        self.activation = activation
        self.num_param_conditioning = num_param_conditioning
        self.diffusion_time_conditioning = diffusion_time_conditioning
        self.use_dt = use_dt
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")
        # Number of resolutions
        n_resolutions = len(ch_mults)

        insize = self.time_history * self.n_input_fields
        n_channels = hidden_channels

        if sinusoidal_embedding:
            embed_dim = hidden_channels
            get_embed_fn = lambda: SinusoidalEmbedding(
                n_channels=hidden_channels, 
                spacing=sine_spacing
            )
        else:
            embed_dim = 1
            get_embed_fn = lambda: nn.Identity()

        if use_dt:
            num_embs = 1
            self.dt_embed = get_embed_fn()
        else:
            num_embs = 0

        if self.diffusion_time_conditioning:
            num_embs += 1
            self.diffusion_time_embed = get_embed_fn()

        if self.num_param_conditioning > 0:
            num_embs += self.num_param_conditioning
            self.pde_embed = nn.ModuleList()
            for _ in range(self.num_param_conditioning):
                self.pde_embed.append(get_embed_fn())

        cond_embed_dim = embed_dim * num_embs

        # Project image into feature map
        if use1x1:
            self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=1)
        else:
            self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=(3, 3), padding=(1, 1), padding_mode=padding_mode)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        cond_embed_dim,
                        padding_mode=padding_mode,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels, padding_mode=padding_mode))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(
            out_channels,
            cond_embed_dim,
            padding_mode=padding_mode,
            has_attn=mid_attn,
            activation=activation,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
        )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        cond_embed_dim,
                        padding_mode=padding_mode,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(
                UpBlock(
                    in_channels,
                    out_channels,
                    cond_embed_dim,
                    padding_mode=padding_mode,
                    has_attn=is_attn[i],
                    activation=activation,
                    norm=norm,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            )
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm and False:
            self.norm = nn.GroupNorm(8, n_channels)
        else:
            self.norm = nn.Identity()
        out_channels = self.time_future * self.n_output_fields
        #
        if use1x1:
            final = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            final = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), padding_mode=padding_mode)
        
        self.final = final

    def forward(self, batch: Batch):
        x = batch.x
        dt = batch.dt
        if self.num_param_conditioning > 0:
            z = batch.z
        else:
            z = None
        diff_time = batch.diff_time if hasattr(batch, 'diff_time') else None
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C
        emb = self.embed(
            dt=dt,
            diff_time=diff_time,
            z=z
        )

        x = self.image_proj(x)

        x, h = self.down_blocks(
            x=x,
            emb=emb
        )

        if self.use_checkpoint:
            x = checkpoint(
                self.middle,
                x, emb,
                use_reentrant=False
            )
        else:
            x = self.middle(x, emb)

        x = self.up_blocks(
            x=x,
            emb=emb,
            h_list=h
        )

        x = self.final(self.activation(self.norm(x)))
        x = x.reshape(
            orig_shape[0], -1, self.n_output_fields, *orig_shape[3:]
        )
        return x

    def embed(
        self,
        dt: torch.Tensor,
        diff_time: torch.Tensor,
        z: torch.Tensor
    ):
        if self.use_dt:
            embs = [self.dt_embed(dt)]
        else:
            embs = []

        if self.diffusion_time_conditioning:
            assert diff_time is not None
            embs.append(self.diffusion_time_embed(diff_time + 1))
        else:
            assert diff_time is None

        if self.num_param_conditioning > 0:
            assert z is not None
            z_list = z.chunk(self.num_param_conditioning, dim=1)
            for i, z in enumerate(z_list):
                z = z.flatten()
                assert z.shape == dt.shape
                embs.append(self.pde_embed[i](z))
        
        embs = [emb.view(len(emb), -1) for emb in embs]
        emb = torch.cat(embs, dim=1)
        return emb


    def down_blocks(
            self,
            x: torch.Tensor,
            emb: torch.Tensor
    ):
        h = [x]
        for m in self.down:
            if isinstance(m, Downsample):
                if self.use_checkpoint:
                    x = checkpoint(m, x, use_reentrant=False)
                else:
                    x = m(x)
            else:
                if self.use_checkpoint:
                    x = checkpoint(m, x, emb, use_reentrant=False)
                else:
                    x = m(x, emb)
            h.append(x)
        return x, h

    def up_blocks(
            self,
            x: torch.Tensor,
            emb: torch.Tensor,
            h_list: torch.Tensor
    ):
        # h = [z.clone() for z in h_list]
        h = h_list
        for m in self.up:
            if isinstance(m, Upsample):
                if self.use_checkpoint:
                    x = checkpoint(m, x, use_reentrant=False)
                else:
                    x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                if self.use_checkpoint:
                    x = checkpoint(m, x, emb, use_reentrant=False)
                else:
                    x = m(x, emb)
        return x