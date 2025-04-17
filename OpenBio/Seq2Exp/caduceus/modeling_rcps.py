"""Reverse-complement equivariant modules.

"""
from collections import OrderedDict
from typing import Optional

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class RCPSEmbedding(nn.Module):
    """Embedding layer that supports reverse-complement equivariance."""
    def __init__(self, vocab_size: int, d_model: int, complement_map: dict, **factory_kwargs):
        """
        Args:
            vocab_size: Size of vocabulary.
            d_model: Dimensionality of embedding (actual embedding matrix will have 1/2 the output dim).
            complement_map: Dictionary mapping each token id to its complement.
        """
        super().__init__()
        self.register_buffer(
            "complement_map",
            torch.tensor(list(OrderedDict(complement_map).values()), dtype=torch.long)
        )
        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

    @property
    def weight(self):
        """Embedding weights."""
        return self.embedding.weight

    def set_weight(self, value):
        """Set embedding weights."""
        self.embedding.weight = value

    def rc(self, x):
        """Reverse-complement a tensor of input_ids by flipping along length dimension and complementing the ids."""
        return torch.gather(
            self.complement_map.unsqueeze(0).expand(x.shape[0], -1),
            dim=1,
            index=torch.flip(x, dims=[-1])
        )

    def forward(self, input_ids):
        """Reverse-complement equivariant forward pass.

        This embedding module doubles the output dimensionality to support reverse-complement equivariance.

        Args:
            input_ids: Input tensor of shape (batch_size, seq_len)
        Returns:
            Embedding tensor of shape (batch_size, seq_len, d_model * 2)
        """
        fwd_out = self.embedding(input_ids)
        rc_out = torch.flip(self.embedding(self.rc(input_ids)), dims=[-2, -1])

        return torch.cat([fwd_out, rc_out], dim=-1)  # torch.Size([2, 131072, 512])


class RCPSWrapper(nn.Module):
    """Wrapper to convert arbitrary nn.Module into a reverse-complement equivariant module.

    See ref. "Towards a Better Understanding of Reverse-Complement Equivariance for Deep Learning Models in Regulatory
    Genomics", Zhou et al. (2022), https://proceedings.mlr.press/v165/zhou22a.html for more details.
    """
    def __init__(self, submodule: nn.Module):
        super().__init__()
        self.submodule = submodule

    @staticmethod
    def rc(x):
        """Reverse-complement a tensor by flipping the length (dim=-2) and channel (dim=-1) dimensions."""
        return torch.flip(x, dims=[-2, -1])

    def forward(self, x, **kwargs):
        """Reverse-complement equivariant forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, channels)
        Returns:
            Output tensor of shape (batch_size, seq_len, channels * 2)
        """
        n_channels = x.shape[-1]  # 512 dim
        # Run submodule along sequence
        fwd_out = self.submodule(x[..., :n_channels // 2], **kwargs)  # 256 dim
        # Run submodule along rc-sequence
        rc_out = self.submodule(self.rc(x[..., n_channels // 2:]), **kwargs)
        # Concatenate along channel dimension (dim=-1)
        return torch.cat([fwd_out, self.rc(rc_out)], dim=-1)  # 512 dim again


class RCPSAddNormWrapper(RCPSWrapper):
    """RC equivariant AddNorm layer."""
    def __init__(self, submodule: nn.Module):
        super().__init__(submodule)

    def forward(self, x, residual=None, prenorm=False):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, channels)
            residual: Residual tensor of shape (batch_size, seq_len, channels) or None.
            prenorm: Whether to return residual.
        """
        n_channels = x.shape[-1]
        if residual is None:
            residual = x
            x_fwd = self.submodule(x[..., :n_channels // 2].to(dtype=self.submodule.weight.dtype))
            x_rc = self.submodule(self.rc(x[..., n_channels // 2:]).to(dtype=self.submodule.weight.dtype))
            x = torch.cat([x_fwd, self.rc(x_rc)], dim=-1)
        else:
            residual_fwd = x[..., :n_channels // 2] + residual[..., :n_channels // 2]
            x_fwd = self.submodule(residual_fwd.to(dtype=self.submodule.weight.dtype))

            residual_rc = self.rc(x[..., n_channels // 2:]) + self.rc(residual[..., n_channels // 2:])
            x_rc = self.submodule(residual_rc.to(dtype=self.submodule.weight.dtype))

            residual = torch.cat([residual_fwd, self.rc(residual_rc)], dim=-1)
            x = torch.cat([x_fwd, self.rc(x_rc)], dim=-1)

        return x if not prenorm else (x, residual)


class RCPSMambaBlock(nn.Module):
    def __init__(
            self,
            dim,
            mixer_cls,
            norm_cls=nn.LayerNorm,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,  # Keep for consistency with original Mamba Block
            dtype=None,  # Keep for consistency with original Mamba Block
    ):
        """RCPS version of simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection.

        Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = RCPSWrapper(mixer_cls(dim))
        norm_f = norm_cls(dim)
        self.norm = norm_f if fused_add_norm else RCPSAddNormWrapper(norm_f)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual)).
            inference_params: inference parameters for mixer.
        """
        if not self.fused_add_norm:
            hidden_states, residual = self.norm(hidden_states, residual=residual, prenorm=True)
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn

            hidden_states_fwd, residual_fwd = fused_add_norm_fn(
                hidden_states[..., hidden_states.shape[-1] // 2:],
                self.norm.weight,
                self.norm.bias,
                residual=residual[..., hidden_states.shape[-1] // 2:] if residual is not None else None,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )

            hidden_states_rc, residual_rc = fused_add_norm_fn(
                hidden_states[..., :hidden_states.shape[-1] // 2].flip(dims=[-2, -1]),
                self.norm.weight,
                self.norm.bias,
                residual=residual[..., :hidden_states.shape[-1] // 2].flip(dims=[-2, -1]) if residual is not None else None,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
            hidden_states = torch.cat([hidden_states_fwd, hidden_states_rc.flip(dims=[-2, -1])], dim=-1)
            residual = torch.cat([residual_fwd, residual_rc.flip(dims=[-2, -1])], dim=-1)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocate inference cache for mixer.

        Keep for compatibility with original Mamba Block.
        """
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class RCPSLMHead(nn.Module):
    """LM Head for reverse-complement equivariant inputs, which have dim * 2 relative to standard inputs."""
    def __init__(self, true_dim: int, vocab_size: int, complement_map: dict, **factory_kwargs):
        """
        `true_dim` corresponds to the actual dimensionality of the input were it not reverse-complement
        equivariant, i.e. 0.5 times the actual input dim.
        """
        super().__init__()
        self.register_buffer(
            "complement_map",
            torch.tensor(list(OrderedDict(complement_map).values()), dtype=torch.long)
        )
        self.true_dim = true_dim
        self.lm_head = nn.Linear(true_dim, vocab_size, bias=False, **factory_kwargs)

    @property
    def weight(self):
        """LM head weights."""
        return self.lm_head.weight

    def set_weight(self, value):
        """Set LM head weights."""
        self.lm_head.weight = value

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim), where dim = 2 * true_dim.
        """
        n_channels = x.shape[-1]
        assert n_channels == 2 * self.true_dim, "Input must have 2 * true_dim channels."
        fwd_logits = F.linear(x[..., :n_channels // 2], self.weight, bias=self.lm_head.bias)
        rc_logits = F.linear(
            torch.flip(x[..., n_channels // 2:], dims=[-1]),
            self.weight[self.complement_map, :],
            bias=self.lm_head.bias
        )
        return fwd_logits + rc_logits
