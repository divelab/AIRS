import copy
import math
import re
from collections import namedtuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flash_attn.modules.block import Block
from flash_attn.modules.embedding import GPT2Embeddings, ParallelGPT2Embeddings
from flash_attn.modules.mha import MHA, ParallelMHA
from flash_attn.modules.mlp import Mlp, FusedMLP, ParallelFusedMLP
from flash_attn.utils.distributed import sync_shared_params, all_gather_raw
from flash_attn.utils.generation import GenerationMixin
from torch.utils.checkpoint import checkpoint

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

from src.utils import instantiate
import src.utils.registry as registry


class CheckpointedModule(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return checkpoint(self.layer, x)


def create_mixer_cls(
    layer=None,
    process_group=None,
    attn_layer_idx=None,
    attn_cfg=None,
    layer_idx=None,
    sequence_parallel=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    parallel_kwargs = (
        {"process_group": process_group, "sequence_parallel": sequence_parallel}
        if process_group is not None
        else {}
    )
    if attn_layer_idx is not None and layer_idx in attn_layer_idx:
        causal = True if attn_cfg is None else attn_cfg.pop("causal", True)
        fused_bias_fc = (
            False if attn_cfg is None else attn_cfg.get("fused_bias_fc", False)
        )
        if not fused_bias_fc:
            assert process_group is None, "TensorParallel MHA requires fused_bias_fc"
        mha_cls = MHA if process_group is None else ParallelMHA
        # ParallelMHA doesn't take 'fused_bias_fc', it is assumed that we fuse matmul + bias
        if process_group is not None:
            attn_cfg = copy.deepcopy(attn_cfg)  # Don't modify the original cfg
            attn_cfg.pop("fused_bias_fc", None)
        mixer_cls = partial(
            mha_cls,
            causal=causal,
            layer_idx=layer_idx,
            **(attn_cfg if attn_cfg is not None else {}),
            **parallel_kwargs,
            **factory_kwargs,
        )
    else:
        fused_bias_fc = False if layer is None else layer.get("fused_bias_fc", False)
        if process_group is not None:
            assert fused_bias_fc, "TensorParallel SSM requires fused_bias_fc"
        mixer_cls = instantiate(
            registry.layer,
            layer,
            partial=True,
            layer_idx=layer_idx,
            **factory_kwargs,
            **parallel_kwargs,
        )
    return mixer_cls


def create_mlp_cls(
    d_model,
    d_inner=None,
    process_group=None,
    fused_mlp=False,
    sequence_parallel=True,
    identity_mlp=False,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    inner_dim = d_inner if d_inner is not None else 4 * d_model
    if process_group is not None:
        assert fused_mlp, "Tensor Parallel is only implemented for FusedMLP"

    if not fused_mlp and not identity_mlp:
        mlp_cls = partial(
            Mlp,
            hidden_features=inner_dim,
            activation=partial(F.gelu, approximate="tanh"),
            **factory_kwargs,
        )
    elif fused_mlp:
        mlp_cls = FusedMLP if process_group is None else ParallelFusedMLP
        parallel_kwargs = (
            {"process_group": process_group, "sequence_parallel": sequence_parallel}
            if process_group is not None
            else {}
        )
        mlp_cls = partial(
            mlp_cls, hidden_features=inner_dim, **parallel_kwargs, **factory_kwargs
        )
    else:
        mlp_cls = nn.Identity
    return mlp_cls


def create_block(
    d_model,
    d_inner=None,
    process_group=None,
    layer=None,
    attn_layer_idx=None,
    attn_cfg=None,
    layer_norm_epsilon=1e-5,
    resid_dropout1=0.0,
    resid_dropout2=0.0,
    residual_in_fp32=False,
    fused_mlp=False,
    identity_mlp=False,
    fused_dropout_add_ln=False,
    layer_idx=None,
    sequence_parallel=True,
    checkpoint_mlp=False,
    checkpoint_mixer=False,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = create_mixer_cls(
        layer=layer,
        process_group=process_group,
        attn_layer_idx=attn_layer_idx,
        attn_cfg=attn_cfg,
        layer_idx=layer_idx,
        sequence_parallel=sequence_parallel,
        **factory_kwargs,
    )
    mlp_cls = create_mlp_cls(
        d_model,
        d_inner=d_inner,
        process_group=process_group,
        fused_mlp=fused_mlp,
        identity_mlp=identity_mlp,
        sequence_parallel=sequence_parallel,
        **factory_kwargs,
    )
    norm_cls = partial(nn.LayerNorm, eps=layer_norm_epsilon, **factory_kwargs)
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        prenorm=True,
        resid_dropout1=resid_dropout1,
        resid_dropout2=resid_dropout2,
        fused_dropout_add_ln=fused_dropout_add_ln,
        residual_in_fp32=residual_in_fp32,
        sequence_parallel=sequence_parallel and process_group is not None,
        mark_shared_params=process_group is not None,
    )

    block.layer_idx = layer_idx

    if checkpoint_mlp:
        block.mlp = CheckpointedModule(block.mlp)
    if checkpoint_mixer:
        block.mixer = CheckpointedModule(block.mixer)
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    glu_act=False,
):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(
                    p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer)
                )
            # If using GLU activation for now, we scale the std by 2
            elif name in ["output_linear.0.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                if not glu_act:
                    nn.init.normal_(
                        p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer)
                    )
                else:
                    out_features = p.shape[0]
                    # Multiplying the first half of the matrix by 2 since sigmoid scales it down by 0.5
                    # on average.
                    nn.init.normal_(
                        p[: out_features // 2],
                        mean=0.0,
                        std=initializer_range / math.sqrt(2 * n_layer) * 2,
                    )


class LMBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_inner: int,
        vocab_size: int,
        process_group=None,
        layer=None,
        attn_layer_idx=None,
        attn_cfg=None,
        max_position_embeddings=0,
        resid_dropout: float = 0.0,
        embed_dropout: float = 0.1,
        dropout_cls=nn.Dropout,
        layer_norm_epsilon: float = 1e-5,
        initializer_cfg=None,
        fused_mlp=False,
        identity_mlp=False,
        fused_dropout_add_ln=False,
        residual_in_fp32=False,
        sequence_parallel=True,
        checkpoint_mlp=False,
        checkpoint_mixer=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.residual_in_fp32 = residual_in_fp32

        if process_group is None:
            self.embeddings = GPT2Embeddings(
                d_model, vocab_size, max_position_embeddings, **factory_kwargs
            )
        else:
            self.embeddings = ParallelGPT2Embeddings(
                d_model,
                vocab_size,
                max_position_embeddings,
                process_group=process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.fused_dropout_add_ln = fused_dropout_add_ln
        if self.fused_dropout_add_ln and dropout_add_layer_norm is None:
            raise ImportError("dropout_add_layer_norm is not installed")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_inner=d_inner,
                    process_group=process_group,
                    layer=layer,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    layer_norm_epsilon=layer_norm_epsilon,
                    resid_dropout1=embed_dropout if i == 0 else resid_dropout,
                    resid_dropout2=resid_dropout,
                    residual_in_fp32=residual_in_fp32,
                    fused_mlp=fused_mlp,
                    identity_mlp=identity_mlp,
                    fused_dropout_add_ln=fused_dropout_add_ln,
                    layer_idx=i,
                    sequence_parallel=self.sequence_parallel,
                    checkpoint_mlp=checkpoint_mlp,
                    checkpoint_mixer=checkpoint_mixer,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.drop_f = nn.Dropout(resid_dropout)
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_epsilon, **factory_kwargs)

        if process_group is not None:
            for p in self.ln_f.parameters():
                # Mark the norm parameters as "shared_params" so that we sync their values at init.
                p._shared_params = True
                # Mark the norm params as "sequence_parallel" so we run all-reduce on their grads.
                if self.sequence_parallel:
                    p._sequence_parallel = True

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.process_group is not None:
            sync_shared_params(self, self.process_group)

    def forward(self, input_ids, position_ids=None, inference_params=None):
        # If using Tensor Parallel with sequence parallel, we combine the batch and the seqlen
        # dimensions so that we can split on it easily, in case of small batch size.
        # Only the attention/SSM layers need to know the seqlen.
        embedding_kwargs = (
            {"combine_batch_seqlen_dim": True}
            if self.process_group is not None and self.sequence_parallel
            else {}
        )
        hidden_states = self.embeddings(
            input_ids, position_ids=position_ids, **embedding_kwargs
        )
        residual = None
        mixer_kwargs = (
            {"seqlen": input_ids.shape[1]}
            if self.process_group is not None and self.sequence_parallel
            else {}
        )
        if inference_params is not None:
            mixer_kwargs["inference_params"] = inference_params
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, mixer_kwargs=mixer_kwargs
            )
        if not self.fused_dropout_add_ln:
            dropped = self.drop_f(hidden_states)
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = dropout_add_layer_norm(
                hidden_states,
                residual,
                self.ln_f.weight,
                self.ln_f.bias,
                self.drop_f.p if self.training else 0.0,
                self.ln_f.eps,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


class ConvLMHeadModel(nn.Module, GenerationMixin):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_inner: int,
        vocab_size: int,
        process_group=None,
        layer=None,
        attn_layer_idx=None,
        attn_cfg=None,
        max_position_embeddings=0,
        resid_dropout: float = 0.0,
        embed_dropout: float = 0.1,
        dropout_cls=nn.Dropout,
        layer_norm_epsilon: float = 1e-5,
        initializer_cfg=None,
        fused_mlp=False,
        fused_dropout_add_ln=False,
        residual_in_fp32=False,
        pad_vocab_size_multiple: int = 1,
        sequence_parallel=True,
        checkpoint_mlp=False,
        checkpoint_mixer=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.process_group = process_group
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (
                vocab_size % pad_vocab_size_multiple
            )
        self.backbone = LMBackbone(
            d_model=d_model,
            n_layer=n_layer,
            d_inner=d_inner,
            vocab_size=vocab_size,
            process_group=process_group,
            layer=layer,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            max_position_embeddings=max_position_embeddings,
            resid_dropout=resid_dropout,
            embed_dropout=embed_dropout,
            dropout_cls=dropout_cls,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_cfg=initializer_cfg,
            fused_mlp=fused_mlp,
            fused_dropout_add_ln=fused_dropout_add_ln,
            residual_in_fp32=residual_in_fp32,
            sequence_parallel=sequence_parallel,
            checkpoint_mlp=checkpoint_mlp,
            checkpoint_mixer=checkpoint_mixer,
            **factory_kwargs,
            **kwargs,
        )
        if process_group is None:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
        else:
            if ColumnParallelLinear is None:
                raise ImportError("fused_dense_lib is not installed")
            self.lm_head = ColumnParallelLinear(
                d_model,
                vocab_size,
                process_group,
                bias=False,
                sequence_parallel=sequence_parallel,
                **factory_kwargs,
            )
        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight
        if self.process_group is not None:
            sync_shared_params(self, self.process_group)

    def forward(
        self, input_ids, position_ids=None, inference_params=None, state=None
    ):  # state for the repo interface
        hidden_states = self.backbone(
            input_ids, position_ids=position_ids, inference_params=inference_params
        )
        lm_logits = self.lm_head(hidden_states)
        # During inference, we want the full logit for sampling
        if ColumnParallelLinear is not None and inference_params is not None:
            if isinstance(self.lm_head, ColumnParallelLinear):
                lm_logits, _ = all_gather_raw(lm_logits, self.lm_head.process_group)
                lm_logits = rearrange(
                    lm_logits, "(n b) s d -> b s (n d)", b=hidden_states.shape[0]
                )
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits), None
