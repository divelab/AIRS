"""Tests for RCPS modules.

"""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from caduceus.modeling_rcps import (
    RCPSEmbedding, RCPSAddNormWrapper, RCPSLMHead, RCPSWrapper
)

from caduceus.modeling_caduceus import CaduceusConfig, CaduceusMixerModel, CaduceusForMaskedLM, create_block


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("seq_len", [512])
@pytest.mark.parametrize("d_model", [256])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_rcps_embedding(batch_size, seq_len, d_model, dtype):
    # Set tolerance
    device = torch.device("cpu")
    rtol, atol = (6e-4, 2e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    # Set seed
    torch.random.manual_seed(0)

    # Define complement map
    str_to_id = {"[CLS]": 0, "[MASK]": 1, "A": 2, "C": 3, "G": 4, "T": 5, "N": 6}
    complement_map = {"A": "T", "C": "G", "G": "C", "T": "A"}
    complement_map = {
        str_to_id[k]: str_to_id[complement_map[k]] if k in complement_map.keys() else v
        for k, v in str_to_id.items()
    }
    vocab_size = 12
    pad_vocab_size_multiple = 8
    if vocab_size % pad_vocab_size_multiple != 0:
        vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
    if vocab_size > len(complement_map):
        for i in range(len(complement_map), vocab_size):
            complement_map[i] = i

    # Generate random sequences
    input_ids = torch.randint(low=1, high=len(str_to_id), size=(batch_size, seq_len), device=device)
    rc_input_ids = torch.flip(input_ids, dims=[-1]).to("cpu").apply_(lambda t: complement_map[t]).to(device)

    # Test RC equivariance of embedding layer
    factory_kwargs = {"device": device, "dtype": dtype}
    embedding = RCPSEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        complement_map=complement_map,
        **factory_kwargs
    ).to(device)
    out_embed = embedding(input_ids)
    rc_out_embed = torch.flip(embedding(rc_input_ids), dims=[-2, -1])
    # Test that channels are 2 * d_model
    assert tuple(out_embed.size()) == (batch_size, seq_len, d_model * 2)
    assert tuple(rc_out_embed.size()) == (batch_size, seq_len, d_model * 2)
    # Test that RC equivariance holds
    assert torch.allclose(out_embed.detach(), rc_out_embed.detach(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("d_model", [128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rcps_wrapper(batch_size, seq_len, d_model, dtype):
    # Set tolerance
    device = torch.device("cuda")
    rtol, atol = (6e-4, 2e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    # Set seed
    torch.random.manual_seed(0)

    # Generate random sequence with 2 * d_model channels
    x = torch.randn(batch_size, seq_len, d_model * 2, device=device, dtype=dtype)
    rc_x = torch.flip(x, dims=[-2, -1])

    factory_kwargs = {"device": device, "dtype": dtype}
    module = nn.Sequential(
        nn.Linear(d_model, d_model, bias=False, **factory_kwargs),
        nn.ReLU(),
        nn.Linear(d_model, d_model*2, bias=True, **factory_kwargs),
        nn.ReLU(),
        nn.Linear(d_model * 2, d_model, bias=True, **factory_kwargs)
    )

    # Test RC equivariance of wrapper
    rcps_module = RCPSWrapper(module).to(device)
    out = rcps_module(x)
    rc_out = torch.flip(rcps_module(rc_x), dims=[-2, -1])
    assert out.size() == x.size()
    assert rc_out.size() == x.size()
    assert torch.allclose(out.detach(), rc_out.detach(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("d_model", [128])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_rcps_add_norm_wrapper(batch_size, seq_len, d_model, dtype):
    # Set tolerance
    device = torch.device("cuda")
    rtol, atol = (6e-4, 2e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    # Set seed
    torch.random.manual_seed(0)

    # Generate random sequence with 2 * d_model channels
    x = torch.randn(batch_size, seq_len, d_model * 2, device=device, dtype=dtype)
    rc_x = torch.flip(x, dims=[-2, -1])

    factory_kwargs = {"device": device, "dtype": dtype}
    norm = RMSNorm(d_model, eps=1e-5, **factory_kwargs)

    # Test RC equivariance of wrapper
    rcps_module = RCPSAddNormWrapper(norm).to(device)
    out = rcps_module(x)
    rc_out = tuple([torch.flip(r, dims=[-2, -1]) for r in rcps_module(rc_x)])
    for f, r in zip(out, rc_out):
        assert f.size() == x.size()
        assert r.size() == x.size()
        assert torch.allclose(f.detach(), r.detach(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("d_model", [128])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("fused_add_norm", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_rcps_mamba_block_wrapper(batch_size, seq_len, d_model, bidirectional, fused_add_norm, dtype):
    # Set tolerance
    device = torch.device("cuda")
    rtol, atol = (6e-4, 2e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    # Set seed
    torch.random.manual_seed(0)

    # Generate random sequence with 2 * d_model channels
    x = torch.randn(batch_size, seq_len, d_model * 2, device=device, dtype=dtype)
    rc_x = torch.flip(x, dims=[-2, -1])

    ssm_cfg = {
        "d_state": 16, "d_conv": 4, "expand": 2, "dt_rank": "auto", "dt_min": 0.001, "dt_max": 0.1, "dt_init": "random",
        "dt_scale": 1.0, "dt_init_floor": 1e-4, "conv_bias": True, "bias": False, "use_fast_path": True
    }
    factory_kwargs = {"device": device, "dtype": dtype}

    mamba_block = create_block(
        d_model,
        ssm_cfg=ssm_cfg,
        norm_epsilon=1e-5,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=fused_add_norm,
        layer_idx=0,
        bidirectional=bidirectional,
        bidirectional_strategy="add",
        bidirectional_weight_tie=True,
        rcps=True,
        **factory_kwargs
    )

    # Test RC equivariance of wrapper
    out = mamba_block(x, residual=None)
    rc_out = tuple([torch.flip(r, dims=[-2, -1]) for r in mamba_block(rc_x, residual=None)])
    for f, r in zip(out, rc_out):
        assert f.size() == x.size()
        assert r.size() == x.size()
        assert torch.allclose(f.detach(), r.detach(), rtol=rtol, atol=atol)

    out = mamba_block(x, residual=x)
    rc_out = tuple([torch.flip(r, dims=[-2, -1]) for r in mamba_block(rc_x, residual=rc_x)])
    for f, r in zip(out, rc_out):
        assert f.size() == x.size()
        assert r.size() == x.size()
        assert torch.allclose(f.detach(), r.detach(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("seq_len", [1, 1024, 2048])
@pytest.mark.parametrize("d_model", [2, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rcps_lm_head(batch_size, seq_len, d_model, dtype):
    # Set tolerance
    device = torch.device("cuda")
    rtol, atol = (6e-4, 2e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2

    # Set seed
    torch.random.manual_seed(0)

    # Define complement map
    str_to_id = {"[CLS]": 0, "[MASK]": 1, "A": 2, "C": 3, "G": 4, "T": 5, "N": 6}
    complement_map = {"A": "T", "C": "G", "G": "C", "T": "A"}
    complement_map = {
        str_to_id[k]: str_to_id[complement_map[k]] if k in complement_map.keys() else v
        for k, v in str_to_id.items()
    }
    factory_kwargs = {"device": device, "dtype": dtype}
    vocab_size = 12
    if vocab_size > len(complement_map):
        for i in range(len(complement_map), vocab_size):
            complement_map[i] = i

    # Instantiate LM head
    lm_head = RCPSLMHead(
        complement_map=complement_map,
        vocab_size=vocab_size,
        true_dim=d_model,
        **factory_kwargs
    )

    # Generate random sequence with 2 * d_model channels
    x = torch.randn(batch_size, seq_len, d_model * 2, device=device, dtype=dtype)
    rc_x = torch.flip(x, dims=[-2, -1])

    # Test RC equivariance of LM head
    out = lm_head(x)
    rc_out = lm_head(rc_x)
    assert tuple(out.size()) == (batch_size, seq_len, vocab_size)
    assert tuple(rc_out.size()) == (batch_size, seq_len, vocab_size)
    assert torch.allclose(
        out.detach(),
        torch.flip(rc_out.detach()[..., lm_head.complement_map], dims=[1]),
        rtol=rtol,
        atol=atol
    )
    assert torch.allclose(
        F.softmax(out, dim=-1).detach(),
        torch.flip(F.softmax(rc_out, dim=-1).detach()[..., lm_head.complement_map], dims=[1]),
        rtol=rtol,
        atol=atol
    )


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("seq_len", [1024, 2048])
@pytest.mark.parametrize("n_layer", [1, 2])
@pytest.mark.parametrize("d_model", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("fused_add_norm", [True, False])
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("bidirectional_weight_tie", [False, True])
def test_rcps_backbone(batch_size, seq_len, n_layer, d_model, dtype, fused_add_norm,
                       bidirectional, bidirectional_weight_tie):
    # Set tolerance
    device = torch.device("cuda")
    rtol, atol = (6e-4, 2e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2

    # Set seed
    torch.random.manual_seed(0)

    # Define complement map
    str_to_id = {"[CLS]": 0, "[MASK]": 1, "A": 2, "C": 3, "G": 4, "T": 5, "N": 6}
    complement_map = {"A": "T", "C": "G", "G": "C", "T": "A"}
    complement_map = {
        str_to_id[k]: str_to_id[complement_map[k]] if k in complement_map.keys() else v
        for k, v in str_to_id.items()
    }

    # Setup CaduceusConfig
    initializer_cfg = {"initializer_range": 0.02, "rescale_prenorm_residual": True, "n_residuals_per_layer": 1}
    ssm_cfg = {
        "d_state": 16, "d_conv": 4, "expand": 2, "dt_rank": "auto", "dt_min": 0.001, "dt_max": 0.1, "dt_init": "random",
        "dt_scale": 1.0, "dt_init_floor": 1e-4, "conv_bias": True, "bias": False, "use_fast_path": True
    }
    config = CaduceusConfig(
        d_model=d_model,
        n_layer=n_layer,
        vocab_size=12,
        ssm_cfg=ssm_cfg,
        rms_norm=True,
        residual_in_fp32=False,
        fused_add_norm=fused_add_norm,
        pad_vocab_size_multiple=8,
        norm_epsilon=1e-5,
        initializer_cfg=initializer_cfg,
        bidirectional=bidirectional,
        bidirectional_strategy="add",
        bidirectional_weight_tie=bidirectional_weight_tie,
        rcps=True,
        complement_map=complement_map,
    )
    factory_kwargs = {"device": device, "dtype": dtype}

    # Instantiate model
    backbone = CaduceusMixerModel(
        config,
        **factory_kwargs,
    ).to(device)

    # Generate random sequences
    input_ids = torch.randint(low=1, high=len(str_to_id), size=(batch_size, seq_len), device=device)
    rc_input_ids = torch.flip(input_ids, dims=[-1]).to("cpu").apply_(lambda t: complement_map[t]).to(device)

    # Test RC equivariance of rc backbone
    out = backbone(input_ids)[0]
    rc_out = backbone(rc_input_ids)[0]
    if isinstance(rc_out, tuple):
        rc_out = tuple([torch.flip(r, dims=[1, 2]) for r in rc_out])
        for f, r in zip(out, rc_out):
            assert f.size() == (batch_size, seq_len, d_model * 2)
            assert r.size() == (batch_size, seq_len, d_model * 2)
            assert torch.allclose(f.detach(), r.detach(), rtol=rtol, atol=atol)
    else:
        # Hidden state size should double
        assert tuple(out.size()) == (batch_size, seq_len, d_model * 2)
        assert tuple(rc_out.size()) == (batch_size, seq_len, d_model * 2)
        assert torch.allclose(out.detach(), torch.flip(rc_out.detach(), dims=[1, 2]), rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("seq_len", [1024, 2048])
@pytest.mark.parametrize("n_layer", [1, 4])
@pytest.mark.parametrize("d_model", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("bidirectional_weight_tie", [False, True])
def test_rcps_mamba_lm(batch_size, seq_len, n_layer, d_model, dtype, bidirectional, bidirectional_weight_tie):
    # Set tolerance
    device = torch.device("cuda")
    rtol, atol = (6e-4, 2e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2

    # Set seed
    torch.random.manual_seed(0)

    # Define complement map
    str_to_id = {"[CLS]": 0, "[MASK]": 1, "A": 2, "C": 3, "G": 4, "T": 5, "N": 6}
    complement_map = {"A": "T", "C": "G", "G": "C", "T": "A"}
    complement_map = {
        str_to_id[k]: str_to_id[complement_map[k]] if k in complement_map.keys() else v
        for k, v in str_to_id.items()
    }

    # Setup CaduceusConfig
    initializer_cfg = {"initializer_range": 0.02, "rescale_prenorm_residual": True, "n_residuals_per_layer": 1}
    ssm_cfg = {
        "d_state": 16, "d_conv": 4, "expand": 2, "dt_rank": "auto", "dt_min": 0.001, "dt_max": 0.1, "dt_init": "random",
        "dt_scale": 1.0, "dt_init_floor": 1e-4, "conv_bias": True, "bias": False, "use_fast_path": True
    }
    config = CaduceusConfig(
        d_model=d_model,
        n_layer=n_layer,
        vocab_size=12,
        ssm_cfg=ssm_cfg,
        rms_norm=True,
        residual_in_fp32=False,
        fused_add_norm=True,
        pad_vocab_size_multiple=8,
        norm_epsilon=1e-5,
        initializer_cfg=initializer_cfg,
        bidirectional=bidirectional,
        bidirectional_strategy="add",
        bidirectional_weight_tie=bidirectional_weight_tie,
        rcps=True,
        complement_map=complement_map,
    )
    factory_kwargs = {"device": device, "dtype": dtype}

    # Instantiate model
    mamba_lm = CaduceusForMaskedLM(
        config=config,
        **factory_kwargs,
    ).to(device)

    # Generate random sequences
    input_ids = torch.randint(low=1, high=len(str_to_id), size=(batch_size, seq_len), device=device)
    rc_input_ids = torch.flip(input_ids, dims=[-1]).to("cpu").apply_(lambda t: complement_map[t]).to(device)

    # Test RC equivariance of rc backbone
    out = mamba_lm(input_ids)
    rc_out = mamba_lm(rc_input_ids)
    if config.vocab_size % config.pad_vocab_size_multiple != 0:
        config.vocab_size += config.pad_vocab_size_multiple - (config.vocab_size % config.pad_vocab_size_multiple)
    assert tuple(out.logits.size()) == (batch_size, seq_len, config.vocab_size)
    assert tuple(rc_out.logits.size()) == (batch_size, seq_len, config.vocab_size)
    assert torch.allclose(
        out.logits.detach(),
        torch.flip(rc_out.logits.detach()[..., mamba_lm.lm_head.complement_map], dims=[1]),
        rtol=rtol,
        atol=atol
    )
    assert torch.allclose(
        F.softmax(out.logits, dim=-1).detach(),
        torch.flip(F.softmax(rc_out.logits, dim=-1).detach()[..., mamba_lm.lm_head.complement_map], dims=[1]),
        rtol=rtol,
        atol=atol
    )


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("n_layer", [2])
@pytest.mark.parametrize("d_model", [128])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("bidirectional_weight_tie", [True])
def test_collapse_invariance(batch_size, seq_len, n_layer, d_model, dtype, bidirectional, bidirectional_weight_tie):
    # Set tolerance
    device = torch.device("cuda")
    rtol, atol = (6e-4, 2e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2

    # Set seed
    torch.random.manual_seed(0)

    # Define complement map
    str_to_id = {"[CLS]": 0, "[MASK]": 1, "A": 2, "C": 3, "G": 4, "T": 5, "N": 6}
    complement_map = {"A": "T", "C": "G", "G": "C", "T": "A"}
    complement_map = {
        str_to_id[k]: str_to_id[complement_map[k]] if k in complement_map.keys() else v
        for k, v in str_to_id.items()
    }

    # Setup CaduceusConfig
    initializer_cfg = {"initializer_range": 0.02, "rescale_prenorm_residual": True, "n_residuals_per_layer": 1}
    ssm_cfg = {
        "d_state": 16, "d_conv": 4, "expand": 2, "dt_rank": "auto", "dt_min": 0.001, "dt_max": 0.1, "dt_init": "random",
        "dt_scale": 1.0, "dt_init_floor": 1e-4, "conv_bias": True, "bias": False, "use_fast_path": True
    }
    config = CaduceusConfig(
        d_model=d_model,
        n_layer=n_layer,
        vocab_size=12,
        ssm_cfg=ssm_cfg,
        rms_norm=True,
        residual_in_fp32=False,
        fused_add_norm=True,
        pad_vocab_size_multiple=8,
        norm_epsilon=1e-5,
        initializer_cfg=initializer_cfg,
        bidirectional=bidirectional,
        bidirectional_strategy="add",
        bidirectional_weight_tie=bidirectional_weight_tie,
        rcps=True,
        complement_map=complement_map,
    )
    factory_kwargs = {"device": device, "dtype": dtype}

    # Instantiate model
    backbone = CaduceusMixerModel(
        config,
        **factory_kwargs,
    ).to(device)

    # Generate random sequences
    input_ids = torch.randint(low=1, high=len(str_to_id), size=(batch_size, seq_len), device=device)
    rc_input_ids = torch.flip(input_ids, dims=[-1]).to("cpu").apply_(lambda t: complement_map[t]).to(device)

    # Test RC Invariance when collapsing output of backbone
    out = backbone(input_ids)[0]
    out_collapse = (out[..., :d_model] + torch.flip(out[..., d_model:], dims=[1, 2])) / 2
    rc_out = backbone(rc_input_ids)[0]
    rc_out_collapse = (rc_out[..., :d_model] + torch.flip(rc_out[..., d_model:], dims=[1, 2])) / 2
    # Hidden state size should be d_model
    assert tuple(out_collapse.size()) == (batch_size, seq_len, d_model)
    assert tuple(rc_out_collapse.size()) == (batch_size, seq_len, d_model)
    assert torch.allclose(out_collapse.detach(), rc_out_collapse.detach(), rtol=rtol, atol=atol)
