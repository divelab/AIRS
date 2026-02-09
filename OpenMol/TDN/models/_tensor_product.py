from __future__ import annotations

import collections
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3._tensor_product._codegen import _sum_tensors
from e3nn.util.jit import script
from opt_einsum_fx import optimize_einsums_full
from scipy.spatial.transform import Rotation
from torch import fx, nn


@dataclass(frozen=True)
class _Instruction:
    i_in1: int
    i_in2: int
    i_out: int


def _generate_fully_connected_instructions(lmax: int):
    """
    FullyConnectedTensorProduct rule:
        |l1 - l2| <= l_out <= l1 + l2
    for all l1, l2, l_out in [0, ..., lmax].
    """
    instr = []
    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):
            lmin = abs(l1 - l2)
            lmax_pair = min(lmax, l1 + l2)
            for l_out in range(lmin, lmax_pair + 1):
                instr.append(_Instruction(i_in1=l1, i_in2=l2, i_out=l_out))
    return instr


def _check_multiplicities(ch_in1: int, ch_in2: int, ch_out: int, mode: str):
    if mode == "uvw":
        return
    elif mode == "uuu":
        if not (ch_in1 == ch_in2 == ch_out):
            raise ValueError(
                f"mode='uuu' requires channels_in1 == channels_in2 == channels_out, "
                f"got {ch_in1}, {ch_in2}, {ch_out}"
            )
    elif mode == "uuw":
        if ch_in1 != ch_in2:
            raise ValueError(
                f"mode='uuw' requires channels_in1 == channels_in2, got {ch_in1} and {ch_in2}"
            )
    elif mode == "uvu":
        if ch_in1 != ch_out:
            raise ValueError(
                f"mode='uvu' requires channels_out == channels_in1, got {ch_out} and {ch_in1}"
            )
    elif mode == "uvv":
        if ch_in2 != ch_out:
            raise ValueError(
                f"mode='uvv' requires channels_out == channels_in2, got {ch_out} and {ch_in2}"
            )
    else:
        raise ValueError(
            f"Unsupported connection_mode '{mode}'. "
            f"Supported: 'uvw', 'uvu', 'uvv', 'uuw', 'uuu'."
        )


def _example_weight_shape(
    batchdim: int,
    channels_in1: int,
    channels_in2: int,
    channels_out: int,
    mode: str,
):
    if mode == "uvw":
        return (batchdim, channels_in1, channels_in2, channels_out)
    if mode in ("uvu", "uvv"):
        return (batchdim, channels_in1, channels_in2)
    if mode == "uuw":
        return (batchdim, channels_in1, channels_out)
    if mode == "uuu":
        return (batchdim, channels_in1)
    raise RuntimeError(f"Unsupported connection_mode '{mode}' in optimize_einsums.")


# Exact CG tensor product (FX codegen)
def codegen_tensor_product(
    lmax: int,
    channels_in1: int,
    channels_in2: int,
    channels_out: int,
    connection_mode: str = "uvu",
    optimize_einsums: bool = True,
) -> fx.GraphModule:
    """
    Exact Clebsch–Gordan tensor product (fully-connected paths).
    """
    mode = connection_mode.lower()
    _check_multiplicities(channels_in1, channels_in2, channels_out, mode)

    instructions = _generate_fully_connected_instructions(lmax)

    graph = fx.Graph()
    tracer = fx.proxy.GraphAppendingTracer(graph)
    constants = collections.OrderedDict()

    # placeholders
    x1 = fx.Proxy(
        graph.placeholder("x1", torch.Tensor), tracer=tracer
    )  # (B, d, ch_in1)
    x2 = fx.Proxy(
        graph.placeholder("x2", torch.Tensor), tracer=tracer
    )  # (B, d, ch_in2)
    w = fx.Proxy(graph.placeholder("w", torch.Tensor), tracer=tracer)

    batch = x1.shape[0]

    dims_per_l = [2 * l + 1 for l in range(lmax + 1)]
    x1_slices = [0, *np.cumsum(dims_per_l).tolist()]
    x2_slices = [0, *np.cumsum(dims_per_l).tolist()]

    x1_by_l = [x1[:, x1_slices[l] : x1_slices[l + 1], :] for l in range(lmax + 1)]
    x2_by_l = [x2[:, x2_slices[l] : x2_slices[l + 1], :] for l in range(lmax + 1)]

    path_outputs = []

    for ins in instructions:
        l1, l2, l_out = ins.i_in1, ins.i_in2, ins.i_out

        w3j_key = f"_w3j_{l1}_{l2}_{l_out}"
        w3j = fx.Proxy(graph.get_attr(w3j_key), tracer=tracer)

        x1_l = x1_by_l[l1]
        x2_l = x2_by_l[l2]

        if mode == "uvw":
            out_l = torch.einsum("ijk,ziu,zjv,zuvw->zkw", w3j, x1_l, x2_l, w)
        elif mode == "uvu":
            out_l = torch.einsum("ijk,ziu,zjv,zuv->zku", w3j, x1_l, x2_l, w)
        elif mode == "uvv":
            out_l = torch.einsum("ijk,ziu,zjv,zuv->zkv", w3j, x1_l, x2_l, w)
        elif mode == "uuw":
            out_l = torch.einsum("ijk,ziu,zju,zuw->zkw", w3j, x1_l, x2_l, w)
        elif mode == "uuu":
            out_l = torch.einsum("ijk,ziu,zju,zu->zku", w3j, x1_l, x2_l, w)
        else:
            raise RuntimeError(f"Unsupported connection_mode '{mode}'.")

        path_outputs.append(out_l)

        if len(w3j.node.users) == 0:
            graph.erase_node(w3j.node)
        else:
            if w3j_key not in constants:
                constants[w3j_key] = o3.wigner_3j(l1, l2, l_out)

    out_chunks = []
    for l_out in range(lmax + 1):
        outs_for_l = [
            t for ins_, t in zip(instructions, path_outputs) if ins_.i_out == l_out
        ]
        if not outs_for_l:
            continue

        dim_out = 2 * l_out + 1
        out_chunks.append(
            _sum_tensors(
                outs_for_l,
                shape=(batch, dim_out, channels_out),
                like=x1,
            )
        )

    out = torch.cat(out_chunks, dim=1) if len(out_chunks) > 1 else out_chunks[0]

    graph.output(out.node, torch.Tensor)
    graph.lint()

    constants_root = torch.nn.Module()
    for k, v in constants.items():
        constants_root.register_buffer(k, v)

    graphmod = fx.GraphModule(constants_root, graph, class_name="clebsch_gordan")

    if optimize_einsums:
        d = sum(2 * l + 1 for l in range(lmax + 1))
        batchdim = 4

        w_shape = _example_weight_shape(
            batchdim, channels_in1, channels_in2, channels_out, mode
        )
        example_inputs = (
            torch.zeros((batchdim, d, channels_in1)),
            torch.zeros((batchdim, d, channels_in2)),
            torch.zeros(w_shape),
        )
        graphmod = optimize_einsums_full(graphmod, example_inputs)

    return graphmod


# CP approximation (FX codegen)
def initialize_abc(d1: int, d2: int, d3: int, rank: int):
    """
    CP factors for a 3rd-order tensor M[i,j,c] ≈ Σ_r A[i,r] B[j,r] C[c,r]
    """
    A = nn.Parameter(torch.empty(d1, rank))
    B = nn.Parameter(torch.empty(d2, rank))
    C = nn.Parameter(torch.empty(d3, rank))
    gain = 1.0 / rank ** (1 / 3)
    nn.init.xavier_uniform_(A, gain=gain)
    nn.init.xavier_uniform_(B, gain=gain)
    nn.init.xavier_uniform_(C, gain=gain)
    return A, B, C


def codegen_tensor_product_cp_approximate(
    lmax: int,
    channels_in1: int,
    channels_in2: int,
    channels_out: int,
    rank: int,
    connection_mode: str = "uvu",
    optimize_einsums: bool = True,
) -> fx.GraphModule:
    """
    CP-approximate tensor product.
    """
    mode = connection_mode.lower()
    _check_multiplicities(channels_in1, channels_in2, channels_out, mode)

    d = sum(2 * l + 1 for l in range(lmax + 1))

    graph = fx.Graph()
    tracer = fx.proxy.GraphAppendingTracer(graph)
    constants = collections.OrderedDict()

    # placeholders
    x1 = fx.Proxy(
        graph.placeholder("x1", torch.Tensor), tracer=tracer
    )  # (B, d, ch_in1)
    x2 = fx.Proxy(
        graph.placeholder("x2", torch.Tensor), tracer=tracer
    )  # (B, d, ch_in2)
    w = fx.Proxy(graph.placeholder("w", torch.Tensor), tracer=tracer)

    A_cp, B_cp, C_cp = initialize_abc(d, d, d, rank)
    constants["cp_a"] = A_cp
    constants["cp_b"] = B_cp
    constants["cp_c"] = C_cp

    A = fx.Proxy(graph.get_attr("cp_a"), tracer=tracer)  # (d, r)
    B = fx.Proxy(graph.get_attr("cp_b"), tracer=tracer)  # (d, r)
    C = fx.Proxy(graph.get_attr("cp_c"), tracer=tracer)  # (d, r)

    if mode == "uvw":
        out = torch.einsum("biu,ir,bjv,jr,cr,buvo->bco", x1, A, x2, B, C, w)
    elif mode == "uvu":
        out = torch.einsum("bio,ir,bjv,jr,cr,bov->bco", x1, A, x2, B, C, w)
    elif mode == "uvv":
        out = torch.einsum("biu,ir,bjo,jr,cr,buo->bco", x1, A, x2, B, C, w)
    elif mode == "uuw":
        out = torch.einsum("biu,ir,bju,jr,cr,buo->bco", x1, A, x2, B, C, w)
    elif mode == "uuu":
        out = torch.einsum("bio,ir,bjo,jr,cr,bo->bco", x1, A, x2, B, C, w)
    else:
        raise RuntimeError(f"Unsupported connection_mode '{mode}'.")

    graph.output(out.node, torch.Tensor)
    graph.lint()

    constants_root = nn.Module()
    for k, v in constants.items():
        constants_root.register_parameter(k, v)

    graphmod = fx.GraphModule(constants_root, graph, class_name="tp_forward_cp_abc")

    if optimize_einsums:
        batchdim = 4
        w_shape = _example_weight_shape(
            batchdim, channels_in1, channels_in2, channels_out, mode
        )
        example_inputs = (
            torch.zeros((batchdim, d, channels_in1)),
            torch.zeros((batchdim, d, channels_in2)),
            torch.zeros(w_shape),
        )
        graphmod = optimize_einsums_full(graphmod, example_inputs)

    return graphmod


# CP init / fitting utilities
def _sample_weights(
    mode: str,
    batch: int,
    ch_in1: int,
    ch_in2: int,
    ch_out: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mode = mode.lower()
    shape = _example_weight_shape(batch, ch_in1, ch_in2, ch_out, mode)
    return torch.randn(shape, device=device, dtype=dtype)


def _make_sh_irreps(lmax: int) -> o3.Irreps:
    return o3.Irreps.spherical_harmonics(lmax, p=1)


def _sample_rep_tensor(
    *,
    init: str,
    batch: int,
    d: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
    sh_irreps: Optional[o3.Irreps] = None,
) -> torch.Tensor:
    init = init.lower()
    if init == "random":
        return torch.randn(batch, d, channels, device=device, dtype=dtype)

    if init == "sh":
        if sh_irreps is None:
            raise ValueError("sh_irreps must be provided when init='sh'.")

        edge_vec = torch.randn(batch, 3, device=device, dtype=dtype)
        y = o3.spherical_harmonics(
            l=sh_irreps,
            x=edge_vec,
            normalize=True,
            normalization="component",
        ).to(device=device, dtype=dtype)

        y = y.unsqueeze(-1)

        # don’t just repeat the same SH in every channel;
        # excite channels with random per-channel mixing.
        coeff = torch.randn(batch, 1, channels, device=device, dtype=dtype)
        return (y * coeff).contiguous()

    raise ValueError(f"Unsupported init='{init}', expected 'random' or 'sh'.")


def fit_cp_to_exact_tp(
    cp_tp: nn.Module,
    exact_tp: nn.Module,
    *,
    lmax: int,
    channels_in1: int,
    channels_in2: int,
    channels_out: int,
    connection_mode: str,
    n_steps: int = 50_000,
    lr: float = 1e-4,
    batch_size: int = 128,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float32,
    log_every: int = 100,
    stop_loss: float = 1e-3,
    max_steps: int = 50_000,
    do_equiv_check: bool = False,
    equiv_every: int = 10_000,
    equiv_trials: int = 50,
    use_amp: bool = True,
    x1_init: str = "random",
    x2_init: str = "sh",
) -> Dict[str, float]:
    """
    Train cp_tp to match exact_tp on random data. exact_tp is frozen.
    Returns a small dict of final metrics.
    """
    mode = connection_mode.lower()
    _check_multiplicities(channels_in1, channels_in2, channels_out, mode)

    device = torch.device(device)
    cp_tp.to(device=device)
    exact_tp.to(device=device)

    exact_tp.eval()
    for p in exact_tp.parameters():
        p.requires_grad_(False)

    cp_tp.train()
    for p in cp_tp.parameters():
        p.requires_grad_(True)

    opt = torch.optim.AdamW(cp_tp.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(
        device.type, enabled=(use_amp and device.type == "cuda")
    )

    d = sum(2 * l + 1 for l in range(lmax + 1))
    sh_irreps = _make_sh_irreps(lmax)

    def _equivariance_error(
        x1: torch.Tensor, x2: torch.Tensor, w: torch.Tensor
    ) -> float:
        with torch.no_grad():
            base = cp_tp(x1, x2, w)

        errs = []
        for _ in range(equiv_trials):
            R = torch.from_numpy(Rotation.random().as_matrix()).to(dtype=torch.float32)
            D = sh_irreps.D_from_matrix(R).to(device=x1.device, dtype=x1.dtype)

            x1_rot = (x1.permute(0, 2, 1) @ D).permute(0, 2, 1).contiguous()
            x2_rot = (x2.permute(0, 2, 1) @ D).permute(0, 2, 1).contiguous()

            with torch.no_grad():
                out_rot = cp_tp(x1_rot, x2_rot, w)
                out_then_rot = (base.permute(0, 2, 1) @ D).permute(0, 2, 1)

            errs.append(torch.sqrt(((out_rot - out_then_rot) ** 2).mean()).item())

        return float(sum(errs) / len(errs))

    last_loss = float("inf")

    for step in range(min(n_steps, max_steps)):
        x1 = _sample_rep_tensor(
            init=x1_init,
            batch=batch_size,
            d=d,
            channels=channels_in1,
            device=device,
            dtype=dtype,
            sh_irreps=sh_irreps,
        )

        x2 = _sample_rep_tensor(
            init=x2_init,
            batch=batch_size,
            d=d,
            channels=channels_in2,
            device=device,
            dtype=dtype,
            sh_irreps=sh_irreps,
        )

        w = _sample_weights(
            mode, batch_size, channels_in1, channels_in2, channels_out, device, dtype
        )

        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast(device.type, enabled=(scaler.is_enabled())):
            pred = cp_tp(x1, x2, w)
            with torch.no_grad():
                target = exact_tp(x1, x2, w)

            mse = F.mse_loss(pred, target)
            loss = torch.sqrt(mse) / torch.sqrt((target**2).mean().clamp_min(1e-12))

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        last_loss = float(loss.detach().cpu().item())

        if step % log_every == 0:
            print(f"[fit_cp] step={step:06d}  rel_rmse={last_loss:.6g}")

        if do_equiv_check and (step > 0) and (step % equiv_every == 0):
            eq_err = _equivariance_error(x1, x2, w)
            print(f"[fit_cp] equiv_err={eq_err:.6g}")

        if last_loss < stop_loss:
            break

    return {"final_rel_rmse": last_loss, "steps": step + 1}


# Rank scheduling
def rank_scheduler(rank_scheduler_type, L):
    if rank_scheduler_type == "linear":
        return 7 * L
    elif rank_scheduler_type == "quadratic":
        return 7 * (L**2)
    elif rank_scheduler_type == "log":
        return 16 * int(math.log(L + 1) ** 2)
    elif rank_scheduler_type == "full":
        return (L + 1) ** 4
    else:
        raise ValueError("Unsupported rank scheduler type.")


class CPTensorProduct(nn.Module):
    def __init__(
        self,
        channels_in1: int,
        channels_in2: int,
        channels_out: int,
        lmax: int,
        connection_mode: str = "uvu",
        rank_scheduler_type: Optional[str] = None,
        init_cache_dir: str = "tmp",
        init_steps: int = 50_000,
        init_lr: float = 1e-4,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()

        self.lmax = int(lmax)
        self.channels_in1 = int(channels_in1)
        self.channels_in2 = int(channels_in2)
        self.channels_out = int(channels_out)
        self.mode = connection_mode.lower()

        _check_multiplicities(
            self.channels_in1, self.channels_in2, self.channels_out, self.mode
        )

        if rank_scheduler_type is None:
            rank = rank_scheduler("quadratic", self.lmax)
        else:
            rank = rank_scheduler(rank_scheduler_type, self.lmax)
        self.rank = int(rank)

        exact_tp = codegen_tensor_product(
            lmax=self.lmax,
            channels_in1=self.channels_in1,
            channels_in2=self.channels_in2,
            channels_out=self.channels_out,
            connection_mode=self.mode,
            optimize_einsums=True,
        )

        self.cp_tp = codegen_tensor_product_cp_approximate(
            lmax=self.lmax,
            channels_in1=self.channels_in1,
            channels_in2=self.channels_in2,
            channels_out=self.channels_out,
            rank=self.rank,
            connection_mode=self.mode,
            optimize_einsums=True,
        )

        os.makedirs(init_cache_dir, exist_ok=True)
        cache_path = os.path.join(
            init_cache_dir,
            f"cp_init_lmax{self.lmax}_cin1{self.channels_in1}_cin2{self.channels_in2}"
            f"_cout{self.channels_out}_rank{self.rank}_mode{self.mode}.pt",
        )

        if os.path.exists(cache_path):
            state = torch.load(cache_path, map_location="cpu")
            self.cp_tp.load_state_dict(state, strict=False)
        else:
            metrics = fit_cp_to_exact_tp(
                cp_tp=self.cp_tp,
                exact_tp=exact_tp,
                lmax=self.lmax,
                channels_in1=self.channels_in1,
                channels_in2=self.channels_in2,
                channels_out=self.channels_out,
                connection_mode=self.mode,
                n_steps=init_steps,
                lr=init_lr,
                batch_size=128,
                device=device,
                do_equiv_check=False,
            )
            torch.save(self.cp_tp.state_dict(), cache_path)
            print(f"[TensorProductRescaleIn] saved CP init to: {cache_path}")
            print(f"[TensorProductRescaleIn] init metrics: {metrics}")

        del exact_tp

        for p in self.cp_tp.parameters():
            p.requires_grad_(False)

        self.cp_tp = script(self.cp_tp)

        self.weight_numel = math.prod(
            _example_weight_shape(
                1, channels_in1, channels_in2, channels_out, self.mode
            )
        )

    def _reshape_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Make weight shape consistent with connection_mode.
        """
        if weight is None:
            raise ValueError("weight must be provided for this tensor product.")

        mode = self.mode
        b = weight.shape[0]
        target_shape = _example_weight_shape(
            b, self.channels_in1, self.channels_in2, self.channels_out, mode
        )
        return weight.view(*target_shape)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        w = self._reshape_weight(weight) if weight is not None else None
        return self.cp_tp(x, y, w)


class CPTensorProductSH(CPTensorProduct):
    def __init__(
        self,
        hidden_channels: int,
        lmax: int,
        rank_scheduler_type: Optional[str] = None,
        init_cache_dir: str = "tmp",
        init_steps: int = 50_000,
        init_lr: float = 1e-4,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__(
            channels_in1=int(hidden_channels),
            channels_in2=1,
            channels_out=int(hidden_channels),
            lmax=int(lmax),
            connection_mode="uvu",
            rank_scheduler_type=rank_scheduler_type,
            init_cache_dir=init_cache_dir,
            init_steps=init_steps,
            init_lr=init_lr,
            device=device,
        )
