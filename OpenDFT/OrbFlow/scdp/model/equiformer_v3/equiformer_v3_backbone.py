"""
EquiformerV3 backbone adapted for OrbFlow GTO coefficient prediction.

Vendored transformer blocks from https://github.com/atomicarchitects/equiformer_v3
(MIT / Meta FAIR lineage). Graph edges come from the dataset (same as eSCN), not fairchem OTF.

Interface matches ``scdp.model.scn.eSCN``: ``forward(data) -> (coeffs, expo_scaling)`` with
optional ``coeff_xt`` / ``bridge_t`` for flow matching.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from e3nn import o3

from scdp.common.utils import get_edge_vectors_and_lengths
from scdp.model.coeff_flow_matching import SinusoidalTimeEmbedding

from .edge_rot_mat import init_edge_rot_mat
from .envelope import PolynomialEnvelope
from .input_block import EdgeDegreeEmbedding
from .layer_norm import get_normalization_layer
from .radial_function import GaussianSmearing
from .so3 import SO3Rotation
from .transformer_block import TransBlockV3


class EquiformerV3(nn.Module):
    """SE(3)-equivariant graph attention backbone with GTO coefficient readout."""

    def __init__(
        self,
        max_n_Ls: torch.Tensor,
        max_n_orbitals_per_L: torch.Tensor,
        cutoff: float = 6.0,
        max_num_elements: int = 100,
        num_layers: int = 8,
        lmax_list: List[int] | None = None,
        mmax_list: List[int] | None = None,
        sphere_channels: int = 128,
        hidden_channels: int = 256,
        edge_channels: int = 128,
        num_radial_basis: int | None = None,
        distance_resolution: float = 0.02,
        basis_width_scalar: float = 2.0,
        expo_trainable: bool = False,
        enable_flow_coeff_entangle: bool = False,
        flow_bridge_time_dim: int = 0,
        num_neighbors: float | int | None = None,
        attn_hidden_channels: int = 64,
        num_heads: int = 8,
        attn_alpha_channels: int = 32,
        attn_value_channels: int = 16,
        attn_grid_resolution_list: List[int] | None = None,
        ffn_grid_resolution_list: List[int] | None = None,
        norm_type: str = "merge_layer_norm",
        use_atom_edge_embedding: bool = True,
        use_envelope: bool = True,
        use_attn_renorm: bool = True,
        use_add_merge: bool = False,
        use_grid_mlp: bool = True,
        attn_activation: str = "sep-merge_gates2_swiglu",
        ffn_activation: str = "sep-merge_gates2_swiglu",
        alpha_drop: float = 0.0,
        attn_mask_rate: float = 0.0,
        attn_weights_drop: float = 0.1,
        value_drop: float = 0.0,
        drop_path_rate: float = 0.05,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
        gradient_checkpointing_block_list: List[int] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        if lmax_list is None:
            lmax_list = [6]
        if mmax_list is None:
            mmax_list = [2]
        if len(lmax_list) != 1 or len(mmax_list) != 1:
            raise ValueError("EquiformerV3 currently requires len(lmax_list)==len(mmax_list)==1")
        if attn_grid_resolution_list is None:
            attn_grid_resolution_list = [20, 8]
        if ffn_grid_resolution_list is None:
            ffn_grid_resolution_list = [20, 20]

        self.cutoff = float(cutoff)
        self.max_num_elements = int(max_num_elements)
        self.num_layers = int(num_layers)
        self.lmax_list = list(lmax_list)
        self.mmax_list = list(mmax_list)
        self.lmax = int(lmax_list[0])
        self.mmax = int(mmax_list[0])
        self.sphere_channels = int(sphere_channels)
        self.num_channels = self.sphere_channels
        self.sphere_channels_all = self.sphere_channels
        self.hidden_channels = int(hidden_channels)
        self.edge_channels = int(edge_channels)
        self.ffn_hidden_channels = int(hidden_channels)
        self.attn_hidden_channels = int(attn_hidden_channels)
        self.num_heads = int(num_heads)
        self.attn_alpha_channels = int(attn_alpha_channels)
        self.attn_value_channels = int(attn_value_channels)
        self.attn_grid_resolution_list = list(attn_grid_resolution_list)
        self.ffn_grid_resolution_list = list(ffn_grid_resolution_list)
        self.norm_type = norm_type
        self.use_atom_edge_embedding = bool(use_atom_edge_embedding)
        self.use_envelope = bool(use_envelope)
        self.use_attn_renorm = bool(use_attn_renorm)
        self.use_add_merge = bool(use_add_merge)
        self.use_grid_mlp = bool(use_grid_mlp)
        self.attn_activation = attn_activation
        self.ffn_activation = ffn_activation
        self.alpha_drop = float(alpha_drop)
        self.attn_mask_rate = float(attn_mask_rate)
        self.attn_weights_drop = float(attn_weights_drop)
        self.value_drop = float(value_drop)
        self.drop_path_rate = float(drop_path_rate)
        self.proj_drop = float(proj_drop)
        self.ffn_drop = float(ffn_drop)

        avg_degree = float(num_neighbors) if num_neighbors is not None else 23.4
        self.avg_degree = avg_degree

        if num_radial_basis is None:
            num_radial_basis = max(int(self.cutoff / distance_resolution), 1)
        self.num_radial_basis = int(num_radial_basis)

        self.max_n_orbitals_per_L = max_n_orbitals_per_L
        self.expo_trainable = bool(expo_trainable)

        self.sphere_embedding = nn.Embedding(self.max_num_elements, self.num_channels)
        self.distance_expansion = GaussianSmearing(
            0.0,
            self.cutoff,
            self.num_radial_basis,
            float(basis_width_scalar),
        )
        edge_input_channels = int(self.distance_expansion.num_output)
        self.edge_channels_list = [edge_input_channels] + [self.edge_channels] * 2
        self.envelope_func = (
            PolynomialEnvelope(cutoff=self.cutoff, exponent=5) if self.use_envelope else None
        )
        self.so3_rotation = SO3Rotation(self.lmax, self.mmax, use_rotation_mask=False)
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            num_channels=self.num_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            so3_rotation=self.so3_rotation,
            max_num_elements=self.max_num_elements,
            edge_channels_list=self.edge_channels_list,
            use_atom_edge_embedding=self.use_atom_edge_embedding,
            rescale_factor=self.avg_degree,
        )

        if gradient_checkpointing_block_list is None:
            gradient_checkpointing_block_list = [0] * self.num_layers
        if len(gradient_checkpointing_block_list) != self.num_layers:
            raise ValueError("gradient_checkpointing_block_list length must match num_layers")
        self.gradient_checkpointing_block_list = list(gradient_checkpointing_block_list)

        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            if self.gradient_checkpointing_block_list[i] == 1:
                attn_act = self.attn_activation.replace("_mem", "")
                ffn_act = self.ffn_activation.replace("_mem", "")
            else:
                attn_act = self.attn_activation
                ffn_act = self.ffn_activation
            self.blocks.append(
                TransBlockV3(
                    num_in_channels=self.num_channels,
                    attn_hidden_channels=self.attn_hidden_channels,
                    num_heads=self.num_heads,
                    attn_alpha_channels=self.attn_alpha_channels,
                    attn_value_channels=self.attn_value_channels,
                    ffn_hidden_channels=self.ffn_hidden_channels,
                    num_out_channels=self.num_channels,
                    lmax=self.lmax,
                    mmax=self.mmax,
                    so3_rotation=self.so3_rotation,
                    attn_grid_resolution_list=self.attn_grid_resolution_list,
                    ffn_grid_resolution_list=self.ffn_grid_resolution_list,
                    max_num_elements=self.max_num_elements,
                    edge_channels_list=self.edge_channels_list,
                    use_atom_edge_embedding=self.use_atom_edge_embedding,
                    attn_activation=attn_act,
                    use_attn_renorm=self.use_attn_renorm,
                    use_add_merge=self.use_add_merge,
                    use_rad_l_parametrization=True,
                    softcap=None,
                    attn_eps=1e-16,
                    ffn_activation=ffn_act,
                    use_grid_mlp=self.use_grid_mlp,
                    norm_type=self.norm_type,
                    alpha_drop=self.alpha_drop,
                    attn_mask_rate=self.attn_mask_rate,
                    attn_weights_drop=self.attn_weights_drop,
                    value_drop=self.value_drop,
                    drop_path_rate=self.drop_path_rate,
                    proj_drop=self.proj_drop,
                    ffn_drop=self.ffn_drop,
                )
            )

        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=self.lmax,
            num_channels=self.num_channels,
        )

        irreps_in = o3.Irreps(
            [
                (self.sphere_channels_all, (l, 1 if l % 2 == 0 else -1))
                for l in range(self.lmax + 1)
            ]
        )
        irreps_out = o3.Irreps(
            [
                (int(x), (l, 1))
                for l, x in enumerate(self.max_n_orbitals_per_L)
                if int(x) > 0
            ]
        )
        self.orbit_readout = o3.FullyConnectedTensorProduct(irreps_in, irreps_in, irreps_out)
        if self.expo_trainable:
            expo_irreps_out = o3.Irreps([(int(hidden_channels), (0, 1))])
            self.expo_scaling_readout = o3.FullyConnectedTensorProduct(
                irreps_in, irreps_in, expo_irreps_out
            )
            self.expo_scaling_linear = nn.Linear(hidden_channels, int(max_n_Ls))
            nn.init.zeros_(self.expo_scaling_linear.weight)
            nn.init.zeros_(self.expo_scaling_linear.bias)

        self.enable_flow_coeff_entangle = bool(enable_flow_coeff_entangle)
        self._flow_xt_inject: Optional[nn.ModuleList] = None
        if self.enable_flow_coeff_entangle:
            irreps_c = o3.Irreps(
                [
                    (int(x), (ell, 1))
                    for ell, x in enumerate(self.max_n_orbitals_per_L)
                    if int(x) > 0
                ]
            )
            irreps_s = o3.Irreps(
                [
                    (self.sphere_channels_all, (ell, 1 if ell % 2 == 0 else -1))
                    for ell in range(self.lmax + 1)
                ]
            )
            self._flow_xt_inject = nn.ModuleList(
                [o3.Linear(irreps_c, irreps_s) for _ in range(self.num_layers)]
            )

        self.flow_bridge_time_dim = int(flow_bridge_time_dim)
        self._flow_time_emb: Optional[SinusoidalTimeEmbedding] = None
        self._flow_time_inject: Optional[nn.Linear] = None
        if self.flow_bridge_time_dim > 0:
            self._flow_time_emb = SinusoidalTimeEmbedding(self.flow_bridge_time_dim)
            self._flow_time_inject = nn.Linear(self.flow_bridge_time_dim, self.num_channels)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)

    def _forward_edge(self, edge_distance: torch.Tensor, edge_distance_vec: torch.Tensor):
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec, use_rotation_mask=False)
        self.so3_rotation.set_wigner(edge_rot_mat)
        edge_envelope_weight = (
            self.envelope_func(edge_distance) if self.envelope_func is not None else None
        )
        edge_features = self.distance_expansion(edge_distance)
        return edge_features, edge_envelope_weight

    def _forward_embedding(
        self,
        atomic_numbers: torch.Tensor,
        edge_distance: torch.Tensor,
        edge_index: torch.Tensor,
        edge_envelope_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        num_atoms = len(atomic_numbers)
        x = torch.zeros(
            (num_atoms, (self.lmax + 1) ** 2, self.num_channels),
            device=atomic_numbers.device,
            dtype=edge_distance.dtype,
        )
        x[:, 0, :] = self.sphere_embedding(atomic_numbers)
        edge_degree = self.edge_degree_embedding(
            atomic_numbers,
            edge_distance,
            edge_index,
            edge_envelope_weight,
        )
        return x + edge_degree

    def _inject_bridge_time(
        self,
        x: torch.Tensor,
        bridge_t: Optional[torch.Tensor],
        trace: Optional[dict],
    ) -> torch.Tensor:
        if bridge_t is None or self._flow_time_emb is None or self._flow_time_inject is None:
            if trace is not None:
                trace["time_inject_rms"] = 0.0
                trace["l0_embed_rms_before_time"] = float(
                    x[:, 0, :].pow(2).mean().sqrt().item()
                )
                trace["time_inject_over_l0"] = 0.0
            return x

        te = self._flow_time_emb(bridge_t)
        inj_t = self._flow_time_inject(te.to(dtype=x.dtype))
        if trace is not None:
            l0_before = x[:, 0, :]
            trace["time_inject_rms"] = float(inj_t.pow(2).mean().sqrt().item())
            trace["l0_embed_rms_before_time"] = float(l0_before.pow(2).mean().sqrt().item())
            trace["time_inject_over_l0"] = trace["time_inject_rms"] / (
                trace["l0_embed_rms_before_time"] + 1e-12
            )
        x = x.clone()
        x[:, 0, :] = x[:, 0, :] + inj_t
        return x

    def _forward_blocks(
        self,
        x: torch.Tensor,
        atomic_numbers: torch.Tensor,
        edge_distance: torch.Tensor,
        edge_index: torch.Tensor,
        edge_envelope_weight: Optional[torch.Tensor],
        batch: torch.Tensor,
        coeff_xt: Optional[torch.Tensor],
        trace: Optional[dict],
        bridge_xt_layer_mask: Optional[Sequence[bool]] = None,
    ) -> torch.Tensor:
        source_atomic_numbers = atomic_numbers[edge_index[0]]
        target_atomic_numbers = atomic_numbers[edge_index[1]]

        if trace is not None:
            trace["xt_inject_rms_per_layer"] = []
            trace["latent_rms_after_layer"] = []
            trace["xt_inject_over_latent_pre_layer"] = []

        if bridge_xt_layer_mask is not None and len(bridge_xt_layer_mask) != self.num_layers:
            raise ValueError(
                f"bridge_xt_layer_mask length {len(bridge_xt_layer_mask)} != "
                f"num_layers {self.num_layers}"
            )

        for i in range(self.num_layers):
            inject_xt = (
                coeff_xt is not None
                and self._flow_xt_inject is not None
                and (bridge_xt_layer_mask is None or bool(bridge_xt_layer_mask[i]))
            )
            if inject_xt:
                mod = self._flow_xt_inject[i]
                inj = mod(coeff_xt.to(device=x.device, dtype=mod.weight.dtype))
                inj = inj.to(dtype=x.dtype).view_as(x)
                if trace is not None:
                    pre_rms = float(x.pow(2).mean().sqrt().item())
                    inj_rms = float(inj.pow(2).mean().sqrt().item())
                    trace["xt_inject_rms_per_layer"].append(inj_rms)
                    trace["xt_inject_over_latent_pre_layer"].append(
                        inj_rms / (pre_rms + 1e-12)
                    )
                x = x + inj
            elif trace is not None:
                trace["xt_inject_rms_per_layer"].append(0.0)
                trace["xt_inject_over_latent_pre_layer"].append(0.0)

            block = self.blocks[i]
            if self.gradient_checkpointing_block_list[i] == 0:
                x = block(
                    x,
                    source_atomic_numbers,
                    target_atomic_numbers,
                    edge_distance,
                    edge_index,
                    edge_envelope_weight,
                    batch,
                )
            else:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    source_atomic_numbers,
                    target_atomic_numbers,
                    edge_distance,
                    edge_index,
                    edge_envelope_weight,
                    batch,
                    use_reentrant=False,
                )

            if trace is not None:
                trace["latent_rms_after_layer"].append(float(x.pow(2).mean().sqrt().item()))

        return self.norm(x)

    def forward(
        self,
        data,
        return_latent: bool = False,
        return_trace: bool = False,
        coeff_xt: Optional[torch.Tensor] = None,
        bridge_t: Optional[torch.Tensor] = None,
        bridge_xt_layer_mask: Optional[Sequence[bool]] = None,
    ):
        device = data.coords.device
        dtype = data.coords.dtype
        atomic_numbers = data.atom_types.long()
        edge_index = data.edge_index

        edge_distance_vec, edge_distance = get_edge_vectors_and_lengths(
            positions=data.coords,
            edge_index=edge_index,
            shifts=data.shifts,
        )

        edge_distance, edge_envelope_weight = self._forward_edge(edge_distance, edge_distance_vec)
        x = self._forward_embedding(
            atomic_numbers,
            edge_distance,
            edge_index,
            edge_envelope_weight,
        )

        trace = {} if return_trace else None
        x = self._inject_bridge_time(x, bridge_t, trace)
        x = self._forward_blocks(
            x,
            atomic_numbers,
            edge_distance,
            edge_index,
            edge_envelope_weight,
            data.batch,
            coeff_xt,
            trace,
            bridge_xt_layer_mask=bridge_xt_layer_mask,
        )

        x_pt = x.reshape(x.shape[0], -1)
        coeffs = self.orbit_readout(x_pt, x_pt)
        if self.expo_trainable:
            expo_scaling = self.expo_scaling_linear(
                self.expo_scaling_readout(x_pt, x_pt)
            )
        else:
            expo_scaling = None

        if return_latent:
            if return_trace:
                trace["readout_coeff_rms"] = float(coeffs.pow(2).mean().sqrt().item())
                trace["final_latent_rms"] = float(x_pt.pow(2).mean().sqrt().item())
                return coeffs, expo_scaling, x_pt, trace
            return coeffs, expo_scaling, x_pt
        return coeffs, expo_scaling
