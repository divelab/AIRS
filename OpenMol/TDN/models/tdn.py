import math

import torch
import torch.nn as nn
from torch.nn import init
import torch_geometric
from torch_cluster import radius_graph
from torch_geometric.utils import remove_isolated_nodes
from torch_scatter import scatter

from e3nn import o3
from e3nn.util.jit import compile_mode

from ._tensor_product import CPTensorProductSH


_MAX_ATOM_TYPE = 80
# Statistics of Subset
_AVG_NUM_NODES = 36.6
_AVG_DEGREE = 18.6

# Statistics of Whole Set
# _AVG_NUM_NODES = 30.0
# _AVG_DEGREE = 16.7


@torch.jit.script
def gaussian(x, mean, std):
    a = (2 * math.pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


# From Graphormer
class GaussianRadialBasisLayer(torch.nn.Module):
    def __init__(self, num_basis, cutoff):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        self.mean = torch.nn.Parameter(torch.zeros(1, self.num_basis))
        self.std = torch.nn.Parameter(torch.zeros(1, self.num_basis))
        self.weight = torch.nn.Parameter(torch.ones(1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, 1))

        self.std_init_max = 1.0
        self.std_init_min = 1.0 / self.num_basis
        self.mean_init_max = 1.0
        self.mean_init_min = 0

        torch.nn.init.uniform_(self.mean, self.mean_init_min, self.mean_init_max)
        torch.nn.init.uniform_(self.std, self.std_init_min, self.std_init_max)
        torch.nn.init.constant_(self.weight, 1)
        torch.nn.init.constant_(self.bias, 0)

    def forward(self, dist):
        x = (dist / self.cutoff).unsqueeze(-1)
        x = self.weight * x + self.bias
        return gaussian(x, self.mean, self.std.abs() + 1e-5)

    def extra_repr(self):
        return "mean_init_max={}, mean_init_min={}, std_init_max={}, std_init_min={}".format(
            self.mean_init_max, self.mean_init_min, self.std_init_max, self.std_init_min
        )


# MLP from Equiformer
class RadialProfile(nn.Module):
    def __init__(self, ch_list, use_layer_norm=True, use_offset=True):
        super().__init__()
        layers = []
        last_i = len(ch_list) - 1

        for i in range(1, len(ch_list)):
            in_ch, out_ch = ch_list[i - 1], ch_list[i]
            is_last = i == last_i
            use_bias = not (is_last and use_offset)
            layers.append(nn.Linear(in_ch, out_ch, bias=use_bias))

            if not is_last:
                if use_layer_norm:
                    layers.append(nn.LayerNorm(out_ch))
                layers.append(nn.SiLU())

        self.net = nn.Sequential(*layers)

        self.offset = None
        if use_offset:
            self.offset = nn.Parameter(torch.zeros(ch_list[-1]))
            fan_in = ch_list[-2]
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            init.uniform_(self.offset, -bound, bound)

    def forward(self, f_in):
        f_out = self.net(f_in)
        if self.offset is not None:
            f_out = f_out + self.offset.unsqueeze(0)
        return f_out


@compile_mode("script")
class Linear(torch.nn.Module):
    """
    Linear over the multiplicity (last) dim, with an optional bias applied ONLY to the scalar (0e) channel.
    Input/Output: [N, (lmax+1)^2, C]
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.linear = torch_geometric.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(1, out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.bias is not None:
            # scalar channel only (index 0 along the component axis)
            x[:, 0, :].add_(self.bias)
        return x


@compile_mode("script")
class Activation(torch.nn.Module):
    """Apply `act` ONLY to the scalar (0e) channel."""

    def __init__(self, act: torch.nn.Module):
        super().__init__()
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.act(x[:, :1, :])
        return torch.cat((x0, x[:, 1:, :]), dim=1)


@compile_mode("script")
class Gate(torch.nn.Module):
    """
    Apply `act` to scalar (0e) channel and `act_gate` to higher-l channel

    Input:  [*, (lmax+1)^2, 2C]
    Output: [*, (lmax+1)^2, C]

    - scalar (0e): act(value)
    - higher-l:    value * act_gate(gate_scalar)
    """

    def __init__(self, lmax: int, act: torch.nn.Module, act_gate: torch.nn.Module):
        super().__init__()
        self.act = act
        self.act_gate = act_gate  # lmax kept for API compatibility

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.size(-1) // 2
        value = x[..., :c]  # [*, comp, C]
        gate = self.act_gate(x[:, 0, c:]).unsqueeze(1)

        v0 = self.act(value[:, :1, :])
        v1 = value[:, 1:, :] * gate
        return torch.cat((v0, v1), dim=1)


class SmoothLeakyReLU(torch.nn.Module):

    def __init__(self, negative_slope: float = 0.2):
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2

    def extra_repr(self) -> str:
        return f"negative_slope={self.alpha}"


def _l_slices(lmax: int):
    """Slices of the (lmax+1)^2 component axis: [0e | 1e | 2e | ...]."""
    s = 0
    out = []
    for l in range(lmax + 1):
        dim = 2 * l + 1
        out.append(slice(s, s + dim))
        s += dim
    return out


@compile_mode("script")
class LayerNorm(torch.nn.Module):
    """
    Per-l normalization (component or norm) based on EquivariantLayerNormV2.
    Input/Output: [N, (lmax+1)^2, C]
    """

    def __init__(
        self,
        hidden_channels: int,
        lmax: int,
        eps: float = 1e-5,
        affine: bool = True,
        normalization: str = "component",
    ):
        super().__init__()
        assert normalization in ("norm", "component")
        self.lmax = lmax
        self.slices = _l_slices(lmax)
        self.eps = eps
        self.affine = affine
        self.normalization = normalization

        if affine:
            self.affine_weight = torch.nn.Parameter(
                torch.ones((lmax + 1, 1, hidden_channels))
            )
            self.affine_bias = torch.nn.Parameter(torch.zeros((1, 1, hidden_channels)))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.Tensor, **_) -> torch.Tensor:
        fields = []
        for l, sl in enumerate(self.slices):
            field = x[:, sl, :]

            if l == 0:
                field = field - field.mean(dim=2, keepdim=True)

            if self.normalization == "norm":
                field_norm = field.pow(2).sum(dim=1)
            else:
                field_norm = field.pow(2).mean(dim=1)

            field_norm = field_norm.mean(dim=1, keepdim=True)
            field_norm = (field_norm + self.eps).pow(-0.5)

            if self.affine:
                field_norm = field_norm * self.affine_weight[l]

            field = field * field_norm.unsqueeze(1)

            if self.affine and l == 0:
                field = field + self.affine_bias

            fields.append(field)

        return torch.cat(fields, dim=1)


@compile_mode("script")
class GraphAttention(torch.nn.Module):
    """
    Nonlinear-message attention based on Equiformer:
      message -> tp -> alpha path + value path -> scatter -> proj
    """

    def __init__(
        self,
        hidden_channels: int,
        lmax: int,
        fc_neurons,
        num_heads: int,
    ):
        super().__init__()
        assert hidden_channels % num_heads == 0
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        self.merge_src = Linear(hidden_channels, hidden_channels, bias=True)
        self.merge_dst = Linear(hidden_channels, hidden_channels, bias=False)

        # tp + radial weights
        self.tp1 = CPTensorProductSH(hidden_channels, lmax)
        self.rad = RadialProfile(fc_neurons + [self.tp1.weight_numel])

        # alpha path
        self.lin_alpha = Linear(hidden_channels, hidden_channels)
        self.alpha_act = SmoothLeakyReLU(0.2)
        self.alpha_dot = torch.nn.Parameter(torch.randn(num_heads, self.head_dim))
        torch_geometric.nn.inits.glorot(self.alpha_dot)
        self.alpha_dropout = torch.nn.Dropout(0.2)

        # value path
        self.lin_value = Linear(hidden_channels, 2 * hidden_channels)
        self.gate = Gate(lmax, torch.nn.SiLU(), torch.nn.Sigmoid())
        self.tp2 = CPTensorProductSH(hidden_channels, lmax)

        # output projection
        self.proj = Linear(hidden_channels, hidden_channels)

    def forward(
        self,
        node_input: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_scalars: torch.Tensor,
        batch,
        **_,
    ) -> torch.Tensor:
        # merge
        message = (
            self.merge_src(node_input)[edge_src] + self.merge_dst(node_input)[edge_dst]
        )

        # tp
        weight = self.rad(edge_scalars)
        message = self.tp1(message, edge_attr, weight)

        # alpha
        alpha = self.lin_alpha(message)[:, 0, :]
        alpha = alpha.reshape(alpha.size(0), self.num_heads, self.head_dim)
        alpha = self.alpha_act(alpha).unsqueeze(1)
        alpha = torch.einsum("bikd,kd->bik", alpha, self.alpha_dot)
        alpha = torch_geometric.utils.softmax(alpha, edge_dst)
        alpha = self.alpha_dropout(alpha).unsqueeze(-1)

        # value
        value = self.gate(self.lin_value(message))
        value = self.tp2(value, edge_attr, weight)
        value = value.reshape(
            value.size(0), value.size(1), self.num_heads, self.head_dim
        )

        # aggregate
        out = scatter(value * alpha, index=edge_dst, dim=0, dim_size=node_input.size(0))
        out = out.reshape(out.size(0), out.size(1), -1)

        return self.proj(out)


@compile_mode("script")
class TransBlock(torch.nn.Module):
    """Pre-norm transformer block based on Equiformer."""

    def __init__(
        self,
        hidden_channels: int,
        lmax: int,
        fc_neurons,
        num_heads: int,
    ):
        super().__init__()
        self.norm1 = LayerNorm(hidden_channels, lmax)
        self.attn = GraphAttention(
            hidden_channels=hidden_channels,
            lmax=lmax,
            fc_neurons=fc_neurons,
            num_heads=num_heads,
        )
        self.norm2 = LayerNorm(hidden_channels, lmax)
        self.ffn = torch.nn.Sequential(
            Linear(hidden_channels, 2 * hidden_channels),
            Gate(lmax, torch.nn.SiLU(), torch.nn.Sigmoid()),
            Linear(hidden_channels, hidden_channels),
        )

    def forward(
        self,
        node_input: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_scalars: torch.Tensor,
        batch,
        **_,
    ) -> torch.Tensor:
        x = node_input
        x = x + self.attn(
            node_input=self.norm1(x, batch=batch),
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_attr,
            edge_scalars=edge_scalars,
            batch=batch,
        )
        x = x + self.ffn(self.norm2(x, batch=batch))
        return x


class ScaledScatter(torch.nn.Module):
    """Scaled scatter block based on Equiformer."""

    def __init__(self, avg_aggregate_num: float):
        super().__init__()
        self.avg_aggregate_num = float(avg_aggregate_num)

    def forward(self, x: torch.Tensor, index: torch.Tensor, **kwargs) -> torch.Tensor:
        out = scatter(x, index, **kwargs)
        return out.div(self.avg_aggregate_num**0.5)

    def extra_repr(self) -> str:
        return f"avg_aggregate_num={self.avg_aggregate_num}"


class NodeEmbeddingNetwork(torch.nn.Module):
    """Node embedding block."""

    def __init__(
        self,
        hidden_channels: int,
        lmax: int,
        max_atom_type: int = _MAX_ATOM_TYPE,
        bias: bool = True,
    ):
        super().__init__()
        self.max_atom_type = max_atom_type
        self.lmax = lmax
        self.hidden_channels = hidden_channels
        self.atom_type_lin = torch_geometric.nn.Linear(
            max_atom_type, hidden_channels, bias=bias
        )
        self.atom_type_lin.weight.data.mul_(self.max_atom_type**0.5)
        self.num_components = (lmax + 1) ** 2

    def forward(self, node_atom: torch.Tensor) -> torch.Tensor:
        onehot = torch.nn.functional.one_hot(node_atom, self.max_atom_type).to(
            dtype=torch.float32
        )
        node_embedding = self.atom_type_lin(onehot)
        out = node_embedding.new_zeros(
            (node_embedding.size(0), self.num_components, self.hidden_channels)
        )
        out[:, 0, :] = node_embedding
        return out


class NodeDegreeEmbeddingNetwork(torch.nn.Module):
    """Node degree embedding block."""

    def __init__(
        self, hidden_channels: int, lmax: int, fc_neurons, avg_aggregate_num: float
    ):
        super().__init__()
        self.exp = Linear(hidden_channels, hidden_channels)
        self.tp = CPTensorProductSH(hidden_channels, lmax)
        self.rad = RadialProfile(fc_neurons + [self.tp.weight_numel])
        self.proj = Linear(hidden_channels, hidden_channels)
        self.scale_scatter = ScaledScatter(avg_aggregate_num)

    def forward(
        self,
        node_input: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_scalars: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> torch.Tensor:
        ones = node_input.new_ones((node_input.size(0), 1, node_input.size(2)))
        node_features = node_input.new_zeros(node_input.shape)
        node_features[:, 0:1, :] = self.exp(ones)

        weight = self.rad(edge_scalars)
        edge_features = self.tp(node_features[edge_src], edge_attr, weight)
        edge_features = self.proj(edge_features)
        return self.scale_scatter(
            edge_features, edge_dst, dim=0, dim_size=node_features.size(0)
        )


class TensorDecompositionNetwork(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_interactions: int = 4,
        num_gaussians: int = 128,
        cutoff: float = 4.5,
        max_num_neighbors: int = 100,
        gradient_force: bool = False,
        lmax: int = 1,
        fc_neurons=(64, 64),
        num_heads: int = 4,
        compile_blocks: bool = True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.number_of_basis = num_gaussians
        self.num_layers = num_interactions
        self.num_heads = num_heads
        self.gradient_force = gradient_force

        self.lmax = lmax
        self.irreps_edge_attr = o3.Irreps(
            "+".join([f"1x{l}e" for l in range(self.lmax + 1)])
        )

        fc_neurons = [self.number_of_basis, *fc_neurons]
        self.atom_embed = NodeEmbeddingNetwork(
            hidden_channels, self.lmax, _MAX_ATOM_TYPE
        )
        self.edge_embed = NodeDegreeEmbeddingNetwork(
            hidden_channels, self.lmax, fc_neurons, _AVG_DEGREE
        )
        self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.cutoff)

        self.blocks = torch.nn.ModuleList(
            [
                TransBlock(
                    hidden_channels=self.hidden_channels,
                    lmax=self.lmax,
                    fc_neurons=fc_neurons,
                    num_heads=self.num_heads,
                )
                for _ in range(self.num_layers)
            ]
        )

        if compile_blocks and hasattr(torch, "compile"):
            self.blocks = torch.nn.ModuleList(
                [torch.compile(b, dynamic=True) for b in self.blocks]
            )

        self.norm = LayerNorm(hidden_channels, self.lmax)
        self.head = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            Activation(torch.nn.SiLU()),
            Linear(hidden_channels, 1),
        )
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)

        self.force_head = GraphAttention(
            hidden_channels, self.lmax, fc_neurons, self.num_heads
        )
        self.force_proj = Linear(hidden_channels, 1)

    def forward(self, data) -> torch.Tensor:
        if self.gradient_force:
            data.pos.requires_grad_(True)

        batch_size = int(data.batch.max().item()) + 1

        edge_index = radius_graph(
            data.pos,
            r=self.cutoff,
            batch=data.batch,
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_index, _, mask = remove_isolated_nodes(
            edge_index, num_nodes=data.num_nodes
        )
        edge_src, edge_dst = edge_index

        atomic_numbers = data.x.long().squeeze(-1)[mask]
        batch = data.batch[mask]
        pos = data.pos[mask]

        edge_vec = pos[edge_src] - pos[edge_dst]
        edge_length = edge_vec.norm(dim=1)

        edge_sh = o3.spherical_harmonics(
            l=self.irreps_edge_attr,
            x=edge_vec,
            normalize=True,
            normalization="component",
        ).unsqueeze(-1)

        edge_length_embedding = self.rbf(edge_length).unsqueeze(1)

        node_embedding = self.atom_embed(atomic_numbers)
        node_degree_embedding = self.edge_embed(
            node_embedding, edge_sh, edge_length_embedding, edge_src, edge_dst
        )
        x = node_embedding + node_degree_embedding

        for block in self.blocks:
            x = block(
                node_input=x,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=edge_length_embedding,
                batch=batch,
            )

        x = self.norm(x, batch=batch)

        # energy
        e_node = self.head(x)
        e_graph = self.scale_scatter(e_node, batch, dim=0, dim_size=batch_size)
        energy = e_graph[:, 0, 0]

        # force
        if not self.gradient_force:
            f_node = self.force_head(
                node_input=x,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=edge_length_embedding,
                batch=batch,
            )
            forces = self.force_proj(f_node).squeeze(-1)[:, 1:4]
        else:
            forces = -torch.autograd.grad(
                energy,
                pos,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
            )[0]

        return energy, forces, mask
