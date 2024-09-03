import torch.nn.functional
import torch_scatter
import e3nn
from e3nn import nn, o3

from .embs import MLP
from .irreps_tools import (
    reshape_irreps,
    get_feasible_irrep,
    recollect_features,
)
from .symmetric_contraction import SymmetricContraction


class EdgeBoosterBlock(torch.nn.Module):
    def __init__(
        self,
        edge_attrs_irreps,
        edge_feats_irreps,
        node_feats_irreps,
        edge_attrs_irreps_out,
        boost_level,
        num_elements,
    ):
        super().__init__()
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.node_feats_irreps = node_feats_irreps
        self.boost_level = boost_level
        self.edge_attrs_irreps_out = edge_attrs_irreps_out
        self.num_elements = num_elements

        x_hidden = node_feats_irreps.count(o3.Irrep(0, 1)) // 2
        self.x_init = MLP(input_dim=num_elements * 2, hidden_dim=x_hidden, output_dim=x_hidden)
        self.edge_hidden_irreps = o3.Irreps(f"{x_hidden}x0e+{x_hidden}x1o+{x_hidden}x2e+{x_hidden}x3o")
        self.x_irrep = o3.Irreps(f"{x_hidden}x0e")
        irreps_mid, instructions = get_feasible_irrep(
            self.x_irrep,
            self.edge_attrs_irreps,
            self.edge_hidden_irreps,
            mode='uvu',
            trainable=True,
        )

        self.tp = o3.TensorProduct(
                self.x_irrep,
                self.edge_attrs_irreps,
                irreps_mid,
                instructions=instructions,
                shared_weights=False,
                internal_weights=False,
        )

        self.weight = e3nn.nn.FullyConnectedNet(
            [self.edge_feats_irreps.num_irreps] + 3 * [64] + [self.tp.weight_numel],
            torch.nn.functional.silu,
        )

        input_irreps = self.edge_hidden_irreps
        self.edge_boost_tps = torch.nn.ModuleList()
        self.edge_boost_tps_weights = torch.nn.ModuleList()
        self.edge_boost_tps_weights_2 = torch.nn.ModuleList()
        self.skip_tp = torch.nn.ModuleList()
        for _ in range(self.boost_level):
            irreps_mid, instructions = get_feasible_irrep(
                input_irreps,
                self.edge_attrs_irreps,
                self.edge_hidden_irreps,
                mode='uvu',
                trainable=True
            )

            self.edge_boost_tps.append(
                o3.TensorProduct(
                    input_irreps,
                    self.edge_attrs_irreps,
                    irreps_mid,
                    instructions=instructions,
                    shared_weights=False,
                    internal_weights=False,
                )
            )

            self.edge_boost_tps_weights.append(
                e3nn.nn.FullyConnectedNet(
                    [self.edge_feats_irreps.num_irreps] + 3 * [64] + [self.edge_boost_tps[-1].weight_numel],
                    torch.nn.functional.silu,
                )
            )

            self.edge_boost_tps_weights_2.append(e3nn.nn.FullyConnectedNet(
                [2 * num_elements] + 3 * [64] + [self.edge_boost_tps[-1].weight_numel],
                torch.nn.functional.silu,
                )
            )

    def forward(self, edge_attrs, edge_feats, x):
        input_edge_attrs = self.tp(self.x_init(x), edge_attrs, weight=self.weight(edge_feats))
        out_edge_attr_list = [input_edge_attrs]
        for tp_idx, edge_boost_tp in enumerate(self.edge_boost_tps):
            weights = self.edge_boost_tps_weights[tp_idx](edge_feats)
            weights_2 = self.edge_boost_tps_weights_2[tp_idx](x)
            input_edge_attrs = edge_boost_tp(input_edge_attrs, edge_attrs, weights + weights_2)
            out_edge_attr_list.append(input_edge_attrs)
        out_edge_attr = torch.cat(out_edge_attr_list, dim=-1)
        out_edge_attr, _ = recollect_features(
            out_edge_attr, self.edge_hidden_irreps * len(out_edge_attr_list))
        return out_edge_attr


class AtomicBaseUpdateBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps,
        node_feats_irreps,
        edge_attrs_irreps,
        edge_feats_irreps,
        target_irreps,
        hidden_irreps,
        num_elements,
        avg_num_neighbors=10,
        use_sc=True,
    ):
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors

        # TensorProduct
        irreps_mid, instructions = get_feasible_irrep(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
            mode='uuu'
        )

        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps

        self.conv_tp_weights = e3nn.nn.FullyConnectedNet(
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        self.conv_tp_weights_2 = e3nn.nn.FullyConnectedNet(
            [2 * num_elements] + 3 * [64] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        self.irreps_out = self.target_irreps

        # Selector TensorProduct
        self.use_sc = use_sc
        if self.use_sc:
            self.skip_tp = o3.FullyConnectedTensorProduct(
                self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
            )
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
        self,
        node_attrs,
        node_feats,
        edge_attrs,
        edge_feats,
        edge_index,
        x,
        node_num,
        no_tp=False,
    ):
        if self.use_sc:
            sc = self.skip_tp(node_feats, node_attrs)
        else:
            sc = None

        sender, receiver = edge_index
        if not no_tp:
            tp_weights = self.conv_tp_weights(edge_feats)
            tp_weights_2 = self.conv_tp_weights_2(x)
            mji = self.conv_tp(node_feats[sender], edge_attrs, tp_weights + tp_weights_2)
            edge_message = mji
        else:
            edge_message = edge_attrs
        bases = torch_scatter.scatter(
            edge_message, index=receiver, dim=0, dim_size=node_num) / self.avg_num_neighbors
        return bases, sc


class PolynomialManyBodyBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps,
        target_irreps,
        correlation,
        use_sc=True,
        num_elements=None,
        num_examples=20,
    ):
        super().__init__()

        # linears to make 3 features
        self.lin1 = o3.Linear(
            node_feats_irreps,
            node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.lin2 = o3.Linear(
            node_feats_irreps,
            node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.lin3 = o3.Linear(
            node_feats_irreps,
            node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.reshape = reshape_irreps(node_feats_irreps)

        self.use_sc = use_sc
        self.symmetric_contractions = SymmetricContraction(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=num_elements,
            num_examples=num_examples,
        )
        # Update linear
        self.linear = o3.Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(self, node_feats, sc, node_attrs, use_direct=True):
        if use_direct:
            return self.lin1(node_feats)
        node_feats_1 = self.lin1(node_feats)
        node_feats_2 = self.lin2(node_feats)
        node_feats_3 = self.lin3(node_feats)
        node_feats = [
            self.reshape(node_feats_1),
            self.reshape(node_feats_2),
            self.reshape(node_feats_3),
        ]
        node_feats = self.symmetric_contractions(node_feats, node_attrs)
        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)


#############################################
### Readout
#############################################
class NonLinearReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in, MLP_irreps, gate):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.hidden_irreps)
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=[gate])
        self.linear_2 = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=o3.Irreps("0e")
        )

    def forward(self, x):  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]


#############################################
### Scaleshift
#############################################
class AtomicScaleShiftBlock(torch.nn.Module):
    def __init__(self, scale, shift, num_elements):
        super().__init__()
        self.register_buffer(
            "scale", torch.ones((1, num_elements) , dtype=torch.get_default_dtype()) * scale
        )
        self.register_buffer(
            "shift", torch.ones((1, num_elements), dtype=torch.get_default_dtype()) * shift
        )

    def forward(self, x, node_feats):
        return (self.scale * node_feats).sum(dim=-1) * x + (self.shift * node_feats).sum(dim=-1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(scale={self.scale:.6f}, shift={self.shift:.6f})"
        )
