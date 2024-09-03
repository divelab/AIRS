import torch
from torch_scatter import scatter_sum
from e3nn import o3

from .blocks import (
    EdgeBoosterBlock,
    AtomicBaseUpdateBlock,
    PolynomialManyBodyBlock,
    NonLinearReadoutBlock,
    AtomicScaleShiftBlock
)

from .utils import get_edge_vectors_and_lengths, compute_forces
from .embs import (
    MLP,
    BesselBasis,
    LinearNodeEmbeddingBlock,
    RadialEmbeddingBlock,
    ExponentialBernsteinRadialBasisFunctions,
)


class PACE(torch.nn.Module):
    def __init__(
        self,
        average_energy,
        atomic_inter_scale,
        atomic_inter_shift,
        r_max=5,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=3,
        num_interactions=2,
        num_elements=None,
        booster_dim=256,
        hidden_irreps=o3.Irreps("256x0e + 256x1o + 256x2e + 256x3o"),
        MLP_irreps=o3.Irreps("16x0e"),
        avg_num_neighbors=10,
        correlation=3,
        gate=torch.nn.functional.silu,
        edge_emb="radial",
        num_examples=20,
    ):
        super().__init__()
        # Node embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.average_energy = average_energy

        # Edge embedding
        ### dist emb
        if edge_emb == "expbern":
            self.radial_embedding = ExponentialBernsteinRadialBasisFunctions(
                num_basis_functions=num_bessel,
                cutoff=r_max,
            )
        elif edge_emb == 'bessel':
            self.radial_embedding = BesselBasis(
                r_max=r_max, num_basis = 20, trainable = False
            )
        else:
            self.radial_embedding = RadialEmbeddingBlock(
                r_max=r_max,
                num_bessel=num_bessel,
                num_polynomial_cutoff=num_polynomial_cutoff,
            )
        edge_feats_irreps = o3.Irreps(f"{num_bessel}x0e")
        ### vec sph
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        edge_booster_irreps = (o3.Irreps.spherical_harmonics(max_ell) * booster_dim).sort().irreps.simplify()
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (o3.Irreps.spherical_harmonics(max_ell) * num_features).sort()[0].simplify()

        self.edge_booster = EdgeBoosterBlock(
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            node_feats_irreps=node_feats_irreps,
            boost_level=1,
            edge_attrs_irreps_out=edge_booster_irreps,
            num_elements=num_elements,
        )

        self.abases = torch.nn.ModuleList([])
        self.pmbs = torch.nn.ModuleList([])
        self.readouts = torch.nn.ModuleList([])
        for i in range(num_interactions):
            # if first layer
            if i == 0:
                node_feats_irreps_in = node_feats_irreps
            else:
                node_feats_irreps_in = hidden_irreps

            # if last layer
            if i == num_interactions - 1:
                hidden_irreps_out = o3.Irreps(hidden_irreps[0].__repr__())
            else:
                hidden_irreps_out = hidden_irreps

            abase = AtomicBaseUpdateBlock(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps_in,
                edge_attrs_irreps=edge_booster_irreps,
                edge_feats_irreps=edge_feats_irreps,
                num_elements=num_elements,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
            )

            pmb = PolynomialManyBodyBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
                num_examples=num_examples,
            )
            readout = NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate)

            self.abases.append(abase)
            self.pmbs.append(pmb)
            self.readouts.append(readout)

        # from ScaleShiftMACE
        self.scale_shift = AtomicScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift, num_elements=num_elements
        )

        # Interactions and readout
        print("node_attr_irreps.shape: ", node_attr_irreps)  # 3x0e
        print("node_feats_irreps.shape: ", node_feats_irreps)  # 256x0e
        print("sh_irreps.shape: ", sh_irreps)  # 1x0e+1x1o+1x2e+1x3o
        print("edge_booster_irreps.shape: ", edge_booster_irreps)  # 2x0e+2x1o+2x2e+2x3o
        print("edge_feats_irreps.shape: ", edge_feats_irreps)  # 8x0e
        print("interaction_irreps.shape: ", interaction_irreps)  # 256x0e+256x1o+256x2e+256x3o
        print("hidden_irreps.shape: ", hidden_irreps)  # 256x0e+256x1o+256x2e

    def forward(self, data, training=False):
        # Setup
        data.pos.requires_grad_(True)
        num_graphs = data.ptr.numel() - 1

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.pos,
            edge_index=data.edge_index,
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # initialize invariant edge feature x
        sender, receiver = data.edge_index
        cat_x = torch.cat([data.node_attrs[sender], data.node_attrs[receiver]], dim=1)

        # Interactions
        node_es_list = []
        
        edge_attrs_new = self.edge_booster(
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            x=cat_x,
        )

        for layer_idx, (atomic_base, p_many_body, readout) in enumerate(zip(
            self.abases, self.pmbs, self.readouts
        )):
            node_feats, sc = atomic_base(
                node_attrs=data.node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs_new,
                edge_feats=edge_feats,
                edge_index=data.edge_index,
                x=cat_x,
                node_num=node_feats.shape[0],
                no_tp=True if layer_idx == 0 else False,
            )
            node_feats = p_many_body(
                node_feats=node_feats, sc=sc, node_attrs=data.node_attrs, use_direct=False
            )
            node_es_list.append(readout(node_feats).squeeze(-1))

        # Energy
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )
        node_inter_es = self.scale_shift(node_inter_es, data.node_attrs)
        inter_e = scatter_sum(
            src=node_inter_es, index=data.batch, dim=-1, dim_size=num_graphs
        )  # [n_graphs,]
        total_energy = self.average_energy + inter_e

        # Force
        forces = compute_forces(
            energy=inter_e,
            positions=data.pos,
            training=training,
        )
        return {"energy": total_energy, "force": forces}
