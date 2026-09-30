import torch
import copy

from .radial_function import RadialFunction
from .utils import reduce_edge


class EdgeDegreeEmbedding(torch.nn.Module):
    """
        Args:
            num_channels (int):             Number of spherical channels

            lmax (int):                     Maximum degree (l)
            mmax (int):                     Maximum order (m)

            so3_rotation (SO3Rotation):     Class to calculate Wigner-D matrices and rotate embeddings
            
            max_num_elements (int):         Maximum number of atomic numbers
            edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                            The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
            use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features

            rescale_factor (float):         Rescale the sum aggregation
    """
    def __init__(
        self,
        num_channels,

        lmax,
        mmax,

        so3_rotation,

        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding,

        rescale_factor
    ):
        super(EdgeDegreeEmbedding, self).__init__()
        self.num_channels = num_channels
        self.lmax = lmax
        self.mmax = mmax
        self.so3_rotation = so3_rotation
        
        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.use_atom_edge_embedding = use_atom_edge_embedding

        if self.use_atom_edge_embedding:
            self.source_embedding = torch.nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = torch.nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            torch.nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            torch.nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None

        # Embedding function of distance
        self.edge_channels_list.append((self.lmax + 1) * self.num_channels)
        self.rad_func = RadialFunction(
            self.edge_channels_list, 
            lmax=self.lmax, 
            mmax=self.mmax, 
            use_rad_l_parametrization=True,
            use_expand=False
        )

        self.rescale_factor = rescale_factor


    def forward(
        self,
        atomic_numbers,
        edge_distance,
        edge_index,
        edge_envelope_weight=None
    ):
        if self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)
        else:
            x_edge = edge_distance

        x_edge_m0 = self.rad_func(x_edge)

        if edge_envelope_weight is not None:
            x_edge_m0 = x_edge_m0 * edge_envelope_weight

        x_edge_m0 = x_edge_m0.view(x_edge_m0.shape[0], (self.lmax + 1), self.num_channels)
        x_edge = torch.bmm(
            self.so3_rotation.wigner_inv.narrow(2, 0, (self.lmax + 1)), 
            x_edge_m0
        )   # (num_edges, (self.lmax + 1) ** 2, self.num_channels)
        #x_edge = torch.einsum(
        #    'bij, bjc -> bic', 
        #    self.so3_rotation.wigner_inv.narrow(2, 0, (self.lmax + 1)),
        #    x_edge_m0
        #)   # (num_edges, (self.lmax + 1) ** 2, self.num_channels)

        outputs = reduce_edge(
            inputs=x_edge,
            edge_index=edge_index[1],
            output_shape=[atomic_numbers.shape[0], x_edge.shape[1], x_edge.shape[2]]
        )
        outputs = outputs / self.rescale_factor
        return outputs
    

    def extra_repr(self):
        return 'rescale_factor={}'.format(self.rescale_factor)