import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius_graph
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool, InstanceNorm, avg_pool_x, BatchNorm


class Swish(nn.Module):
    """
    Swish activation function
    """

    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class GNN_Layer(MessagePassing):
    """
    Message passing layer
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 time_window: int):
        """
        Initialize message passing layers
        Args:
            in_features (int): number of node input features
            out_features (int): number of node output features
            hidden_features (int): number of hidden features
            time_window (int): number of input/output timesteps (temporal bundling)
            n_variables (int): number of equation specific parameters used in the solver
        """
        super(GNN_Layer, self).__init__(node_dim=-2, aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + time_window + 1, hidden_features),
                                           Swish()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                           Swish()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features, hidden_features),
                                          Swish()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features),
                                          Swish()
                                          )
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, u, pos, edge_index, batch):
        """
        Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j):
        """
        Message update following formula 8 of the paper
        """
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, torch.norm(pos_i - pos_j, dim=1, keepdim=True)), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x):
        """
        Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message), dim=-1))
        update = self.update_net_2(update)
        if self.in_features == self.out_features:
            return x + update
        else:
            return update


class MP_PDE_Solver(torch.nn.Module):
    """
    MP-PDE solver class
    """

    def __init__(self,
                 n_input_scalar_components: int,
                 n_input_vector_components: int,
                 n_output_scalar_components: int,
                 n_output_vector_components: int,
                 time_history: int,
                 time_future: int,
                 activation: str,
                 hidden_features: int = 128,
                 hidden_layer: int = 6,
                 ):
        """
        Initialize MP-PDE solver class.
        It contains 6 MP-PDE layers with skip connections
        The input graph to the forward pass has the shape [batch*n_nodes, time_window].
        The output graph has the shape [batch*n_nodes, time_window].
        Args:
            time_window (int): number of input/output timesteps (temporal bundling)
            hidden features (int): number of hidden features
            hidden_layer (int): number of hidden layers
        """
        super(MP_PDE_Solver, self).__init__()
        # 1D decoder CNN is so far designed time_window = [20,25,50]
        # assert time_history == time_future
        # assert (time_history == 25 or time_history == 20 or time_history == 50)
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_history = time_history
        self.time_future = time_future

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_history * (n_input_scalar_components + n_input_vector_components * 2),
        ) for _ in range(self.hidden_layer))) #- 1)))

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        # self.gnn_layers.append(GNN_Layer(in_features=self.hidden_features,
        #                                  hidden_features=self.hidden_features,
        #                                  out_features=self.time_future * (n_output_scalar_components + n_output_vector_components * 2),
        #                                  time_window=self.time_history * (n_input_scalar_components + n_input_vector_components * 2),
        #                                  )
        #                        )

        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.time_history * (n_input_scalar_components + n_input_vector_components * 2) + 2,
                      self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
        )

        out_channels = time_future * (self.n_output_scalar_components + self.n_output_vector_components * 2)
        self.output_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, out_channels)
        )


    def image_to_graph(self, image):
        B, C, H, W = image.shape
        nodes = image.view(B, C, H * W).transpose(1, 2)

        neighbor_offsets = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0), (1, 0),
            (-1, 1), (0, 1), (1, 1)
        ]

        edges = [torch.stack([torch.arange(H * W)] * 2).cuda(device=image.device)] # self loops
        for dx, dy in neighbor_offsets:
            # generate all grid positions
            nx, ny = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='ij')
            nx = nx.cuda(device=image.device)
            ny = ny.cuda(device=image.device)
            nx, ny = nx + dx, ny + dy

            # flatten all points
            # source is neighbors and target is the center
            src = ny * W + nx
            dst = (ny - dy) * W + (nx - dx)

            # remove out of boundary points
            valid = (0 <= nx) & (nx < W) & (0 <= ny) & (ny < H)
            src, dst = src[valid], dst[valid]
            edges.append(torch.stack([src, dst], dim=0))

        edge_index = torch.cat(edges, dim=1).cuda(device=image.device)

        # nx,ny=torch.meshgrid(torch.arange(W), torch.arange(H), indexing='ij')
        # grid = torch.concat([nx[None], ny[None]]).flatten(1).transpose(0, 1).float()
        # from torch_geometric.nn import radius_graph
        # edges2=radius_graph(grid.cpu(), r=2**0.5+1e-7, loop=True)
        # set1 = set(tuple(x) for x in edge_index.transpose(0,1).tolist())
        # set2 = set(tuple(x) for x in edges2.transpose(0,1).tolist())
        # assert set1 == set2

        # generate all grid positions
        pos_x, pos_y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='ij')
        pos = torch.stack([pos_x.flatten(), pos_y.flatten()], dim=1).float()

        return Batch.from_data_list(
            [Data(x=nodes[b], edge_index=edge_index.clone(), pos=pos.clone()).cuda(device=image.device) for b in
             range(B)])

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of MP-PDE solver class.
        The input graph has the shape [batch*n_nodes, time_window].
        The output tensor has the shape [batch*n_nodes, time_window].
        Args:
            data (Data): Pytorch Geometric data graph
        Returns:
            torch.Tensor: data output
        """
        assert x.dim() == 5
        B, T, C, H, W = x.shape
        x_reshape = x.reshape(x.size(0), -1, *x.shape[3:])
        data = self.image_to_graph(x_reshape)

        u = data.x
        # Encode and normalize coordinate information
        pos = data.pos / (H - 1)
        edge_index = data.edge_index
        batch = data.batch

        # Encoder and processor (message passing)
        node_input = torch.cat((u, pos), -1)
        h = self.embedding_mlp(node_input)
        for i in range(self.hidden_layer):
            h = self.gnn_layers[i](h, u, pos, edge_index, batch)
        
        # Decoder (formula 10 in the paper)
        dt = 1
        h = h.view(B, H, W, -1)
        diff = self.output_mlp(h).permute(0, 3, 1, 2)
        out = x[:, -1] + dt * diff

        return out.unsqueeze(1)
