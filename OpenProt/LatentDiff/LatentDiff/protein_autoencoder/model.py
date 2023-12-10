import torch
import numpy as np
import torch_geometric
from EGNN_SE3 import EGNN as EGNNSE3
import torch.nn.functional as F
import torch.nn as nn
from utils import build_edge_idx
from torch_geometric.utils import from_networkx, unbatch_edge_index
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.nn import BatchNorm, LayerNorm
from torch_scatter import scatter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_printoptions(profile="full")


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, bn=False):
        '''
        3 layer MLP

        Args:
            input_dim: # input layer nodes
            hidden_dim: # hidden layer nodes
            output_dim: # output layer nodes
            activation: activation function
            layer_norm: bool; if True, apply LayerNorm to output
        '''

        # init superclass and hidden/ output layers
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, output_dim)

        self.bn = bn
        if self.bn:
            self.bn = nn.BatchNorm1d(hidden_dim, eps=1e-5, momentum=0.997)

        # init activation function reset parameters
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self):

        # reset model parameters
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, x):

        # forward prop x
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        if self.bn:
            x = self.bn(x)
        x = self.activation(x)
        x = self.lin3(x)

        return x


class EGNNPooling(torch.nn.Module):
    def __init__(self, hidden_dim=32, stride=2, kernel=3, padding=1, attn=False):
        super(EGNNPooling, self).__init__()

        self.hidden_dim = hidden_dim
        self.stride = stride
        self.kernel = kernel
        self.padding = padding

        self.egnnse3 = EGNNSE3(in_node_nf=hidden_dim, hidden_nf=hidden_dim, out_node_nf=hidden_dim, in_edge_nf=hidden_dim,
                         attention=attn, reflection_equiv=False)


        self.edge_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, F.relu)
        self.edge_mlp_after_pooling = MLP(hidden_dim * 2, hidden_dim, hidden_dim, F.relu)

        self.bn_edge = LayerNorm(in_channels=hidden_dim, mode="node")
        self.bn_h = LayerNorm(in_channels=hidden_dim, mode="node")
        self.bn_edge_after_pooling = LayerNorm(in_channels=hidden_dim, mode="node")

    def forward(self, h, coords, batch=None, batched_data=None, edge_index=None):

        ## get initial h and coords for pooling node

        # get number of node in one input graph and number of pooling node
        num_node = int(torch.div(h.shape[0], (batch[-1] + 1), rounding_mode='floor'))
        num_pool_node = int(torch.div(num_node + 2 * self.padding - self.kernel, self.stride, rounding_mode='floor')) + 1

        # build mapping matrix to map original node to initial pooling node
        M = torch.zeros((num_pool_node, num_node + 2 * self.padding)).double().to(device)
        for i in range(num_pool_node):
            M[i, i * self.stride:(i * self.stride + self.kernel)] = 1 / self.kernel
        # create index to get coords for one graph and padding node (padding mode: same)
        h = h.view((batch[-1] + 1), num_node, -1) # B x n x F
        coords = coords.view((batch[-1] + 1), num_node, -1)  # B x n x 3
        index = [0] * self.padding + list(range(0, num_node)) + [num_node - 1] * self.padding
        coords = coords[:, index, :]
        h = h[:, index, :]
        coords_pool = M @ coords # broadcast matrix multiplication
        h_pool = M @ h


        if edge_index is None:
            edge_index = from_networkx(nx.complete_graph(coords.shape[1])).edge_index.to(device)
            edge_index_unbatch = edge_index.unsqueeze(0).repeat((batch[-1] + 1), 1, 1)
        else:
            edge_index_unbatch = torch.stack(unbatch_edge_index(edge_index, batch), dim=0)

        edge_index_unbatch = edge_index_unbatch + self.padding 


        # connect pooling nodes to input graph nodes
        row, col = torch.where(M > 0)
        index_pool = torch.vstack((row + num_node + 2 * self.padding, col))
        index_pool = torch.cat((index_pool, torch.flip(index_pool, dims=[0])), dim=1)
        edge_index_unbatch = torch.cat((edge_index_unbatch, index_pool.unsqueeze(0).repeat(edge_index_unbatch.shape[0], 1, 1)), dim=2) # B x 2 x num_edges
        h = torch.cat((h, h_pool), dim=1) # B x (n + n_pool) x F
        coords = torch.cat((coords, coords_pool), dim=1) # B x (n + n_pool) x 3

        datalist = []
        for i in range(batch[-1] + 1):
            datalist.append(Data(edge_index=edge_index_unbatch[i]))


        # perform egnn
        data = Batch.from_data_list(datalist).to(device)
        h = h.view(-1, self.hidden_dim)
        coords = coords.view(-1, 3)
        row, col = data.edge_index
        out = torch.cat([h[row], h[col]], dim=1)
        edge_attr = self.edge_mlp(out)
        edge_attr = self.bn_edge(edge_attr)
        h, coords = self.egnnse3(h, coords, data.edge_index, edge_attr, data.batch)

        h = self.bn_h(h)


        # keep pooling node
        h = h.view(batch[-1] + 1, -1, h.shape[1])
        h_pool = h[:, (num_node + 2 * self.padding):, :].reshape(-1, h.shape[2])
        coords = coords.view(batch[-1] + 1, -1, coords.shape[1])
        coords_pool = coords[:, (num_node + 2 * self.padding):, :].reshape(-1, coords.shape[2])


        return h_pool, coords_pool

class EGNNUnPooling(torch.nn.Module):
    def __init__(self, hidden_dim=32, stride=2, kernel=3, padding=1, output_padding=1, attn=False):
        super(EGNNUnPooling, self).__init__()

        self.hidden_dim = hidden_dim
        self.stride = stride
        self.kernel = kernel
        self.padding = padding
        self.output_padding = output_padding


        self.egnnse3 = EGNNSE3(in_node_nf=hidden_dim, hidden_nf=hidden_dim, out_node_nf=hidden_dim, in_edge_nf=hidden_dim,
                         attention=attn, reflection_equiv=False)


        self.edge_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, F.relu)
        self.edge_mlp_after_pooling = MLP(hidden_dim * 2, hidden_dim, hidden_dim, F.relu)

        self.bn_edge = LayerNorm(in_channels=hidden_dim, mode="node")
        self.bn_h = LayerNorm(in_channels=hidden_dim, mode="node")
        self.bn_edge_after_pooling = LayerNorm(in_channels=hidden_dim, mode="node")

    def forward(self, h, coords, batch=None, batched_data=None, edge_index=None):

        # initialize coords for pooling node
        num_node = int(torch.div(h.shape[0], (batch[-1] + 1), rounding_mode='floor'))

        # size after padding
        aug_size = (num_node * self.stride - 1) + 2 * (self.kernel - self.padding - 1) + self.output_padding
        out_size = (num_node - 1) * self.stride - 2 * self.padding + (self.kernel - 1) + self.output_padding + 1
        M = torch.zeros((out_size, aug_size)).double().to(device)
        for i in range(out_size):
            M[i, i:(i + self.kernel)] = 1 / self.kernel

        h = h.view((batch[-1] + 1), num_node, -1) # B x n x F
        coords = coords.view((batch[-1] + 1), num_node, -1)  # B x n x 3

        ##### add same position and h on boundry, add average position and h in between #####
        avg = torch.stack([coords[:, 0:-1, :], coords[:, 1:, :]], dim=2).mean(dim=2) # B x (n-1) x 3
        tmp = torch.stack([coords[:, 0:-1, :], avg], dim=2) # B x (n-1) x 2 x 3
        tmp = torch.flatten(tmp, start_dim=1, end_dim=2) # B x 2*(n-1) x 3
        coords = torch.cat([coords[:, 0:1, :],
                            tmp,
                            coords[:, -1:, :].repeat(1,3,1)], dim=1)

        avg = torch.stack([h[:, 0:-1, :], h[:, 1:, :]], dim=2).mean(dim=2) # B x (n-1) x F
        tmp = torch.stack([h[:, 0:-1, :], avg], dim=2) # B x (n-1) x 2 x F
        tmp = torch.flatten(tmp, start_dim=1, end_dim=2) # B x 2*(n-1) x F
        h = torch.cat([h[:, 0:1, :],
                       tmp,
                       h[:, -1:, :].repeat(1,3,1)], dim=1)

        assert h.shape[1] == M.shape[1]


        coords_pool = M @ coords
        h_pool = M @ h


        edge_index = from_networkx(nx.complete_graph(coords.shape[1])).edge_index.to(device)
        edge_index_unbatch = edge_index.unsqueeze(0).repeat((batch[-1] + 1), 1, 1)


        row, col = torch.where(M > 0)
        index = torch.vstack((row + aug_size, col))
        index = torch.cat((index, torch.flip(index, dims=[0])), dim=1)
        edge_index_unbatch = torch.cat((edge_index_unbatch, index.unsqueeze(0).repeat(edge_index_unbatch.shape[0], 1, 1)), dim=2)  # B x 2 x num_edges
        h = torch.cat((h, h_pool), dim=1)
        coords = torch.cat((coords, coords_pool), dim=1)

        datalist = []
        for i in range(batch[-1] + 1):
            datalist.append(Data(edge_index=edge_index_unbatch[i]))

        # perform egnn
        data = Batch.from_data_list(datalist).to(device)

        h = h.view(-1, self.hidden_dim)
        coords = coords.view(-1, 3)
        row, col = data.edge_index
        out = torch.cat([h[row], h[col]], dim=1)
        edge_attr = self.edge_mlp(out)
        edge_attr = self.bn_edge(edge_attr)
        h, coords = self.egnnse3(h, coords, data.edge_index, edge_attr, data.batch)
        h = self.bn_h(h)


        # keep pooling node
        h = h.view(batch[-1] + 1, -1, h.shape[1])
        h_pool = h[:, aug_size:, :].reshape(-1, h.shape[2])
        coords = coords.view(batch[-1] + 1, -1, coords.shape[1])
        coords_pool = coords[:, aug_size:, :].reshape(-1, coords.shape[2])


        return h_pool, coords_pool


class Encoder(torch.nn.Module):
    def __init__(self, n_feat=1, hidden_dim=32, out_node_dim=32, in_edge_dim=32, max_length=256, layers=1,
                 egnn_layers=4, pooling=True, residual=True, attn=False, stride=2, kernel=3, padding=1):
        super(Encoder, self).__init__()

        self.max_length = max_length
        self.out_node_dim = out_node_dim
        self.layers = layers
        self.pooling = pooling


        # Initialize EGNN
        if self.pooling:
            self.poolings = nn.ModuleList()
        for i in range(self.layers):

            if self.pooling:
                self.poolings.append(
                    EGNNPooling(hidden_dim=hidden_dim, stride=stride, kernel=kernel, padding=padding, attn=attn) # original is 2 2 0
                )

        if self.pooling:
            self.bn_pool = torch.nn.ModuleList([LayerNorm(in_channels=hidden_dim, mode="node") for i in range(self.layers)])
        if not self.pooling:
            self.edge_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, F.relu)
            self.bn_edge = LayerNorm(in_channels=hidden_dim, mode="node")

    def forward(self, coords, h, edge_index, batch, batched_data):

        # EGNN
        for i in range(self.layers):

            if not self.pooling:
                row, col = edge_index
                out = torch.cat([h[row], h[col]], dim=1)
                edge_attr = self.edge_mlp(out)
                edge_attr = self.bn_edge(edge_attr)

            if self.pooling:
                if i == 0:
                    h, coords = self.poolings[i](h, coords, batched_data.batch, batched_data, edge_index)
                else:
                    h, coords = self.poolings[i](h, coords, batched_data.batch, batched_data)
                h = self.bn_pool[i](h)

        return coords, h, batched_data, edge_index

class DecoderTranspose(torch.nn.Module):
    def __init__(self, hidden_dim=32, ratio=2, layers=1, attn=False, out_node_dim=32, in_edge_dim=32, egnn_layers=4, residual=True):
        super(DecoderTranspose, self).__init__()

        self.hidden_dim = hidden_dim
        self.ratio = ratio
        self.layers = layers

        self.unpooling = nn.ModuleList()

        for i in range(layers):

            self.unpooling.append(
                EGNNUnPooling(hidden_dim=self.hidden_dim, stride=2, kernel=3, padding=1, output_padding=1, attn=attn)
            )

        self.bn = torch.nn.ModuleList([LayerNorm(in_channels=hidden_dim, mode="node") for i in range(self.layers)])

    def forward(self, coords, h, batch, batched_data, edge_index=None):

        for i in range(self.layers):
            # unpooling
            h, coords = self.unpooling[i](h, coords, batch)
            h = self.bn[i](h)

        return coords, h


class ProAuto(torch.nn.Module):
    def __init__(self, layers=3, mp_steps=4, num_types=27, type_dim=32, hidden_dim=32, out_node_dim=32, in_edge_dim=32,
                 output_pad_dim=1, output_res_dim=26, pooling=True, up_mlp=False, residual=True, noise=False, transpose=False, attn=False,
                 stride=2, kernel=3, padding=1):
        super(ProAuto, self).__init__()

        self.pooling = pooling
        self.noise = noise
        self.transpose = transpose

        self.encoder = Encoder(n_feat=type_dim, hidden_dim=hidden_dim, out_node_dim=hidden_dim,
                               in_edge_dim=hidden_dim, egnn_layers=mp_steps, layers=layers, pooling=self.pooling, residual=residual, attn=attn,
                               stride=stride, kernel=kernel, padding=padding)

        self.decoder = DecoderTranspose(hidden_dim=hidden_dim, ratio=2, layers=layers, attn=attn)

        self.residue_type_embedding = torch.nn.Embedding(num_types, hidden_dim)


        self.edge_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, F.relu)

        self.bn_edge = LayerNorm(in_channels=hidden_dim, mode="node")

        self.mlp_padding = MLP(hidden_dim, hidden_dim, output_pad_dim, F.relu)

        self.mlp_residue = MLP(hidden_dim, hidden_dim * 4, output_res_dim, F.relu)

        self.sigmoid = nn.Sigmoid()

        # VAE
        self.mlp_mu_h = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_sigma_h = nn.Linear(hidden_dim, hidden_dim)

        self.N = torch.distributions.Normal(0,1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl_x = 0
        self.kl_h = 0

    def add_noise(self, inputs, noise_factor=2):
        noisy = inputs + torch.randn_like(inputs) * noise_factor
        return noisy

    def forward(self, batched_data):
        # forward prop

        x, coords_ca, edge_index, batch = batched_data.x, batched_data.coords_ca, batched_data.edge_index, batched_data.batch

        if self.noise:
            coords_ca = self.add_noise(coords_ca)

        h = self.residue_type_embedding(x.squeeze(1).long()).to(device)

        if self.pooling:
            # encoder
            emb_coords_ca, emb_h, batched_data, edge_index = self.encoder(coords_ca, h, edge_index, batch, batched_data)

            mu_h = self.mlp_mu_h(emb_h)
            sigma_h = self.mlp_sigma_h(emb_h)

            z_h = mu_h + torch.exp(sigma_h / 2) * self.N.sample(mu_h.shape)
            self.kl_h = -0.5 * (1 + sigma_h - mu_h ** 2 - torch.exp(sigma_h)).sum() / (batch[-1] + 1)

            assert z_h.shape == emb_h.shape

            # decoder
            coords_ca_pred, h = self.decoder(emb_coords_ca, z_h, batched_data.batch, batched_data)

        else:
            coords_ca_pred, h, batched_data = self.encoder(coords_ca, h, edge_index, batch, batched_data)

        # predict padding
        pad_pred = self.sigmoid(self.mlp_padding(h))

        # predict residue type
        aa_pred = self.mlp_residue(h)

        return coords_ca_pred, aa_pred, pad_pred, self.kl_x, self.kl_h
