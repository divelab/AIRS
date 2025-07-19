import copy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[...]


class CNNModel(nn.Module):
    def __init__(self, args, alphabet_size, num_cls, classifier=False):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.args = args
        self.classifier = classifier
        self.num_cls = num_cls

        if self.args.clean_data:
            self.linear = nn.Embedding(self.alphabet_size, embedding_dim=args.hidden_dim)
        else:
            expanded_simplex_input = args.cls_expanded_simplex or not classifier and (args.mode == 'dirichlet' or args.mode == 'riemannian')
            inp_size = self.alphabet_size * (2 if expanded_simplex_input else 1)
            if (args.mode == 'ardm' or args.mode == 'lrar') and not classifier:
                inp_size += 1 # plus one for the mask token of these models
            self.linear = nn.Conv1d(inp_size, args.hidden_dim, kernel_size=9, padding=4)
            self.time_embedder = nn.Sequential(GaussianFourierProjection(embed_dim= args.hidden_dim),nn.Linear(args.hidden_dim, args.hidden_dim))

        self.num_layers = 5 * args.num_cnn_stacks
        self.convs = [nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, padding=4),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, padding=4),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, dilation=64, padding=256)]
        self.convs = nn.ModuleList([copy.deepcopy(layer) for layer in self.convs for i in range(args.num_cnn_stacks)])
        self.time_layers = nn.ModuleList([Dense(args.hidden_dim, args.hidden_dim) for _ in range(self.num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(args.hidden_dim) for _ in range(self.num_layers)])
        self.final_conv = nn.Sequential(nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv1d(args.hidden_dim, args.hidden_dim if classifier else self.alphabet_size, kernel_size=1))
        self.dropout = nn.Dropout(args.dropout)
        if classifier:
            self.cls_head = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(args.hidden_dim, self.num_cls))

        if self.args.cls_free_guidance and not self.classifier:
            self.cls_embedder = nn.Embedding(num_embeddings=self.num_cls + 1, embedding_dim=args.hidden_dim)
            self.cls_layers = nn.ModuleList([Dense(args.hidden_dim, args.hidden_dim) for _ in range(self.num_layers)])

    def forward(self, seq, t, cls = None, return_embedding=False):
        if self.args.clean_data:
            feat = self.linear(seq)
            feat = feat.permute(0, 2, 1)
        else:
            time_emb = F.relu(self.time_embedder(t))
            feat = seq.permute(0, 2, 1)
            feat = F.relu(self.linear(feat))

        if self.args.cls_free_guidance and not self.classifier:
            cls_emb = self.cls_embedder(cls)

        for i in range(self.num_layers):
            h = self.dropout(feat.clone())
            if not self.args.clean_data:
                h = h + self.time_layers[i](time_emb)[:, :, None]
            if self.args.cls_free_guidance and not self.classifier:
                h = h + self.cls_layers[i](cls_emb)[:, :, None]
            h = self.norms[i]((h).permute(0, 2, 1))
            h = F.relu(self.convs[i](h.permute(0, 2, 1)))
            if h.shape == feat.shape:
                feat = h + feat
            else:
                feat = h
        feat = self.final_conv(feat)
        feat = feat.permute(0, 2, 1)
        if self.classifier:
            feat = feat.mean(dim=1)
            if return_embedding:
                embedding = self.cls_head[:1](feat)
                return self.cls_head[1:](embedding), embedding
            else:
                return self.cls_head(feat)
        return feat
