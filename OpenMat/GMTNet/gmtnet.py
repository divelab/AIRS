import torch
from torch import nn
from utils import RBFExpansion
from transformer import ComformerConv, ComformerConvEqui, Gradient_block, Piezo_block, Elastic_block
from torch_scatter import scatter



def bond_cosine(r1, r2):
    bond_cosine = torch.sum(r1 * r2, dim=-1) / (
        torch.norm(r1, dim=-1) * torch.norm(r2, dim=-1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine

def equality_adjustment(equality, batch):
    """
    Adjust the second batch of matrices based on the equality of entries in the first batch.
    """
    b, l1, l2 = batch.size()
    batch = batch.reshape(b, l1 * l2)
    for i in range(b):
        mask = equality[i]
        for j in range(l1 * l2):
            for k in range(j + 1, l1 * l2):
                if mask[j, k]:
                    # Average the entries in the second batch
                    batch[i, j] = batch[i, k] = (batch[i, j] + batch[i, k]) / 2
    return batch.reshape(b, l1, l2)

class GMTNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        embsize = 128
        self.atom_embedding = nn.Linear(
            92, embsize
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=-4.0,
                vmax=0.0,
                bins=512,
            ),
            nn.Linear(512, embsize),
            nn.Softplus(),
        )

        self.att_layers = nn.ModuleList(
            [
                ComformerConv(in_channels=embsize, out_channels=embsize, heads=1, edge_dim=embsize)
                for _ in range(2)
            ]
        )

        self.equi_update = ComformerConvEqui(in_channels=embsize, edge_dim=embsize)

        if args.target == 'dielectric':
            self.output_block = Gradient_block()
        elif args.target == 'piezoelectric':
            self.output_block = Piezo_block()
        elif args.target == 'elastic':
            self.output_block = Elastic_block()
        else:
            print(args.target," property not implemented!")
        
        self.mask = args.use_mask

        self.reduce = args.reduce_cell
        self.etgnn_linear = nn.Linear(embsize, 1)
        

    def forward(self, data, feat_mask, equality) -> torch.Tensor:
        node_features = self.atom_embedding(data.x)
        edge_feat = -0.75 / torch.norm(data.edge_attr, dim=1)
        # edge_feat = torch.norm(data.edge_attr, dim=1)
        edge_features = self.rbf(edge_feat)

        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        # node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        # node_features = self.att_layers[3](node_features, data.edge_index, edge_features)

        node_features = self.equi_update(data, node_features, data.edge_index, edge_features)
        # global mean pooling for high rotation order features
        crystal_features = scatter(node_features, data.batch, dim=0, reduce="mean")
        if self.mask:
            crystal_features = torch.bmm(feat_mask, crystal_features.unsqueeze(-1)).squeeze(-1)
        
        outputs = self.output_block(crystal_features)
        outputs = equality_adjustment(equality, outputs)

        return outputs