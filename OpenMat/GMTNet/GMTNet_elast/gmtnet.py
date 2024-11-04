import torch
from torch import nn
from utils import RBFExpansion
from transformer import ComformerConv, ComformerConvEqui, Piezo_block, Elastic_block
from torch_scatter import scatter



def bond_cosine(r1, r2):
    bond_cosine = torch.sum(r1 * r2, dim=-1) / (
        torch.norm(r1, dim=-1) * torch.norm(r2, dim=-1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine

def equality_adjustment(equality, batch):

    b, l1, l2 = batch.size()
    batch = batch.reshape(b, l1 * l2)
    for i in range(b):
        mask = equality[i, 0]
        for j in range(l1 * l2):
            batch[i, mask[j]] = batch[i, mask[j]].mean()
    
    for i in range(b):
        mask = equality[i, 1]
        for j in range(l1 * l2):
            for k in range(j + 1, l1 * l2):
                if mask[j, k]:
                    # Average the entries in the second batch
                    abs_value = torch.abs(batch[i, j] - batch[i, k]) / 2
                    if batch[i, j] < 0:
                        batch[i, j] = - abs_value
                        batch[i, k] = abs_value
                    else:
                        batch[i, j] = abs_value
                        batch[i, k] = - abs_value
                    
    return batch.reshape(b, l1, l2)

class ComformerEquivariant(nn.Module):
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

        self.output_block = Elastic_block()
        
        self.mask = args.use_mask

        self.reduce = args.reduce_cell

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
        crystal_features = scatter(node_features, data.batch, dim=0, reduce="mean")
        if self.mask:
            crystal_features = torch.bmm(feat_mask, crystal_features.unsqueeze(-1)).squeeze(-1)
        
        outputs = self.output_block(crystal_features)
        outputs = equality_adjustment(equality, outputs)

        return outputs