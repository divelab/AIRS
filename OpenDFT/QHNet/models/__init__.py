from .ori_QHNet_wo_bias import QHNet as QHNet_wo_bias
from .ori_QHNet_with_bias import QHNet as QHNet_w_bias


__all__ = ['QHNet_w_bias', 'get_model', 'QHNet_wo_bias']

# version: wo bias and with bias model are used to load the model for the paper reproduction
# QHNet is the clean version, and we use QHNet to build the benchmark performance

def get_model(args):
    if args.version.lower() == 'QHNet_wo_bias'.lower():
        return QHNet_wo_bias(
            in_node_features=1,
            sh_lmax=4,
            hidden_size=128,
            bottle_hidden_size=32,
            num_gnn_layers=5,
            max_radius=15,
            num_nodes=10,
            radius_embed_dim=16
        )
    elif args.version.lower() == 'QHNet_w_bias'.lower():
        return QHNet_w_bias(
            in_node_features=1,
            sh_lmax=4,
            hidden_size=128,
            bottle_hidden_size=32,
            num_gnn_layers=5,
            max_radius=15,
            num_nodes=10,
            radius_embed_dim=16
        )
    else:
        raise NotImplementedError(
            f"the version {args.version} is not implemented.")
