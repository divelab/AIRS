from e3nn import o3
from .pace.pace import PACE as PACE_official


def get_model(args, statistics, device):
    model_config = dict(
        r_max=args.cutoff,
        num_bessel=args.num_bessel, 
        num_elements=len(statistics["z_table"]),
        average_energy=statistics["average_energy"],
        hidden_irreps=o3.Irreps(args.hidden),
        avg_num_neighbors=statistics["avg_num_neighbors"],
        atomic_inter_scale=statistics["std"],
        atomic_inter_shift=statistics["mean"],
        edge_emb=args.edge_emb,
        num_examples=args.examples,
    )

    if args.model == "pace":
        model = PACE_official(**model_config)
    elif args.model == "pace_3bpa":
        model_config["hidden_irreps"] = o3.Irreps("256x0e + 256x1o + 256x2e")
        model = PACE_official(**model_config)
    elif args.model == "pace_acac":
        model_config["hidden_irreps"] = o3.Irreps("256x0e + 256x1o + 256x2e")
        model = PACE_official(**model_config)
    else:
        raise RuntimeError(f"Unknown model: '{args.model}'")
    model.to(device)
    return model
