import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PACE development")
    parser.add_argument("--device", type=int, default=6)
    parser.add_argument("--seed", help="random seed", type=int, default=3)
    parser.add_argument("--split", help="train/test split", type=int, default=1)
    parser.add_argument("--task", type=str, default="benzene")
    parser.add_argument("--model", type=str, default="pace")
    parser.add_argument("--hidden", type=str, default="256x0e + 256x1o + 256x2e + 256x3o")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--val_batch_size", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=2)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--num_bessel", type=int, default=8)
    parser.add_argument("--edge_emb", type=str, default="radial")
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_factor", type=float, default=0.8)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--scheduler_patience", type=int, default=100)
    parser.add_argument("--force_weight", type=float, default=1000.0)
    parser.add_argument("--energy_weight", type=float, default=9.0)
    parser.add_argument("--output_dir", type=str, default="./results/debug/")
    parser.add_argument("--continue_run", action="store_true", default=False)
    return parser
