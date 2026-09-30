#!/usr/bin/env python3
"""Average Network / Density / Total across independent efficiency_test repeats."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_results import _density_s, _mean_s, _network_s


def _load_dir(results_dir: Path):
    by_model_mol = {}
    gpu = None
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("summary_"):
            continue
        d = json.loads(path.read_text())
        tot = _mean_s(d)
        if tot is None:
            continue
        model = d.get("model_name") or path.stem.rsplit("_", 1)[0]
        mol = d.get("molecule") or path.stem.rsplit("_", 1)[-1]
        by_model_mol[(model, mol)] = {
            "network": _network_s(d),
            "density": _density_s(d),
            "total": tot,
            "nmape": d.get("nmape_mean"),
            "gpu": d.get("gpu"),
        }
        gpu = d.get("gpu") or gpu
    return by_model_mol, gpu


def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_dirs", nargs="+", type=Path)
    p.add_argument("--json_out", type=Path, default=None)
    args = p.parse_args()

    repeats = []
    for d in args.results_dirs:
        rows, gpu = _load_dir(d)
        repeats.append({"dir": str(d), "gpu": gpu, "rows": rows})

    models = sorted({m for r in repeats for m, _ in r["rows"]})
    mols = sorted({mol for r in repeats for _, mol in r["rows"]})

    print("\n=== Per-repeat molecule-averaged times ===")
    print(
        f"{'model':22s} {'repeat':8s} {'network_s':>10s} {'density_s':>10s} "
        f"{'total_s':>10s}  gpu"
    )
    per_model_repeat = defaultdict(lambda: {"network": [], "density": [], "total": []})
    for i, rep in enumerate(repeats, 1):
        by_model = defaultdict(lambda: {"network": [], "density": [], "total": []})
        for (model, mol), vals in rep["rows"].items():
            for k in ("network", "density", "total"):
                if vals[k] is not None:
                    by_model[model][k].append(vals[k])
        for model in models:
            nets = by_model[model]["network"]
            dens = by_model[model]["density"]
            tots = by_model[model]["total"]
            net = statistics.mean(nets) if nets else None
            den = statistics.mean(dens) if dens else None
            tot = statistics.mean(tots) if tots else None
            if net is not None:
                per_model_repeat[model]["network"].append(net)
            if den is not None:
                per_model_repeat[model]["density"].append(den)
            if tot is not None:
                per_model_repeat[model]["total"].append(tot)
            print(
                f"{model:22s} {i:<8d} "
                f"{(net if net is not None else float('nan')):10.4f} "
                f"{(den if den is not None else float('nan')):10.4f} "
                f"{(tot if tot is not None else float('nan')):10.4f}  "
                f"{rep['gpu'] or '?'}"
            )

    print("\n=== Mean ± std over repeats (6-molecule average) ===")
    print(
        f"{'model':22s} {'network_s':>18s} {'density_s':>18s} {'total_s':>18s}"
    )
    averages = {}
    for model in models:
        net_m, net_s = _mean_std(per_model_repeat[model]["network"])
        den_m, den_s = _mean_std(per_model_repeat[model]["density"])
        tot_m, tot_s = _mean_std(per_model_repeat[model]["total"])
        averages[model] = {
            "network_mean": net_m,
            "network_std": net_s,
            "density_mean": den_m,
            "density_std": den_s,
            "total_mean": tot_m,
            "total_std": tot_s,
            "n_repeats": len(per_model_repeat[model]["total"]),
        }
        def _fmt(m, s):
            if m is None:
                return f"{'nan':>18s}"
            return f"{m:8.4f} ± {s:6.4f}"

        print(
            f"{model:22s} {_fmt(net_m, net_s)} {_fmt(den_m, den_s)} {_fmt(tot_m, tot_s)}"
        )

    payload = {
        "results_dirs": [str(d) for d in args.results_dirs],
        "molecules": mols,
        "averages": averages,
        "repeats": [
            {"dir": r["dir"], "gpu": r["gpu"]} for r in repeats
        ],
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
