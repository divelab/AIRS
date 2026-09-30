#!/usr/bin/env python3
"""Aggregate per-molecule efficiency_test JSON results into a table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _mean_s(d: dict) -> float | None:
    s = d.get("summaries") or {}
    tot = s.get("total_per_mol") or {}
    return tot.get("mean_s")


def _network_s(d: dict) -> float | None:
    s = d.get("summaries") or {}
    for key in ("euler_per_mol", "coeff_per_mol", "forward_per_mol"):
        val = (s.get(key) or {}).get("mean_s")
        if val is not None:
            return val
    return None


def _density_s(d: dict) -> float | None:
    s = d.get("summaries") or {}
    return (s.get("density_per_mol") or {}).get("mean_s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=Path, required=True)
    p.add_argument("--json_out", type=Path, default=None)
    p.add_argument("--csv_out", type=Path, default=None)
    args = p.parse_args()

    rows = []
    for path in sorted(args.results_dir.glob("*.json")):
        if path.name.startswith("summary_"):
            continue
        d = json.loads(path.read_text())
        mean = _mean_s(d)
        if mean is None:
            continue
        row = {
            "model": d.get("model_name") or path.stem.rsplit("_", 1)[0],
            "molecule": d.get("molecule") or path.stem.rsplit("_", 1)[-1],
            "network_s_per_mol": _network_s(d),
            "density_s_per_mol": _density_s(d),
            "mean_s_per_mol": mean,
            "median_s_per_mol": (d.get("summaries") or {})
            .get("total_per_mol", {})
            .get("median_s"),
            "nmape_mean": d.get("nmape_mean"),
            "n_graphs": d.get("max_n_graphs"),
            "batch_size": d.get("batch_size"),
            "gpu": d.get("gpu"),
            "ckpt_file": d.get("ckpt_file"),
            "source_json": str(path),
        }
        rows.append(row)

    if not rows:
        print(f"no result JSONs in {args.results_dir}")
        return

    print(f"\n=== efficiency_test summary ({args.results_dir}) ===")
    print(
        f"{'model':22s} {'molecule':16s} {'network_s':>10s} {'density_s':>10s} "
        f"{'total_s':>10s} {'nMAPE%':>10s} {'n':>6s}"
    )
    by_model: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        bucket = by_model.setdefault(
            r["model"], {"network": [], "density": [], "total": []}
        )
        if r["network_s_per_mol"] is not None:
            bucket["network"].append(r["network_s_per_mol"])
        if r["density_s_per_mol"] is not None:
            bucket["density"].append(r["density_s_per_mol"])
        bucket["total"].append(r["mean_s_per_mol"])
        nm = r["nmape_mean"]
        print(
            f"{r['model']:22s} {r['molecule']:16s} "
            f"{(r['network_s_per_mol'] or float('nan')):10.4f} "
            f"{(r['density_s_per_mol'] or float('nan')):10.4f} "
            f"{r['mean_s_per_mol']:10.4f} "
            f"{(nm if nm is not None else float('nan')):10.4f} "
            f"{(r['n_graphs'] or 0):6d}"
        )
    print("-" * 96)
    print(
        f"{'model':22s} {'AVERAGE':16s} {'network_s':>10s} {'density_s':>10s} "
        f"{'total_s':>10s}"
    )
    averages = {}
    for model, vals in sorted(by_model.items()):
        net = sum(vals["network"]) / len(vals["network"]) if vals["network"] else None
        den = sum(vals["density"]) / len(vals["density"]) if vals["density"] else None
        tot = sum(vals["total"]) / len(vals["total"])
        averages[model] = {
            "network_s": net,
            "density_s": den,
            "total_s": tot,
        }
        print(
            f"{model:22s} {'AVERAGE':16s} "
            f"{(net if net is not None else float('nan')):10.4f} "
            f"{(den if den is not None else float('nan')):10.4f} "
            f"{tot:10.4f}"
        )

    payload = {
        "results_dir": str(args.results_dir),
        "rows": rows,
        "averages": averages,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json_out}")
    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "molecule",
                    "network_s_per_mol",
                    "density_s_per_mol",
                    "mean_s_per_mol",
                    "median_s_per_mol",
                    "nmape_mean",
                    "n_graphs",
                    "batch_size",
                    "gpu",
                    "ckpt_file",
                    "source_json",
                ],
            )
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv_out}")


if __name__ == "__main__":
    main()
