#!/usr/bin/env python3
"""Build a fixed 10% QM9 train + 10% test split for ablation studies.

Reads the official ``datasplits.json`` (train / validation / test), writes:

  ablation/data/datasplits_ablation_10pct.json

with:
  - train: 10% of official train (seeded shuffle)
  - validation: official validation (50) for light training monitors
  - test: 10% of official test (seeded shuffle; ablation table metrics)

Example::

  python ablation/scripts/make_ablation_split.py
  python ablation/scripts/make_ablation_split.py --fraction 0.1 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "ablation" / "data" / "datasplits_ablation_10pct.json"


def _subsample(ids: list, fraction: float, rng: np.random.Generator) -> list[int]:
    order = rng.permutation(len(ids))
    n = int(round(len(ids) * float(fraction)))
    if n < 1:
        raise SystemExit(f"fraction {fraction} too small for n={len(ids)}")
    return sorted(int(ids[i]) for i in order[:n])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Official datasplits.json (default: $DATAPATH/datasplits.json)",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Fraction of official train AND official test to keep (default 0.1)",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    src = args.source
    if src is None:
        datapath = Path(
            os.environ.get("DATAPATH", "data")
        )
        src = datapath / "datasplits.json"
    if not src.is_file():
        raise SystemExit(f"missing source split: {src}")

    splits = json.loads(src.read_text(encoding="utf-8"))
    for key in ("train", "validation", "test"):
        if key not in splits:
            raise SystemExit(f"source missing key {key!r}: {src}")

    train = list(splits["train"])
    val = list(splits["validation"])
    test = list(splits["test"])

    # Independent RNG streams so train/test shuffles stay decoupled.
    rng_train = np.random.default_rng(int(args.seed))
    rng_test = np.random.default_rng(int(args.seed) + 1)

    train_idx = _subsample(train, args.fraction, rng_train)
    test_idx = _subsample(test, args.fraction, rng_test)

    meta = {
        "created_by": "ablation/scripts/make_ablation_split.py",
        "source_split": str(src.resolve()),
        "seed": int(args.seed),
        "fraction": float(args.fraction),
        "notes": {
            "train": f"{args.fraction:.0%} of official train (ablation training)",
            "validation": "official validation (monitor only; n=50)",
            "test": f"{args.fraction:.0%} of official test (ablation table metrics)",
        },
        "counts": {
            "source_train": len(train),
            "source_test": len(test),
            "source_validation": len(val),
            "train": len(train_idx),
            "validation": len(val),
            "test": len(test_idx),
        },
    }
    # Datamodule expects ONLY train/validation/test list keys (filters all keys).
    out = {
        "train": train_idx,
        "validation": [int(x) for x in val],
        "test": test_idx,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {meta_path}")
    print(
        f"  train={len(train_idx)}/{len(train)}  "
        f"validation={len(val)}/{len(val)}  "
        f"test={len(test_idx)}/{len(test)}"
    )
    print(f"  fraction={args.fraction}  seed={args.seed}")


if __name__ == "__main__":
    main()
