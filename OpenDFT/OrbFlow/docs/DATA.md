# Data

All data is hosted on Hugging Face: **[divelab/OrbFlow-data](https://huggingface.co/datasets/divelab/OrbFlow-data)**.
Everything lives under one root, `DATAPATH` (set in `.env`), with this layout:

```text
$DATAPATH/
  datasplits.json                                         # QM9 split (= data_splits/qm9_datasplits.json)
  lmdb/                                                   # QM9 LMDB: geometry + density probes
  lmdb_gt_ridge/                                          # QM9 target coefficients (β=1.3, ridge 1e-6)
  MD/scdp_lmdb/<mol>/                                     # MD LMDB (+ datasplits.json)
  MD/scdp_lmdb_gt_ridge_beta1.3/<mol>/                    # MD target coefficients
  MD/flow_prior_ridge1e6_beta1.3/<mol>/gt_coeff_flow_prior_stats.json   # MD flow-prior µ/σ
```

| Asset | Size | Needed for |
|-------|------|------------|
| `datasplits.json` | 1 MB | QM9 training, evaluation and ablations |
| `lmdb/` (134 shards + `metadata.json`) | 1.4 TB | QM9 training, evaluation and ablations |
| `lmdb_gt_ridge/` (134 shards) | 12 GB | QM9 training and ablations |
| `MD/scdp_lmdb/` (6 molecules) | 24 GB | MD training and evaluation |
| `MD/scdp_lmdb_gt_ridge_beta1.3/` | 0.6 GB | MD training |
| `MD/flow_prior_ridge1e6_beta1.3/` | < 1 MB | MD training |

## Download

```bash
# everything (≈ 1.46 TB)
hf download divelab/OrbFlow-data --repo-type dataset --local-dir "$DATAPATH"

# only MD (≈ 25 GB)
hf download divelab/OrbFlow-data --repo-type dataset --local-dir "$DATAPATH" --include "MD/*"

# QM9 without the 1.4 TB LMDB (e.g. if you rebuild the LMDB yourself, below)
hf download divelab/OrbFlow-data --repo-type dataset --local-dir "$DATAPATH" --include "lmdb_gt_ridge/*" --include datasplits.json
```

`hf` comes with `huggingface_hub` (`pip install -U huggingface_hub`). Interrupted downloads resume when re-run.

## Rebuild instead of downloading (optional)

Every file can be regenerated with the scripts used for the paper.

### QM9 LMDB (SCDP pipeline)

1. Download all 134 `*.tar` files from the [DTU QM9 VASP dataset](https://data.dtu.dk/articles/dataset/QM9_Charge_Densities_and_Energies_Calculated_with_VASP/16794500)
   (≈ 1.1 TB) directly into `$DATAPATH`, keeping the original file names.
2. Build the LMDB (CPU only; bond-midpoint virtual nodes, same settings as [SCDP](https://github.com/kyonofx/scdp)):

   ```bash
   sbatch vision/preprocess_qm9_lmdb.slurm
   cp data_splits/qm9_datasplits.json "$DATAPATH/datasplits.json"
   ```

   The work can be split by tarball index, e.g. `--start_index 0 --end_index 66` and `--start_index 67 --end_index 133`;
   add `--skip_existing` to resume without redoing finished shards.

> [!IMPORTANT]
> `lmdb_gt_ridge/` is matched to `lmdb/` **by shard file name and entry order**, not by molecule ID. If you combine a
> rebuilt LMDB with the hosted target coefficients, build it with the unmodified script from **all 134 original, unrenamed
> DTU tarballs**: tarball *i* (sorted by name) becomes `data.%04d.lmdb` *i*, with entries in tarball order. A mismatch
> stops training with `sidecar shards must match main LMDB shard names` or `Missing gt sidecar entry`.

### Target coefficients and MD priors

```bash
sbatch vision/prepare_flow_gt_coeffs_ridge1e6_gpu.slurm          # QM9  -> lmdb_gt_ridge/
sbatch vision/prepare_md_gt_coeffs_ridge1e6_beta13_gpu.slurm     # MD   -> MD/scdp_lmdb_gt_ridge_beta1.3/<mol>
sbatch vision/compute_md_flow_prior_stats_beta13.slurm           # MD   -> MD/flow_prior_ridge1e6_beta1.3/<mol>
```

Target coefficients are a per-molecule ridge fit of the DFT density (ridge 1e-6, β = 1.3, def2-QZVPPD-derived basis,
full probe grid, float64); the scripts are resumable and write one sidecar shard per LMDB shard. The MD prior is the
per-molecule µ/σ of the train-split coefficients. QM9 needs no prior file: its train recipe sets µ/σ directly.
The MD scripts accept `MD_MOLECULE="benzene ethane"` to limit the molecules.

## Splits

- QM9: `data_splits/qm9_datasplits.json` (identical to the hosted `datasplits.json`)
- MD: `datasplits.json` inside each `MD/scdp_lmdb/<mol>/` (validation = test)
- Ablations (10% of QM9): `ablation/data/datasplits_ablation_10pct.json` (regenerate with `python ablation/scripts/make_ablation_split.py`)
