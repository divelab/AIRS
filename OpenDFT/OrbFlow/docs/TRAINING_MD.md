# MD — k3l3_slim

Script: [`vision/train_md_all6_10k100k_k3l3_slim.slurm`](../vision/train_md_all6_10k100k_k3l3_slim.slurm)

| Item | Value |
|------|--------|
| Molecules | benzene, ethanol, phenol, resorcinol, ethane, malonaldehyde |
| Backbone | EquiformerV3 slim: \(K{=}3\), \(\ell_{\max}{=}3\), channels 96/192/96 |
| Model | `qm9_flow_qhflow` + `data=md_bond` |
| β | 1.3 |
| Phase 1 / 2 | 10k endpoint → 110k integrated density, Euler \(K{=}2\) |
| GPUs | 4 (seeds 0, 42); 2 (seed 505) |
| Seeds | 0, 42, 505 |

Per molecule under `$DATAPATH/MD/`: `scdp_lmdb/<mol>/`, `scdp_lmdb_gt_ridge_beta1.3/<mol>/`, `flow_prior_ridge1e6_beta1.3/<mol>/gt_coeff_flow_prior_stats.json`. See [`DATA.md`](DATA.md).

```bash
sbatch vision/train_md_all6_10k100k_k3l3_slim.slurm
sbatch vision/train_md_all6_10k100k_k3l3_slim_seed42.slurm
sbatch vision/train_md_all6_10k100k_k3l3_slim_seed505.slurm

MD_MOLECULE=phenol sbatch --array=0 vision/train_md_all6_10k100k_k3l3_slim.slurm
```

| Array | Molecule |
|-------|----------|
| 0–5 | benzene, ethanol, phenol, resorcinol, ethane, malonaldehyde |

Outputs: `$PROJECT_ROOT/md_<mol>_k3l3_slim_10k100k/` (and `_seed42` / `_seed505`). Re-submit the same array task to resume.

```bash
sbatch vision/test_flow_md_paper_2phase_beta13_all6.slurm
```
