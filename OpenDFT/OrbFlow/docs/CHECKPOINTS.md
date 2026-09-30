# Checkpoints

Trained checkpoints are hosted on Hugging Face: **[divelab/OrbFlow](https://huggingface.co/divelab/OrbFlow)**.

| Runs | Folders | Seeds | Size per run |
|------|---------|-------|--------------|
| QM9 2-phase final | `qm9_qhf_eqv3_gt_2ph_final/` (+ `_s42`, `_s123`) | 0, 42, 123 | 1.2 GB |
| MD k3l3_slim × 6 molecules | `md_<mol>_k3l3_slim_10k100k/` (+ `_seed42`, `_seed505`) | 0, 42, 505 | 155 MB |

Each folder holds the run's **best-validation checkpoint** (`epoch=*-step=*.ckpt`, selected by `nmape/val_flow`),
its `config.yaml` and `metadata.json`. Optimizer state is removed, so the files are for evaluation and inference only;
to continue training, retrain with the recipes in `vision/`. The seed-505 phenol and resorcinol runs stopped at
~60k of their 110k steps (job time limit); their checkpoints are from that point.

## Download

Download into the repository root, where the evaluation scripts look for the run folders. Exclude the model card so it
does not overwrite the code repository's `README.md`:

```bash
hf download divelab/OrbFlow --local-dir . --exclude README.md --exclude .gitattributes   # all runs (≈ 6.4 GB)
hf download divelab/OrbFlow --local-dir . --include "qm9_qhf_eqv3_gt_2ph_final/*"         # a single run
```

## Evaluate

The test scripts pick the best-validation checkpoint in each folder, as used for the paper numbers.

```bash
CKPT_PATH=qm9_qhf_eqv3_gt_2ph_final sbatch vision/test_qhf_eqv3_gt_2ph_final_full.slurm
sbatch vision/test_flow_md_paper_2phase_beta13_all6.slurm                                  # MD, seed 0
EXP_SUFFIX=md_%s_k3l3_slim_10k100k_seed42 sbatch vision/test_flow_md_paper_2phase_beta13_all6.slurm
```

Training-seed ± is over `{0, 42, 123}` for QM9 and `{0, 42, 505}` for MD.

## Expected results

Test nMAPE (%) of the released checkpoints with Euler K = 2, mean ± std over training seeds.

| QM9 (seeds 0 / 42 / 123) | nMAPE (%) |
|---|---|
| Full test set (10,000 molecules) | 0.1518 ± 0.0008 |
| Last 1,600 test molecules | 0.1646 ± 0.0008 |

| MD (seeds 0 / 42 / 505) | nMAPE (%) |
|---|---|
| benzene | 0.4210 ± 0.0017 |
| ethane | 0.8706 ± 0.0084 |
| ethanol | 1.0784 ± 0.0229 |
| malonaldehyde | 1.1771 ± 0.0127 |
| phenol | 0.5753 ± 0.0083 |
| resorcinol | 0.6626 ± 0.0136 |
| average | 0.7975 ± 0.0067 |
