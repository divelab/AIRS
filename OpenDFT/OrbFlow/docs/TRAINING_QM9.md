# QM9 — 2-phase final

Script: [`vision/train_qhf_eqv3_gt_2phase_final.slurm`](../vision/train_qhf_eqv3_gt_2phase_final.slurm)

| Item | Value |
|------|--------|
| Backbone | EquiformerV3, \(K{=}8\), \(\ell_{\max}{=}6\), channels 128/256/128 |
| Model | `qm9_flow_qhflow` |
| Prior | `irrep_gaussian` (µ/σ set in the script) |
| β | 1.3 |
| Virtual nodes | bond midpoints (`data=qm9_bond`) |
| Phase 1 | 0–75k, endpoint density |
| Phase 2 | 75k–450k, integrated density, Euler \(K{=}2\) |
| GPUs | 8 (default) |
| Seeds | 0, 42, 123 |

Needs `.env` and `$DATAPATH/{lmdb,lmdb_gt_ridge,datasplits.json}`. Copy `data_splits/qm9_datasplits.json` to `$DATAPATH/datasplits.json` if you rebuilt the LMDB — see [`DATA.md`](DATA.md).

```bash
sbatch vision/train_qhf_eqv3_gt_2phase_final.slurm
sbatch vision/train_qhf_eqv3_gt_2phase_final_s42.slurm
sbatch vision/train_qhf_eqv3_gt_2phase_final_s123.slurm
```

Re-submit the same script / `EXP_NAME` to resume. Outputs: `$PROJECT_ROOT/qm9_qhf_eqv3_gt_2ph_final/`.

```bash
sbatch vision/test_qhf_eqv3_gt_2ph_final_full.slurm
```
