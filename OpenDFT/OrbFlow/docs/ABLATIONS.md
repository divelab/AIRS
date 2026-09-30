# Ablations (10% QM9)

Same architecture as the QM9 2-phase recipe, on a **10%** subset, budget **15k → 90k**.

## Scripts

| Paper row | Script |
|-----------|--------|
| Full curriculum \(K{=}2\) | `ablation/scripts/train_curriculum_full.slurm` |
| Phase 1 only | `train_curriculum_phase1_only.slurm` |
| Phase 2 only \(K{=}2\) | `train_curriculum_phase2_only.slurm` |
| Joint single-stage \(K{=}2\) | `train_joint_endpoint_k2.slurm` |
| Rollout \(K{=}1\) (warm start) | `train_curriculum_phase2_k1.slurm` |
| Rollout \(K{=}3\) (warm start) | `train_curriculum_phase2_k3.slurm` |
| Direct regression | `train_direct_regression.slurm` |

Warm-start rows (`phase2_k1`, `phase2_k3`) reuse a seed-matched `curriculum_full` phase-1 checkpoint when present.

## Train

```bash
python ablation/scripts/make_ablation_split.py   # needs $DATAPATH/datasplits.json
sbatch ablation/scripts/train_curriculum_full.slurm
bash ablation/scripts/submit_ablation_train_seeds.sh   # seeds 42 and 123
```

Outputs: `ablation/runs/<name>/` and `ablation/runs/<name>_seed{42,123}/`.

## Eval

```bash
CKPT_PATH=ablation/runs/curriculum_full sbatch ablation/scripts/eval_nmape.slurm
CKPT_PATH=ablation/runs/direct_regression sbatch ablation/scripts/eval_direct.slurm
sbatch ablation/scripts/eval_nfe.slurm
```

Report mean ± std over training seeds `{0, 42, 123}` using best `nmape/val_flow` (or the direct val metric) within 90k steps.
