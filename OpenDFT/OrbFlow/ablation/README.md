# Ablations

Paper protocol and commands: [`docs/ABLATIONS.md`](../docs/ABLATIONS.md).

```bash
python ablation/scripts/make_ablation_split.py
sbatch ablation/scripts/train_curriculum_full.slurm
CKPT_PATH=ablation/runs/curriculum_full sbatch ablation/scripts/eval_nmape.slurm
```

Scripts: `ablation/scripts/`. Splits: `ablation/data/`.
