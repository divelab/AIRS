#!/usr/bin/env bash
# Submit TRAIN_SEED=42 and 123 for main-text ablation rows.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
mkdir -p ablation/runs/logs

SPECS=(
  "ablation/scripts/train_curriculum_full.slurm|ablation/runs/curriculum_full|abl_cur"
  "ablation/scripts/train_curriculum_phase1_only.slurm|ablation/runs/curriculum_phase1_only|abl_p1"
  "ablation/scripts/train_curriculum_phase2_only.slurm|ablation/runs/curriculum_phase2_only|abl_p2o"
  "ablation/scripts/train_curriculum_phase2_k1.slurm|ablation/runs/curriculum_phase2_k1|abl_p2k1"
  "ablation/scripts/train_curriculum_phase2_k3.slurm|ablation/runs/curriculum_phase2_k3|abl_p2k3"
  "ablation/scripts/train_joint_endpoint_k2.slurm|ablation/runs/joint_endpoint_k2|abl_jk2"
  "ablation/scripts/train_direct_regression.slurm|ablation/runs/direct_regression|abl_dir"
)

SEEDS=(42 123)
JOBIDS=()

for seed in "${SEEDS[@]}"; do
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r script base_exp jname <<< "${spec}"
    exp="${base_exp}_seed${seed}"
    tag="$(basename "${base_exp}")_seed${seed}"
    jid="$(sbatch --parsable \
      --job-name="${jname}_s${seed}" \
      --output="ablation/runs/logs/${tag}_%j.out" \
      --error="ablation/runs/logs/${tag}_%j.err" \
      --export=ALL,TRAIN_SEED="${seed}",EXP_NAME="${exp}" \
      "${script}")"
    JOBIDS+=("${jid}")
  done
done
printf '%s\n' "${JOBIDS[@]}"
