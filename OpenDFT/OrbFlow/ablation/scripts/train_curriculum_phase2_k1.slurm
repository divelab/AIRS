#!/usr/bin/env bash
#SBATCH --job-name=abl_ph2_k1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:2
#SBATCH --output=ablation/runs/logs/curriculum_phase2_k1_%j.out
#SBATCH --error=ablation/runs/logs/curriculum_phase2_k1_%j.err

# Same phase-1 as curriculum_full; phase-2 Euler K=1.
mkdir -p ablation/runs/logs

export ABLATION_MODE=full
export EXP_NAME="${EXP_NAME:-ablation/runs/curriculum_phase2_k1}"
export FLOW_INTEGRATE_TRAIN_STEPS="${FLOW_INTEGRATE_TRAIN_STEPS:-1}"
export FLOW_VAL_INTEGRATE_STEPS="${FLOW_VAL_INTEGRATE_STEPS:-1}"

SEED_TAG=""
if [[ "${TRAIN_SEED:-0}" != "0" ]]; then
  SEED_TAG="_seed${TRAIN_SEED}"
fi
DEFAULT_P1="ablation/runs/curriculum_full${SEED_TAG}/epoch=10-step=15000.ckpt"
if [[ -z "${PHASE1_INIT_CKPT:-}" && -f "${DEFAULT_P1}" ]]; then
  export PHASE1_INIT_CKPT="${DEFAULT_P1}"
fi

exec bash ablation/scripts/train_2phase_common.sh "$@"
