#!/usr/bin/env bash
#SBATCH --job-name=abl_joint_k2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:4
#SBATCH --output=ablation/runs/logs/joint_endpoint_k2_%j.out
#SBATCH --error=ablation/runs/logs/joint_endpoint_k2_%j.err

# One-stage joint loss (endpoint + integrated density, K=2).
mkdir -p ablation/runs/logs

export ABLATION_MODE=joint_ep_k1
export EXP_NAME="${EXP_NAME:-ablation/runs/joint_endpoint_k2}"
export FLOW_INTEGRATE_TRAIN_STEPS="${FLOW_INTEGRATE_TRAIN_STEPS:-2}"
export FLOW_VAL_INTEGRATE_STEPS="${FLOW_VAL_INTEGRATE_STEPS:-2}"
export FLOW_COMBINED_INTEGRATED_WEIGHT="${FLOW_COMBINED_INTEGRATED_WEIGHT:-1.0}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec bash ablation/scripts/train_2phase_common.sh "$@"
