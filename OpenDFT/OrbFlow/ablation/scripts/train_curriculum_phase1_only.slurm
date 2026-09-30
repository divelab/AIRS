#!/usr/bin/env bash
#SBATCH --job-name=abl_ph1_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:2
#SBATCH --output=ablation/runs/logs/curriculum_phase1_only_%j.out
#SBATCH --error=ablation/runs/logs/curriculum_phase1_only_%j.err

# Phase 1 only (endpoint density for the full budget).
mkdir -p ablation/runs/logs
export ABLATION_MODE=phase1_only
export EXP_NAME="${EXP_NAME:-ablation/runs/curriculum_phase1_only}"
exec bash ablation/scripts/train_2phase_common.sh "$@"
