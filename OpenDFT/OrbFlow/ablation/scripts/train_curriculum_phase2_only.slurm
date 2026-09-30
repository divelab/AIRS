#!/usr/bin/env bash
#SBATCH --job-name=abl_ph2_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:2
#SBATCH --output=ablation/runs/logs/curriculum_phase2_only_%j.out
#SBATCH --error=ablation/runs/logs/curriculum_phase2_only_%j.err

# Phase 2 only, no warm start.
mkdir -p ablation/runs/logs
export ABLATION_MODE=phase2_only
export EXP_NAME="${EXP_NAME:-ablation/runs/curriculum_phase2_only}"
exec bash ablation/scripts/train_2phase_common.sh "$@"
