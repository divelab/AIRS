#!/usr/bin/env bash
#SBATCH --job-name=abl_cur_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:2
#SBATCH --output=ablation/runs/logs/curriculum_full_%j.out
#SBATCH --error=ablation/runs/logs/curriculum_full_%j.err

# Full two-phase curriculum on 10% QM9.
mkdir -p ablation/runs/logs
export ABLATION_MODE=full
export EXP_NAME="${EXP_NAME:-ablation/runs/curriculum_full}"
exec bash ablation/scripts/train_2phase_common.sh "$@"
