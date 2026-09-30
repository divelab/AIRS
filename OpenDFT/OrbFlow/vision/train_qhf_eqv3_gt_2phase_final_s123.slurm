#!/usr/bin/env bash
#SBATCH --job-name=eqv3_2ph_s123
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=144G
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:8
#SBATCH --output=train_qhf_eqv3_gt_2phase_final_s123_%j.out

# Seed 123 replicate of train_qhf_eqv3_gt_2phase_final.slurm
export TRAIN_SEED="${TRAIN_SEED:-123}"
export EXP_NAME="${EXP_NAME:-qm9_qhf_eqv3_gt_2ph_final_s123}"
exec bash vision/train_qhf_eqv3_gt_2phase_final.slurm "$@"
