#!/usr/bin/env bash
#SBATCH --job-name=md_k3l3_s42
#SBATCH --array=0-5
#SBATCH --cpus-per-task=32
#SBATCH --mem=144G
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:4
#SBATCH --output=train_md_qhf_2ph_all6_k3l3_slim_10k100k_seed42_%A_%a.out

# Seed 42 replicate of train_md_all6_10k100k_k3l3_slim.slurm
export TRAIN_SEED="${TRAIN_SEED:-42}"
export EXP_TAG="${EXP_TAG:-seed42}"
exec bash vision/train_md_all6_10k100k_k3l3_slim.slurm "$@"
