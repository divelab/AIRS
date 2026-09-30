#!/usr/bin/env bash
#SBATCH --job-name=md_k3l3_s505
#SBATCH --array=0-5
#SBATCH --cpus-per-task=16
#SBATCH --mem=72G
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:2
#SBATCH --output=train_md_qhf_2ph_all6_k3l3_slim_10k100k_seed505_%A_%a.out

# Seed 505 replicate of train_md_all6_10k100k_k3l3_slim.slurm (paper runs used 2 GPUs).
export TRAIN_SEED="${TRAIN_SEED:-505}"
export EXP_TAG="${EXP_TAG:-seed505}"
exec bash vision/train_md_all6_10k100k_k3l3_slim.slurm "$@"
