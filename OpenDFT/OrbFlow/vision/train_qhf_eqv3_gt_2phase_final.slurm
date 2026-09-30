#!/usr/bin/env bash
#SBATCH --job-name=eqv3_2ph_final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=144G
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:8
#SBATCH --output=train_qhf_eqv3_gt_2phase_final_%j.out

# QM9 paper recipe: 75k endpoint density -> 450k integrated density (Euler K=2).
# Re-submit the same script / EXP_NAME to resume.

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f .env ]] && set -a && source .env && set +a
: "${DATAPATH:?set DATAPATH in .env}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
if command -v conda >/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-orbflow}"
fi

EXP_NAME="${EXP_NAME:-qm9_qhf_eqv3_gt_2ph_final}"
STORAGE_DIR="${EXP_NAME}"
LMDB_PATH="${LMDB_PATH:-${DATAPATH}/lmdb}"
GT_COEFFS_PATH="${GT_COEFFS_PATH:-${DATAPATH}/lmdb_gt_ridge}"
SPLIT_FILE="${SPLIT_FILE:-${DATAPATH}/datasplits.json}"
PHASE1_MAX_STEPS="${PHASE1_MAX_STEPS:-75000}"
PHASE2_MAX_STEPS="${PHASE2_MAX_STEPS:-450000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WANDB_MODE="${WANDB_MODE:-online}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-8}}"
TRAIN_STRATEGY=ddp_find_unused_parameters_true
[[ "${NUM_GPUS}" -eq 1 ]] && TRAIN_STRATEGY=auto

resume_step() {
  python -c "from scdp.common.checkpoint_utils import best_resume_global_step as s; print(s('$1'))"
}

COMMON_TRAIN_ARGS=(
  core.expname="${EXP_NAME}"
  data=qm9_bond
  train=flow
  model=qm9_flow_qhflow
  model/model=equiformer_v3
  model.beta=1.3
  model.magnitude_weighting=false
  model.model.num_layers=8
  "model.model.lmax_list=[6]"
  "model.model.mmax_list=[2]"
  model.model.sphere_channels=128
  model.model.hidden_channels=256
  model.model.edge_channels=128
  model.model.cutoff=6.0
  model.model.num_radial_basis=300
  model.bridge_mode=cfm
  model.flow_head=escn
  model.flow_loss_mode=endpoint
  model.flow_target_mode=gt
  model.flow_use_t_scale=false
  model.flow_prior_mode=irrep_gaussian
  model.flow_prior_sigma=0.85
  model.flow_prior_mu=0.0025
  model.flow_integrate_min_t=0.01
  model.flow_integrate_schedule=uniform
  model.flow_integrate_t_hi=0.99
  data.dataset.path="${LMDB_PATH}"
  data.dataset.gt_coeffs_path="${GT_COEFFS_PATH}"
  data.split_file="${SPLIT_FILE}"
  data.batch_size.train="${BATCH_SIZE}"
  data.batch_size.val="${BATCH_SIZE}"
  data.batch_size.test="${BATCH_SIZE}"
  train.optim.lr=1e-3
  train.lr_scheduler.beta=4433
  train.deterministic=true
  train.seed="${TRAIN_SEED:-0}"
  train.trainer.devices="${NUM_GPUS}"
  train.trainer.strategy="${TRAIN_STRATEGY}"
  train.logging.wandb.mode="${WANDB_MODE}"
)

# Phase 1: endpoint density loss (skipped once resumed past PHASE1_MAX_STEPS).
if [[ "$(resume_step "${STORAGE_DIR}")" -lt "${PHASE1_MAX_STEPS}" ]]; then
  python scdp/scripts/train.py \
    "${COMMON_TRAIN_ARGS[@]}" \
    model.flow_endpoint_loss=density \
    model.flow_val_integrate_steps=2 \
    train.trainer.max_steps="${PHASE1_MAX_STEPS}" \
    "$@"
fi

# Phase 2: integrated density loss through K=2 Euler steps.
[[ "$(resume_step "${STORAGE_DIR}")" -ge "${PHASE2_MAX_STEPS}" ]] && exit 0
python scdp/scripts/train.py \
  "${COMMON_TRAIN_ARGS[@]}" \
  model.flow_endpoint_loss=integrated_density \
  model.flow_integrate_train_steps=2 \
  model.flow_val_integrate_steps=2 \
  model.flow_val_integrate_sweep_steps=[1,2,3] \
  train.trainer.max_steps="${PHASE2_MAX_STEPS}" \
  "$@"
