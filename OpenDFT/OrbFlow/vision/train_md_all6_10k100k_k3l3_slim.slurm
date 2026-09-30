#!/usr/bin/env bash
#SBATCH --job-name=md_k3l3_slim
#SBATCH --array=0-5
#SBATCH --cpus-per-task=32
#SBATCH --mem=144G
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:4
#SBATCH --output=train_md_qhf_2ph_all6_k3l3_slim_10k100k_%A_%a.out

# MD paper recipe: 10k endpoint -> 110k integrated density, slim EqV3 (K=3, lmax=3).
# Array 0-5: benzene ethanol phenol resorcinol ethane malonaldehyde

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f .env ]] && set -a && source .env && set +a
: "${DATAPATH:?set DATAPATH in .env}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
if command -v conda >/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-orbflow}"
fi

MD_MOLECULES=(benzene ethanol phenol resorcinol ethane malonaldehyde)
MD_MOLECULE="${MD_MOLECULE:-${MD_MOLECULES[${SLURM_ARRAY_TASK_ID:-0}]}}"

LMDB_PATH="${LMDB_PATH:-${DATAPATH}/MD/scdp_lmdb/${MD_MOLECULE}}"
GT_COEFFS_PATH="${GT_COEFFS_PATH:-${DATAPATH}/MD/scdp_lmdb_gt_ridge_beta1.3/${MD_MOLECULE}}"
SPLIT_FILE="${SPLIT_FILE:-${LMDB_PATH}/datasplits.json}"
PRIOR_JSON="${PRIOR_JSON:-${DATAPATH}/MD/flow_prior_ridge1e6_beta1.3/${MD_MOLECULE}/gt_coeff_flow_prior_stats.json}"
PHASE1_MAX_STEPS="${PHASE1_MAX_STEPS:-10000}"
PHASE2_MAX_STEPS="${PHASE2_MAX_STEPS:-110000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WANDB_MODE="${WANDB_MODE:-online}"

read -r FLOW_PRIOR_MU FLOW_PRIOR_SIGMA < <(
  python - <<'PY' "${PRIOR_JSON}"
import json, sys
d = json.load(open(sys.argv[1]))
h = d.get("hydra_overrides") or {}
print(h["model.flow_prior_mu"], h["model.flow_prior_sigma"])
PY
)

EXP_NAME="${EXP_NAME:-md_${MD_MOLECULE}_k3l3_slim_10k100k}"
[[ -n "${EXP_TAG:-}" ]] && EXP_NAME="${EXP_NAME}_${EXP_TAG}"
STORAGE_DIR="${EXP_NAME}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}"
TRAIN_STRATEGY=ddp_find_unused_parameters_true
[[ "${NUM_GPUS}" -eq 1 ]] && TRAIN_STRATEGY=auto

resume_step() {
  python -c "from scdp.common.checkpoint_utils import best_resume_global_step as s; print(s('$1'))"
}

COMMON_TRAIN_ARGS=(
  core.expname="${EXP_NAME}"
  data=md_bond
  train=flow
  model=qm9_flow_qhflow
  model/model=equiformer_v3
  model.beta=1.3
  model.magnitude_weighting=false
  model.model.num_layers=3
  model.model.lmax_list=[3]
  model.model.mmax_list=[2]
  model.model.sphere_channels=96
  model.model.hidden_channels=192
  model.model.edge_channels=96
  model.model.cutoff=6.0
  model.model.num_radial_basis=300
  model.bridge_mode=cfm
  model.flow_head=escn
  model.flow_loss_mode=endpoint
  model.flow_target_mode=gt
  model.flow_use_t_scale=false
  model.flow_prior_mode=irrep_gaussian
  model.flow_prior_sigma="${FLOW_PRIOR_SIGMA}"
  model.flow_prior_mu="${FLOW_PRIOR_MU}"
  model.flow_prior_nelectron_normalize=false
  data.dataset.path="${LMDB_PATH}"
  data.dataset.gt_coeffs_path="${GT_COEFFS_PATH}"
  data.split_file="${SPLIT_FILE}"
  data.batch_size.train="${BATCH_SIZE}"
  data.batch_size.val="${BATCH_SIZE}"
  data.batch_size.test="${BATCH_SIZE}"
  data.n_probe.train=125000
  data.n_probe.val=125000
  data.n_probe.test=125000
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

python scdp/scripts/train.py \
  "${COMMON_TRAIN_ARGS[@]}" \
  model.flow_endpoint_loss=integrated_density \
  model.flow_integrate_train_steps=2 \
  model.flow_val_integrate_steps=2 \
  model.flow_val_integrate_sweep_steps=[1,2,3] \
  train.trainer.max_steps="${PHASE2_MAX_STEPS}" \
  "$@"
