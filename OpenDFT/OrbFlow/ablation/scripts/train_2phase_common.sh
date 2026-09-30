#!/usr/bin/env bash
# Shared ablation trainer. Wrappers set ABLATION_MODE + EXP_NAME.
# Modes: full | phase1_only | phase2_only | joint_ep_k1

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f .env ]] && set -a && source .env && set +a
: "${DATAPATH:?set DATAPATH in .env}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
if command -v conda >/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-orbflow}"
fi

ABLATION_MODE="${ABLATION_MODE:?set ABLATION_MODE}"
EXP_NAME="${EXP_NAME:?set EXP_NAME}"
STORAGE_DIR="${EXP_NAME}"
LMDB_PATH="${LMDB_PATH:-${DATAPATH}/lmdb}"
GT_COEFFS_PATH="${GT_COEFFS_PATH:-${DATAPATH}/lmdb_gt_ridge}"
SPLIT_FILE="${SPLIT_FILE:-ablation/data/datasplits_ablation_10pct.json}"
PHASE1_MAX_STEPS="${PHASE1_MAX_STEPS:-15000}"
PHASE2_MAX_STEPS="${PHASE2_MAX_STEPS:-90000}"
FLOW_INTEGRATE_TRAIN_STEPS="${FLOW_INTEGRATE_TRAIN_STEPS:-2}"
FLOW_VAL_INTEGRATE_STEPS="${FLOW_VAL_INTEGRATE_STEPS:-${FLOW_INTEGRATE_TRAIN_STEPS}}"
PHASE1_INIT_CKPT="${PHASE1_INIT_CKPT:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WANDB_MODE="${WANDB_MODE:-online}"
BACKBONE="${BACKBONE:-equiformer_v3}"
FLOW_PRIOR_MODE="${FLOW_PRIOR_MODE:-irrep_gaussian}"
FLOW_PRIOR_MU="${FLOW_PRIOR_MU:-0.0025}"
FLOW_PRIOR_SIGMA="${FLOW_PRIOR_SIGMA:-0.85}"
FLOW_COMBINED_INTEGRATED_WEIGHT="${FLOW_COMBINED_INTEGRATED_WEIGHT:-1.0}"
TRAIN_SEED="${TRAIN_SEED:-0}"

NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-2}}"
TRAIN_STRATEGY=ddp_find_unused_parameters_true
[[ "${NUM_GPUS}" -eq 1 ]] && TRAIN_STRATEGY=auto
resume_step() {
  python -c "from scdp.common.checkpoint_utils import best_resume_global_step as s; print(s('$1'))"
}

[[ -f "${SPLIT_FILE}" ]] || exit 1
mkdir -p "${STORAGE_DIR}"

if [[ -n "${PHASE1_INIT_CKPT}" && -f "${PHASE1_INIT_CKPT}" ]]; then
  CUR0="$(resume_step "${STORAGE_DIR}")"
  if [[ "${CUR0}" -lt 0 ]]; then
    dest="${STORAGE_DIR}/$(basename "${PHASE1_INIT_CKPT}")"
    cp -a "${PHASE1_INIT_CKPT}" "${dest}"
    ln -sfn "$(basename "${dest}")" "${STORAGE_DIR}/last.ckpt"  # relative to STORAGE_DIR
  fi
fi

COMMON_TRAIN_ARGS=(
  core.expname="${EXP_NAME}"
  data=qm9_bond
  train=flow
  model=qm9_flow_qhflow
  model/model="${BACKBONE}"
  model.beta=1.3
  model.magnitude_weighting=false
  model.model.num_layers=8
  "model.model.lmax_list=[6]"
  "model.model.mmax_list=[2]"
  model.model.sphere_channels=128
  model.model.hidden_channels=256
  model.model.edge_channels=128
  model.model.cutoff=6.0
  model.bridge_mode=cfm
  model.flow_head=escn
  model.flow_loss_mode=endpoint
  model.flow_target_mode=gt
  model.flow_use_t_scale=false
  model.flow_prior_mode="${FLOW_PRIOR_MODE}"
  model.flow_prior_sigma="${FLOW_PRIOR_SIGMA}"
  model.flow_prior_mu="${FLOW_PRIOR_MU}"
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
  train.seed="${TRAIN_SEED}"
  train.trainer.devices="${NUM_GPUS}"
  train.trainer.strategy="${TRAIN_STRATEGY}"
  train.logging.wandb.mode="${WANDB_MODE}"
  train.run_post_fit_test=false
)
[[ "${BACKBONE}" == "equiformer_v3" ]] && COMMON_TRAIN_ARGS+=(model.model.num_radial_basis=300)

case "${ABLATION_MODE}" in
  joint_ep_k1)
    CUR_STEP="$(resume_step "${STORAGE_DIR}")"
    [[ "${CUR_STEP}" -ge "${PHASE2_MAX_STEPS}" ]] && exit 0
    python scdp/scripts/train.py \
      "${COMMON_TRAIN_ARGS[@]}" \
      model.flow_endpoint_loss=density_and_integrated \
      model.flow_integrate_train_steps="${FLOW_INTEGRATE_TRAIN_STEPS}" \
      model.flow_val_integrate_steps="${FLOW_VAL_INTEGRATE_STEPS}" \
      model.flow_val_integrate_sweep_steps=[1,2,3] \
      model.flow_combined_integrated_weight="${FLOW_COMBINED_INTEGRATED_WEIGHT}" \
      train.trainer.max_steps="${PHASE2_MAX_STEPS}" \
      "$@"
    ;;
  phase1_only)
    CUR_STEP="$(resume_step "${STORAGE_DIR}")"
    [[ "${CUR_STEP}" -ge "${PHASE2_MAX_STEPS}" ]] && exit 0
    python scdp/scripts/train.py \
      "${COMMON_TRAIN_ARGS[@]}" \
      model.flow_endpoint_loss=density \
      model.flow_val_integrate_steps=2 \
      train.trainer.max_steps="${PHASE2_MAX_STEPS}" \
      "$@"
    ;;
  phase2_only)
    CUR_STEP="$(resume_step "${STORAGE_DIR}")"
    [[ "${CUR_STEP}" -ge "${PHASE2_MAX_STEPS}" ]] && exit 0
    python scdp/scripts/train.py \
      "${COMMON_TRAIN_ARGS[@]}" \
      model.flow_endpoint_loss=integrated_density \
      model.flow_integrate_train_steps="${FLOW_INTEGRATE_TRAIN_STEPS}" \
      model.flow_val_integrate_steps="${FLOW_VAL_INTEGRATE_STEPS}" \
      model.flow_val_integrate_sweep_steps=[1,2,3] \
      train.trainer.max_steps="${PHASE2_MAX_STEPS}" \
      "$@"
    ;;
  full)
    PHASE1_STEP="$(resume_step "${STORAGE_DIR}")"
    if [[ "${PHASE1_STEP}" -lt "${PHASE1_MAX_STEPS}" ]]; then
      python scdp/scripts/train.py \
        "${COMMON_TRAIN_ARGS[@]}" \
        model.flow_endpoint_loss=density \
        model.flow_val_integrate_steps=2 \
        train.trainer.max_steps="${PHASE1_MAX_STEPS}" \
        "$@" || exit $?
      PHASE1_STEP="$(resume_step "${STORAGE_DIR}")"
      [[ "${PHASE1_STEP}" -ge "${PHASE1_MAX_STEPS}" ]] || exit 1
    fi
    PHASE2_STEP="$(resume_step "${STORAGE_DIR}")"
    [[ "${PHASE2_STEP}" -ge "${PHASE1_MAX_STEPS}" ]] || exit 1
    [[ "${PHASE2_STEP}" -ge "${PHASE2_MAX_STEPS}" ]] && exit 0
    python scdp/scripts/train.py \
      "${COMMON_TRAIN_ARGS[@]}" \
      model.flow_endpoint_loss=integrated_density \
      model.flow_integrate_train_steps="${FLOW_INTEGRATE_TRAIN_STEPS}" \
      model.flow_val_integrate_steps="${FLOW_VAL_INTEGRATE_STEPS}" \
      model.flow_val_integrate_sweep_steps=[1,2,3] \
      train.trainer.max_steps="${PHASE2_MAX_STEPS}" \
      "$@"
    ;;
  *)
    exit 1
    ;;
esac
