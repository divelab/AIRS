#!/usr/bin/env bash
#SBATCH --job-name=abl_direct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:2
#SBATCH --output=ablation/runs/logs/direct_regression_%j.out
#SBATCH --error=ablation/runs/logs/direct_regression_%j.err

# Direct coefficient regression (no flow).
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f .env ]] && set -a && source .env && set +a
: "${DATAPATH:?set DATAPATH in .env}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
if command -v conda >/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-orbflow}"
fi

mkdir -p ablation/runs/logs

EXP_NAME="${EXP_NAME:-ablation/runs/direct_regression}"
STORAGE_DIR="${EXP_NAME}"
SPLIT_FILE="${SPLIT_FILE:-ablation/data/datasplits_ablation_10pct.json}"
LMDB_PATH="${LMDB_PATH:-${DATAPATH}/lmdb}"
GT_COEFFS_PATH="${GT_COEFFS_PATH:-${DATAPATH}/lmdb_gt_ridge}"

MAX_STEPS="${MAX_STEPS:-90000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TRAIN_LR="${TRAIN_LR:-1e-3}"
TRAIN_LR_BETA="${TRAIN_LR_BETA:-4433}"
E3NN_K="${E3NN_K:-8}"
E3NN_L="${E3NN_L:-6}"
ETB_BETA="${ETB_BETA:-1.3}"
TRAIN_SEED="${TRAIN_SEED:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
N_PROBE="${N_PROBE:-60000}"

if [[ ! -f "${SPLIT_FILE}" ]]; then
  echo "ERROR: missing split file: ${SPLIT_FILE}" >&2
  echo "  Run: python ablation/scripts/make_ablation_split.py" >&2
  exit 1
fi

NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-2}}"
TRAIN_STRATEGY=ddp_find_unused_parameters_true
[[ "${NUM_GPUS}" -eq 1 ]] && TRAIN_STRATEGY=auto
mkdir -p "${STORAGE_DIR}"

CUR_STEP="$(python -c "from scdp.common.checkpoint_utils import best_resume_global_step as s; print(s('${STORAGE_DIR}'))")"
if [[ "${CUR_STEP}" -ge "${MAX_STEPS}" ]]; then
  exit 0
fi

GT_ARGS=()
if [[ -d "${GT_COEFFS_PATH}" ]]; then
  GT_ARGS+=(data.dataset.gt_coeffs_path="${GT_COEFFS_PATH}")
fi

python scdp/scripts/train.py \
  core.expname="${EXP_NAME}" \
  data=qm9_bond \
  train=default \
  model=qm9 \
  model/model=equiformer_v3 \
  model.beta="${ETB_BETA}" \
  model.magnitude_weighting=false \
  model.expo_trainable=false \
  model.model.num_layers="${E3NN_K}" \
  "model.model.lmax_list=[${E3NN_L}]" \
  "model.model.mmax_list=[2]" \
  model.model.sphere_channels=128 \
  model.model.hidden_channels=256 \
  model.model.edge_channels=128 \
  model.model.cutoff=6.0 \
  model.model.num_radial_basis=300 \
  data.dataset.path="${LMDB_PATH}" \
  "${GT_ARGS[@]}" \
  data.split_file="${SPLIT_FILE}" \
  data.batch_size.train="${BATCH_SIZE}" \
  data.batch_size.val="${BATCH_SIZE}" \
  data.batch_size.test="${BATCH_SIZE}" \
  data.n_probe.train="${N_PROBE}" \
  data.n_probe.val=120000 \
  data.n_probe.test=120000 \
  train.optim.lr="${TRAIN_LR}" \
  train.lr_scheduler.beta="${TRAIN_LR_BETA}" \
  train.deterministic=true \
  train.seed="${TRAIN_SEED}" \
  train.trainer.max_steps="${MAX_STEPS}" \
  train.trainer.devices="${NUM_GPUS}" \
  train.trainer.strategy="${TRAIN_STRATEGY}" \
  train.logging.wandb.mode="${WANDB_MODE}" \
  train.run_post_fit_test=false \
  "$@"

