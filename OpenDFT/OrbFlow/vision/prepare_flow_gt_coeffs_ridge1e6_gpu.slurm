#!/usr/bin/env bash
#SBATCH --job-name=gt_r1e6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:8
#SBATCH --output=vision_prepare_gt_coeffs_ridge1e6_gpu_%j.out

# QM9 target coefficients (ridge=1e-6, beta=1.3) -> $DATAPATH/lmdb_gt_ridge (one sidecar shard per LMDB shard).
# Resumable: already-computed graphs are skipped (--skip_existing).

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f .env ]] && set -a && source .env && set +a
: "${DATAPATH:?set DATAPATH in .env}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
if command -v conda >/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-orbflow}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LMDB_IN="${LMDB_IN:-${DATAPATH}/lmdb}"
LMDB_OUT="${LMDB_OUT:-${DATAPATH}/lmdb_gt_ridge}"
SPLIT_FILE="${SPLIT_FILE:-${DATAPATH}/datasplits.json}"

# metadata.json (basis / atom types) is written by the datamodule on first setup.
if [[ ! -f "${LMDB_IN}/metadata.json" ]]; then
  python - "${LMDB_IN}" "${SPLIT_FILE}" <<'PY'
import sys
from pathlib import Path
import hydra
from hydra.utils import instantiate
lmdb, split = sys.argv[1:3]
with hydra.initialize_config_dir(config_dir=str(Path.cwd() / "scdp" / "config"), version_base="1.1"):
    cfg = hydra.compose("default", overrides=[f"data.dataset.path={lmdb}", f"data.split_file={split}"])
instantiate(cfg.data, _recursive_=False).setup("fit")
PY
fi

mkdir -p "${LMDB_OUT}"
python scdp/scripts/compute_gt_coeffs_lmdb.py \
  --lmdb_in "${LMDB_IN}" \
  --lmdb_out "${LMDB_OUT}" \
  --metadata_json "${LMDB_IN}/metadata.json" \
  --beta 1.3 \
  --dft_basis_set def2-qzvppd \
  --orb_cutoff 5.0 \
  --ridge 1e-6 \
  --ridge_diag_frac 0 \
  --probe_weighting uniform \
  --max_probe_samples -1 \
  --device cuda \
  --dtype float64 \
  --gpu_workers "${SLURM_GPUS_ON_NODE:-1}" \
  --skip_existing \
  "$@"
