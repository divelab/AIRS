#!/usr/bin/env bash
#SBATCH --job-name=md_gt_b13
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:8
#SBATCH --output=vision_prepare_md_gt_coeffs_ridge1e6_beta13_gpu_%j.out

# MD target coefficients (ridge=1e-6, beta=1.3) -> $DATAPATH/MD/scdp_lmdb_gt_ridge_beta1.3/<mol>.
# Needs the MD LMDBs in $DATAPATH/MD/scdp_lmdb/<mol>. MD_MOLECULE="benzene ethane" limits the molecules.

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

MD_LMDB="${MD_LMDB:-${DATAPATH}/MD/scdp_lmdb}"
MD_LMDB_GT="${MD_LMDB_GT:-${DATAPATH}/MD/scdp_lmdb_gt_ridge_beta1.3}"
MOLECULES="${MD_MOLECULE:-benzene ethanol phenol resorcinol ethane malonaldehyde}"

for mol in ${MOLECULES}; do
  lmdb_in="${MD_LMDB}/${mol}"
  lmdb_out="${MD_LMDB_GT}/${mol}"
  if [[ ! -f "${lmdb_in}/metadata.json" ]]; then
    python - "${lmdb_in}" "${lmdb_in}/datasplits.json" <<'PY'
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

  mkdir -p "${lmdb_out}"
  python scdp/scripts/compute_gt_coeffs_lmdb.py \
    --lmdb_in "${lmdb_in}" \
    --lmdb_out "${lmdb_out}" \
    --metadata_json "${lmdb_in}/metadata.json" \
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
    --graph_timeout_sec 1200 \
    --skip_existing \
    "$@"
done
