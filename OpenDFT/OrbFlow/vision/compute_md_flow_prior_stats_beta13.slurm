#!/usr/bin/env bash
#SBATCH --job-name=md_prior_b13
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --output=compute_md_flow_prior_stats_beta13_%j.out

# Per-molecule flow prior mu/sigma from the MD train-split target coefficients
# -> $DATAPATH/MD/flow_prior_ridge1e6_beta1.3/<mol>/gt_coeff_flow_prior_stats.json (read by the MD train recipe).
# (QM9 does not need this: its train recipe sets the prior mu/sigma directly.)

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f .env ]] && set -a && source .env && set +a
: "${DATAPATH:?set DATAPATH in .env}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
if command -v conda >/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-orbflow}"
fi

MD_LMDB="${MD_LMDB:-${DATAPATH}/MD/scdp_lmdb}"
MD_LMDB_GT="${MD_LMDB_GT:-${DATAPATH}/MD/scdp_lmdb_gt_ridge_beta1.3}"
OUT_DIR="${OUT_DIR:-${DATAPATH}/MD/flow_prior_ridge1e6_beta1.3}"
MOLECULES="${MD_MOLECULE:-benzene ethanol phenol resorcinol ethane malonaldehyde}"

for mol in ${MOLECULES}; do
  LMDB_PATH="${MD_LMDB}/${mol}"
  SPLIT_FILE="${LMDB_PATH}/datasplits.json"
  FIRST_TRAIN="$(python -c "import json; print(json.load(open('${SPLIT_FILE}'))['train'][0])")"
  # Prior stats are computed over the train split; the 1-graph reconstruction is only a side check.
  python scdp/scripts/validate_gt_coeffs.py \
    --lmdb_path "${LMDB_PATH}" \
    --gt_coeffs_path "${MD_LMDB_GT}/${mol}" \
    --metadata_json "${LMDB_PATH}/metadata.json" \
    --split_file "${SPLIT_FILE}" \
    --graph_indices "${FIRST_TRAIN}" \
    --tag flow_prior_smoke \
    --max_n_graphs 1 \
    --batch_size 4 \
    --beta 1.3 \
    --ridge 1e-6 \
    --ridge_diag_frac 0 \
    --max_probe_samples -1 \
    --device cuda \
    --recompute_n 0 \
    --nmape_threshold 100 \
    --out_dir "${OUT_DIR}/${mol}" \
    --compute_flow_prior_stats \
    --prior_stats_tag train \
    --prior_stats_max_graphs 10000
  rm -f "${OUT_DIR:?}/${mol:?}"/coeff_validate_flow_prior_smoke_* "${OUT_DIR:?}/${mol:?}"/coeff_nmape_flow_prior_smoke_*.txt
done
