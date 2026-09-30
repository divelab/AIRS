#!/usr/bin/env bash
#SBATCH --job-name=md_b13_all6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --output=test_flow_md_paper_2phase_beta13_all6_%j.out
#SBATCH --error=test_flow_md_paper_2phase_beta13_all6_%j.err

# MD test eval for the 6 k3l3_slim runs (best val ckpt).
# Extra args go to test_flow.py (e.g. --use_last).

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
MOLECULES="${MOLECULES:-benzene ethanol phenol resorcinol ethane malonaldehyde}"
EXP_SUFFIX="${EXP_SUFFIX:-md_%s_k3l3_slim_10k100k}"   # e.g. md_%s_k3l3_slim_10k100k_seed42
EULER_STEPS="${EULER_STEPS:-2}"

for mol in ${MOLECULES}; do
  # shellcheck disable=SC2059
  CKPT_PATH="$(printf "${EXP_SUFFIX}" "${mol}")"
  LMDB_PATH="${MD_LMDB}/${mol}"
  [[ -f "${CKPT_PATH}/config.yaml" ]] || continue
  for n_steps in ${EULER_STEPS}; do
    python scdp/scripts/test_flow.py \
      --ckpt_path "${CKPT_PATH}" \
      --data_path "${LMDB_PATH}" \
      --split_file "${LMDB_PATH}/datasplits.json" \
      --tag test \
      --max_n_graphs 10000 \
      --batch_size 4 \
      --max_n_probe 125000 \
      --n_steps "${n_steps}" \
      --integrate_schedule uniform \
      "$@"
  done
done
