#!/usr/bin/env bash
#SBATCH --job-name=qhf_gt2ph_tst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --output=test_qhf_eqv3_gt_2ph_final_full_%j.out
#SBATCH --error=test_qhf_eqv3_gt_2ph_final_full_%j.err

# QM9 test eval: full test split + last 1600 test molecules, Euler s=2 and s=3.
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

CKPT_PATH="${CKPT_PATH:-qm9_qhf_eqv3_gt_2ph_final}"
LMDB_PATH="${LMDB_PATH:-${DATAPATH}/lmdb}"
SPLIT_FILE="${SPLIT_FILE:-${DATAPATH}/datasplits.json}"
TAIL_SPLIT_FILE=".cache/qm9_test_tail1600_datasplits.json"

mkdir -p .cache
python - "${SPLIT_FILE}" "${TAIL_SPLIT_FILE}" <<'PY'
import json, sys
test = json.load(open(sys.argv[1]))["test"]
json.dump({"test": test[-1600:]}, open(sys.argv[2], "w"))
PY

run_test() {  # split_file max_n_graphs output_tag n_steps [extra args]
  python scdp/scripts/test_flow.py \
    --ckpt_path "${CKPT_PATH}" \
    --data_path "${LMDB_PATH}" \
    --split_file "$1" \
    --tag test \
    --output_tag "$3" \
    --max_n_graphs "$2" \
    --batch_size 4 \
    --max_n_probe 500000 \
    --n_steps "$4" \
    --integrate_schedule uniform \
    "${@:5}"
}

for n_steps in 2 3; do
  run_test "${SPLIT_FILE}" 10000 test_full "${n_steps}" "$@"
  run_test "${TAIL_SPLIT_FILE}" 1600 test_tail1600 "${n_steps}" "$@"
done
