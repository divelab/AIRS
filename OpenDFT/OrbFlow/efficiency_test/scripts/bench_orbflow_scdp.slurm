#!/usr/bin/env bash
#SBATCH --job-name=eff_orb_scdp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --output=bench_orbflow_scdp_%j.out
#SBATCH --error=bench_orbflow_scdp_%j.err

# CUDA-synced MD test-set inference timing (seconds / molecule):
#   OrbFlow k3l3_slim (Euler K=2, Triton GTO kernel on) and SCDP eSCN (original kernel).
# Run dirs are printf patterns relative to the repo root; missing runs are skipped.

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
ORBFLOW_RUN="${ORBFLOW_RUN:-md_%s_k3l3_slim_10k100k}"
SCDP_RUN="${SCDP_RUN:-md_%s_escn_k4l3_beta1.5_gpwno_vo}"
MODELS="${MODELS:-orbflow_k3l3_slim scdp_gpwno}"
MOLECULES="${MOLECULES:-benzene ethane ethanol malonaldehyde phenol resorcinol}"
OUT_DIR="${OUT_DIR:-efficiency_test/results/${SLURM_JOB_ID:-local}}"
MAX_N_GRAPHS="${MAX_N_GRAPHS:-0}"   # 0 = full test split after warmup
WARMUP="${WARMUP:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_N_PROBE="${MAX_N_PROBE:-500000}"
N_STEPS="${N_STEPS:-2}"

mkdir -p "${OUT_DIR}"
nvidia-smi -L || true

read -r -a MOL_ARR <<< "${MOLECULES}"
read -r -a MODEL_ARR <<< "${MODELS}"

for model in "${MODEL_ARR[@]}"; do
  for mol in "${MOL_ARR[@]}"; do
    case "${model}" in
      orbflow_k3l3_slim) CKPT_PATH="$(printf "${ORBFLOW_RUN}" "${mol}")" ;;
      scdp_gpwno)        CKPT_PATH="$(printf "${SCDP_RUN}" "${mol}")" ;;
      *) echo "unknown model: ${model}" >&2; exit 1 ;;
    esac
    LMDB_PATH="${MD_LMDB}/${mol}"
    SPLIT_FILE="${LMDB_PATH}/datasplits.json"
    JSON_OUT="${OUT_DIR}/${model}_${mol}.json"
    if [[ ! -f "${CKPT_PATH}/config.yaml" ]]; then
      echo "skip ${model}/${mol}: no run at ${CKPT_PATH}"
      continue
    fi
    echo "== ${model} / ${mol}: ${CKPT_PATH}"

    COMMON=(
      --ckpt_path "${CKPT_PATH}"
      --data_path "${LMDB_PATH}"
      --split_file "${SPLIT_FILE}"
      --tag test
      --max_n_graphs "${MAX_N_GRAPHS}"
      --warmup "${WARMUP}"
      --batch_size "${BATCH_SIZE}"
      --max_n_probe "${MAX_N_PROBE}"
      --use_latest_epoch
      --json_out "${JSON_OUT}"
    )
    if [[ "${model}" == orbflow_* ]]; then
      python scdp/scripts/bench_flow_infer.py "${COMMON[@]}" --n_steps "${N_STEPS}"
      python - "${JSON_OUT}" "${model}" "${mol}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text())
d["model_name"], d["molecule"] = sys.argv[2], sys.argv[3]
p.write_text(json.dumps(d, indent=2))
PY
    else
      python efficiency_test/scripts/bench_scdp_infer.py "${COMMON[@]}" \
        --model_name "${model}" --molecule "${mol}"
    fi
  done
done

python efficiency_test/scripts/summarize_results.py \
  --results_dir "${OUT_DIR}" \
  --json_out "${OUT_DIR}/summary_orbflow_scdp.json" \
  --csv_out "${OUT_DIR}/summary_orbflow_scdp.csv"
