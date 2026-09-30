#!/usr/bin/env bash
#SBATCH --job-name=abl_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --output=ablation/runs/logs/eval_nmape_%j.out
#SBATCH --error=ablation/runs/logs/eval_nmape_%j.err

# Flow ablation eval on the 10% test split.
#   CKPT_PATH=ablation/runs/curriculum_full sbatch ablation/scripts/eval_nmape.slurm

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f .env ]] && set -a && source .env && set +a
: "${DATAPATH:?set DATAPATH in .env}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
if command -v conda >/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-orbflow}"
fi

mkdir -p ablation/runs/logs ablation/results

CKPT_PATH="${CKPT_PATH:?set CKPT_PATH=ablation/runs/<exp>}"
SPLIT_FILE="${SPLIT_FILE:-ablation/data/datasplits_ablation_10pct.json}"
LMDB_PATH="${LMDB_PATH:-${DATAPATH}/lmdb}"
TAG="${TAG:-test}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_N_PROBE="${MAX_N_PROBE:-500000}"
COEFF_MODE="${COEFF_MODE:-integrate}"
EULER_STEPS="${EULER_STEPS:-2}"
MAX_N_GRAPHS="${MAX_N_GRAPHS:-1000}"
OUTPUT_TAG="${OUTPUT_TAG:-ablation_test10pct}"
RESULTS_CSV="${RESULTS_CSV:-ablation/results/summary.csv}"
LABEL="${LABEL:-$(basename "${CKPT_PATH}")}"

[[ -d "${CKPT_PATH}" && -f "${SPLIT_FILE}" ]] || exit 1

run_log="$(mktemp)"
python scdp/scripts/test_flow.py \
  --ckpt_path "${CKPT_PATH}" \
  --data_path "${LMDB_PATH}" \
  --split_file "${SPLIT_FILE}" \
  --tag "${TAG}" \
  --output_tag "${OUTPUT_TAG}" \
  --max_n_graphs "${MAX_N_GRAPHS}" \
  --batch_size "${BATCH_SIZE}" \
  --max_n_probe "${MAX_N_PROBE}" \
  --n_steps "${EULER_STEPS}" \
  --coeff_mode "${COEFF_MODE}" \
  --eval_log - \
  --no_nmape_txt \
  --integrate_schedule "${FLOW_INTEGRATE_SCHEDULE:-uniform}" \
  --integrate_t_hi "${FLOW_INTEGRATE_T_HI:-0.99}" \
  --integrate_power "${FLOW_INTEGRATE_POWER:-4.0}" \
  2>&1 | tee "${run_log}"

python - <<PY
import re
from pathlib import Path

label = """${LABEL}"""
ckpt = """${CKPT_PATH}"""
steps = """${EULER_STEPS}"""
csv_path = Path("""${RESULTS_CSV}""")
text = Path("""${run_log}""").read_text()
mean = std = None
for line in text.splitlines():
    # Preferred: "flow test nMAPE ...: mean=0.31 std=0.62 ..."
    m = re.search(
        r"(?:flow test nMAPE|nmape).*?\bmean\s*=\s*([0-9.]+).*?\bstd\s*=\s*([0-9.]+)",
        line,
        re.I,
    )
    if m:
        mean, std = m.group(1), m.group(2)
        break
    # Legacy: "nMAPE: 1.2345 ± 0.01"
    m = re.search(r"(?:flow test nMAPE|nmape)\s*[:=]\s*([0-9.]+)\s*±\s*([0-9.]+)", line, re.I)
    if m:
        mean, std = m.group(1), m.group(2)
        break
if mean is None:
    raise SystemExit("ERROR: could not parse nMAPE from eval log")

csv_path.parent.mkdir(parents=True, exist_ok=True)
if not csv_path.exists():
    csv_path.write_text(
        "label,ckpt_path,eval_split,n_steps,nmape_mean,nmape_std,notes\n"
    )
with csv_path.open("a") as f:
    f.write(f"{label},{ckpt},test_10pct,{steps},{mean},{std},\n")
print(f"appended -> {csv_path}: {label} nMAPE={mean}±{std} (K={steps})")
PY
