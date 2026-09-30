#!/usr/bin/env bash
#SBATCH --job-name=abl_nfe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --output=ablation/runs/logs/eval_nfe_%j.out
#SBATCH --error=ablation/runs/logs/eval_nfe_%j.err

# Inference-time NFE sweep for tab:ablation_nfe.
#
# Protocol (fixed):
#   - Model: curriculum_full (Phase-2 train-matched K=2), best-val ckpt (same as main ablation table)
#   - Split: ablation test_10pct (1000 mols)
#   - Sweep inference Euler steps K in {1,2,4,8,16}
#   - One timed eval pass per K (same graphs/batch); relative cost = wall_K / wall_1
#
#   sbatch ablation/scripts/eval_nfe.slurm
#   CKPT_PATH=ablation/runs/curriculum_full EULER_STEPS="1 2 4 8 16" sbatch ...

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

CKPT_PATH="${CKPT_PATH:-ablation/runs/curriculum_full}"
SPLIT_FILE="${SPLIT_FILE:-ablation/data/datasplits_ablation_10pct.json}"
LMDB_PATH="${LMDB_PATH:-${DATAPATH}/lmdb}"
TAG="${TAG:-test}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_N_PROBE="${MAX_N_PROBE:-500000}"
COEFF_MODE="${COEFF_MODE:-integrate}"
EULER_STEPS="${EULER_STEPS:-1 2 4 8 16}"
MAX_N_GRAPHS="${MAX_N_GRAPHS:-1000}"
RESULTS_CSV="${RESULTS_CSV:-ablation/results/nfe_sweep.csv}"
META_TXT="${META_TXT:-ablation/results/nfe_sweep_meta.txt}"
USE_LAST="${USE_LAST:-0}"
FLOW_INTEGRATE_SCHEDULE="${FLOW_INTEGRATE_SCHEDULE:-uniform}"
FLOW_INTEGRATE_T_HI="${FLOW_INTEGRATE_T_HI:-0.99}"
FLOW_INTEGRATE_POWER="${FLOW_INTEGRATE_POWER:-4.0}"

if [[ ! -d "${CKPT_PATH}" ]]; then
  echo "ERROR: checkpoint dir not found: ${CKPT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SPLIT_FILE}" ]]; then
  echo "ERROR: missing split: ${SPLIT_FILE}" >&2
  exit 1
fi

USE_LAST_FLAG=()
if [[ "${USE_LAST}" == "1" ]]; then
  USE_LAST_FLAG=(--use_last)
fi

# Archive previous CSV if present.
if [[ -f "${RESULTS_CSV}" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  cp -a "${RESULTS_CSV}" "${RESULTS_CSV}.bak_${ts}"
  echo "Archived previous results -> ${RESULTS_CSV}.bak_${ts}"
fi

read -r -a STEPS_ARR <<< "${EULER_STEPS}"

{
  echo "date=$(date -Is)"
  echo "jobid=${SLURM_JOB_ID:-local}"
  echo "CKPT_PATH=${CKPT_PATH}"
  echo "USE_LAST=${USE_LAST}  (best-val when 0; fit-end last.ckpt when 1)"
  echo "BATCH_SIZE=${BATCH_SIZE}  COEFF_MODE=${COEFF_MODE}"
  echo "schedule=${FLOW_INTEGRATE_SCHEDULE} t_hi=${FLOW_INTEGRATE_T_HI} power=${FLOW_INTEGRATE_POWER}"
  echo "note: relative cost = wall_seconds(K) / wall_seconds(K=1) on the same ${MAX_N_GRAPHS}-mol pass"
} | tee "${META_TXT}"

echo "n_steps,nmape_mean,nmape_std,wall_seconds,seconds_per_mol,rel_cost,resolved_ckpt" > "${RESULTS_CSV}"

base_wall=""
for FLOW_INTEGRATE_STEPS in "${STEPS_ARR[@]}"; do
  run_log="$(mktemp)"
  /usr/bin/time -f "WALL_SECONDS %e" -o "${run_log}.time" \
    python scdp/scripts/test_flow.py \
      --ckpt_path "${CKPT_PATH}" \
      --data_path "${LMDB_PATH}" \
      --split_file "${SPLIT_FILE}" \
      --tag "${TAG}" \
      --output_tag "nfe_k${FLOW_INTEGRATE_STEPS}" \
      --max_n_graphs "${MAX_N_GRAPHS}" \
      --batch_size "${BATCH_SIZE}" \
      --max_n_probe "${MAX_N_PROBE}" \
      --n_steps "${FLOW_INTEGRATE_STEPS}" \
      --coeff_mode "${COEFF_MODE}" \
      --eval_log - \
      --no_nmape_txt \
      "${USE_LAST_FLAG[@]}" \
      --integrate_schedule "${FLOW_INTEGRATE_SCHEDULE}" \
      --integrate_t_hi "${FLOW_INTEGRATE_T_HI}" \
      --integrate_power "${FLOW_INTEGRATE_POWER}" \
      2>&1 | tee "${run_log}"

  python - <<PY
import re
from pathlib import Path

steps = int("""${FLOW_INTEGRATE_STEPS}""")
n_graphs = int("""${MAX_N_GRAPHS}""")
text = Path("""${run_log}""").read_text()

mean = std = None
resolved = ""
for line in text.splitlines():
    if "loaded checkpoint:" in line:
        resolved = line.split("loaded checkpoint:", 1)[-1].strip()
    if "flow test nMAPE" not in line and not re.search(r"\bnmape\b", line, re.I):
        continue
    m = re.search(r"\bmean\s*=\s*([0-9.]+).*?\bstd\s*=\s*([0-9.]+)", line, re.I)
    if m:
        mean, std = float(m.group(1)), float(m.group(2))
        continue
    m = re.search(r"(?:flow test nMAPE|nmape)\s*[:=]\s*([0-9.]+)\s*±\s*([0-9.]+)", line, re.I)
    if m:
        mean, std = float(m.group(1)), float(m.group(2))
if mean is None:
    raise SystemExit(f"ERROR: parse nMAPE failed for K={steps}")

# Prefer /usr/bin/time wall; fall back to logged elapsed_s.
wall = None
time_path = Path("""${run_log}.time""")
if time_path.is_file():
    wall_txt = time_path.read_text().strip()
    m = re.search(r"WALL_SECONDS\s+([0-9]+(?:\.[0-9]+)?)", wall_txt)
    if m:
        wall = float(m.group(1))
if wall is None:
    m = re.search(r"elapsed_s\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        wall = float(m.group(1))
if wall is None:
    raise SystemExit(f"ERROR: could not parse wall time for K={steps}")

spm = wall / max(n_graphs, 1)
base_path = Path("""${RESULTS_CSV}.base_wall""")
if not base_path.exists():
    base_path.write_text(str(wall))
base = float(base_path.read_text())
rel = wall / base if base > 0 else float("nan")

out = Path("""${RESULTS_CSV}""")
with out.open("a") as f:
    f.write(
        f"{steps},{mean:.6f},{std:.6f},{wall:.3f},{spm:.6f},{rel:.4f},{resolved}\n"
    )
print(
    f"K={steps}: nMAPE={mean:.4f}±{std:.4f}  "
    f"wall={wall:.1f}s  s/mol={spm:.4f}  rel={rel:.3f}x  ckpt={resolved}"
)
PY
done

rm -f "${RESULTS_CSV}.base_wall"
cat "${RESULTS_CSV}"
cat "${META_TXT}"
