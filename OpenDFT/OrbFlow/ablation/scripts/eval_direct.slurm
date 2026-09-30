#!/usr/bin/env bash
#SBATCH --job-name=abl_eval_dr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --output=ablation/runs/logs/eval_direct_%j.out
#SBATCH --error=ablation/runs/logs/eval_direct_%j.err

# Direct-regression eval on the 10% test split.
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

CKPT_PATH="${CKPT_PATH:-ablation/runs/direct_regression}"
SPLIT_FILE="${SPLIT_FILE:-ablation/data/datasplits_ablation_10pct.json}"
LMDB_PATH="${LMDB_PATH:-${DATAPATH}/lmdb}"
TAG="${TAG:-test}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_N_PROBE="${MAX_N_PROBE:-500000}"
MAX_N_GRAPHS="${MAX_N_GRAPHS:-1000}"
RESULTS_CSV="${RESULTS_CSV:-ablation/results/summary.csv}"
LABEL="${LABEL:-direct_regression}"

[[ -d "${CKPT_PATH}" ]] || exit 1

run_log="$(mktemp)"
set +e
python scdp/scripts/test.py \
  --ckpt_path "${CKPT_PATH}" \
  --data_path "${LMDB_PATH}" \
  --split_file "${SPLIT_FILE}" \
  --tag "${TAG}" \
  --max_n_graphs "${MAX_N_GRAPHS}" \
  --batch_size "${BATCH_SIZE}" \
  --max_n_probe "${MAX_N_PROBE}" \
  --use_best \
  2>&1 | tee "${run_log}"
py_rc=${PIPESTATUS[0]}
set -e
[[ "${py_rc}" -eq 0 ]] || exit "${py_rc}"

python - <<PY
import re
from pathlib import Path

import numpy as np

label = """${LABEL}"""
ckpt = Path("""${CKPT_PATH}""")
tag = """${TAG}"""
csv_path = Path("""${RESULTS_CSV}""")
text = Path("""${run_log}""").read_text()

mean = std = None
# Preferred: per-graph scores written by test.py
nmape_txt = ckpt / f"nmape_{tag}.txt"
if nmape_txt.is_file():
    vals = np.loadtxt(nmape_txt)
    if vals.ndim == 0:
        vals = np.array([float(vals)])
    if vals.size > 0:
        mean = f"{float(vals.mean()):.6f}"
        std = f"{float(vals.std()):.6f}"
        print(f"parsed from {nmape_txt}: n={vals.size} nMAPE={mean}±{std}")

if mean is None:
    for line in text.splitlines():
        m = re.search(
            r"nmape.*?\bmean\s*=\s*([0-9.]+).*?\bstd\s*=\s*([0-9.]+)",
            line,
            re.I,
        )
        if m:
            mean, std = m.group(1), m.group(2)
            break
        m = re.search(r"nmape\s*[:=]\s*([0-9.]+)\s*±\s*([0-9.]+)", line, re.I)
        if m:
            mean, std = m.group(1), m.group(2)
            break
if mean is None:
    for line in text.splitlines():
        m = re.search(r"([0-9.]+)\s*±\s*([0-9.]+)", line)
        if m and "nmape" in line.lower():
            mean, std = m.group(1), m.group(2)
            break
if mean is None:
    raise SystemExit(
        "ERROR: could not parse nMAPE (checked stdout and "
        f"{nmape_txt})"
    )

# Dedup: replace existing row for this label if present
header = "label,ckpt_path,eval_split,n_steps,nmape_mean,nmape_std,notes\n"
row = f"{label},{ckpt},test_10pct,NA,{mean},{std},direct_regression\n"
if csv_path.exists():
    lines = csv_path.read_text().splitlines()
    keep = [lines[0]] if lines else [header.strip()]
    for line in lines[1:]:
        if not line.startswith(f"{label},"):
            keep.append(line)
    keep.append(row.rstrip("\n"))
    csv_path.write_text("\n".join(keep) + "\n")
else:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(header + row)
print(f"wrote -> {csv_path}: {label} nMAPE={mean}±{std}")
PY
