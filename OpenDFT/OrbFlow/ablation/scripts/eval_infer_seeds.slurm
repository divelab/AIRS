#!/usr/bin/env bash
#SBATCH --job-name=abl_infer3seed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --output=ablation/runs/logs/eval_infer_seeds_%j.out
#SBATCH --error=ablation/runs/logs/eval_infer_seeds_%j.err

# Inference-seed error bars on the 10% test split (3 RNG seeds).
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

SPLIT_FILE="${SPLIT_FILE:-ablation/data/datasplits_ablation_10pct.json}"
LMDB_PATH="${LMDB_PATH:-${DATAPATH}/lmdb}"
TAG="${TAG:-test}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_N_PROBE="${MAX_N_PROBE:-500000}"
COEFF_MODE="${COEFF_MODE:-integrate}"
USE_LAST="${USE_LAST:-0}"
MAX_N_GRAPHS="${MAX_N_GRAPHS:-1000}"
INFER_SEEDS="${INFER_SEEDS:-0 1 2}"

# label:relative_ckpt_dir:euler_steps
ABLATION_SPECS="${ABLATION_SPECS:-curriculum_full:ablation/runs/curriculum_full:2 phase1_only:ablation/runs/curriculum_phase1_only:2 phase2_only:ablation/runs/curriculum_phase2_only:2 phase2_k1:ablation/runs/curriculum_phase2_k1:1 phase2_k3:ablation/runs/curriculum_phase2_k3:3 joint_endpoint_k2:ablation/runs/joint_endpoint_k2:2}"

JOB_TAG="${SLURM_JOB_ID:-local}"
OUT_DIR="${OUT_DIR:-ablation/results}"
RESULTS_CSV="${RESULTS_CSV:-${OUT_DIR}/infer_seeds_${JOB_TAG}.csv}"
SUMMARY_TXT="${SUMMARY_TXT:-${OUT_DIR}/infer_seeds_${JOB_TAG}_summary.txt}"
RESULTS_TSV="${RESULTS_TSV:-${OUT_DIR}/.cache/abl_infer_seeds_${JOB_TAG}.tsv}"

FLOW_INTEGRATE_SCHEDULE="${FLOW_INTEGRATE_SCHEDULE:-uniform}"
FLOW_INTEGRATE_T_HI="${FLOW_INTEGRATE_T_HI:-0.99}"
FLOW_INTEGRATE_POWER="${FLOW_INTEGRATE_POWER:-4.0}"

[[ -f "${SPLIT_FILE}" ]] || exit 1

USE_LAST_ARGS=()
[[ "${USE_LAST}" == 1 || "${USE_LAST}" == true ]] && USE_LAST_ARGS=(--use_last)

read -r -a SEED_ARR <<< "${INFER_SEEDS}"
read -r -a SPEC_ARR <<< "${ABLATION_SPECS}"
[[ ${#SEED_ARR[@]} -gt 0 && ${#SPEC_ARR[@]} -gt 0 ]] || exit 1

mkdir -p "$(dirname "${RESULTS_TSV}")"
: > "${RESULTS_TSV}"


for spec in "${SPEC_ARR[@]}"; do
  IFS=':' read -r label ckpt_rel n_steps <<< "${spec}"
  if [[ -z "${label}" || -z "${ckpt_rel}" || -z "${n_steps}" ]]; then
    echo "ERROR: bad ABLATION_SPECS entry: ${spec} (want label:ckpt:n_steps)" >&2
    exit 1
  fi
  CKPT_PATH="${ckpt_rel}"


  [[ -d "${CKPT_PATH}" && -f "${CKPT_PATH}/config.yaml" ]] || exit 1
  compgen -G "${CKPT_PATH}/*.ckpt" > /dev/null || exit 1

  for seed in "${SEED_ARR[@]}"; do
    output_tag="ablation_test10pct_seed${seed}"
    run_log="$(mktemp)"
    python scdp/scripts/test_flow.py \
      --ckpt_path "${CKPT_PATH}" \
      --data_path "${LMDB_PATH}" \
      --split_file "${SPLIT_FILE}" \
      --tag "${TAG}" \
      --output_tag "${output_tag}" \
      --max_n_graphs "${MAX_N_GRAPHS}" \
      --batch_size "${BATCH_SIZE}" \
      --max_n_probe "${MAX_N_PROBE}" \
      --n_steps "${n_steps}" \
      --coeff_mode "${COEFF_MODE}" \
      --seed "${seed}" \
      --eval_log - \
      --no_nmape_txt \
      "${USE_LAST_ARGS[@]}" \
      --integrate_schedule "${FLOW_INTEGRATE_SCHEDULE}" \
      --integrate_t_hi "${FLOW_INTEGRATE_T_HI}" \
      --integrate_power "${FLOW_INTEGRATE_POWER}" \
      2>&1 | tee "${run_log}"
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
      exit "${PIPESTATUS[0]}"
    fi
    python - <<PY >> "${RESULTS_TSV}"
import re
import sys
from pathlib import Path

label = """${label}"""
ckpt = """${CKPT_PATH}"""
seed = """${seed}"""
n_steps_expected = """${n_steps}"""
text = Path("""${run_log}""").read_text()
summary = None
for line in text.splitlines():
    if line.startswith("flow test nMAPE"):
        summary = line
if summary is None:
    print(f"ERROR: missing flow test summary for {label} seed={seed}", file=sys.stderr)
    sys.exit(1)
m = re.search(
    r"n_steps=(\d+).*mean=([\d.]+) std=([\d.]+).* n=(\d+)",
    summary,
)
if m is None:
    print(f"ERROR: could not parse summary: {summary}", file=sys.stderr)
    sys.exit(1)
got = m.group(1)
if got != n_steps_expected:
    print(
        f"ERROR: n_steps mismatch for {label}: expected {n_steps_expected} got {got}",
        file=sys.stderr,
    )
    sys.exit(1)
print(
    f"{label}\t{ckpt}\t{seed}\t{m.group(1)}\t{m.group(2)}\t{m.group(3)}\t{m.group(4)}",
    flush=True,
)
PY
    rm -f "${run_log}"
  done
done

[[ -s "${RESULTS_TSV}" ]] || exit 1

python - <<PY
import csv
import math
from collections import defaultdict
from pathlib import Path

tsv = Path("""${RESULTS_TSV}""")
csv_path = Path("""${RESULTS_CSV}""")
summary_txt = Path("""${SUMMARY_TXT}""")
seeds = [int(x) for x in """${INFER_SEEDS}""".split()]
spec_order = []
for spec in """${ABLATION_SPECS}""".split():
    label = spec.split(":", 1)[0]
    if label not in spec_order:
        spec_order.append(label)

rows = []
for line in tsv.read_text().splitlines():
    if not line.strip():
        continue
    label, ckpt, seed, s, mean, std, n = line.split("\t")
    rows.append(
        {
            "label": label,
            "ckpt_path": ckpt,
            "infer_seed": int(seed),
            "euler_steps": int(s),
            "mean_nmape_pct": float(mean),
            "within_run_std_pct": float(std),
            "n_graphs": int(n),
        }
    )

csv_path.parent.mkdir(parents=True, exist_ok=True)
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "label",
            "ckpt_path",
            "infer_seed",
            "euler_steps",
            "mean_nmape_pct",
            "within_run_std_pct",
            "n_graphs",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

grouped = defaultdict(list)
meta = {}
for r in rows:
    key = (r["label"], r["euler_steps"])
    grouped[key].append(r["mean_nmape_pct"])
    meta[r["label"]] = (r["ckpt_path"], r["euler_steps"])

def mean_std(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)

ordered_keys = []
seen = set()
for label in spec_order:
    for s in sorted({k[1] for k in grouped if k[0] == label}):
        key = (label, s)
        if key in grouped:
            ordered_keys.append(key)
            seen.add(key)
for key in sorted(grouped.keys()):
    if key not in seen:
        ordered_keys.append(key)

lines = [
    "Ablation — inference-seed error bars (10% QM9 test)",
    "=" * 88,
    f"infer_seeds: {seeds}",
    f"max_n_graphs: {rows[0]['n_graphs'] if rows else 'NA'}",
    "note: ± is sample std of per-seed mean nMAPE% (ddof=1)",
    "",
    f"{'label':<24} {'K':>3} {'mean':>10} {'±seed':>10} {'n':>5}  table-ready",
    "-" * 88,
]
for label, s in ordered_keys:
    vals = grouped[(label, s)]
    m, sd = mean_std(vals)
    per = ", ".join(f"{v:.6f}" for v in vals)
    lines.append(
        f"{label:<24} {s:>3} {m:>10.6f} {sd:>10.6f} {len(vals):>5d}  {m:.3f} ± {sd:.3f}"
    )
    ckpt, _ = meta.get(label, ("", s))
    lines.append(f"  ckpt: {ckpt}")
    lines.append(f"  per-seed: [{per}]")

text = "\n".join(lines) + "\n"
print(text, flush=True)
summary_txt.write_text(text)
PY

