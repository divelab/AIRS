#!/usr/bin/env bash
#SBATCH --job-name=preqm9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --output=vision_preprocess_qm9_%j.out

# Build the QM9 LMDB from the DTU VASP tarballs in $DATAPATH (SCDP pipeline, bond-midpoint virtual nodes).
# Tarball i (sorted by name) -> shard data.%04d.lmdb i; order within a shard follows the tarball.
# Extra args go to preprocess.py, e.g. --start_index 0 --end_index 9 --skip_existing.

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f .env ]] && set -a && source .env && set +a
: "${DATAPATH:?set DATAPATH in .env}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
if command -v conda >/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-orbflow}"
fi

mkdir -p "${DATAPATH}/lmdb"
python scdp/scripts/preprocess.py \
  --tar \
  --disable_pbc \
  --data_path "${DATAPATH}" \
  --out_path "${DATAPATH}/lmdb" \
  --device cpu \
  --atom_cutoff 6 \
  --vnode_method bond \
  --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
  "$@"
