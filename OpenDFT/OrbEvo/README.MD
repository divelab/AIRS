# OrbEvo

Official codebase for **Orbital Transformers for Predicting Wavefunctions in Time-Dependent Density Functional Theory** (ICLR 2026).

- Paper (OpenReview): https://openreview.net/pdf?id=06I7jcrkW2
- Datasets (Hugging Face): https://huggingface.co/datasets/divelab/OrbEvo

## Overview

This repository contains code for training and evaluating autoregressive models for TDDFT dynamics.

<!-- Core entry points:
- Training: `python -m orbevo.train` -->
<!-- - Evaluation: `python -m orbevo.eval` -->

## Installation

### Option 1: Conda

```bash
conda create -n orbevo python=3.11 pip
conda activate orbevo

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
pip install torch_geometric
pip install numpy==1.26.4 hydra-core tensorboard e3nn tqdm huggingface_hub
```

### Option 2: uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

uv venv --python 3.11
source .venv/bin/activate

uv pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
uv pip install torch_geometric
uv pip install numpy==1.26.4 hydra-core tensorboard e3nn tqdm huggingface_hub
```

<!-- You can also install from `requirements.txt` if needed. -->

## Data

Please check the [data release notes](DATA_RELEASE.md).

<!-- `huggingface_hub` is required if you want `orbevo/datasets/pyg_dataset.py` to automatically download QM9 raw files from Hugging Face into:
- `data/QM9_tddft/raw` -->

Data will be downloaded and processed automatically when running training for the first time.

<!-- 
Raw-data generation and preprocessing references:
- `orbevo/datagen_abacus/preprocess` -->


Dataset conventions, initialization, and `collate_fn` usage are documented in [orbevo/datasets/README.md](orbevo/datasets/README.md)

## Training

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m orbevo.train +experiment=debug
```

For custom launches, edit experiment configs in `orbevo/configs/experiment/` and run:

```bash
python -m orbevo.train +experiment=<experiment_name>
```

<!-- ## Evaluation

Example:

```bash
python -m orbevo.eval \
  --dataset QM9_ood \
  --ckpt_path /path/to/ckpt/best_s1.pt \
  --split test
```

Supported datasets in eval are:
- `MDA`
- `QM9`
- `QM9_ood` -->


## Checkpoints

- OrbEvo-DM, QM9: https://huggingface.co/divelab/orbevo_model/blob/main/qm9_dm_pf_tb8_s8_395k_bs16.pt

## Citation

```bibtex
@inproceedings{
  zhang2026orbital,
  title={Orbital Transformers for Predicting Wavefunctions in Time-Dependent Density Functional Theory},
  author={Xuan Zhang and Haiyang Yu and Chengdong Wang and Jacob Helwig and Xiaofeng Qian and Shuiwang Ji},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=06I7jcrkW2}
}
```
