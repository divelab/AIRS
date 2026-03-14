# OrbEvo

Official codebase for **OrbEvo**, accepted at **ICLR 2026**.

- Paper (OpenReview PDF): https://openreview.net/pdf?id=06I7jcrkW2
- Datasets (Hugging Face): https://huggingface.co/divelab

## Overview

This repository contains code for training and evaluating autoregressive models for TDDFT dynamics.

Core entry points:
- Training: `python -m orbevo.train`
<!-- - Evaluation: `python orbevo/eval.py` -->

## Installation

### Option 1: Conda

```bash
conda create -n orbevo python=3.11 pip
conda activate orbevo

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
pip install torch_geometric
pip install numpy==1.26.4 hydra-core tensorboard e3nn tqdm
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
uv pip install numpy==1.26.4 hydra-core tensorboard e3nn tqdm
```

You can also install from `requirements.txt` if needed.

## Data

Processed datasets are available on Hugging Face:
- https://huggingface.co/divelab

Place data under the project `data/` directory, for example:
- `data/MDA_train_tddftv2`
- `data/MDA_test_tddftv2`
- `data/QM9_tddftv2`

Raw-data generation and preprocessing references:
- `orbevo/datagen_abacus/preprocess`

Dataset conventions, initialization, and `collate_fn` usage are documented in:
- `orbevo/datasets/README.md`

## Training

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m dft_v2.train +experiment=debug
```

For custom launches, edit experiment configs in `dft_v2/configs/experiment/` and run:

```bash
python -m orbevo.train +experiment=<experiment_name>
```

<!-- ## Evaluation

Example:

```bash
python orbevo/eval.py \
  --dataset QM9_ood \
  --ckpt_path /path/to/ckpt/best_s1.pt \
  --split test
```

Supported datasets in eval are:
- `MDA`
- `QM9`
- `QM9_ood` -->


## Citation

```bibtex
@inproceedings{zhang2026orbital,
  title={Orbital Transformers for Predicting Wavefunctions in Time-Dependent Density Functional Theory},
  author={Xuan Zhang and Haiyang Yu and Chengdong Wang and Jacob Helwig and Shuiwang Ji and Xiaofeng Qian},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=06I7jcrkW2}
}
```

## Acknowledgments
This work was supported in part by the U.S. Department of Energy Office of Basic Energy Sciences under grant DE-SC0023866, National Science Foundation under grant MOMS-2331036, National Institutes of Health under grant U01AG070112, and Advanced Research Projects Agency for Health under 1AY1AX000053.
