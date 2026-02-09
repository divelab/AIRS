# TDN
Official code repository of paper “[Tensor Decomposition Networks for Fast Machine Learning Force Field Computations](https://arxiv.org/abs/2507.01131)” by Yuchao Lin, Cong Fu, Zachary Krueger, Haiyang Yu, Maho Nakata, Jianwen Xie, Emine Kucukbenli, Xiaofeng Qian, Shuiwang Ji. [[NeurIPS 2025 Poster](https://openreview.net/forum?id=9vKJyCUfMH)]

![tdn](CPDecompFig-1.png)

This repository contains code for **training and evaluation** on the **PubChemQCR** dataset.

## Environment Setup

See [ENVIRONMENT.md](ENVIRONMENT.md) for environment setup instructions.



## Dataset Preparation for PubChemQCR

PubChemQCR [1] provides relaxation trajectories for ~3.5M small molecules (PM3 → HF → DFT), totaling ~300M snapshots (≈105M at DFT). The dataset is available as a **subset** and a **full** split. For comprehensive details and benchmarks, please refer to the [preprint](https://arxiv.org/abs/2506.23008) and the [original repository](https://huggingface.co/datasets/divelab/PubChemQCR). **Before training, download the LMDB shards to your target directory.**

#### Important Flags

- `root`: path to the directory containing LMDB files
- `stage`: which optimization stage to load
  - `"pm3"` — PM3
  - `"hf"` — Hartree–Fock
  - `"1st"` — DFT first substage (Firefly/SMASH)
  - `"1st_smash"` — SMASH-only portion of the first DFT substage
  - `"2nd"` — DFT second substage (GAMESS)
  - `"mixing"` — DFT first **and** second substages
- `total_traj` (bool): load the full trajectory per molecule
- `subset` (bool): load only the subset split

In this repository, all training uses **DFT first-stage** data with **full trajectories** (`stage="1st"`, `total_traj=True`).



## Training and Evaluation

This project trains and evaluates atomistic machine-learning potentials that predict:

- **Energies** (graph-level target: `data.y`)
- **Forces** (atom-level target: `data.y_force`)

Supported models (selected by `--model_name`):

- `schnet` → `models/schnet.py::SchNet`
- `painn` → `models/painn.py::PaiNN`
- `tdn` → `models/tdn.py::TensorDecompositionNetwork`



### Splits

The split ratios are determined by `--subset`:

- If `--subset` is set:
  - Train = 60%
  - Val   = 20%
  - Test  = remaining 20%
- Otherwise (full):
  - Train = 80%
  - Val   = 10%
  - Test  = remaining 10%



### Loss metrics

- **Energy criterion**: `nn.L1Loss()`
- **Force criterion**: `RMSE()`
- **Total training objective:**

```
loss = energy_weight * energy_loss + force_weight * force_loss
```



### Tensor product

TDN uses `uvu` connection mode and saves the precomputed tensor decomposition result to `tmp` as default, corresponding to `CPTensorProductSH`. Before training, remember `mkdir tmp`.



### Single GPU training

Example of subset TDN training:

```
python main.py \
  --root data/ \
  --model_name tdn \
  --stage 1st \
  --optimizer_name adam \
  --batch_size 64 \
  --cutoff 4.5 \
  --num_gaussians 256 \
  --hidden_channels 256 \
  --num_interactions 6 \
  --lr 5e-4 \
  --epochs 100 \
  --energy_weight 1.0 \
  --force_weight 1.0 \
  --num_workers 16 \
  --total_traj \
  --subset
```

Additionally, use `--resume_from_checkpoint` pointing to a `*_checkpoint_*.pth` file or `--EMA` to enable exponential moving average training. For TDN, change `_AVG_NUM_NODES` and `_AVG_DEGREE` according to the subset or full set.



### Distributed Training

Example of full set TDN training

```
torchrun --standalone --nproc_per_node=4 main.py \
  --root data/ \
  --model_name tdn \
  --stage 1st \
  --optimizer_name adam \
  --batch_size 64 \
  --cutoff 4.5 \
  --num_gaussians 256 \
  --hidden_channels 256 \
  --num_interactions 6 \
  --lr 5e-4 \
  --epochs 100 \
  --energy_weight 1.0 \
  --force_weight 1.0 \
  --num_workers 16 \
  --total_traj \
  --data_parallel
```



### Evaluation

Example of TDN evaluation with checkpoint `tdn_checkpoint_True_6_64_0.0005.pth`

```
python main.py \
  --root data/ \
  --model_name tdn \
  --stage 1st \
  --optimizer_name adam \
  --batch_size 64 \
  --cutoff 4.5 \
  --num_gaussians 256 \
  --hidden_channels 256 \
  --num_interactions 6 \
  --lr 5e-4 \
  --epochs 0 \
  --energy_weight 1.0 \
  --force_weight 1.0 \
  --num_workers 16 \
  --total_traj \
  --subset \
  --resume_from_checkpoint tdn_checkpoint_True_6_64_0.0005.pth
```



## Acknowledgement

The TDN architecture is based on [Equiformer](https://github.com/atomicarchitects/equiformer) [2], and the tensor product operation is based on e3nn [tensor product](https://github.com/e3nn/e3nn/tree/main/e3nn/o3/_tensor_product) [3].  This work is supported by ARPA-H under grant 1AY1AX000053, National Institutes of Health under grant U01AG070112, National Science Foundation under grant IIS-2243850, and the Air Force Office of Scientific Research (AFOSR) under Grant FA9550-24-1-0207. We acknowledge the support of Lambda, Inc. and NVIDIA for providing the computational resources for this project.



## Reference

[1] **A Benchmark for Quantum Chemistry Relaxations via Machine Learning Interatomic Potentials.** *Cong Fu, Yuchao Lin, Zachary Krueger, Wendi Yu, Xiaoning Qian, Byung-Jun Yoon, Raymundo Arróyave, Xiaofeng Qian, Toshiyuki Maeda, Maho Nakata, Shuiwang Ji.*

[2] **Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs.** *Yi-Lun Liao, Tess Smidt.*

[3] **e3nn: Euclidean Neural Networks.** *Mario Geiger, Tess Smidt.*



## Citation

```latex
@inproceedings{
lin2025tensor,
title={Tensor Decomposition Networks for Accelerating Machine Learning Force Field Computations},
author={Yuchao Lin and Cong Fu and Zachary Krueger and Haiyang Yu and Maho Nakata and Jianwen Xie and Emine Kucukbenli and Xiaofeng Qian and Shuiwang Ji},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=9vKJyCUfMH}
}
```
