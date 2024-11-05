# PACE

This is the official implement of PACE Paper [Equivariant Graph Network Approximations of High-Degree Polynomials for Force Field Prediction](https://openreview.net/pdf?id=7DAFwp0Vne).

## Installation

- clone this repo
- create the env and install the requirements
  
  ```bash
  $ conda create --name pace python=3.8
  $ conda activate pace
  $ conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
  $ conda install pyg -c pyg
  $ pip install e3nn
  $ pip install torch-ema
  $ pip install prettytable
  $ pip install ase
  ```

## Data
The raw data in xyz format is provided in `AIRS/OpenMol/PACE/dataset`. The rMD17 data is downloaded from its [official source](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038). 3BPA and AcAc datasets are from [BOTNet-datasets](https://github.com/davkovacs/BOTNet-datasets).

## Usage

Training and evaluation of PACE. Note that `$DEVICE` is the GPU number.

### rMD17 Datasets

```bash
python main_rmd17.py --device $DEVICE --task aspirin --output_dir ./results/pace/aspirin --num_bessel 6
```

```bash
python main_rmd17.py --device $DEVICE --task azobenzene --output_dir ./results/pace/azobenzene --num_bessel 4
```

```bash
python main_rmd17.py --device $DEVICE --task benzene --output_dir ./results/pace/benzene --energy_weight 15
```

```bash
python main_rmd17.py --device $DEVICE --task ethanol --output_dir ./results/pace/ethanol  --num_bessel 20 --edge_emb expbern
```

```bash
python main_rmd17.py --device $DEVICE --task malonaldehyde --output_dir ./results/pace/malonaldehyde  --num_bessel 12 --edge_emb expbern --cutoff 6
```

```bash
python main_rmd17.py --device $DEVICE --task naphthalene --output_dir ./results/pace/naphthalene  --num_bessel 8
```

```bash
python main_rmd17.py --device $DEVICE --task paracetamol --output_dir ./results/pace/paracetamol  --num_bessel 4
```

```bash
python main_rmd17.py --device $DEVICE --task salicylic --output_dir ./results/pace/salicylic  --num_bessel 10 --edge_emb expbern
```

```bash
python main_rmd17.py --device $DEVICE --task toluene --output_dir ./results/pace/toluene --num_bessel 8
```

```bash
python main_rmd17.py --device $DEVICE --task uracil --output_dir ./results/pace/uracil  --num_bessel 4 --cutoff 6
```

### 3BPA Dataset
Seeds 2, 3 and 4 are used for three runs.
```bash
python main_3bpa.py --model pace_3bpa --device $DEVICE --output_dir ./results/pace/3bpa/seed2 --examples 10 --num_bessel 4 --energy_weight 15 --eval_interval 100 --seed 2
```

### AcAc Dataset
Seeds 2, 3 and 4 are used for three runs.
```bash
python main_acac.py --model pace_acac --device $DEVICE --output_dir ./results/pace/acac/seed2 --examples 10 --num_bessel 4  --energy_weight 15 --eval_interval 100 --seed 2
```

## Citation

Please cite our paper if you find our paper useful.
```
@article{
    xu2024equivariant,
    title={Equivariant Graph Network Approximations of High-Degree Polynomials for Force Field Prediction},
    author={Zhao Xu and Haiyang Yu and Montgomery Bohde and Shuiwang Ji},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2024},
    url={https://openreview.net/forum?id=7DAFwp0Vne},
    note={Featured Certification}
}
```

## Acknowledgments

This work was supported in part by National Science Foundation grant IIS-2243850 and National Institutes of Health grant U01AG070112.