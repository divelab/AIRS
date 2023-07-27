# Group Equivariant Fourier Neural Operators for Partial Differential Equations

This is the official implementation of *G*-FNO:

Jacob Helwig*, Xuan Zhang*, Cong Fu, Jerry Kurtin, Stephan Wojtowytsch and Shuiwang Ji. "[Group Equivariant Fourier Neural Operators for Partial Differential Equations](https://icml.cc/virtual/2023/poster/23875)". [ICML 2023 Poster]

*Equal contribution

![](assets/network_visual.png)


## Requirements

To create a `GFNO` `conda` environment, run:

```bash
source setup.sh
```

## Preparing Data

* The Navier-Stokes data with a non-symmetric forcing term (NS) are available via the 
[FNO GitHub](https://github.com/neuraloperator/neuraloperator/tree/master). Note that we use both the dataset 
`ns_data_V1e-4_N20_T50_R256test.mat` (20 super-resolution test trajectories) and `ns_V1e-4_N10000_T30.mat`
(10,000 downsampled trajectories). 

* We use [`ns_2d_rt.py`](data_generation/navier_stokes/ns_2d_rt.py) to generate the Navier-Stokes data with a symmetric 
forcing term (NS-Sym). To generate this data (`ns_V0.001_N1200_T30_cos4.mat` for 1,200 downsampled trajectories and 
`ns_V0.001_N1200_T30_cos4_super.mat` for 100 super-resolution test trajectories), run:

```python
python ns_2d_rt.py --nu=1e-4 --T=30 --N=1200 --save_path=./data --ntest=100 --period=4
```

* Instructions for generating the shallow water equations data from PDEArena (SWE) are available via the [PDEArena data generation instructions](https://microsoft.github.io/pdearena/data/).

* The shallow water equations data from PDEBench (SWE) is available via the [PDEBench GitHub](https://github.com/pdebench/PDEBench).

## Run

We use the shell script [`run_experiment.sh`](run_experiment.sh) to run all experiments on all datasets and models. 
Below are commands for training *G*-FNO2d-*p4* on each of the datasets.

NS:

```python
python experiments.py --seed=1 --data_path=./data/ns_V1e-4_N10000_T30.mat \ 
    --results_path=./results/ns_V1e-4_N10000_T30.mat/GFNO2d_p4/ --strategy=teacher_forcing \ 
    --T=20 --ntrain=1000 --nvalid=100 --ntest=100 --model_type=GFNO2d_p4 --modes=12 --width=10 \
    --batch_size=20 --epochs=100 --suffix=seed1 --txt_suffix=ns_V1e-4_N10000_T30.mat_GFNO2d_p4_seed1 \ 
    --learning_rate=1e-3 --early_stopping=100 --verbose --super \
    --super_path=./data/ns_data_V1e-4_N20_T50_R256test.mat
```

NS-Sym:

```python
python experiments.py --seed=1 --data_path=./data/ns_V0.0001_N1200_T30_cos4.mat \ 
    --results_path=./results/ns_V0.0001_N1200_T30_cos4.mat/GFNO2d_p4/ --strategy=teacher_forcing \ 
    --T=10 --ntrain=1000 --nvalid=100 --ntest=100 --model_type=GFNO2d_p4 --modes=12 --width=10 \
    --batch_size=20 --epochs=100 --suffix=seed1 --txt_suffix=ns_V0.0001_N1200_T30_cos4.mat_GFNO2d_p4_seed1 \ 
    --learning_rate=1e-3 --early_stopping=100 --verbose --super \ 
    --super_path=./data/ns_V0.0001_N1200_T30_cos4_super.mat
```

SWE:

```python
python experiments.py --seed=1 --data_path=./data/ShallowWater2D \ 
    --results_path=./results/ShallowWater2D/GFNO2d_p4/ --strategy=teacher_forcing \ 
    --T=9 --ntrain=5600 --nvalid=1120 --ntest=1120 --model_type=GFNO2d_p4 --modes=32 --width=10 \ 
    --batch_size=20 --epochs=100 --suffix=seed1 --txt_suffix=ShallowWater2D_GFNO2d_p4_seed1 \ 
    --learning_rate=1e-3 --early_stopping=100 --verbose --time_pad
```

SWE-Sym:

```python
python experiments.py --seed=1 --data_path=./data/2D_rdb_NA_NA.h5 \
    --results_path=./results/2D_rdb_NA_NA.h5/GFNO2d_p4/ --strategy=teacher_forcing \ 
    --T=24 --ntrain=800 --nvalid=100 --ntest=100 --model_type=GFNO2d_p4 --modes=12 --width=10 \ 
    --batch_size=20 --epochs=100 --suffix=seed1 --txt_suffix=2D_rdb_NA_NA.h5_GFNO2d_p4_seed1 \ 
    --learning_rate=1e-3 --early_stopping=100 --verbose --super
```

## Citation
```latex
@inproceedings{helwig2023group,
author = {Jacob Helwig and Xuan Zhang and Cong Fu and Jerry Kurtin and Stephan Wojtowytsch and Shuiwang Ji},
title = {Group Equivariant {Fourier} Neural Operators for Partial Differential Equations},
booktitle = {Proceedings of the 40th International Conference on Machine Learning},
year = {2023},
}
```

## Acknowledgments
This work was supported in part by National Science Foundation grant IIS-2006861, and by state allocated funds for the Water Exceptional Item through Texas A&M AgriLife Research facilitated by the Texas Water Resources Institute.
