# PotNet
Official code repository of paper “[Efficient Approximations of Complete Interatomic Potentials for Crystal Property Prediction](https://arxiv.org/abs/2306.10045)” by Yuchao Lin, Keqiang Yan, Youzhi Luo, Yi Liu, Xiaoning Qian, and Shuiwang Ji. [ICML 2023 Poster]

![graph](graph.png)

## Environment Setup

- We'll use `conda` to install dependencies and set up the environment. We recommend using the [Python 3.9 Miniconda installer](https://docs.conda.io/en/latest/miniconda.html#linux-installers).
- After installing `conda`, run

```shell
conda env create -f environment.yml
```

- Then run below code to activate the environment

```
conda activate potnet
```

- Be aware that we are using an old version of JARVIS toolkits `jarvis-tools==2022.9.16`. The newest JARVIS toolkits will contain new versions of datasets that contain more data than the one we present in the paper.

## Running Summation Algorithm

To run the summation algorithm, please run below commands in order to install the algorithm package

```shell
cd functions
tar xzvf gsl-latest.tar.gz
cd gsl-2.7.1
./configure --prefix=TARGET PATH
make
make install
```

Then edit `~/.bashrc` by adding

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:TARGET PATH/lib/
```

Now we back to `functions` directory and run

```shell
python setup.py build_ext --inplace
```

Then the algorithm is installed as Cython package. A simple way to test if it is successfully installed is to run below in the root directory.

```shell
python test_algorithm.py
```

## Using Summation Algorithm

We provide six infinite summations of

- Coulomb potential
- London dispersion potential
- Pauli repulsion potential
- Lennard-Jones potential
- Morse potential
- Screened Coulomb potential

and they are achieved in `algorithm.py` by

- `zeta` function referring to summation $\sum_{\mathbf{k}\in \mathbb{Z}^d, \Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert\ne 0}\frac{1}{\Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert^{2p}}$
- `exp` function referring to summation $\sum_{\mathbf{k}\in \mathbb{Z}^d } e^{-\alpha \Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert}$
- `lj` function referring to summation $\sum_{\mathbf{k}\in \mathbb{Z}^d, \Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert\ne 0}(\frac{\sigma^{12}}{\Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert^{12}} - \frac{\sigma^6}{\Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert^6} )$
- `morse` function referring to summation $\sum_{\mathbf{k}\in \mathbb{Z}^d} (e^{-2\alpha \Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert - r_e } - 2e^{-\alpha \Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert - r_e})$
- `screened_coulomb` function referring to summation $\sum_{\mathbf{k}\in\mathbb{Z}^d, \Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert\ne 0} \frac{e^{-\alpha \Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert}}{\Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert} $

Each function requires the input of vectors `v` and lattice matrix `Omega`, dimension `d` and a specific `param`. We also provide a parameter `R` denoting half of the grid length and the corresponding error bound calculation enabled by `verbose`. 

To parallelize a single computation, set `parallel` as `True` and it will use `NUM_CPUS` (default 32) cpus to compute the function parallelly. Note that this will be set as `False` when doing data processing in our model because we provide a faster parallelization by `pandarallel`. 

For more details, see Appendix C.5 in the paper on error bound computation.

## Train and Evaluate Models

In this code base, the datasets are directly provided by JARVIS toolkits and there is no need to download the JARVIS or Materials Project dataset from the official site. To change between different datasets and among different properties, go to `config.yaml` and set the corresponding entries such as

```yaml
dataset: dft_3d
# JARVIS dataset: dft_3d
# MP dataset: megnet
target: formation_energy_peratom
# JARVIS dataset entries: formation_energy_peratom, mbj_bandgap, optb88vdw_bandgap, optb88vdw_total_energy, ehull
# MP dataset entries: e_form, gap pbe
```

To train our model, use the script

```shell
python main.py --config configs/potnet.yaml --output_dir xxx --checkpoint xxx
```

Here, `output_dir` denotes the output directory of checkpoints and processed data files, and `checkpoint` denotes the path of a checkpoint meaning restarting training from a certain checkpoint. One can also omit `checkpoint` in this script.

To evaluate our model, use the script

```shell
python main.py --config configs/potnet.yaml --output_dir xxx --checkpoint xxx --testing
```

and here `checkpoint` denotes the path of a checkpoint and `testing` denotes the evaluation phase.

## Train on Custom Dataset

We are supporting custom datasets in the same format as datasets in [JARVIS Leaderboard](https://github.com/usnistgov/jarvis_leaderboard/tree/main). Once one has the corresponding dataset, use the script

```shell
python main.py --config configs/potnet.yaml --output_dir xxx --checkpoint xxx --data_root xxx
```

where `data_root` denotes the path of the custom dataset. And then the code will read the data from `dataset_info.json` in the dataset file.

## Acknowledgement

The underlying training part is based on [ALIGNN](https://github.com/usnistgov/alignn) [2] and the incomplete Bessel Function is based on [ScaFaCoS](https://github.com/scafacos/scafacos) [3].



## Reference

[1] Crandall, R. E. (1998). Fast evaluation of Epstein zeta functions. 

[2] Choudhary, K. and DeCost, B. (2021). Atomistic line graph neural network for improved materials property predictions. *npj Computational Materials*, *7*(1), p.185.

[3] Sutmann, G. (2014). ScaFaCoS–A Scalable library of Fast Coulomb Solvers for particle Systems.



## Citation

```latex
@inproceedings{lin2023efficient,
author = {Yuchao Lin and Keqiang Yan and Youzhi Luo and Yi Liu and Xiaoning Qian and Shuiwang Ji},
title = {Efficient Approximations of Complete Interatomic Potentials for Crystal Property Prediction},
booktitle = {Proceedings of the 40th International Conference on Machine Learning},
year = {2023},
}
```

