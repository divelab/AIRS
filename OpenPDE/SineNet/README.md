# SineNet: Learning Temporal Dynamics in Time-Dependent Partial Differential Equations

Code for our [ICLR 2024 paper](https://openreview.net/pdf?id=LSYhE2hLWG) on deep-learning-based fluid dynamics simulation. Our code is based on the [PDEArena](https://github.com/microsoft/pdearena) library [1].

<!-- ![Alt Text](./assets/171.gif) -->
Incompressible Navier-Stokes        |  Compressible Navier-Stokes |  Shallow Water Equations
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/divelab/AIRS/blob/main/OpenPDE/SineNet/assets/NavierStokes2D_1218.gif)  |  ![](https://github.com/divelab/AIRS/blob/main/OpenPDE/SineNet/assets/CFD_1198.gif) | ![](https://github.com/divelab/AIRS/blob/main/OpenPDE/SineNet/assets/ShallowWater2DVel-2Day_193.gif)


## Model

We employ a multi-stage UNet model. Checkout our paper for details.

<p align="center">
<img src="https://github.com/divelab/AIRS/blob/main/OpenPDE/SineNet/assets/SineNet_arch.jpg" width="800" class="center" alt=""/>
    <br/>
</p>

## Data

For **INS** and **SWE**, please download from PDEArena [here](https://microsoft.github.io/pdearena/datadownload/). The **SWE** data are converted from `.nc` to `.h5` using the `h5_conv.py` script.

The **CNS** data were generated using PDEBench [2] [here](https://github.com/pdebench/PDEBench/tree/main/pdebench/data_download). This data can be generated using their code with the modified files and data generation script `PDEBench_gen.sh` in the `PDEbench` folder of this repo as:

```shell
bash data_gen.sh --mode train --nsamples 5600 --batch_size 50 --run && bash data_gen.sh --mode valid --nsamples 1400 --batch_size 50 --run && bash data_gen.sh --mode test --nsamples 1400 --batch_size 50 --run
```
Note that the solver rarely but consistently exhibits instability resulting in trajectories of all 0, which is why the split we presented in the paper is 5400/1300/1300.

## Setup
```shell
cd pdearena
source setup.sh
```
This will create a new conda environment named pdearena, install this code base locally, and install other necessary packages.

## Training

```shell
python scripts/train.py -c configs/<config>.yaml \
            --data.data_dir=<data_dir> \
            --data.num_workers=8 \
            --data.batch_size=32 \
            --model.name=<model_name> \
            --model.lr=2e-4 --optimizer.lr=2e-4
```
where `<config>` can be `navierstokes2d`, `cfd` or `shallowwater2d_2day`.
Valid `<model_name>`'s can be found in `pdearena/models/registry.py`. E.g., `sinenet8-dual`.


## Testing
```shell
python scripts/test.py test -c configs/<config>.yaml \
            --data.data_dir=<data_dir> \
            --trainer.devices=1 \
            --data.num_workers=8 \
            --data.batch_size=32 \
            --model.name=<model_name> \
            --ckpt_path=<ckpt_path>
```


[1] Gupta, Jayesh K., and Johannes Brandstetter. "Towards multi-spatiotemporal-scale generalized pde modeling." arXiv preprint arXiv:2209.15616 (2022).

[2] Takamoto, Makoto, et al. "PDEBench: An extensive benchmark for scientific machine learning." Advances in Neural Information Processing Systems 35 (2022): 1596-1611.

## Citation

```bibtex
@inproceedings{zhang2024sinenet,
    title={SineNet: Learning Temporal Dynamics in Time-Dependent Partial Differential Equations},
    author={Xuan Zhang and Jacob Helwig and Yuchao Lin and Yaochen Xie and Cong Fu and Stephan Wojtowytsch and Shuiwang Ji},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=LSYhE2hLWG}
}
```

## Acknowledgements

This work was supported in part by National Science Foundation grants IIS-2243850 and IIS-2006861.
