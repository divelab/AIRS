

# :sparkles: Invariant function learning for ODEs: Disentanglement of Invariant Functions :sparkles:

[license-url]: https://github.com/divelab/AIRS/blob/main/LICENSE
[license-image]:https://img.shields.io/badge/license-GPL3.0-green.svg

![Last Commit](https://img.shields.io/github/last-commit/divelab/AIRS)
[![License][license-image]][license-url]

[**ICML 2025 Paper**](https://openreview.net/forum?id=hzYHxtIn23) | [Preprint](https://www.arxiv.org/abs/2502.04495) 

## Table of contents

* [Overview](#overview)
* [Installation](#installation)
* [Quick tutorial](#quick-tutorial)
* [Citing DIF](#citing-dif)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)

## Overview


## Installation 

* Unbuntu >= 18.04

```bash
conda env create -f environment.yml
conda activate CEL
```
## Quick Tutorial

### Download datasets

Let's say that your storage dir is <storage_dir>.

Download the datasets [HERE](https://drive.google.com/file/d/1aFahCl7RNoQQrcWsRP4UKnlHBhoqaQcE/view?usp=sharing) and put it under `<storage_dir>/`.

```bash
cd <storage_dir>/
unzip datasets.zip
```

Then you are ready to go!

### Initial run

When running the following command, you will be prompted to input your storage folder `<storage_dir>`, which will be connected by a soft link for future use. (You only need to do this once.)

```bash
python -m CEL.kernel.main --config configs/json_config/DampedPendulum/InvariantFuncEncMI.yaml --exp.init_args.wandb_logger False
```

You may notice two things:
- The `wandb_logger` is set to `False`, which means that we will not log the results to Weights & Biases (wandb) for this run. After you log in to wandb, you can set it to `True` to log the results.
- The config file is a default one, basically including a lot of parameter placeholders (not the one you want to reproduce, so you can just make sure it is runnable and skip it), class arguments, and structures. Since we use jsonargparse to parse the config file and instantiate classes, so please read its [DOCS](https://jsonargparse.readthedocs.io/en/v4.40.0/#class-type-and-sub-classes) first if you are not familiar with it or you may feel there are a lot of magic.

As for July 3rd, 2025, the above steps should be fully reproducible.

### Running IFL-DIF

In our paper, you can find our results including a lot of runs. These runs are obtained from the following sweeping config files, which are located at `configs/sweep_config/`.
Now, we need to use `wandb sweep` and use `wandb agent` to run the sweeping jobs.

```bash
wandb sweep --project InvariantFunctionLearning configs/sweep_config/Pendulum/Pendulum_ETSD_MI.yaml
wandb sweep --project InvariantFunctionLearning configs/sweep_config/LV2/V2F2LO2_lf2_MI.yaml
wandb sweep --project configs/sweep_config/SIREpidemic/SIR_FOND_MI.yaml

CUDA_VISIBLE_DEVICES=0 wandb agent <sweep_id>
```

Please definitely check [wandb sweep](https://docs.wandb.ai/guides/sweeps/parallelize-agents/) and launch your sweeping jobs with LOTS OF `wandb agent` on all the GPUs you have.

### Checking results

When you are running the sweeping jobs, you can check the results in your wandb dashboard. You may check in the sweep session or use a sweep filter in your main workspace (I prefer the latter one).

Then you can get something like this:

![wandb dashboard1](/docs/figures/wandb_fig_Page_1.jpg)
![wandb dashboard2](/docs/figures/wandb_fig_Page_2.jpg)

### Visualization

You can paint the figures like the ones in our paper by running the plot.ipynb notebook in `notebooks/`. 

## Additional experiments and visualization

All the experiment configs in `configs/sweep_config/` and visualization notebooks in `notebooks/` are runnable. Have fun!



## Citing DIF
If you find this repository helpful, please cite our [paper](https://www.arxiv.org/abs/2502.04495).
```
@inproceedings{
gui2025discovering,
title={Discovering Physics Laws of Dynamical Systems via Invariant Function Learning},
author={Shurui Gui and Xiner Li and Shuiwang Ji},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=hzYHxtIn23}
}
```

## Discussion

Please submit [new issues](/../../issues/new) or start [a new discussion](/../../discussions/new) for any technical or other questions.

## Contact

Please feel free to contact [Shurui Gui](mailto:shurui.gui@tamu.edu) or [Shuiwang Ji](mailto:sji@tamu.edu)!

## Acknowledgements

This work was supported in part by National Science Foundation under grant CNS-2328395 and ARPA-H under grant 1AY1AX000053.




[//]: # (### Mutual information theory based method.)

[//]: # ()
[//]: # (- Maximize: I&#40;F_C;Y&#41;, I&#40;F;Y&#41;, I&#40;F_E^theta;E&#41;.)

[//]: # (- Minimize: I&#40;F_C;E&#41;, H&#40;F_E^theta|E&#41;.)

[//]: # (- Discriminator: input: F_C and F_E^theta)

[//]: # (- Adversarial: optimize hypernetwork to minimize I&#40;F_C;E&#41; === Maximize H&#40;E|F_C&#41; === Discriminator predict uniform distributed E given F_C.)

[//]: # ()
[//]: # ()
[//]: # (wandb sweep --project InvariantFunctionLearning configs/sweep_config/Pendulum/Pendulum_ETSD_MI.yaml)