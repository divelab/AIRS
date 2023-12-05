# Semi-Supervised Learning for High-Fidelity Fluid Flow Reconstruction

This is the official implementation of the **CFRF** method proposed in the following paper. 

Cong Fu, Jacob Helwig, Shuiwang Ji ["Semi-Supervised Learning for High-Fidelity Fluid Flow Reconstruction"](https://openreview.net/pdf?id=695IYJh1Ba), Learning on Graphs Conference (LoG) 2023

<p align="center">
<img src="https://github.com/divelab/AIRS/blob/main/OpenPDE/CFRF/assets/CFRF.png" width="800" class="center" alt=""/>
    <br/>
</p>


## Requirements

To create a `CFRF` `conda` environment, run:
```bash
source setup.sh
```

## Preparing Data
* We use three PDEs: 2D Incompressible Navier-Stokes Equation, 2D Shallow Water Equation, and 2D Diffusion-Reaction Equation.
* 2D Incompressible Navier-Stokes Equation data is generated using the script provided in [FNO Github](https://github.com/neuraloperator/neuraloperator/blob/master/data_generation/navier_stokes/ns_2d.py).
* The Shallow Water and Diffusion-Reaction equations data are generated using scripts provided in [PDEBench](https://github.com/pdebench/PDEBench).
* We put all the generated data in [google drive](https://drive.google.com/drive/folders/1UUjsH48WnH6mc38wUXVbTIjvQuXeaVoH?usp=sharing) for downloading.

Download the data from the google drive and unzip all the datasets in the `data` folder.


## Run
The training process contains two stages: 1. supervised training for the proposal network 2. unsupervised training for the refinement network.  We use the Navier-Stokes equation as an example, the training on the other two datasets are similar.

* Train CFRF on Navier-Stokes data:

First, we train the proposal network:
```linux
cd scripts
bash train_proposal_iCFD.sh
```
After training the proposal network, a folder will be created containing the trained model, evaluation results, and arguments. The name of the folder is `<time stamp> + <suffix>`, where suffix is defined in the bash file.

Then, we can train the refinement network:
```linux
cd scripts
bash train_refinement_iCFD.sh
```
Note that in the `train_refinement_iCFD.sh`, you need to replace `"path to the root folder of the trained model"` with the folder just created when training the proposal network,

## Citation
```latex
@inproceedings{fu2023semi,
  title={Semi-Supervised Learning for High-Fidelity Fluid Flow Reconstruction},
  author={Fu, Cong and Helwig, Jacob and Ji, Shuiwang},
  booktitle={The Second Learning on Graphs Conference},
  year={2023}
}
```

## Acknowledgments
This work was supported in part by National Science Foundation grant IIS-2006861.
