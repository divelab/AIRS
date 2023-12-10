# A Latent Diffusion Model for Protein Structure Generation

This is the official implementation of the **LatentDiff** method proposed in the following paper.

Cong Fu*, Keqiang Yan*, Limei Wang, Wing Yee Au, Michael McThrow, Tao Komikado, Koji Maruhashi, Kanji Uchino, Xiaoning Qian, Shuiwang Ji ["A Latent Diffusion Model for Protein Structure Generation"](https://openreview.net/pdf?id=MBZVrtbi06), the Second Learning on Graphs Conference (LoG) 2023

<p align="center">
<img src="https://github.com/divelab/AIRS/blob/main/OpenProt/LatentDiff/assets/LatentDiff.png" width="800" class="center" alt=""/>
    <br/>
</p>


## Requirements

We include key dependencies below. The versions we used are in the parentheses. Our detailed environmental setup is available in [environment.yml]().
* PyTorch (1.11.0)
* PyTorch Geometric (2.1.0)
* biopython (1.79)
* biotite (0.34.1)
* tmalign (20170708)


## Preparing Data
* We curate protein data from [Protein Data Bank](https://www.rcsb.org/?ref=nav_home) and [AlphaFold DB](https://alphafold.ebi.ac.uk/).
* We put all the curated data in [google drive](https://drive.google.com/drive/folders/1iqaYFDfmrhjGRvmfwUrEnG4dqhruGGGl?usp=sharing) for downloading.

Download the data from the google drive and unzip all the datasets in the `data` folder.


## Run
The training process contains two stages: 1. train the protein autoencoder 2. train the diffusion model in the latent protein space.

* ### Train LatentDiff:

First, we train the protein autoencoder:
```linux
cd scripts
bash train_autoencoder.sh
```
After training the protein autoencoder, a folder will be created containing the trained model. The name of the folder is `<time stamp> + <suffix>`, where the suffix is defined in `train_autoencoder.sh`.

Next, we need to generate training data in the latent protein space using the trained encoder:
```linux
cd data
Run all the cells in gen_data_for_diffusion.ipynb
```
Replace the `<path of protein autoencoder checkpoint>` and `<latent_data_name>` in the `gen_data_for_diffusion.ipynb`. Please save the latent protein data in `data` folder to avoid triggering path error when running the following steps.

Then, we can start training the latent diffusion model:
```linux
cd scripts
bash train_diffusion.sh
```
Note that `latent_dataname` in `train_diffusion.sh` is the same with `<latent_data_name>` you just set in the previous step.

Diffusion model framework is adapted from [EDM](https://github.com/ehoogeboom/e3_diffusion_for_molecules).


* ### Generate proteins and evaluation results:
```linux
cd scripts
source gen_diffusion_analysis.sh
```
There are some variables you need to set in the `gen_diffusion_analysis.sh`: \
`autoencoder_path`: name of the root folder containing the trained autoencoder model (automatically created when training the autoencoder, with name `<time stamp> + <suffix>`) \
`latent_data_name`: this is the same as `latent_dataname` in the `train_diffusion.sh`\
`diffusion_model_path`: name of the root folder containing the trained diffusion model


In order to run `gen_diffusion_analysis.sh`, you also need to install [OmegaFold](https://github.com/HeliXonProtein/OmegaFold). OmegeFold should be installed in another conda environment with the name `omegafold`.


## Citation
```latex
@inproceedings{fu2023latent,
  title={A Latent Diffusion Model for Protein Structure Generation},
  author={Fu, Cong and Yan, Keqiang and Wang, Limei and Au, Wing Yee and McThrow, Michael and Komikado, Tao and Maruhashi, Koji and Uchino, Kanji and Qian, Xiaoning and Ji, Shuiwang},
  booktitle={The Second Learning on Graphs Conference},
  year={2023}
}
```
