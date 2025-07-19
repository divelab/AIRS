# ATGC-Gen

[//]: # (This is the official implement of Paper [Learning to Discover Regulatory Elements for Gene Expression Prediction]&#40;https://arxiv.org/abs/2502.13991&#41;. You can also find our [paper]&#40;https://huggingface.co/papers/2502.13991&#41; and [material collections]&#40;https://huggingface.co/collections/xingyusu/seq2exp-67daf57c1cfc53d3a4642d44&#41; on <b> 🤗 HuggingFace </b>!)

![framework](images/framework.png)

## Installation

- clone this repo
- create the env and install the requirements
  
```bash
git clone https://github.com/divelab/AIRS.git
cd AIRS/OpenBio/Seq2Exp
source ./install.sh
```

# Dataset

The dataset for this repo can be downloaded from https://huggingface.co/datasets/xingyusu/DNA_Gen. 

Please place the downloaded contents into a newly created folder named `data/`.


# Training and Evaluation

All the training and evaluation codes are under `scripts/`

For example, users could run the training code
```bash
sh scripts/bert_train_promo.sh
```

After the training, run the evaluation codes
```bash
sh scripts/bert_gen_promo.sh
```
Note to potentially change the ROOT_PATH and the model_path.
Other datasets follow similar logics


# Evaluation by Trained Models

The pretrained model can be downloaded from https://huggingface.co/xingyusu/GeneExp_Seq2Exp. 
Set the model_path of the corresponding generation bash file as the downloaded model path (`.ckpt` file).


[//]: # (## Citation)

[//]: # ()
[//]: # ()
[//]: # (Please cite our paper if you find our paper useful.)

[//]: # ()
[//]: # (```)

[//]: # (@article{su2025learning,)

[//]: # (  title={Learning to Discover Regulatory Elements for Gene Expression Prediction},)

[//]: # (  author={Su, Xingyu and Yu, Haiyang and Zhi, Degui and Ji, Shuiwang},)

[//]: # (  journal={arXiv preprint arXiv:2502.13991},)

[//]: # (  year={2025})

[//]: # (})

[//]: # (```)

## Acknowledgments

This work was supported in part by National Institutes of Health under grant U01AG070112.