# ATGC-Gen

This is the official implement of Paper [Language Models for Controllable DNA Sequence Design](https://arxiv.org/abs/2507.19523). 

![framework](images/framework.png)

## Installation

- clone this repo
- create the env and install the requirements
  
```bash
git clone https://github.com/divelab/AIRS.git
cd AIRS/OpenBio/ATGC_Gen
source ./install.sh
```

# Dataset

The dataset for this repo can be downloaded from https://huggingface.co/datasets/xingyusu/DNA_Gen. \
Please place the downloaded contents into a newly created folder named `data/`.


# Training and Evaluation

All training and evaluation scripts are located in the `scripts/` directory.

For example, to run the training process of promoter design, execute
```bash
sh scripts/bert_train_promo.sh
```

After training completes, you can perform evaluation with
```bash
sh scripts/bert_gen_promo.sh
```

> **Note:** Please ensure that `ROOT_PATH` and `model_path` are correctly set in the script before running.  
> Other datasets follow similar logic and structure.



# Evaluation by Trained Models

The pretrained models are available at: https://huggingface.co/xingyusu/DNA_ATGC_Gen. \
To use it, set the `model_path` in the corresponding generation script to point to the downloaded `.ckpt` file.



## Citation


Please cite our paper if you find our paper useful.

```
@article{su2025language,
  title={Language Models for Controllable DNA Sequence Design},
  author={Su, Xingyu and Li, Xiner and Lin, Yuchao and Xie, Ziqian and Zhi, Degui and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2507.19523},
  year={2025}
}
```

## Acknowledgments

This work was supported in part by National Institutes of Health under grant U01AG070112.
