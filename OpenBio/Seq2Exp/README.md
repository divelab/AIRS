# Seq2Exp

This is the official implement of Paper [Learning to Discover Regulatory Elements for Gene Expression Prediction]().

![causal_figure](images/causal_figure.jpg)

## Installation

- clone this repo
- create the env and install the requirements
  
```bash
git clone https://github.com/divelab/AIRS.git
cd AIRS/OpenBio/Seq2Exp
source ./install.sh
```

# Dataset

The Dataset for this repo can be downloaded from https://huggingface.co/datasets/xingyusu/GeneExp. Set the data directory as $DATA_ROOT.

# Reproduce

To reproduce the results of Seq2Exp
```bash
sh Seq2Exp.sh $DATA_ROOT
```

To reproduce the results of different baselines, run the following
```bash
sh baselines.sh $DATA_ROOT
```


## Citation


Please cite our paper if you find our paper useful.

```

@inproceedings{sulearning,
  title={Learning to Discover Regulatory Elements for Gene Expression Prediction},
  author={Su, Xingyu and Yu, Haiyang and Zhi, Degui and Ji, Shuiwang},
  booktitle={The Thirteenth International Conference on Learning Representations}
}

```

## Acknowledgments

This work was supported in part by the National Institute on Aging of the National Institutes of Health under Award Number U01AG070112 and ARPA-H under Award Number 1AY1AX000053. The content is solely the responsibility of the authors and does not necessarily represent the official views of the funding agencies.