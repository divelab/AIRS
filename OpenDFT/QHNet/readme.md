# QHNet (Quantum Hamiltonian Net)

This is the official implement of QHNet Paper [Efficient and Equivariant Graph Networks for Predicting Quantum Hamiltonian](https://arxiv.org/pdf/2306.04922.pdf).

## Installation

- clone this repo
- create the env and install the requirements
  
  ```bash
  $ git clone https://github.com/divelab/AIRS.git
  $ cd AIRS/OpenDFT/QHNet
  $ source ./install.sh
  ```

## Reproduce

Please download the model parameters from [QHNet_models - Google Drive](https://drive.google.com/drive/folders/1GzJimKOaxU-fiyhgKVWWaJ5juNqbJT3N?usp=sharing).
Then set the model path and version in the configs accroding to the names of the saved models.

## Training

Training the QHNet on the MD17 dataset. Note that `$DATASET` is the name of datasets from water, ethanol, malondialdehyde, uracil.
```bash
python train_wH.py dataset=$DATASET model=QHNet model.version=QHNet_w_bias
```

Training the QHNet on the mixed MD17 dataset.
```bash
python train_mixed.py dataset=all model=QHNet model.version=QHNet_w_bias
```

## Test

Testing the QHNet on the MD17 dataset. Note that `$PATH_TO_SAVED_MODEL` is the path to the downloaded model parameters or saved model paramters. The `$Selected_version$` are in `['QHNet', 'QHNet_wo_bias', 'QHNet_w_bias']` according to the model version name in the model filenames. 
```bash
python test_wH.py dataset=$DATASET model=QHNet model.version=$Selected_version model_path=$PATH_TO_SAVED_MODEL
```

Testing the QHNet on the mixed MD17 dataset. 
```bash
python test_mixed.py dataset=all model=QHNet model.version=$Selected_version model_path=$PATH_TO_SAVED_MODEL
```

## Clarification about the model version - `QHNet` suggested

For the model version, the `QHNet_w_bias` applies self-interaction layer with bias in the `\ell=0` by setting `bias=True` in [self-interaction layer](https://github.com/divelab/AIRS/blob/e1abc14bb9424f563e4ff1a4abd9be73a7b4b52f/OpenDFT/QHNet/models/ori_QHNet_with_bias.py#L178) and normalize the path using [weights from the dictionary](https://github.com/divelab/AIRS/blob/e1abc14bb9424f563e4ff1a4abd9be73a7b4b52f/OpenDFT/QHNet/models/ori_QHNet_with_bias.py#L65). For the `QHNet_wo_bias`, it does not apply such bias and path_normalization operations.

When conducting experiments for the paper, the model is developing and have minior setting difference for different experiments. In order to exactly reproduce the reported results, we provide the model with different versions. After our exploration, we clean our model and provide the suggested `QHNet` for following experiments. We applogize for the inconvience.



## Benchmark

The results of QHNet in the main paper is based on **float64**, the it applies a linear schedule of total 200, 000 steps with warmup 1,000.

| Dataset         | Training strategies  | MAE   | $\epsilon$ | $\psi$ |
|:--------------- | -------------------- |:-----:|:----------:|:------:|
| water           | LSW (1,000, 200,000) | 10.79 | 33.76      | 99.99  |
| Ethanol         | LSW (1,000, 200,000) | 20.91 | 81.03      | 99.99  |
| Malondialdehyde | LSW (1,000, 200,000) | 21.52 | 82.12      | 99.92  |
| Uracil          | LSW (1,000, 200,000) | 20.12 | 113.44     | 99.89  |



To facilitate the development of models to predict the Hamiltonian matrix, 
here we provide the results of QHNet on various total training steps 
for later on comparison. Note that the training procedure is based on **float32**.

| Dataset         | Training strategies     | MAE   | $\epsilon$ | $\psi$ |
| --------------- | ----------------------- |:-----:|:----------:|:------:|
| water           | RLROP                   | 10.36 | 36.21      | 99.99  |
| Ethanol         | LSW (10,000, 1,000,000) | 12.78 | 62.97      | 99.99  |
| Ethanol         | RLROP                   | 13.12 | 51.80      | 99.99  |
| Malondialdehyde | LSW (10,000, 1,000,000) | 11.97 | 55.57      | 99.94  |
| Malondialdehyde | RLROP                   | 13.18 | 51.54      | 99.95  |
| Uracil          | LSW (10,000, 1,000,000  | 9.96  | 66.75      | 99.95  |





## Citation

Please cite our paper if you find our paper useful.
```
@inproceedings{yu2023efficient,
  title={Efficient and Equivariant Graph Networks for Predicting Quantum Hamiltonian},
  author={Yu, Haiyang and Xu, Zhao and Qian, Xiaofeng and Qian, Xiaoning and Ji, Shuiwang},
  booktitle={International Conference on Machine Learning},
  year={2023},
  organization={PMLR}
}
```

## Acknowledgments

This work was supported in part by National Science Foundation grants CCF-1553281, IIS-1812641, OAC-1835690, DMR-2119103, DMR-2103842, IIS-1908220, IIS-2006861, and IIS-2212419, and National Institutes of Health grant U01AG070112. Acknowledgment is made to the donors of the American Chemical Society Petroleum Research Fund for partial support of this research.
