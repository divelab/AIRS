# QH9: A Quantum Hamiltonian Prediction Benchmark

[[Paper]](https://arxiv.org/abs/2306.09549)

## Introduction

QH9 provides precise DFT-calculated Hamiltonian matrices for **2,399 molecular dynamics trajectories** and **130,831  stable molecular geometries**, based on the [QM9](http://quantum-machine.org/datasets/) dataset.

In this repo, we provide both the QH9 dataset and the baseline models, which can be highly valuable for developing machine learning methods and accelerating molecular and materials design for scientific and technological applications.


## Tasks

To comprehensively evaluate the quantum Hamiltonian prediction performance, we define the following tasks based on the obtained stable and dynamic geometries in the QH9 dataset. Please refer to our paper for details of these task setups.

* **QH-stable-iid** 
* **QH-stable-ood** 
* **QH-dynamic-geo** 
* **QH-dynamic-mol** 

| Task | # Total geometries | # Total molecules | # Training/validation/testing geometries|
| -------- | -------- | -------- | -------- |
|**QH-stable-iid** | 130, 831 | 130, 831 | 104, 664/13, 083/13, 084|
|**QH-stable-ood** | 130, 831 | 130, 831 | 104, 001/17, 495/9, 335|
|**QH-dynamic-geo** | 143, 940 | 2, 399 | 119, 950/11, 995/11, 995|
|**QH-dynamic-mol** | 143, 940 | 2, 399 | 115, 140/14, 340/14, 460|

## Requirement

We include key dependencies below. The versions we used are in parentheses. 
* PyTorch (1.11.0)
* PyG (2.0.4)
* e3nn (0.5.1)
* pyscf (2.2.1)
* hydra-core (1.1.2)

## Dataset Usage
We provide the datasets as commonly used PyG datasets. Here are simple examples to load our datasets with a few lines of code. Prior to that, you can download the `datasets` folder, which includes the raw data files `QH9Stable.db` and `QH9Dynamic.db`, via [this Google Drive link](https://drive.google.com/drive/folders/13pPgBh3XvN2FCpowfnA8TT4VJ0OTceNM?usp=sharing).

```python
from torch_geometric.loader import DataLoader
from datasets import QH9Stable, QH9Dynamic

### Use one of the following lines to Load the specific dataset
dataset = QH9Stable(split='random')  # QH-stable-iid
dataset = QH9Stable(split='size_ood')  # QH-stable-ood
dataset = QH9Dynamic(split='geometry')  # QH-dynamic-geo
dataset = QH9Dynamic(split='mol')  # QH-dynamic-mol

### Get the training/validation/testing subsets
train_dataset = dataset[dataset.train_mask]
valid_dataset = dataset[dataset.val_mask]
test_dataset = dataset[dataset.test_mask]

### Get the dataloders
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_data_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

## Baselines
Equivariant quantum tensor network QHNet is selected as the main baseline method in the QH9 benchmark currently. QHNet has an extendable expansion module that is built upon intermediate full orbital matrices, enabling its capability to effectively handle different molecules. This flexibility allows QHNet to accommodate various molecules in the QH9 benchmark.

* Train the QHNet model
```shell script
### Modify the configurations in config/config.yaml (or pass the configurations as args) as needed, and then run
python main.py datasets=QH9-stable datasets.split=random # QH-stable-iid
python main.py datasets=QH9-stable datasets.split=size_ood # QH-stable-ood
python main.py datasets=QH9-dynamic datasets.split=geometry # QH-stable-iid
python main.py datasets=QH9-dynamic datasets.split=mol # QH-stable-iid
```

* Evaluate the trained model (in terms of MAE on Hamiltonian matrix, MAE on occupied orbital energies, and cosine similarity of orbital coefficients)
```shell script
### Modify the configurations in config/config.yaml (or pass the configurations as args) as needed (including the trained_model arg), and then run
python test.py
```

* Evaluate the performance of accelerating DFT calculation
```shell script
### Modify the configurations in config/config.yaml (or pass the configurations as args) as needed (including the trained_model arg), and then run
python test_dft_acceleration.py
```

## Citing QH9
```
@article{yu2023qh9,
      title={{QH9}: A Quantum Hamiltonian Prediction Benchmark for QM9 Molecules}, 
      author={Haiyang Yu and Meng Liu and Youzhi Luo and Alex Strasser and Xiaofeng Qian and Xiaoning Qian and Shuiwang Ji},
      journal={arXiv Preprint, arXiv:2306.09549},
      year={2023}
}
```
