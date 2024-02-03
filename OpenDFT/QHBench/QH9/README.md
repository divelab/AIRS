# QH9: A Quantum Hamiltonian Prediction Benchmark

[[Paper]](https://arxiv.org/abs/2306.09549) (**NeurIPS**, Track on Datasets and Benchmarks, 2023)

## Introduction

QH9 provides precise DFT-calculated Hamiltonian matrices for **2,399 molecular dynamics trajectories** and **130,831  stable molecular geometries**, based on the [QM9](http://quantum-machine.org/datasets/) dataset.

In this repo, we provide both the QH9 dataset and the benchmark code, which can be highly valuable for developing machine learning methods and accelerating molecular and materials design for scientific and technological applications.

![QH9](figs/QH9.png)

## Tasks

To comprehensively evaluate the quantum Hamiltonian prediction performance, we define the following tasks based on the obtained stable and dynamic geometries in the QH9 dataset. Please refer to our paper for details of these task setups.

* **QH9-stable-id** 
* **QH9-stable-ood** 
* **QH9-dynamic-geo** 
* **QH9-dynamic-mol** 

| Task | # Total geometries | # Total molecules | # Training/validation/testing geometries|
| -------- | -------- |-------------------| -------- |
|**QH9-stable-id** | 130, 831 | 130, 831          | 104, 664/13, 083/13, 084|
|**QH9-stable-ood** | 130, 831 | 130, 831          | 104, 001/17, 495/9, 335|
|**QH9-dynamic-geo** | 99, 900 | 999               | 79, 920/9, 990/9, 990|
|**QH9-dynamic-mol** | 99, 900 | 999          | 79, 900/9, 900/10, 100|

**Note that we have updated the dynamic datasets which is shown in new arxiv version. 
As a future plan, we plan to update the dynamic dataset with more MD data, 
we will keep updating and release new dynamic data as different versions.**

## Requirement

We include key dependencies below. The versions we used are in parentheses. 
* PyTorch (1.11.0)
* PyG (2.0.4)
* e3nn (0.5.1)
* pyscf (2.2.1) (Stable)
* pyscf (2.3.0) (Dynamic)
* hydra-core (1.1.2)

Meanwhile, we provide the installation file, and you can build the environment by `sh install.sh`.


## Dataset Usage
We provide the datasets as commonly used PyG datasets. Here are simple examples to load our datasets with a few lines of code. Prior to that, you can download the `datasets` folder, which includes the raw data files `QH9Stable.db` and `QH9Dynamic.db`, via [this Google Drive link](https://drive.google.com/drive/folders/13pPgBh3XvN2FCpowfnA8TT4VJ0OTceNM?usp=sharing) or [OneDrive Link](https://tamucs-my.sharepoint.com/:f:/g/personal/haiyang_tamu_edu/Ev4XIVcumhVFtaI8lUkIHXABHkKnKgWSJ5LYZOo67UKO0g?e=tsXkT1). Meanwhile, we provide the zip files of the datasets in this [google drive link](https://drive.google.com/drive/u/0/folders/1LXTC8uaOQzmb76FsuGfwSocAbK5Hshfj).

```python
from torch_geometric.loader import DataLoader
from datasets import QH9Stable, QH9Dynamic

### Use one of the following lines to Load the specific dataset
dataset = QH9Stable(split='random')  # QH9-stable-id
dataset = QH9Stable(split='size_ood')  # QH9-stable-ood
dataset = QH9Dynamic(split='geometry')  # QH9-dynamic-geo
dataset = QH9Dynamic(split='mol')  # QH9-dynamic-mol

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
Equivariant quantum tensor network [QHNet](https://arxiv.org/abs/2306.04922) is selected as the main baseline method in the QH9 benchmark currently. QHNet has an extendable expansion module that is built upon intermediate full orbital matrices, enabling its capability to effectively handle different molecules. This flexibility allows QHNet to accommodate various molecules in the QH9 benchmark.

* Train the QHNet model

```shell script
### Modify the configurations in config/config.yaml (or pass the configurations as args) as needed, and then run
python main.py datasets=QH9-stable datasets.split=random # QH9-stable-id
python main.py datasets=QH9-stable datasets.split=size_ood # QH9-stable-ood
python main.py datasets=QH9-dynamic datasets.split=geometry # QH9-dynamic-geo
python main.py datasets=QH9-dynamic datasets.split=mol # QH9-dynamic-mol
```

**Trained models**: our trained QHNet models on the defined tasks are available via [this Google Drive link](https://drive.google.com/drive/folders/10ebqIWLrZ672A9bFg9wLe48F-nsz7za3?usp=share_link).

* Evaluate the trained model (in terms of MAE on Hamiltonian matrix, MAE on occupied orbital energies, and  cosine similarity of orbital coefficients)
```shell script
### Modify the configurations in config/config.yaml (or pass the configurations as args) as needed (including the trained_model arg), and then run
python test.py
```

* Evaluate the performance of accelerating DFT calculation
```shell script
### Modify the configurations in config/config.yaml (or pass the configurations as args) as needed (including the trained_model arg), and then run


# Pyscf version 2.2.1 for QH9-Stable; Pyscf version 2.3.0 for QH9-Dynamic
python test_dft_acceleration.py
```

## Customization
Below we provide a brief description on how to customize this benchmark to run model on your own dataset.

#### How to prepare your own dataset
Suppose that you are prepared to generate your own datasets, our current dataset classes, such as `QH9Stable`and `QH9Dynamic`, support to fetch data from `apsw` database.
Therefore, `apsw` database is recommended to save the data.

MUST HAVE:
* `pos`: The coordinates of the atomic 3D positions.
* `atoms`: The atomic number.
* `Ham`: The Hamiltonian matrix for molecular geometries.

For the Hamiltonian matrix, pay attention to the atomic orbital order, and magnetic order $m$.
For current quantum tensor networks in the QHBench such as QHNet, arrangement of atomic orbitals adheres to the sequence of $s$, $p$, $d$, and so forth.
For the magnetic order $m$, it follows the order from low to high. 
For example, when $\ell = 1$, the magnetic order $m$ should be in the order of $-1, 0, 1$.
When $\ell = 2$, the magnetic order $m$ should be in order of $-2, -1, 0, 1, 2$.
To make the Hamiltonian matrix arranged in this order, the convertion should be applied when processing. 
Please add corresponding order information in [the convention dict](https://github.com/divelab/AIRS/blob/46802e963505caef90e57f213314db9800004e01/OpenDFT/QHBench/QH9/datasets.py#L21).
Currently, we provide the convention dict for pyscf_631G, and pyscf_def2svp. Note that the arrangement of $m$ for $\ell=1$ in pyscf is $0, 1, -1$, and convertion is needed.

#### How to add our own Model
Add the model file in the corresponding directory `AIRS/OpenDFT/QHBench/QH9/models/`, and the add the corresponding configuration information in `AIRS/OpenDFT/QHBench/QH9/config/`. 

## Citation
```
@article{yu2023qh9,
      title={{QH9}: A Quantum Hamiltonian Prediction Benchmark for QM9 Molecules}, 
      author={Haiyang Yu and Meng Liu and Youzhi Luo and Alex Strasser and Xiaofeng Qian and Xiaoning Qian and Shuiwang Ji},
      journal={arXiv Preprint, arXiv:2306.09549},
      year={2023}
}
```

## Acknowledgments
This work was supported in part by National Science Foundation grant IIS-2006861, CCF-1553281, DMR-2119103, DMR-1753054, DMR-2103842, and IIS-2212419. Acknowledgment is also made to the donors of the American Chemical Society Petroleum Research Fund for partial support of this research.
