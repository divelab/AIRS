# QH9: A Quantum Hamiltonian Prediction Benchmark

[[Paper]](https://github.com/divelab/AIRS/blob/main/OpenDFT/QHBench/QH9/QH9__A_Quantum_Hamiltonian_Prediction_Benchmark_for_QM9_Molecules.pdf) (**NeurIPS**, Track on Datasets and Benchmarks, 2023)

## Introduction

QH9 provides precise DFT-calculated Hamiltonian matrices for **999 or 2,998 molecular dynamics trajectories** and **130,831  stable molecular geometries**, based on the [QM9](http://quantum-machine.org/datasets/) dataset.

In this repo, we provide both the QH9 dataset and the benchmark code, which can be highly valuable for developing machine learning methods and accelerating molecular and materials design for scientific and technological applications.

![QH9](figs/QH9.png)
## News
* We have released QH9-dyn-300k with 2,998 molecular trajectories. The time step is 50 a.u. (~1.2fs) and each trajectory has 100 different geometries.
* The code has implemented the automatically downloading for datasets and checkpoints. For load the pretrained model parameters, please refer to  function [load_pretrained_model_parameters](https://github.com/divelab/AIRS/blob/635ef085b68f2cb6fccc15c5a590534196176f16/OpenDFT/QHBench/QH9/load_pretrained_models.py#L44).
* The [dataset generation example code](https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench/QH9/dataset_generation_examples) is provided for both stable and dynamic dataset based on PySCF.


## Tasks

To comprehensively evaluate the quantum Hamiltonian prediction performance, we define the following tasks based on the obtained stable and dynamic geometries in the QH9 dataset. Please refer to our paper for details of these task setups.

* **QH9-stable-id** 
* **QH9-stable-ood** 
* **QH9-dynamic-300k-geo** 
* **QH9-dynamic-300k-mol** 
* **QH9-dynamic-100k-geo** 
* **QH9-dynamic-100k-mol** 

| Task | # Total geometries | # Total molecules | # Training/validation/testing geometries|
| -------- | -------- |-------------------| -------- |
|**QH9-stable-id** | 130, 831 | 130, 831          | 104, 664/13, 083/13, 084|
|**QH9-stable-ood** | 130, 831 | 130, 831          | 104, 001/17, 495/9, 335|
|**QH9-dynamic-100k-geo** | 99, 900 | 999               | 79, 920/9, 990/9, 990|
|**QH9-dynamic-100k-mol** | 99, 900 | 999          | 79, 900/9, 900/10, 100|
|**QH9-dynamic-300k-geo** | 299, 800 | 2,998               | 239,840 / 29,980 / 29,980 |
|**QH9-dynamic-300k-mol** | 299, 800 | 2,998          | 239,840 /29, 900/30, 100|

**Note that the cost of training on QH9-dynamic-300k is similar compared to QH9-dynamic-100k, while it contains more data and achieves higher performance in molecule-wise split. Therefore, it is recommended to use QH9-dynamic-300k.**

## Requirement

We include key dependencies below. The versions we used are in parentheses. 
* PyTorch (1.11.0)
* PyG (2.0.4)
* e3nn (0.5.1)
* pyscf (2.2.1) (QH9-Stable, QH9-Dynamic-300k)
* pyscf (2.3.0) (QH9-Dynamic-100k)
* hydra-core (1.1.2)

Meanwhile, we provide the installation file, and you can build the environment by `source install.sh`.


## Dataset Usage
We provide the datasets as commonly used PyG datasets. Here are simple examples to load our datasets with a few lines of code. Prior to that, you can download the `datasets` folder, which includes the raw data files `QH9Stable.db` and `QH9Dynamic.db`, via [this Google Drive link](https://drive.google.com/drive/folders/13pPgBh3XvN2FCpowfnA8TT4VJ0OTceNM?usp=sharing) or [OneDrive Link](https://tamucs-my.sharepoint.com/:f:/g/personal/haiyang_tamu_edu/Ev4XIVcumhVFtaI8lUkIHXABHkKnKgWSJ5LYZOo67UKO0g?e=tsXkT1). Meanwhile, we provide the zip files of the datasets in this [google drive link](https://drive.google.com/drive/u/0/folders/1LXTC8uaOQzmb76FsuGfwSocAbK5Hshfj).

```python
from torch_geometric.loader import DataLoader
from datasets import QH9Stable, QH9Dynamic

### Use one of the following lines to Load the specific dataset
dataset = QH9Stable(split='random')  # QH9-stable-id
dataset = QH9Stable(split='size_ood')  # QH9-stable-ood
dataset = QH9Dynamic(split='geometry', version='300k')  # QH9-dynamic-geo
dataset = QH9Dynamic(split='mol', version='300k')  # QH9-dynamic-mol

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
python main.py datasets=QH9-dynamic datasets.split=geometry datasets.version=300k # QH9-dynamic-300k-geo
python main.py datasets=QH9-dynamic datasets.split=mol datasets.version=300k # QH9-dynamic-300k-mol
```

**Trained models**: our trained QHNet models on the defined tasks are available via [this Google Drive link](https://drive.google.com/drive/folders/10ebqIWLrZ672A9bFg9wLe48F-nsz7za3?usp=share_link).

* Evaluate the trained model (in terms of MAE on Hamiltonian matrix, MAE on occupied orbital energies, and  cosine similarity of orbital coefficients). The eigen decoposition cost lots of time to run it.
```shell script
### Modify the configurations in config/config.yaml (or pass the configurations as args) as needed (including the trained_model arg), and then run
python test.py
```

* Evaluate the performance of accelerating DFT calculation, it needs to run DFT for 50 molecules with high computatioinal cost.
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
