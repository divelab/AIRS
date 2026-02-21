# Augmenting Molecular Graphs with Geometries via Machine Learning Interatomic Potentials

This is the official repository for the following paper.

Cong Fu*, Yuchao Lin*, Zachary Krueger, Haiyang Yu, Maho Nakata, Jianwen Xie, Emine Kucukbenli, Xiaofeng Qian, Shuiwang Ji ["Augmenting Molecular Graphs with Geometries via Machine Learning Interatomic Potentials"](https://openreview.net/forum?id=JwxhHTISJL), Transactions on Machine Learning Research (TMLR)

<p align="center">
<img src="https://github.com/divelab/AIRS/blob/main/OpenMol/Force2Geo/assets/Pre-Trained_model.png" width="800" class="center" alt=""/>
    <br/>
</p>

<p align="center">
<img src="https://github.com/divelab/AIRS/blob/main/OpenMol/Force2Geo/assets/Geometric_finetune.png" width="800" class="center" alt=""/>
    <br/>
</p>

## Requirements

We include key dependencies below. The versions we used are in the parentheses.
* PyTorch (2.2.0)
* PyTorch Geometric (2.4.0)
* PyTorch Lightning (2.1.4)
* RDKit (2024.9.5)

## Curated Large-Scale Small Molecule Relaxation Dataset (PubChemQCR)

We curate 3.5 million relaxation trajectories of organic small molecules, with high-fidelity energy and force labels for each snapshot. Raw trajectory data is from the PubChemQC project.

Dataset details can be found on Hugging Face: [link](https://huggingface.co/datasets/divelab/PubChemQCR)

## MLIP Pre-Trained Model

In the [README](https://huggingface.co/datasets/divelab/PubChemQCR) file of the pre-training dataset, we provide code to easily load the data and train the model. Users can either train on our provided small subset for model benchmarking or pre-train model on the full dataset.

We also provide the checkpoint of pre-trained PaiNN model (used in this paper), which can be downloaded via [this link](https://drive.google.com/file/d/1-Dq9_MJsCbnBVhWiEOo5wtBwTJDrfl-s/view?usp=sharing).

## Geometric Fine-Tuning

<!-- data need to be uploaded:
processed_relax_subset_new -->

MLIP-relaxed geometries can be used as inputs to downstream 3DGNNs for molecular property prediction. In this setting, ground truth geometries are available during training but not at test time.

For geometric fine-tuning, we aim to fine-tune the downstream 3DGNN on the MLIP-relaxed geometries. The 3DGNN is initially pre-trained using ground truth geometries. After geometric fine-tuning, the model can make predictions directly using MLIP-relaxed geometries during inference on the test set.

### Data processing

We use a subset of Molecule3D dataset that contains 600000 training molecules with ground-state geometries and properties. Since Molecule3D and PubChemQCR are both curated from PubChemQC, so we can use CID to align the molecules in both datasets and assign each molecule the geometry after the PM3 and Hartree-Fock optimization. Then we use the pre-trained MLIP model (PaiNN architecture) to relax the molecules in Molecule3D starting from the first snapshot after PM3 and Hartree-Fock optimization. 

Processed data can be downloaded via [this link](https://drive.google.com/drive/folders/1BOGwCDRAshB7pH4Y7ivH6J6V8U2-QQF-?usp=sharing)

After downloading the processed data, create a new folder `data/processed_relax_subset`, and move downloaded data to this folder.

### Pre-Train Downstream 3DGNN

To pre-train the downstream PaiNN model, run:
```
cd Molecule3D
python train.py --config-name subset_painn.yaml
```

### Fine-Tune Downstream 3DGNN on MLIP-Relaxed Geometries

Run:
```
cd Molecule3D
python train.py --config-name subset_painn_relaxed3D_finetune_GP.yaml
```

## Fine-Tune MLIP for Property Prediction

### Molecule3D Dataset

To directly fine-tune pre-trained MLIP model for property prediction on the subset of Molecule3D, run:

```
cd Molecule3D
python train.py --config-name subset_property_finetune.yaml
```

<!-- files need to be uploaded:
/mnt/data/shared/congfu/FFP-finetune-clean/Molecule3D/train.py
/mnt/data/shared/congfu/FFP-finetune-clean/Molecule3D/utils
/mnt/data/shared/congfu/FFP-finetune-clean/Molecule3D/callbacks
/mnt/data/shared/congfu/FFP-finetune-clean/Molecule3D/model/painn
/mnt/data/shared/congfu/FFP-finetune-clean/Molecule3D/dataset/pyg_datasets.py
/mnt/data/shared/congfu/FFP-finetune-clean/Molecule3D/config/datamodule
/mnt/data/shared/congfu/FFP-finetune-clean/Molecule3D/config/foundation_model
/mnt/data/shared/congfu/FFP-finetune-clean/Molecule3D/config/loggers
/mnt/data/shared/congfu/FFP-finetune-clean/Molecule3D/config/model/painn.yaml
/mnt/data/shared/congfu/FFP-finetune-clean/Molecule3D/config/model/painn_finetune.yaml
/mnt/data/shared/congfu/FFP-finetune-clean/Molecule3D/config/model/painn_auxiliary.yaml

/mnt/data/shared/congfu/FFP-finetune-clean/Molecule3D/config/subset_painn.yaml
/mnt/data/shared/congfu/FFP-finetune-clean/Molecule3D/config/subset_property_finetune.yaml
/mnt/data/shared/congfu/FFP-finetune-clean/Molecule3D/config/subset_painn_relaxed3D_finetune_GP.yaml -->

<!-- don't need this:
/mnt/data/shared/congfu/FFP-finetune-clean/Molecule3D/dataset/transform.py -->


### $\nabla^2$ DFT Dataset

First, git clone the $\nabla^2$ DFT repository:
```
git clone git@github.com:AIRI-Institute/nablaDFT.git
```

Copy all files from the `nablaDFT` folder in this repository into the folder you just cloned. Ensure that the directory structure remains unchanged. If any files already exist in the target folder, replace them with the new ones.

Next, we need to download data.
```
cd nablaDFT
mkdir datasets
cd datasets
mkdir train
mkdir test
```
From the original $\nabla^2$ DFT repository, download the `summary.csv` files, `dataset_train_large` training data, and `dataset_test_structures` test data. Place `summary.csv` into the `datasets` folder, and move `dataset_train_large` and `dataset_test_structures` into the `train` and `test` folder, respectively.

<!-- files need to be uploaded:
/mnt/data/shared/congfu/FFP-finetune-clean/nablaDFT/config/painn_finetune.yaml
/mnt/data/shared/congfu/FFP-finetune-clean/nablaDFT/config/datamodule/nablaDFT_pyg_prop.yaml
/mnt/data/shared/congfu/FFP-finetune-clean/nablaDFT/config/datamodule/nablaDFT_pyg_prop_test.yaml
/mnt/data/shared/congfu/FFP-finetune-clean/nablaDFT/config/model/painn.yaml
/mnt/data/shared/congfu/FFP-finetune-clean/nablaDFT/nablaDFT/painn/
/mnt/data/shared/congfu/FFP-finetune-clean/nablaDFT/nablaDFT/pipelines.py
/mnt/data/shared/congfu/FFP-finetune-clean/nablaDFT/nablaDFT/dataset/pyg_property_datasets.py -->

To fine-tune the pre-trained MLIP foundation model on HOMO-LUMO gap prediction, run:
```
python run.py --config-name painn_finetune.yaml
```

To evaluate the test result, change `dataset_name` attribute in painn_finetune.yaml to `dataset_test_structures`, then run the same command as above.


## Citation
```latex
@article{fu2026augmenting,
  title={Augmenting Molecular Graphs with Geometries via Machine Learning Interatomic Potentials},
  author={Fu, Cong and Lin, Yuchao and Krueger, Zachary and Yu, Haiyang and Nakata, Maho and Xie, Jianwen and Kucukbenli, Emine and Qian, Xiaofeng and Ji, Shuiwang},
  journal={Transactions on Machine Learning Research},
  year={2026}
}
```

## Acknowledgments
SJ acknowledges support from SES AI Corporation, ARPA-H under grant 1AY1AX000053, National Institutes of Health under grant U01AG070112, and National Science Foundation under grant IIS-2243850. XFQ acknowledges support from SES AI Corporation, the Center for Reconfigurable Electronic Materials Inspired by Nonlinear Dynamics (reMIND), an Energy Frontier Research Center funded by the U.S. Department of Energy, Basic Energy Sciences, under Award Number DE-SC0023353. We acknowledge the support of Lambda, Inc. and NVIDIA for providing the computational resources for this project.