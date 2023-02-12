# Generating 3D Molecules for Target Protein Binding
This is the official implementation of the **GraphBP** method proposed in the following paper.

Meng Liu, Youzhi Luo, Kanji Uchino, Koji Maruhashi, and Shuiwang Ji. "[Generating 3D Molecules for Target Protein Binding](https://arxiv.org/abs/2204.09410)". [ICML 2022 **Long Presentation**]

![](https://github.com/divelab/GraphBP/blob/main/assets/GraphBP.png)


## Requirements
We include key dependencies below. The versions we used are in the parentheses. Our detailed environmental setup is available in [environment.yml](https://github.com/divelab/GraphBP/blob/main/GraphBP/environment.yml).
* PyTorch (1.9.0)
* PyTorch Geometric (1.7.2)
* rdkit-pypi (2021.9.3)
* biopython (1.79)
* openbabel (3.3.1)


## Preparing Data
* Download and extract the CrossDocked2020 dataset:
```linux
wget https://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.1.tgz -P data/crossdock2020/
tar -C data/crossdock2020/ -xzf data/crossdock2020/CrossDocked2020_v1.1.tgz
wget https://bits.csb.pitt.edu/files/it2_tt_0_lowrmsd_mols_train0_fixed.types -P data/crossdock2020/
wget https://bits.csb.pitt.edu/files/it2_tt_0_lowrmsd_mols_test0_fixed.types -P data/crossdock2020/
```
**Note**: (1) The unzipping process could take a lot of time. Unzipping on SSD is much faster!!! (2) Several samples in the training set cannot be processed by our code. Hence, we recommend replacing the `it2_tt_0_lowrmsd_mols_train0_fixed.types` 
file with a new one, where these samples are deleted. The new one is available [here](https://github.com/divelab/GraphBP/blob/main/GraphBP/data/crossdock2020/it2_tt_0_lowrmsd_mols_train0_fixed.types).

* Split data files:
```linux
python scripts/split_sdf.py data/crossdock2020/it2_tt_0_lowrmsd_mols_train0_fixed.types data/crossdock2020
python scripts/split_sdf.py data/crossdock2020/it2_tt_0_lowrmsd_mols_test0_fixed.types data/crossdock2020
```

## Run
* Train GraphBP from scratch:
```linux
CUDA_VISIBLE_DEVICES=${you_gpu_id} python main.py
```
**Note**: GraphBP can be trained on a `48GB GPU` with `batchsize=16`. Our trained model is available [here](https://github.com/divelab/GraphBP/blob/main/GraphBP/trained_model/model_33.pth).

* Generate atoms in the 3D space with the trained model:
```linux
CUDA_VISIBLE_DEVICES=${you_gpu_id} python main_gen.py
```

* Postprocess and then save the generated molecules:
```linux
CUDA_VISIBLE_DEVICES=${you_gpu_id} python main_eval.py
```



## Reference
```
@inproceedings{liu2022graphbp,
  title={Generating 3D Molecules for Target Protein Binding},
  author={Meng Liu and Youzhi Luo and Kanji Uchino and Koji Maruhashi and Shuiwang Ji},
  booktitle={International Conference on Machine Learning},
  year={2022}
}
```
