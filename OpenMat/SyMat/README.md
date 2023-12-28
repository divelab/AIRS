# Towards Symmetry-Aware Generation of Periodic Materials

This is the official implementation of the **SyMat** method proposed in the following paper.

Youzhi Luo, Chengkai Liu, Shuiwang Ji. "[Towards Symmetry-Aware Generation of Periodic Materials](https://openreview.net/forum?id=Jkc74vn1aZ)".

<p align="center">
<img src="https://github.com/divelab/AIRS/blob/main/OpenMat/SyMat/assets/symat.png" width="800" class="center" alt=""/>
    <br/>
</p>

## Requirements
We include key packages below. The anaconda environment we used is available in environment.yml.
* PyTorch >= 1.7.1
* PyTorch Geometric >= 1.6.3
* pymatgen == 2020.11.11
* matminer == 0.7.4
* smact == 2.3.1

## Run

Model training:
```linux
python train.py --result_path result/ --dataset perov_5
```
Material generation:
```linux
python generate.py --model_path /path/to/your/model --dataset perov_5 --num_gen 10000
```

## Citation
```latex
@inproceedings{
  luo2023towards,
  title={Towards Symmetry-Aware Generation of Periodic Materials},
  author={Youzhi Luo and Chengkai Liu and Shuiwang Ji},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=Jkc74vn1aZ}
}
```

## Acknowledgments
This work was supported in part by National Science Foundation under grants IIS-1908220 and IIS-2006861.
