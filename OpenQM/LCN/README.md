# Lattice Convolutional Networks for Learning Ground States of Quantum Many-Body Systems

This is the official implementation of the **LCN** method proposed in the following paper.

Cong Fu*, Xuan Zhang*, Huixin Zhang, Hongyi Ling, Shenglong Xu, Shuiwang Ji. "[Lattice Convolutional Networks for Learning Ground States of Quantum Many-Body Systems]()", SIAM International Conference on Data Mining (SDM) 2024.

<p align="center">
<img src="https://github.com/divelab/AIRS/blob/main/OpenQM/LCN/assets/LCN.png" width="600" class="center" alt=""/>
    <br/>
</p>


## Requirements

We include key dependencies below. The versions we used are in the parentheses. Our detailed environmental setup is available in [quantum_env.yaml]().
* PyTorch (1.9.0)
* PyTorch Geometric (1.7.2)

## Preparing Data
* We use the same four kinds of lattices adopted in [Kochkov et al., 2021](https://arxiv.org/abs/2110.06390), including square, triangular, honeycomb, and kagome.
* We put all the lattice data in the `dataset` folder.


## Run

### Example

* Train LCN on 6 $\times$ 6 triangular lattice and J2 in the Heisenberg model equals to 0:

```
cd scripts
bash train_triangular_N_36_J_0.sh
```

* Train LCN on 36 nodes kagome lattice and J2 in the Heisenberg model equals to -0.02:

```
cd scripts
bash train_kagome_N_36_J_-002.sh
```



## Citation
```latex
@article{fu2022lattice,
  title={Lattice convolutional networks for learning ground states of quantum many-body systems},
  author={Fu, Cong and Zhang, Xuan and Zhang, Huixin and Ling, Hongyi and Xu, Shenglong and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2206.07370},
  year={2022}
}
```

## Acknowledgments
This work was supported in part by National Science Foundation grant IIS-1908198 and National Institutes of Health grant
U01AG070112.
