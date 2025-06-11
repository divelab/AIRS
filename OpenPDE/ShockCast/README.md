# ShockCast

<!-- ## Introduction -->

This is the official implementation of [A Two-Phase Deep Learning Framework for Adaptive Time-Stepping in High-Speed Flow Modeling](https://arxiv.org/abs/2506.07969).

## Coal Dust Explosion

https://github.com/user-attachments/assets/fceb8e3b-1407-4e2c-b3d8-074c9af5e02c

## Circular Blast

https://github.com/user-attachments/assets/18b675be-444a-4162-b550-8f19fd062a6a

## Environment Setup

```
source setup.sh
```

## Training and Evaluation

```
SETTING=coal
SETTING=blast
NEURAL_SOLVER=ffno_cond_euler_residual
NEURAL_CFL=dtPred_convNeXT-tiny-max-include_deriv_cfl2e-4_noise1e-2

# neural solver training
python scripts/train.py --config ${SETTING}_${NEURAL_SOLVER}

# neural CFL training
python scripts/train.py --config ${SETTING}_${NEURAL_CFL}

# ShockCast eval; add neural CFL ckpt path to ${SETTING}_eval config
python scripts/train.py --config ${SETTING}_eval --ckpt PATH_TO_NEURAL_SOLVER_CKPT --mode test
```

## Data

Supersonic flow datasets can be found on [HuggingFace](https://huggingface.co/datasets/divelab/ShockCast).


## Citation

```bibtex
@article{helwig2025shockcast,
  title={A Two-Phase Deep Learning Framework for Adaptive Time-Stepping in High-Speed Flow Modeling},
  author={Jacob Helwig and Sai Sreeharsha Adavi and Xuan Zhang and Yuchao Lin and Felix S. Chim and Luke Takeshi Vizzini and Haiyang Yu and Muhammad Hasnain and Saykat Kumar Biswas and John J. Holloway and Narendra Singh and N. K. Anand and Swagnik Guhathakurta and Shuiwang Ji},
  journal={arXiv preprint arXiv:2506.07969},
  year={2025},
  url={https://www.arxiv.org/abs/2506.07969}
}
```

## Acknowledgments

This work was supported in part by the National Science Foundation under grant IIS-2243850. The authors express their gratitude to Professor Ryan Houim of the University of Florida for providing access to HyBurn, the computational fluid dynamics (CFD) code utilized for the simulations presented in this study. The CFD calculations presented in this work were partly performed on the Texas A\&M high-performance computing cluster Grace.