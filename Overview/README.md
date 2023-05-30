# AIRS 

## Group Theory, Symmetries, and Equivariance

## Al for Quantum Mechanics

### Spin Systems

- Restricted boltzmann machines
  - [Carleo and Troyer 2017](https://www.science.org/doi/10.1126/science.aag2302)
  - [Gao and Duan 2017](https://www.nature.com/articles/s41467-017-00705-2)
  - [Choo et al. 2018](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.167204)
  - [Chen et al. 202](https://openreview.net/pdf?id=qZUHvvtbzy)
- Feed-forward neural networks
  - [Saito 2017](https://journals.jps.jp/doi/10.7566/JPSJ.86.093001)
  - [Cai and Liu 2018](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.035116)
  - [Saito and Kato 2018](https://journals.jps.jp/doi/10.7566/JPSJ.87.014001)
  - [Saito 2018](https://journals.jps.jp/doi/10.7566/JPSJ.87.074002)
- Convolutional neural networks
  - [Liang et al. 2018](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.104426)
  - [Choo et al. 2019](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.100.125124)
  - [Szabo et al. 2020](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.033075)
  - [Fu et al. 2022](https://arxiv.org/abs/2206.07370)
- Autoregressive and recurrent neural networks
  - [Sharir et al. 2020](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.124.020503)
  - [Hibat-Allah et al. 2020](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.023358)
  - [Luo et al. 2021a](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.013216)
- Graph neural networks
  - [Yang et al. 2020](https://arxiv.org/abs/2011.12453)
  - [Kochkov et al. 2021a](https://arxiv.org/abs/2110.06390)

### Many-electron Systems
- Single Geometry
  - [PauliNet](https://www.nature.com/articles/s41557-020-0544-y)
  - [FermiNet](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.033429)
  - [Gerard et al. 2022](https://openreview.net/pdf?id=nX-gReQ0OT)
  - [PsiFormer](https://openreview.net/pdf?id=xveTeHVlF7j)
  - [FermiNet+DMC](https://www.nature.com/articles/s41467-023-37609-3)
  - [DiffVMC](https://arxiv.org/abs/2305.16540))
- Multiple Geometries
  - [PESNet](https://openreview.net/pdf?id=apv504XsysP)
  - [DeepErwin](https://arxiv.org/abs/2105.08351)
  - [PlaNet](https://openreview.net/pdf?id=Tuk3Pqaizx)
  - [Globe](https://arxiv.org/abs/2302.04168)
  - [Scherbela et al. 2023](https://arxiv.org/abs/2303.09949)
## Al for Density Functional Theory
### Quantum Tensor Prediction
- Invariant Networks
  - SchNorb
  - DeepH
- Equivariant Networks
  - PhiSNet
  - QHNet
  - QHNet
## AI for Small Molecules
### Molecular Representation Learning
- Invariant Methods: $\ell=0$
  - [SchNet](https://pubs.aip.org/aip/jcp/article/148/24/241722/962591/SchNet-A-deep-learning-architecture-for-molecules) 
  - [DimeNet](https://openreview.net/forum?id=B1eWbxStPH)
  - [SphereNet](https://openreview.net/forum?id=givsRXsOt9r)
  - [GemNet](https://openreview.net/forum?id=HS_sOaxS9K-)
  - [ComENet](https://openreview.net/forum?id=mCzMqeWSFJ)
- Equivariant Methods: $\ell=1$
  - [EGNN](https://proceedings.mlr.press/v139/satorras21a.html)
  - [GVP-GNN](https://openreview.net/forum?id=1YLJDvSx6J4)
  - [PaiNN](http://proceedings.mlr.press/v139/schutt21a.html)
  - [Vector Neurons](https://openaccess.thecvf.com/content/ICCV2021/html/Deng_Vector_Neurons_A_General_Framework_for_SO3-Equivariant_Networks_ICCV_2021_paper.html)
- Equivariant Methods: $\ell>1$
  - [TFN](https://arxiv.org/abs/1802.08219)
  - [Cormorant](https://openreview.net/forum?id=SkenhHHeUH)
  - [SE(3)-Transformer](https://proceedings.neurips.cc//paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html)
  - [NequIP](https://www.nature.com/articles/s41467-022-29939-5)
  - [SEGNN](https://openreview.net/forum?id=_xwr8gOBeV1)
  - [Equiformer](https://openreview.net/forum?id=KwmPfARgOTD)
  - [MACE](https://openreview.net/forum?id=YPpSngE-ZU)
### Molecular Conformer Generation
- Learn the Distribution of Low-Energy Geometries
  - [CVGAE](https://www.nature.com/articles/s41598-019-56773-5)
  - [ConfVAE](https://proceedings.mlr.press/v139/xu21f.html)
  - [GeoDiff](https://openreview.net/forum?id=PzcvxEMzvQC)
  - [Torsional Diffusion](https://openreview.net/forum?id=w6fj2r62r_H)
- Predict the Equilibrium Ground-State Geometry
  - [EMPNN](https://arxiv.org/abs/2305.13315)
  - [DeeperGCN-DAGNN+Dist](https://arxiv.org/abs/2110.01717)
### Molecule Generation from Scratch
- Generate Coordinate Matrices
  - [E-NFs](https://openreview.net/forum?id=N5hQI_RowVA)
  - [EDM](https://proceedings.mlr.press/v162/hoogeboom22a.html)
- Generate SE(3)-Invariant Features
  - [EDMNet](https://arxiv.org/abs/1910.03131)
  - [G-SchNet](https://proceedings.neurips.cc/paper_files/paper/2019/hash/a4d8e2a7e0d0c102339f97716d2fdfb6-Abstract.html)
  - [G-SphereNet](https://openreview.net/forum?id=C03Ajc-NS5W)
### Learning to Simulate Molecular Dynamics
### Representation Learning of Stereoisomerism and Conformational Flexibility
## Al for Protein Science
## Al for Material Science
## Al for Molecular Interactions
### Protein-Ligand Binding Prediction
- Predict Coordinates
  - EquiBind
  - E3Bind
- Predict Interatomic Distances
  - TankBind
- Predict Rotation, Translation, and Torsions
  - DiffDock
### Structure-Based Drug Design
- Autoregressively Generate Relative Position-Related Variables
  - AR
  - GraphBP
  - Pocket2Mol
  - FLAG 
- Generate Coordinates with Diffusion Models
  - TargetDiff
  - DiffBP
  - DiffSBDD
### Energy, Force, and Position Prediction for Molecule-Material Pairs
- Invariant Methods
  - SpinConv
  - GemNet-OC
- Equivariant Methods
  - Equiformer
- Approximately Equivariant Methods
  - SCN
  - eSCN
## Al for Partial Differential Equations
## Other Technical Challenges
### Interpretability 

### Out-of-Distribution Generalization
- OOD in AI for Quantum Mechanics
  - [Guan et al. 2021](https://pubs.rsc.org/en/content/articlehtml/2021/sc/d0sc04823b)
  - [Botu and Ramprasad 2015](https://onlinelibrary.wiley.com/doi/abs/10.1002/qua.24836?casa_token=4dRmUPDz_CQAAAAA:jTYJPlzF6e7DNmT73_9ohyUJ6Tu8cff3Y65XDl6kJc7kMFAhxjPg2fpNHZ4mir2T7CW-o011IyQk)
- OOD in AI for Density Functional Theory
  - [Li et al. 2016](https://arxiv.org/abs/1609.03705)
  - [Pereira et al. 2017](https://pubs.acs.org/doi/10.1021/acs.jcim.6b00340)
- OOD in AI for Molecular Science
  - [Yang et al. 2022](https://openreview.net/forum?id=2nWUNTnFijm)
  - [Lee et al. 2022](https://arxiv.org/abs/2206.07632)
- OOD in AI for Protein Science
  - [Nijkamp et al. 2022](https://arxiv.org/abs/2206.13517)
  - [Gruver et al. 2021](https://icml-compbio.github.io/2021/papers/WCBICML2021_paper_61.pdf)
  - [Hamid and Friedberg 2019](https://www.biorxiv.org/content/10.1101/543272v1)
- OOD in AI for Material Science
  - [Murdock et al. 2020](https://link.springer.com/article/10.1007/s40192-020-00179-z)
  - [Kailkhura et al. 2019](https://www.nature.com/articles/s41524-019-0248-2)
  - [Sutton et al. 2020](https://www.nature.com/articles/s41467-020-17112-9)
- OOD in AI for Chemical Interactions
  - [Ji et al. 2022](https://arxiv.org/abs/2201.09637)
  - [Li et al. 2022](https://arxiv.org/abs/2209.07921)
  - [Han et al. 2021](https://arxiv.org/abs/2111.12951)
  - [Cai et al. 2022](https://www.biorxiv.org/content/10.1101/2022.11.15.516682v2)
  - [Cai et al. 2022](https://academic.oup.com/bioinformatics/article/38/9/2561/6547052)
- OOD in AI for Partial Differential Equations
  - [Boussif et al. 2022](https://arxiv.org/abs/2210.05495)
  - [Brandstetter et al. 2022](https://arxiv.org/abs/2202.03376)

### Foundation Models and Self-Supervised Learning
- Self-Supervised Learning
  - Molecule Representation
  - PDE Solvers
- Foundation Models
  - Large Language Models for Science
  - Protein Discovery
  - Molecule Analysis

### Uncertainty Quantification


## Other Examples

- links
- **strong** ~~del~~ *italic* ==highlight==
- multiline
  text
- `inline code`
-
    ```js
    console.log('code block');
    ```
- Katex
  - $x = {-b \pm \sqrt{b^2-4ac} \over 2a}$
  - [More Katex Examples](#?d=gist:af76a4c245b302206b16aec503dbe07b:katex.md)
- Now we can wrap very very very very long text based on `maxWidth` option
