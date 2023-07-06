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
  - [DiffVMC](https://arxiv.org/abs/2305.16540)
- Multiple Geometries
  - [PESNet](https://openreview.net/pdf?id=apv504XsysP)
  - [DeepErwin](https://arxiv.org/abs/2105.08351)
  - [PlaNet](https://openreview.net/pdf?id=Tuk3Pqaizx)
  - [Globe](https://arxiv.org/abs/2302.04168)
  - [Scherbela et al. 2023](https://arxiv.org/abs/2303.09949)
## Al for Density Functional Theory
### Quantum Tensor Prediction
- Invariant Networks
  - [SchNorb](https://www.nature.com/articles/s41467-019-12875-2)
  - [DeepH](https://www.nature.com/articles/s43588-022-00265-6)
- Equivariant Networks
  - [PhiSNet](https://arxiv.org/abs/2106.02347)
  - [QHNet](https://arxiv.org/abs/2306.04922)
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
### Protein Folding
- Two-Stage Learning
  - [RaptorX-Contact](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005324)
  - [AlphaFold1](https://www.nature.com/articles/s41586-019-1923-7)
  - [trRoseTTA](https://www.nature.com/articles/s41596-021-00628-9)
- End-To-End Learning
  - [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2)
  - [RoseTTAFold](https://www.science.org/doi/10.1126/science.abj8754)
  - [ESMFold](https://www.science.org/doi/10.1126/science.ade2574)
  - [OpenFold](https://www.biorxiv.org/content/10.1101/2022.11.20.517210v1)
### Protein Representation Learning
- Invariant Networks
  - [IEConv](https://openreview.net/forum?id=l0mSUROpwY)
  - [HoloProt](https://openreview.net/forum?id=-xEk43f_EO6)
  - [GearNet](https://openreview.net/forum?id=to3qCB3tOh9)
  - [ProNet](https://openreview.net/forum?id=9X-hgLDLYkQ)
  - [PiFold](https://openreview.net/forum?id=oMsN9TYwJ0j)
  - [CDConv](https://openreview.net/forum?id=P5Z-Zl9XJ7)
- Equivariant Networks
  - [GVP-GNN](https://openreview.net/forum?id=1YLJDvSx6J4)
  - [GBPNet](https://dl.acm.org/doi/abs/10.1145/3534678.3539441)
### Protein Backbone Generation
- Structure Representation: Coordinates
  - [ProtDiff](https://arxiv.org/abs/2206.04119)
  - [Chroma](https://www.biorxiv.org/content/10.1101/2022.12.01.518682v1)
  - [LatentDiff](https://arxiv.org/abs/2305.04120)
  - [Genie](https://arxiv.org/abs/2301.12485)
- Structure Representation: Frames
  - [RFdiffusion](https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1)
  - [FrameDiff](https://arxiv.org/abs/2302.02277)
- Structure Representation: Internal Angles
  - [FoldingDiff](https://arxiv.org/abs/2209.15611)
## Al for Material Science
### Material Representation Learning
- Material Representation: Multi-Edge Graphs
  - [CGCNN](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301)
  - [MEGNET](https://arxiv.org/abs/1812.05055)
  - [GATGNN](https://arxiv.org/abs/2003.13379)
  - [SchNet](https://arxiv.org/abs/1706.08566)
  - [ALIGNN](https://www.nature.com/articles/s41524-021-00650-1)
  - [M3GNET](https://www.nature.com/articles/s43588-022-00349-3)
- Material Representation: Multi-Edge Graphs and Fully-Connected Graphs
  - [Matformer](https://openreview.net/forum?id=pqCT3L-BU9T&)
  - PotNet
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
### Multiscale Dynamics
- Sequential Multiscale Processing
  - Dil-ResNet
  - U-Net
- Parallel Multiscale Processing
  - FNO
  - GraphCast
  - FourCastNet
  - 3DEST
### Multiresolution Dynamics
- Geometry Deformation
  - Geo-FNO, 
  - F-FNO
- Learned Adaptive Remeshing
  - MeshGraphNets, 
  - LAMP
### ROllout Stability
- Adversarial Noise Injection
  - GNS, 
  - MPPDE
- Multistep Objective
  - HGNS
- Temporal Bundling
  - MPPDE
### Incorporating Symmetries
- Data Augmentation
  - LPSDA
- Equivariant Architectures
  - Equ-ResNet
  - Equ-Unet
  - RGroup
  - RSteer
  - GCAN
  - GFNO
  - IsoGCN

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
- SSL for Molecule Representation
  - [Survey on 2D Graphs](https://ieeexplore.ieee.org/document/9764632)
  - [GraphMVP](https://openreview.net/forum?id=xQUe1pOKPam)
  - [3D InfoMax](https://proceedings.mlr.press/v162/stark22a.html)
  - [Noisy Node - Pretraining](https://openreview.net/forum?id=tYIMtogyee)
  - [Noisy Node - Auxiliary](https://openreview.net/forum?id=1wVvweK3oIb)
- Large Language Models for Science
  - [Galactica](https://arxiv.org/abs/2211.09085)
  - [BioMedLM](https://crfm.stanford.edu/2022/12/15/biomedlm.html)
  - [med-PALM](https://sites.research.google/med-palm/)
  - [ChemGPT](https://chemrxiv.org/engage/chemrxiv/article-details/627bddd544bdd532395fb4b5)
  - [ChatDrug](https://chao1224.github.io/ChatDrug)
  - [Boiko et al.](https://arxiv.org/abs/2304.05332)
  - [Nori et al.](https://www.microsoft.com/en-us/research/publication/capabilities-of-gpt-4-on-medical-challenge-problems/)
- Foundation Models for Protein Discovery
  - [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2)
  - [RoseTTAFold](https://www.science.org/doi/10.1126/science.abj8754)
  - [RFdiffusion](https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1)
  - [Chroma](https://www.biorxiv.org/content/10.1101/2022.12.01.518682v1)
  - [AlphaFold Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2)
  - [Humphreys et al.](https://www.science.org/doi/10.1126/science.abm4805)
- Foundation Models for Molecule Analysis

### Uncertainty Quantification

## Education, Workforce Developments, and Public Engagements
