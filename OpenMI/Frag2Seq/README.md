# Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language Models

This is the official implementation of the **Frag2Seq** method proposed in the following paper.

Cong Fu*, Xiner Li*, Blake Olson, Heng Ji, Shuiwang Ji ["Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language Models"](https://openreview.net/pdf?id=MBZVrtbi06), The Thirteenth International Conference on Learning Representations (ICLR) 2025

<p align="center">
<img src="https://github.com/divelab/AIRS/blob/main/OpenProt/Frag2Seq/assets/Frag2Seq.png" width="800" class="center" alt=""/>
    <br/>
</p>


## Requirements

We include key dependencies below. The versions we used are in the parentheses. Our detailed environmental setup is available in [environment.yml]().
* PyTorch (2.0.1)
* biopython (1.79)
* rdkit (2023.9.5)


## Preparing Data

We use CrossDocked data to train and test our model. Please download and extract the curated dataset following the instruction of 3DSBDD:\
https://github.com/luost26/3D-Generative-SBDD/blob/main/data/README.md

Then process the raw data:
```
bash process_crossdock.sh
```


## Run

* ### Tokenization:
Run the following script to convert ligand into fragment-based tokens and extract protein embeddings.
```
cd tokenizaion
bash convert_token_frag.sh
```

* ### Train Frag2Seq:
<!-- Please change the path in the `train.sh` -->
Train Frag2Seq from scratch:
```
bash train.sh
```


* ### Generate molecules and evaluate results:
Generate molecule fragment sequences conditioning on the protein pockets in the test set:
```
bash generate.sh
```
Please specify the root folder to the checkpoint in the `generate.sh` by setting `--model_root_path`, and specific checkpoint is choosed based on `epoch` parameter in the `generate.sh`. By default, we generate 100 molecules for each protein pocket. This can be changed by modifying `--sample_repeats`.


To compute docking score using QuickVina, we first need to convert all protein PDB files to PDBQT files using MGLTools, as described in the DiffSBDD:
https://github.com/arneschneuing/DiffSBDD/tree/30358af24215921a869619e9ddf1e387cafceedd

```
conda activate mgltools
cd analysis
python docking_py27.py ../sample_output/test_pdb/ ../sample_output/test_pdbqt/ crossdocked
cd ..
conda deactivate
```

Then, convert sequences to molecules and run evaluation:
```
bash evaluate.sh
```


## Citation
```latex
@article{fu2024fragment,
  title={Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language Models},
  author={Fu, Cong and Li, Xiner and Olson, Blake and Ji, Heng and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2408.09730},
  year={2024}
}
```

## Acknowledgments
This work was supported partially by National Science Foundation grant IIS-2243850 and National
Institutes of Health grant U01AG070112 to S.J., and by the Molecule Maker Lab Institute to H.J.:
an AI research institute program supported by NSF under award No. 2019897. The views and
conclusions contained herein are those of the authors and should not be interpreted as necessarily
representing the official policies, either expressed or implied, of the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding
any copyright annotation therein.