conda create -y -n QHBench python=3.8
source activate QHBench
CUDA="cu102"
conda install -y pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
pip install scipy
conda install pyg -c pyg
pip install tqdm hydra-core pyscf rdkit transformers torch_ema e3nn
