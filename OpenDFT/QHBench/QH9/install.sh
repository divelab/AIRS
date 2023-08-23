conda create -y -n QHBench python=3.8
source activate QHBench
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install scipy
conda install pyg -c pyg
pip install tqdm hydra-core pyscf rdkit transformers torch_ema e3nn
