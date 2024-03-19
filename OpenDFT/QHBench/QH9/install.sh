conda create -y -n QHBench python=3.8
conda activate QHBench
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install scipy
conda install pyg==2.1.0 -c pyg
pip install tqdm hydra-core>=1.2.0 pyscf rdkit transformers torch_ema e3nn lmdb apsw gdown
