conda create -n GeneExp python=3.9
conda activate GeneExp
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch_geometric==2.5.2
pip install pytorch_lightning==1.8.6 wandb collections mamba-ssm==1.1.2 transformers scikit-learn torchmetrics
pip install h5py pyfaidx pyBigWig einops
