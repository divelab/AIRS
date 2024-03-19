conda create -n pdearena python=3.8 -y
conda activate pdearena
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
pip install pytorch-lightning==1.7.7 phiflow
pip install -U 'jsonargparse[signatures]'
pip install torchmetrics==0.11.4
pip install -U rich
