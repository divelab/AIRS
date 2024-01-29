conda create --name PDE python=3.8
conda activate PDE
conda install -c conda-forge tmux -y
conda install -c conda-forge gpustat -y
conda install -c conda-forge tqdm -y
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge tensorboard -y
conda install -c conda-forge einops -y
conda install -c anaconda scipy -y
conda install -c conda-forge matplotlib -y
conda install -c anaconda h5py -y
