# usage: source setup.sh
conda create -n GFNO python=3.8 -y
conda activate GFNO
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge -y
conda install scipy numpy h5py conda gfortran tensorboard gpustat -c anaconda -c conda-forge -y
pip install e2cnn escnn zarr xarray