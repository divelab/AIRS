# usage: source setup.sh
conda create -n GFNO python=3.8 -y
conda activate GFNO
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install scipy numpy h5py conda gfortran tensorboard gpustat -c anaconda -c conda-forge -y
pip install zarr xarray
