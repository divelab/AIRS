conda create -n GeoMPNN python=3.10 -y
conda activate GeoMPNN
pip install lips-benchmark[recommended] airfrans dill torch_geometric gpustat
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
pip install tensorboard --upgrade
pip install -e .
