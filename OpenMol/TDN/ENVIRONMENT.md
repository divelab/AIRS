## Install Training & Evaluation Environment for TDN

### Install Miniforge

We recommend Miniforge. Compared to `conda`, Miniforge is user-friendly for any package in the `conda-forge` channel, and the `mamba` installation is speedy. First, access the [Miniforge GitHub repository](https://github.com/conda-forge/miniforge) to download the suitable Miniforge installation package for your system, e.g., [Miniforge3-Linux-x86_64](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh) for Linux x86.

Run the installer with:

```sh
bash Miniforge-x86_64.sh
```

Specify the directory `PATH_OF_YOUR_MINIFORGE` for installation when prompted.

> Note: This will overwrite your original `conda` environment. If you have an existing `conda` environment, reverse the initialization with:

```sh
conda init --reverse
```

Then, set up the new environment using:

```sh
PATH_OF_YOUR_MINIFORGE/bin/conda init
```

Now that Miniforge is installed, you can use `conda install` as usual, but also take advantage of:

```sh
mamba install
```

which is much faster than `conda install`.

### Create Conda Environment

It is recommended to install PyTorch and torch-geometric (CUDA-related packages, which may have compatibility issues) separately. First, create an environment named `ffp` with Python 3.9:

```sh
conda create -n ffp python=3.9
```

Activate the environment:

```sh
conda activate ffp
```

Install PyTorch with your preferred CUDA version [here](https://pytorch.org/get-started/previous-versions/), e.g., cu118:

```sh
mamba install pytorch=2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

To install torch-geometric, search [PyTorch Geometric WHL](https://data.pyg.org/whl/) for the corresponding installation for your CUDA version. For example, find the installation for torch 2.3.0 and cu118 [here](https://data.pyg.org/whl/torch-2.3.0%2Bcu118.html).

Install `torch-scatter`, `torch-sparse`, and `torch-cluster` with the website we found above:

```sh
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.3.0%2Bcu118.html
```

Then, install torch-geometric:

```sh
pip install torch-geometric
```

Finally, install other required packages:

```sh
pip install -r requirements.txt
```

