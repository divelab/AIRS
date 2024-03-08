# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os.path as osp

from setuptools import find_packages, setup

# with open("README.md", encoding="utf-8") as fh:
#     long_description = fh.read()
long_description = ""

exec(open(osp.join(osp.dirname(__file__), "pdearena", "version.py")).read())

extras = {
    "datagen": [
        "phiflow==2.1",
        "fdtd==0.2.5",
        "joblib",
        "juliapkg",
        "tqdm",
    ],
}

base_requires = [
    # "pytorch-lightning>=1.7",
    "numpy",
    "xarray",
    "h5py",
    "click",
    # "torch>=1.12",
    "torchdata==0.4.1",
    "matplotlib",
    "jsonargparse",
    "omegaconf",
    "tensorboard",
    "pytest",
    "pytest-mock",
    "pytest-explicit",
]

setup(
    name="pdearena",
    description="PyTorch library for PDE Surrogate Benchmarking",
    install_requires=base_requires,
    extras_require=extras,
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    version=__version__,
    author="PDEArena authors, modified by SineNet authors",
    python_requires=">=3.6",
    zip_safe=True,
    project_urls={
        "Documentation": "https://microsoft.github.io/pdearena",
        "Source code": "https://github.com/microsoft/pdearena",
        "Bug Tracker": "https://github.com/microsoft/pdearena/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
