# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os.path as osp

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

exec(open(osp.join(osp.dirname(__file__), "pdearena", "version.py")).read())

setup(
    name="pdearena",
    description="PyTorch library for PDE Surrogate Benchmarking",
    install_requires=[],
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    version=__version__,
    author="Jayesh K. Gupta, Johannes Brandstetter, and contributors",
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
