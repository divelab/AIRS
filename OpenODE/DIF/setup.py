import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    'setuptools>=59.5.0',
    'cilog>=1.2.3',
    'gdown>=4.4.0',
    'matplotlib>=3.5.2',
    'munch>=2.5.0',
    'pytest==7.1.2',
    'pytest-cov~=3.0',
    'pytest-xdist~=2.5',
    'ruamel.yaml==0.17.21',
    'sphinx>=4.5',
    'protobuf==3.20.1',
    'sphinx-rtd-theme==1.0.0',
    'tensorboard==2.8.0',
    'tqdm>=4.64.0',
    'typed-argument-parser>=1.7.2',
    'pynvml>=11.4.1',
    'psutil>=5.9.1',
    'jsonargparse[all]',
    'torchdiffeq',
    'diffrax',
    'equinox',
    'optax',
    'docopt',
    'pysindy',
    'sympy2jax',
    'scikit-learn',
    'wandb',
    'python-dotenv'
]

setuptools.setup(
    name="causal-equation-learning",
    version="0.0.1",
    author="Shurui Gui",
    author_email="shurui.gui@tamu.edu",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='GPLv3',
    url="https://github.com/divelab/AIRS",
    project_urls={
        "Bug Tracker": "https://github.com/divelab/AIRS/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    package_dir={"CEL": "CEL"},
    install_requires=install_requires,
    entry_points = {},
    python_requires=">=3.8",
)