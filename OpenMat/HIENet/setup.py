from setuptools import find_packages, setup

setup(
    name='hienet',
    version='1.0.0',
    description='HIENet',
    author='Keqiang Yan, Montgomery Bohde, Andrii Kryvenko, Ziyu Xiang',
    python_requires='>=3.8',
    packages=find_packages(include=['hienet', 'hienet*']),
    install_requires=[
        'ase',
        'pymatviz',
        'torch-geometric',
        'braceexpand',
        'pyyaml',
        'torch-scatter',
        'e3nn',
        'scikit-learn',
        'pymatgen',
        'wandb',
        'torch-ema'
    ],
)
