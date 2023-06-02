# PotNet
Code repository for paper “Efficient Approximations of Complete Interatomic Potentials for Crystal Property Prediction”. More README coming soon.

## Running Summation Algorithm

To run the summation algorithm, please run below commands in order to install the algorithm package

```shell
cd functions
tar xzvf gsl-latest.tar.gz
cd gsl-2.7.1
./configure --prefix=TARGET PATH
make
make install
```

Then edit `~/.bashrc` by adding

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:TARGET PATH/lib/
```

Now we back to `functions` directory and run

```shell
python setup.py build_ext --inplace
```

Then the algorithm is installed as Cython package. A simple way to test if it is successfully installed is to run below in the root directory.

```shell
python test.py
```



## Acknowledgement

The underlining training part is based on [ALIGNN](https://github.com/usnistgov/alignn) [2] and the incomplete Bessel Function is based on [ScaFaCoS](https://github.com/scafacos/scafacos) [3].



## Reference

[1] Crandall, R. E. (1998). Fast evaluation of Epstein zeta functions. 

[2] Choudhary, K. and DeCost, B. (2021). Atomistic line graph neural network for improved materials property predictions. *npj Computational Materials*, *7*(1), p.185.

[3] Sutmann, G. (2014). ScaFaCoS–A Scalable library of Fast Coulomb Solvers for particle Systems.
