# cython: language_level=2
cimport cython
import numpy


cdef extern from "header.h":
    double upper_bessel_k(double, double, double, double);
    double lower_bessel_k(double, double, double, double);


cdef extern from "gsl/gsl_sf_gamma.h":
    double gsl_sf_gamma_inc(double, double);
    double gsl_sf_gamma(double)

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_gammainc(a, x):
    cdef int num_vectors
    num_vectors = len(x)
    results = numpy.zeros(num_vectors, dtype=numpy.double)
    x = numpy.clip(x, a_min=0, a_max=5e2)
    for i in range(num_vectors):
        results[i] = gsl_sf_gamma_inc(a, x[i])
    return results

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_upper_bessel_k(nu, x, y, eps):
    return upper_bessel_k(nu, x, y, eps)

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_lower_bessel_k(nu, x, y, eps):
    return lower_bessel_k(nu, x, y, eps)

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_gsl_sf_gamma(p):
    return gsl_sf_gamma(p)

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_gsl_sf_gamma_inc(p, x):
    return gsl_sf_gamma_inc(p, x)

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_upper_bessel(nu, x, y, eps):
    cdef int num_vectors
    num_vectors = len(x)
    results = numpy.zeros(num_vectors, dtype=numpy.double)
    for i in range(num_vectors):
        results[i] = upper_bessel_k(nu, x[i], y, eps)
    return results
