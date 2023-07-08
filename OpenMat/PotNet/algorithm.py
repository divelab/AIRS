import itertools

import numpy as np

from functions.series import cython_upper_bessel, cython_upper_bessel_k, cython_gsl_sf_gamma, cython_gsl_sf_gamma_inc

from joblib import delayed, Parallel

NUM_CPUS = 32

def zeta_cal(v, w, vecs, vecs_inv, d, det, p=1.0, eps=1e-12):
    result = sum(
        np.e ** (2 * np.pi * 1.0j * vecs @ w) * cython_upper_bessel(-p, np.linalg.norm(vecs + v, axis=1) ** 2, 0,
                                                                    eps)
        + np.e ** (2 * np.pi * 1.0j * v @ w) / det * np.pi ** (d / 2) * np.e ** (
                -2 * np.pi * 1.0j * vecs_inv @ v)
        * cython_upper_bessel(p - d / 2, np.pi ** 2 * np.linalg.norm(vecs_inv + w, axis=1) ** 2, 0, eps))

    if (v == 0).all():
        result = result - 1.0 / p
    else:
        result = result + cython_upper_bessel_k(-p, np.linalg.norm(v) ** 2, 0, eps)

    if (w == 0).all():
        result = result - np.pi ** (d / 2) / ((d / 2 - p) * det)
    else:
        result = result + np.e ** (2 * np.pi * 1.0j * v @ w) / det * np.pi ** (d / 2) \
                 * cython_upper_bessel_k(p - d / 2, np.pi ** 2 * np.linalg.norm(w) ** 2, 0, eps)
    return result


def epstein(v, w, Omega, param=1.0, R=3, eps=1e-12, parallel=False, verbose=False):
    d = Omega.shape[0]

    assert len(np.shape(v)) == len(np.shape(w))
    if len(np.shape(v)) == 1:
        v = [v]
        w = [w]

    v = np.array(v, dtype=np.double)
    w = np.array(w, dtype=np.double)

    num_vectors = v.shape[0]

    # normalization
    det = np.linalg.det(Omega)
    assert det > 0

    gamma_norm = det ** (1.0 / d)
    Omega = Omega / gamma_norm
    Omega_inv = np.linalg.inv(Omega).T
    v = v / gamma_norm
    w = w * gamma_norm
    det = 1.0

    gamma_p = cython_gsl_sf_gamma(param)

    products = np.array([l for l in itertools.product(*[list(range(-R, R + 1)) for _ in range(d)]) if any(l)])

    vecs = products @ Omega
    vecs_inv = products @ Omega_inv

    if verbose:
        for i in range(num_vectors):
            rounds = np.array([l for l in itertools.product(*[list(range(-1, 2)) for _ in range(d)]) if any(l)])
            _, s1, _ = np.linalg.svd(Omega)
            minor_minus1 = np.clip(s1[-1] * R - np.linalg.norm(v[i]), a_min=0, a_max=np.inf) ** 2
            error_radius1 = np.sqrt(minor_minus1)
            rho1 = np.min(np.linalg.norm(rounds @ Omega, axis=1))
            error1 = d / 2 * (2 / rho1) ** d * cython_gsl_sf_gamma_inc(d / 2, (error_radius1 - rho1 / 2) ** 2)

            _, s2, _ = np.linalg.svd(Omega_inv)
            minor_minus2 = np.clip(s2[-1] * R - np.linalg.norm(w[i]), a_min=0, a_max=np.inf) ** 2
            error_radius2 = np.sqrt(np.pi ** 2 * minor_minus2)
            rho2 = np.pi * np.min(np.linalg.norm(rounds @ Omega_inv, axis=1))
            error2 = d / 2 * (2 / rho2) ** d * cython_gsl_sf_gamma_inc(d / 2, (error_radius2 - rho2 / 2) ** 2)
            print("Error upper bound for " + str(i) + " vector is " + str(error1 + error2))

    if not parallel:
        values = np.array(
            [zeta_cal(v[i], w[i], vecs, vecs_inv, d, det, p=param, eps=eps).real for i in range(num_vectors)],
            dtype=np.double)
    else:
        values = np.array(Parallel(n_jobs=NUM_CPUS)(
            delayed(zeta_cal)(v[i], w[i], vecs, vecs_inv, d, det, param, eps) for i in range(num_vectors)))

    return values * gamma_norm ** (-2 * param) / gamma_p


# Coulomb
def zeta(v, Omega, param=1.0, R=3, eps=1e-12, parallel=False, verbose=False):
    return epstein(v, np.zeros_like(v), Omega, param=param, R=R, eps=eps, parallel=parallel, verbose=verbose)


def exp_cal(v, vecs, vecs_inv, d, det, B, eps=1e-12):
    return sum(np.e ** (2 * np.pi * 1.0j * vecs_inv @ v) / det *
               cython_upper_bessel(- 0.5 - d / 2, B + np.pi * np.linalg.norm(vecs_inv, axis=1) ** 2, 0, eps) +
               cython_upper_bessel(0.5, np.pi * np.linalg.norm(vecs + v, axis=1) ** 2, B, eps)).real


# Pauli
def exp(v, Omega, param=1.0, R=3, eps=1e-12, parallel=False, verbose=False):
    d = Omega.shape[0]

    if len(np.shape(v)) == 1:
        v = [v]

    v = np.array(v, dtype=np.double)
    num_vectors = v.shape[0]

    det = np.linalg.det(Omega)
    assert det > 0

    gamma_norm = det ** (1.0 / d)
    Omega = Omega / gamma_norm
    Omega_inv = np.linalg.inv(Omega).T
    v = v / gamma_norm
    det = 1.0

    param = param * np.sqrt(gamma_norm)

    products = np.array([l for l in itertools.product(*[list(range(-R, R + 1)) for _ in range(d)])])

    vecs = products @ Omega
    vecs_inv = products @ Omega_inv

    if verbose:
        for i in range(num_vectors):
            rounds = np.array([l for l in itertools.product(*[list(range(-1, 2)) for _ in range(d)]) if any(l)])
            _, s1, _ = np.linalg.svd(Omega)
            minor_minus1 = np.clip(s1[-1] * R - np.linalg.norm(v[i]), a_min=0, a_max=np.inf) ** 2
            error_radius1 = np.sqrt(np.pi * minor_minus1)
            rho1 = np.sqrt(np.pi) * np.min(np.linalg.norm(rounds @ Omega, axis=1))
            error1 = d / 2 * (2 / rho1) ** d * cython_gsl_sf_gamma_inc(d / 2, (error_radius1 - rho1 / 2) ** 2)

            _, s2, _ = np.linalg.svd(Omega_inv)
            minor_minus2 = np.clip(s2[-1] * R, a_min=0, a_max=np.inf) ** 2
            error_radius2 = np.sqrt(np.pi * minor_minus2)
            rho2 = np.sqrt(np.pi) * np.min(np.linalg.norm(rounds @ Omega_inv, axis=1))
            error2 = d / 2 * (2 / rho2) ** d * cython_gsl_sf_gamma_inc(d / 2, (error_radius2 - rho2 / 2) ** 2)
            print("Error upper bound for " + str(i) + " vector is " + str(error1 + error2))

    B = param ** 2 / 4.0 / np.pi
    if not parallel:
        values = np.array([exp_cal(v[i], vecs, vecs_inv, d, det, B, eps) for i in range(num_vectors)], dtype=np.double)
    else:
        values = np.array(Parallel(n_jobs=NUM_CPUS)(
            delayed(exp_cal)(v[i], vecs, vecs_inv, d, det, B, eps) for i in range(num_vectors)))

    return values * param / 2.0 / np.pi

# TODO: Add error bound approximation for LJ potential
def lj(v, Omega, param=1.0, R=3, eps=1e-12, parallel=False, verbose=False):
    if verbose:
        raise NotImplementedError("Error bound for LJ potential is not implemented yet")
    return param ** 12 * zeta(v, Omega, param=6.0, R=R, eps=eps, parallel=parallel) - \
           param ** 6 * zeta(v, Omega, param=3.0, R=R, eps=eps, parallel=parallel)

# TODO: Add error bound approximation for morse potential
def morse(v, Omega, param=1.0, re=1.0, R=3, eps=1e-12, parallel=False, verbose=False):
    if verbose:
        raise NotImplementedError("Error bound for morse potential is not implemented yet")
    return np.exp(2.0 * param * re) * exp(v, Omega, param=2.0 * param, R=R, eps=eps, parallel=parallel) - \
           2.0 * np.exp(param * re) * exp(v, Omega, param=param, R=R, eps=eps, parallel=parallel)


def screened_coulomb_cal(v, vecs, vecs_inv, d, det, B, eps=1e-12):
    result = sum(np.e ** (2 * np.pi * 1.0j * vecs_inv @ v) * np.pi ** (d / 2) / det *
                 cython_upper_bessel(0.5 - d / 2, B + np.pi ** 2 * np.linalg.norm(vecs_inv, axis=1) ** 2, 0, eps) +
                 cython_upper_bessel(-0.5, np.linalg.norm(vecs + v, axis=1) ** 2, B, eps)).real
    if (v == 0).all():
        result = result + B ** 0.5 * (cython_gsl_sf_gamma(-0.5) - cython_gsl_sf_gamma_inc(-0.5, B))
    else:
        result = result + cython_upper_bessel_k(-0.5, np.linalg.norm(v) ** 2, B, eps)
    return result

# TODO: Add error bound approximation for screened coulomb potential
def screened_coulomb(v, Omega, param=1.0, R=3, eps=1e-12, parallel=False, verbose=False):
    d = Omega.shape[0]

    if len(np.shape(v)) == 1:
        v = [v]

    v = np.array(v, dtype=np.double)
    num_vectors = v.shape[0]

    det = np.linalg.det(Omega)
    assert det > 0

    gamma_norm = det ** (1.0 / d)
    Omega = Omega / gamma_norm
    Omega_inv = np.linalg.inv(Omega).T
    v = v / gamma_norm
    det = 1.0

    param = param * np.sqrt(gamma_norm)

    products = np.array([l for l in itertools.product(*[list(range(-R, R + 1)) for _ in range(d)]) if any(l)])

    vecs = products @ Omega
    vecs_inv = products @ Omega_inv

    B = param ** 2

    if verbose:
        raise NotImplementedError("Error bound for screened coulomb potential is not implemented yet")

    if not parallel:
        values = np.array([screened_coulomb_cal(v[i], vecs, vecs_inv, d, det, B, eps) for i in range(num_vectors)],
                          dtype=np.double)
    else:
        values = np.array(Parallel(n_jobs=NUM_CPUS)(
            delayed(screened_coulomb_cal)(v[i], vecs, vecs_inv, d, det, B, eps) for i in range(num_vectors)))

    return values / np.sqrt(np.pi) / np.sqrt(gamma_norm)
