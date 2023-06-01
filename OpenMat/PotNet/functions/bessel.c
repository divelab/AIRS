#include <math.h>
#include <stdio.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_lambert.h>


/* Codes are based on https://github.com/scafacos/scafacos */

/* Compute the incomplete Bessel-K function of order nu according to the paper
   Richard M. Slevinsky and Hassan Safouhi. 2010.
   A recursive algorithm for the G transformation and accurate computation of incomplete Bessel functions.
   Appl. Numer. Math. 60, 12 (December 2010), 1411-1417.
   http://dx.doi.org/10.1016/j.apnum.2010.04.005
*/

#define FLOAT_PREC 1.0e-14


int is_equal(double x, double y) {
  return (fabs(x-y) < FLOAT_PREC);
}

int is_zero(double x) {
  return (fabs(x) < FLOAT_PREC);
}

/*************************************************/
/* wrappers for Bessel K and inc. */
/*************************************************/

double upper_bessel_k(double nu, double x, double y, double eps) {
    if (is_zero(x)) {
        return pow(y, -nu) * (gsl_sf_gamma(nu) - gsl_sf_gamma_inc(nu, y));
    }

    int n = 2, n_max = 127;
    double err = 1.0, val_new, val_old;
    double N[4], D[4];

    if (-21 <= nu && nu <= 21) {
        if (x > 111) return 0.0;
        if (x < y) if (x * y > 58.0 * 58.0) return 0.0;
    } else {
        const double bound = 1e-50;

        if (nu >= -1) {
            if (x > gsl_sf_lambert_W0(1 / bound)) return 0.0;
        } else {
            double fak = 1.0;
            for (int t = 1; t < -nu; t++)
                fak *= t;

            if (fak * exp(1 - x) * pow(x, -1) < bound) return 0.0;
        }
    }

    if (is_zero(y))
        return pow(x, nu) * gsl_sf_gamma_inc(-nu, x);

    if (is_zero(nu)) {
        if (pow(x, 2) + pow(y, 2) < pow(0.75, 2)) {
            int k = 0;
            double fak = 1.0;
            double z = 0.0;
            while (exp(-x) * pow(y, k + 1) / (x * (k + 1) * fak) > eps) {
                z += pow(-1, k) * pow(x * y, k) * gsl_sf_gamma_inc(-k, x) / fak;
                k += 1;
                fak *= k;
            }
            return z;
        }
    }

    N[0] = 0.0;
    N[1] = 1.0;
    N[2] = 0.5 * (x + nu + 3.0 - y) * N[1];
    N[3] = (x + nu + 5.0 - y) * N[2] + (2.0 * y - nu - 2.0) * N[1];
    N[3] = N[3] / 3.0;

    D[0] = exp(x + y);
    D[1] = (x + nu + 1.0 - y) * D[0];
    D[2] = 0.5 * (x + nu + 3.0 - y) * D[1] + 0.5 * (2.0 * y - nu - 1.0) * D[0];
    D[3] = (x + nu + 5.0 - y) * D[2] + (2.0 * y - nu - 2.0) * D[1] - y * D[0];
    D[3] = D[3] / 3.0;

    val_old = N[2] / D[2];
    val_new = N[3] / D[3];

    err = fabs(val_new - val_old);

    while (err > eps) {

        if (fabs(val_new) < eps)
            break;

        if (n >= n_max) {
            break;
        }
        n++;

        val_old = val_new;

        N[0] = N[1];
        N[1] = N[2];
        N[2] = N[3];

        D[0] = D[1];
        D[1] = D[2];
        D[2] = D[3];

        N[3] = (x + nu + 1 + 2 * n - y) * N[2] + (2 * y - nu - n) * N[1] - y * N[0];
        N[3] = N[3] / (n + 1);

        D[3] = (x + nu + 1 + 2 * n - y) * D[2] + (2 * y - nu - n) * D[1] - y * D[0];
        D[3] = D[3] / (n + 1);

        val_new = N[3] / D[3];

        if (isnan(val_new)) {
            val_new = val_old;
            break;
        }

        if (is_zero(val_new)) {
            val_new = val_old;
            break;
        }

        if (isinf(val_new)) {
            val_new = val_old;
            break;
        }

        err = fabs(val_new - val_old);
    }

    return val_new;
}

/*************************************************/
/* wrappers for Bessel K and low. */
/*************************************************/

double lower_bessel_k(double nu, double x, double y, double eps) {
    return upper_bessel_k(-nu, y, x, eps);
}

