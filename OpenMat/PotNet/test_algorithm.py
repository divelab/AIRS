from algorithm import *

if __name__ == '__main__':
    x = zeta(np.array([[0.0, 0.0]]),
             np.array([[1.0, 0.0], [0.0, 1.0]]), eps=1e-12, param=2.0, R=4, verbose=True)
    print(x)

    x = exp(np.array([[0.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), eps=1e-12,
            param=1.0, R=2, verbose=True)
    print(x)

    x = screened_coulomb(np.array([[0.0, 0.5, 0.0]]), np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                         param=1.0, R=3, eps=1e-12)
    print(x)