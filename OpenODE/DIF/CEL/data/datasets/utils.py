import numpy as np
from scipy.linalg import solve
from typing import Tuple

# Code from: https://github.com/smrfeld/Total-Variation-Regularization-Derivative-Python/blob/main/python/diff_tvr.py
class DiffTVR:

    def __init__(self, t, alpha):
        """Differentiate with TVR.
        Args:
            n (int): Number of points in data.
            dx (float): Spacing of data.
        """
        self.n = t.shape[0]
        self.dx = t[1] - t[0]
        self.alpha = alpha

        self.d_mat = self._make_d_mat()
        self.a_mat = self._make_a_mat()
        self.a_mat_t = self._make_a_mat_t()

    def _make_d_mat(self) -> np.array:
        """Make differentiation matrix with central differences. NOTE: not efficient!
        Returns:
            np.array: N x N+1
        """
        arr = np.zeros((self.n, self.n + 1))
        for i in range(0, self.n):
            arr[i, i] = -1.0
            arr[i, i + 1] = 1.0
        return arr / self.dx

    def _make_a_mat(self) -> np.array:
        """Make integration matrix with trapezoidal rule. NOTE: not efficient!
        Returns:
            np.array: N x N+1
        """
        arr = np.zeros((self.n + 1, self.n + 1))
        for i in range(0, self.n + 1):
            if i == 0:
                continue
            for j in range(0, self.n + 1):
                if j == 0:
                    arr[i, j] = 0.5
                elif j < i:
                    arr[i, j] = 1.0
                elif i == j:
                    arr[i, j] = 0.5

        return arr[1:] * self.dx

    def _make_a_mat_t(self) -> np.array:
        """Transpose of the integration matirx with trapezoidal rule. NOTE: not efficient!
        Returns:
            np.array: N+1 x N
        """
        smat = np.ones((self.n + 1, self.n))

        cmat = np.zeros((self.n, self.n))
        li = np.tril_indices(self.n)
        cmat[li] = 1.0

        dmat = np.diag(np.full(self.n, 0.5))

        vec = np.array([np.full(self.n, 0.5)])
        combmat = np.concatenate((vec, cmat - dmat))

        return (smat - combmat) * self.dx

    def make_en_mat(self, deriv_curr: np.array) -> np.array:
        """Diffusion matrix
        Args:
            deriv_curr (np.array): Current derivative of length N+1
        Returns:
            np.array: N x N
        """
        eps = pow(10, -6)
        vec = 1.0 / np.sqrt(pow(self.d_mat @ deriv_curr, 2) + eps)
        return np.diag(vec)

    def make_ln_mat(self, en_mat: np.array) -> np.array:
        """Diffusivity term
        Args:
            en_mat (np.array): Result from make_en_mat
        Returns:
            np.array: N+1 x N+1
        """
        return self.dx * np.transpose(self.d_mat) @ en_mat @ self.d_mat

    def make_gn_vec(self, deriv_curr: np.array, data: np.array, alpha: float, ln_mat: np.array) -> np.array:
        """Negative right hand side of linear problem
        Args:
            deriv_curr (np.array): Current derivative of size N+1
            data (np.array): Data of size N
            alpha (float): Regularization parameter
            ln_mat (np.array): Diffusivity term from make_ln_mat
        Returns:
            np.array: Vector of length N+1
        """
        return self.a_mat_t @ self.a_mat @ deriv_curr - self.a_mat_t @ (data - data[0]) + alpha * ln_mat @ deriv_curr

    def make_hn_mat(self, alpha: float, ln_mat: np.array) -> np.array:
        """Matrix in linear problem
        Args:
            alpha (float): Regularization parameter
            ln_mat (np.array): Diffusivity term from make_ln_mat
        Returns:
            np.array: N+1 x N+1
        """
        return self.a_mat_t @ self.a_mat + alpha * ln_mat

    def get_deriv_tvr_update(self, data: np.array, deriv_curr: np.array, alpha: float) -> np.array:
        """Get the TVR update
        Args:
            data (np.array): Data of size N
            deriv_curr (np.array): Current deriv of size N+1
            alpha (float): Regularization parameter
        Returns:
            np.array: Update vector of size N+1
        """

        n = len(data)

        en_mat = self.make_en_mat(
            deriv_curr=deriv_curr
        )

        ln_mat = self.make_ln_mat(
            en_mat=en_mat
        )

        hn_mat = self.make_hn_mat(
            alpha=alpha,
            ln_mat=ln_mat
        )

        gn_vec = self.make_gn_vec(
            deriv_curr=deriv_curr,
            data=data,
            alpha=alpha,
            ln_mat=ln_mat
        )

        return solve(hn_mat, -gn_vec)

    def get_deriv_tvr(self,
                      data: np.array,
                      deriv_guess: np.array,
                      alpha: float,
                      no_opt_steps: int,
                      return_progress: bool = False,
                      return_interval: int = 1
                      ) -> Tuple[np.array, np.array]:
        """Get derivative via TVR over optimization steps
        Args:
            data (np.array): Data of size N
            deriv_guess (np.array): Guess for derivative of size N+1
            alpha (float): Regularization parameter
            no_opt_steps (int): No. opt steps to run
            return_progress (bool, optional): True to return derivative progress during optimization. Defaults to False.
            return_interval (int, optional): Interval at which to store derivative if returning. Defaults to 1.
        Returns:
            Tuple[np.array,np.array]: First is the final derivative of size N+1, second is the stored derivatives if return_progress=True of size no_opt_steps+1 x N+1, else [].
        """

        deriv_curr = deriv_guess

        if return_progress:
            deriv_st = np.full((no_opt_steps + 1, len(deriv_guess)), 0)
        else:
            deriv_st = np.array([])

        for opt_step in range(0, no_opt_steps):
            update = self.get_deriv_tvr_update(
                data=data,
                deriv_curr=deriv_curr,
                alpha=alpha
            )

            deriv_curr += update

            if return_progress:
                if opt_step % return_interval == 0:
                    deriv_st[int(opt_step / return_interval)] = deriv_curr

        return (deriv_curr, deriv_st)

    def _differentiate(self, y, t):
        dy = []
        for j in range(y.shape[1]):
            (deriv, _) = self.get_deriv_tvr(
                data=y[:, j],
                deriv_guess=np.full(self.n + 1, 0.0),
                alpha=self.alpha,
                no_opt_steps=100
            )
            dy.append(deriv[:-1])

        dy = np.stack(dy, axis=-1)
        return dy