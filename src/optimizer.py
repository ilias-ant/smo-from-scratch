import random
from typing import Callable, Union

import numpy as np


class SMO(object):
    """
    Support Vector Machine that uses the
    Sequential Minimal Optimization (SMO)
    algorithm for training.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "linear",
        tol: float = 1e-3,
        max_iter: int = 10000,
    ):

        self.C = C
        self.kernel = kernel
        self.kernel_func = self._get_kernel_func(kernel)
        self.tol = tol
        self.max_iter = max_iter
        self.w = None
        self.b = None

    def fit(self, X, y) -> Union[tuple, None]:

        n, d = X.shape
        alpha = np.zeros((n,))

        count = 0
        while True:
            count += 1
            alpha_prev = np.copy(alpha)
            for j in range(0, n):
                i = self.get_random_int(0, n - 1, j)
                x_i, x_j, y_i, y_j = X[i, :], X[j, :], y[i], y[j]
                # the second derivative of the objective function along the diagonal line can be expressed as
                k_ij = (
                    self.kernel_func(x_i, x_i)
                    + self.kernel_func(x_j, x_j)
                    - 2 * self.kernel_func(x_i, x_j)
                )
                if k_ij == 0:
                    continue
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                L = self.compute_L(alpha_prime_j, alpha_prime_i, y_j, y_i)
                H = self.compute_H(alpha_prime_j, alpha_prime_i, y_j, y_i)
                # Compute model parameters
                self.w = self.calc_w(alpha, y, X)
                self.b = self.calc_b(X, y, self.w)

                E_i = self.E(
                    x_i, y_i, self.w, self.b
                )  # error on the i-th training example
                E_j = self.E(
                    x_j, y_j, self.w, self.b
                )  # error on the j-th training example

                # under normal circumstances, the objective function will be positive definite thus there will be a
                # minimum along the direction of the linear equality constraint, and η will be greater than zero. In
                # this case, SMO computes the minimum along the direction of the constraint.
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j)) / k_ij

                # the constrained minimum is found by clipping the unconstrained minimum to the ends of the line segment
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                # the value of a1 is computed from the new, clipped a2
                alpha[i] = alpha_prime_i + y_i * y_j * (alpha_prime_j - alpha[j])

            # check for convergence
            if np.linalg.norm(alpha - alpha_prev) < self.tol:
                break

            if count >= self.max_iter:
                print(
                    f"Iteration number exceeded the max of {self.max_iter} iterations"
                )
                return

            # compute final model parameters
            self.b = self.calc_b(X, y, self.w)

            if self.kernel == "linear":
                self.w = self.calc_w(alpha, y, X)
            else:
                raise ValueError(
                    f"Optimizer does not yet support kernel: {self.kernel}"
                )

            return self.b, self.w

    def _get_kernel_func(self, kernel: str) -> Callable:

        kernel_mapping = {
            "linear": self.linear_kernel,
        }

        return kernel_mapping[kernel]

    @staticmethod
    def linear_kernel(x1, x2):
        return np.dot(x1, x2.T)

    @staticmethod
    def get_random_int(a: int, b: int, z: int) -> int:
        i = z
        cnt = 0
        while i == z and cnt < 1000:
            i = random.randint(a, b)
            cnt = cnt + 1
        return i

    def E(self, x_k, y_k, w, b) -> int:
        return self.h(x_k, w, b) - y_k

    @staticmethod
    def h(X, w, b) -> int:
        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    def compute_L(self, alpha_prime_j, alpha_prime_i, y_j, y_i) -> float:
        """Computes lower bound for Lagrange multiplier `alpha_prime_j`."""
        return (
            max(0, alpha_prime_j - alpha_prime_i)
            if (y_i != y_j)
            else max(0, alpha_prime_j + alpha_prime_i - self.C)
        )

    def compute_H(self, alpha_prime_j, alpha_prime_i, y_j, y_i) -> float:
        """Computes upper bound for Lagrange multiplier `alpha_prime_j`."""
        return (
            min(self.C, self.C + alpha_prime_j - alpha_prime_i)
            if (y_i != y_j)
            else min(self.C, alpha_prime_i + alpha_prime_j)
        )

    @staticmethod
    def calc_b(X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)

    @staticmethod
    def calc_w(alpha, y, X):
        return np.dot(X.T, np.multiply(alpha, y))
