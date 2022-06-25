from copy import deepcopy
from typing import Callable

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
    ):

        self.C = C
        self.kernel = kernel
        self.kernel_func = self._get_kernel_func(kernel)
        self.tol = tol
        self.alphas = None  # lagrange multipliers
        self.E = None
        self.w = None
        self.b = None

    def fit(self, X, y) -> tuple:
        """
        Implements the training algorithm.

        Args:
            X ((n, m)-shaped array): design matrix
            y ((n,)-shaped array): labels

        Returns:
            A tuple containing:
                - the Lagrange multipliers
                - the intercept
                - the weights
        """
        n, d = X.shape

        # initialize shared parameters
        self.w = np.zeros((d,))
        self.b = 0.0
        self.E = -deepcopy(y)  # E = u - y, but u = 0 in the initial iteration
        self.alphas = np.zeros((n,))

        kernel_matrix = self.kernel_func(X, X)

        num_changed = 0
        examine_all = 1
        while num_changed > 0 or examine_all == 1:
            num_changed = 0

            if examine_all == 1:
                for i in range(n):  # loop over all training examples
                    num_changed += self.examine_example(i, X, y, kernel_matrix)

            else:  # loop over all non-zeros and non-C alphas
                for i in (
                    np.where(self.alphas != 0) and np.where(self.alphas != self.C)
                )[0]:
                    num_changed += self.examine_example(i, X, y, kernel_matrix)

            if examine_all == 1:
                examine_all = 0

            elif num_changed == 0:
                examine_all = 1

        return self.alphas, self.b, self.w

    def examine_example(
        self, i2: int, X: np.array, y: np.array, kernel_matrix: np.array
    ) -> int:
        """Choose the second alpha to optimize according to the second choice heuristic."""
        y_2 = y[i2]
        alpha_2 = self.alphas[i2]
        E_2 = self.E[i2]
        r_2 = E_2 * y_2

        # find positions of non-bound examples
        non_bound = np.where(self.alphas != 0) and np.where(self.alphas != self.C)
        # if error is within tolerance
        if (r_2 < -self.tol and alpha_2 < self.C) or (r_2 > self.tol and alpha_2 > 0):
            # if number of non-zero and non-C alpha is greater than 1
            # use second choice heuristic
            if len(non_bound[0]) > 1:
                if E_2 > 0:
                    # choose minimum error if E2 is positive
                    i1 = np.argmin(self.E)
                elif E_2 < 0:
                    # choose maximum error if E2 is negative
                    i1 = np.argmax(self.E)

                if self.take_step(i1, i2, X, y, kernel_matrix):
                    return 1

            # loop over all non-zero and non-C alpha starting at a random point
            random_index = non_bound[0]
            random_index = random_index[np.random.permutation(len(random_index))]
            for i in random_index:
                # choose identity of current alpha
                if self.alphas[i] == self.alphas[i2]:
                    i1 = i
                    if self.take_step(i1, i2, X, y, kernel_matrix):
                        return 1
            # loop over all possible indices starting at a random point
            temp = np.arange(0, len(self.alphas))
            temp = temp[np.random.permutation(len(temp))]
            for i in temp:
                # loop variable
                i1 = i
                if self.take_step(i1, i2, X, y, kernel_matrix):
                    return 1
        return 0

    def take_step(
        self, i1: int, i2: int, X: np.array, y: np.array, kernel_matrix: np.array
    ) -> bool:
        """Updates threshold, weight vector (if kernel is linear), error cache, alphas."""
        # if alphas are the same return 0
        if i1 == i2:
            return False

        # get label, lagrange multiplier and error for elements in position i1, i2
        alpha1, alpha2 = self.alphas[i1], self.alphas[i2]
        y1, y2 = y[i1], y[i2]
        E1, E2 = self.E[i1], self.E[i2]

        s = y1 * y2

        # compute L, H depending on the values of the labels y1, y2
        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)

        if L == H:
            return False

        # the second derivative of the objective function along the diagonal line can be expressed as
        k11 = kernel_matrix[i1, i1]
        k12 = kernel_matrix[i1, i2]
        k22 = kernel_matrix[i2, i2]
        eta = k11 + k22 - 2 * k12
        # if the second derivative is positive, update alpha2
        # using the following formula
        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2) / eta
            # clip a2 according to bounds
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            # evaluate the objective function at each end of the line
            # segment
            f1 = y1 * (E1 + self.b) - alpha1 * k11 - s * alpha2 * k12
            f2 = y2 * (E2 + self.b) - s * alpha1 * k12 - alpha2 * k22
            L1 = alpha1 + s * (alpha2 - L)
            H1 = alpha1 + s * (alpha2 - H)
            # objective function at a2 = L
            obj_L = (
                L1 * f1
                + L * f2
                + (1 / 2) * (L1**2) * k11
                + (1 / 2) * (L**2) * k22
                + s * L * L1 * k12
            )
            # objective function at a2 = H
            obj_H = (
                H1 * f1
                + H * f2
                + (1 / 2) * (H1**2) * k11
                + (1 / 2) * (H**2) * k22
                + s * H * H1 * k12
            )
            if obj_L < (obj_H - self.tol):
                a2 = L
            elif obj_L > (obj_H + self.tol):
                a2 = H
            else:
                a2 = alpha2

        # if a2 very close to zero or C set a to 0 or C respectively
        if a2 < (10 ** (-8)):
            a2 = 0.0
        elif a2 > self.C - (10**-8):
            a2 = self.C
        # if difference between a_new and a_old is negligible return 0
        if abs(a2 - alpha2) < (self.tol * (a2 + alpha2 + self.tol)):
            return False

        # compute new alpha1
        a1 = alpha1 + s * (alpha2 - a2)

        # compute threshold b
        b1 = E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + self.b
        b2 = E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + self.b

        # According to Platt's SMO paper 2.3 threshold b is b1 if a1 is not in
        #  bounds, b2 is when a2 is not at bounds and (b1+b2)/2
        # when both a1, a2 are at bounds.

        if 0 < a1 < self.C:
            beta_new = b1
        elif 0 < a2 < self.C:
            beta_new = b2
        else:
            beta_new = (b1 + b2) / 2

        t1 = y1 * (a1 - alpha1)
        t2 = y2 * (a2 - alpha2)

        # update weight vector to reflect change in alpha1 and alpha2 if SVM is linear
        if self.kernel == "linear":
            self.w = self.w + t1 * X[i1, :] + t2 * X[i2, :]

        delta_b = self.b - beta_new

        # update error cache using new lagrange multipliers
        for i in range(len(self.E)):
            self.E[i] = (
                self.E[i]
                + t1 * kernel_matrix[i1, i]
                + t2 * kernel_matrix[i2, i]
                + delta_b
            )

        # set error of optimized examples to 0 if new alphas are not at bounds
        for index in [i1, i2]:
            if 0.0 < self.alphas[index] < self.C:
                self.E[index] = 0.0

        # update threshold to reflect change in lagrange multipliers
        self.b = beta_new
        # store a1 in the alpha array
        self.alphas[i1] = a1
        # store a2 in the alpha array
        self.alphas[i2] = a2

        return True

    def _get_kernel_func(self, kernel: str) -> Callable:

        kernel_mapping = {
            "linear": self.linear_kernel,
        }

        return kernel_mapping[kernel]

    @staticmethod
    def linear_kernel(x1, x2):
        return np.dot(x1, x2.T)
